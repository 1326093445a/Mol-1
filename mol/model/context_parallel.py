# Copyright 2026 Molence Bio
# SPDX-License-Identifier: Apache-2.0
"""Inference context parallelism for one complex, with replicated stage boundaries.

Pairformer pair rows and diffusion query tokens are partitioned across ranks.
Triangle products stream peer projections; ending attention transposes the
partition by exchanging row slabs. This deliberately does not claim end-to-end
Fold-CP memory scaling: weights, stage inputs, atom networks and caches remain
replicated. No checkpoint parameters or state-dict names are changed.
"""

from contextlib import contextmanager
from contextvars import ContextVar

import torch
import torch.distributed as dist
import torch.nn.functional as F


_ACTIVE = ContextVar("mol_context_parallel", default=None)


def active_context():
    return _ACTIVE.get()


class ParallelContext:
    def __init__(self, group=None):
        if not dist.is_initialized():
            raise ValueError(
                "Context parallelism requires an initialized process group"
            )
        self.group = group
        self.rank = dist.get_rank(group)
        self.size = dist.get_world_size(group)
        if self.size < 2:
            raise ValueError("Context parallelism requires at least two ranks")
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if dist.get_backend(group) == "nccl"
            else torch.device("cpu")
        )
        self.status = torch.zeros((), dtype=torch.int32, device=device)
        self.stats = {"pairformer_calls": 0, "diffusion_calls": 0}

    def local(self, fn):
        """Propagate rank-local failures before any peer enters the next exchange."""
        result, error = None, None
        try:
            result = fn()
        except Exception as exc:
            error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
            exc.__traceback__ = None
        self.status.fill_(error is not None)
        dist.all_reduce(self.status, op=dist.ReduceOp.MAX, group=self.group)
        if self.status.item():
            errors = [None] * self.size
            dist.all_gather_object(errors, error, group=self.group)
            raise RuntimeError(
                "Context-parallel operation failed: "
                + "; ".join(e for e in errors if e)
            )
        return result

    def extent(self, n, rank=None):
        width = (n + self.size - 1) // self.size
        start = (self.rank if rank is None else rank) * width
        return start, max(0, min(width, n - start)), width

    def shard(self, x, dim, n):
        start, count, width = self.extent(n)
        dim %= x.ndim
        part = x.narrow(dim, min(start, n), count)
        pad = [0, 0] * x.ndim
        pad[2 * (x.ndim - dim - 1) + 1] = width - count
        return F.pad(part, pad).contiguous()

    def gather(self, x, dim, n):
        buffers = self.local(lambda: [torch.empty_like(x) for _ in range(self.size)])
        dist.all_gather(buffers, x.contiguous(), group=self.group)
        return self.local(
            lambda: torch.cat(buffers, dim=dim).narrow(dim, 0, n).contiguous()
        )

    def peers(self, x):
        buffer = self.local(lambda: torch.empty_like(x))
        for rank in range(self.size):
            if rank == self.rank:
                buffer.copy_(x)
            source = (
                rank if self.group is None else dist.get_global_rank(self.group, rank)
            )
            dist.broadcast(buffer, src=source, group=self.group)
            yield rank, buffer

    def transpose_pair(self, x, n):
        """[local rows, all columns, C] -> rows of the transposed global pair."""
        start, count, _ = self.extent(n)
        result = self.local(lambda: torch.zeros_like(x))
        for rank, peer in self.peers(x):
            other_start, other_count, _ = self.extent(n, rank)

            def assign():
                if count and other_count:
                    result[..., :count, other_start : other_start + other_count, :] = (
                        peer[..., :other_count, start : start + count, :].transpose(
                            -3, -2
                        )
                    )

            self.local(assign)
        return result


@contextmanager
def context_parallel(group=None):
    if torch.is_grad_enabled():
        raise ValueError("Context parallelism is inference-only; use torch.no_grad()")
    context = ParallelContext(group)
    token = _ACTIVE.set(context)
    try:
        yield context
    finally:
        _ACTIVE.reset(token)


def _triangle_mult(module, z, mask, ctx, n):
    def project():
        norm = module.layer_norm_in(z)
        a = (
            module.linear_a_p(norm)
            * torch.sigmoid(module.linear_a_g(norm))
            * mask[..., None]
        )
        b = (
            module.linear_b_p(norm)
            * torch.sigmoid(module.linear_b_g(norm))
            * mask[..., None]
        )
        gate = torch.sigmoid(module.linear_g(norm))
        return a, b, gate

    a, b, gate = ctx.local(project)
    start, count, _ = ctx.extent(n)
    product = ctx.local(lambda: torch.zeros_like(a))
    if module._outgoing:
        for rank, peer in ctx.peers(b):
            other_start, other_count, _ = ctx.extent(n, rank)

            def multiply():
                if other_count:
                    tile = a.movedim(-1, -3) @ peer[..., :other_count, :, :].movedim(
                        -1, -3
                    ).transpose(-1, -2)
                    product[..., other_start : other_start + other_count, :] = (
                        tile.movedim(-3, -1)
                    )

            ctx.local(multiply)
    else:
        # Both projections carry the same contracted k slab. Each rank computes
        # only its own output i rows, accumulating contributions from every k.
        packed = ctx.local(lambda: torch.stack((a, b)))
        for rank, peer in ctx.peers(packed):
            _, k_count, _ = ctx.extent(n, rank)

            def multiply():
                if count and k_count:
                    left = (
                        peer[0, ..., :k_count, start : start + count, :]
                        .movedim(-1, -3)
                        .transpose(-1, -2)
                    )
                    right = peer[1, ..., :k_count, :, :].movedim(-1, -3)
                    product[..., :count, :, :].add_((left @ right).movedim(-3, -1))

            ctx.local(multiply)
    return ctx.local(lambda: module.linear_z(module.layer_norm_out(product)) * gate)


def _triangle_attention(module, z, mask, ctx, n, chunk_size):
    norm = ctx.local(lambda: module.layer_norm(z))
    bias = ctx.local(lambda: module.linear(norm))
    bias = ctx.gather(bias, -3, n).movedim(-1, -3).unsqueeze(-4)
    # Bound the materialized attention scores independently of the user's
    # pair-transition chunk size.
    rows = max(
        1, min(chunk_size or z.shape[-3], 32_000_000 // (module.no_heads * n * n))
    )

    def attention():
        output = torch.empty_like(norm)
        for start in range(0, z.shape[-3], rows):
            block = norm[..., start : start + rows, :, :]
            mask_bias = module.inf * (
                mask[..., start : start + rows, None, None, :] - 1
            )
            output[..., start : start + rows, :, :] = module.mha(
                q_x=block,
                kv_x=block,
                biases=[mask_bias, bias],
                triangle_attention="torch",
            )
        return output

    return ctx.local(attention)


def _token_attention(module, a, s, z, ctx, n, *, fusion=False, bias=None, extra=None):
    if module.cross_attention_mode:
        raise ValueError("Context parallelism supports global self attention")
    query = ctx.local(
        lambda: module.layernorm_a(a=a, s=s) if module.has_s else module.layernorm_a(a)
    )
    keys = ctx.gather(query, -2, n)

    def attention():
        result = module.standard_multihead_attention(
            query,
            keys,
            z,
            enable_efficient_fusion=fusion,
            pair_bias=bias,
            extra_attn_bias=extra,
        )
        if module.has_s:
            result = result * torch.sigmoid(module.linear_a_last(s))
        return result

    return ctx.local(attention)


def pairformer_parallel(
    stack, s, z, mask, chunk_size, mixer_chunk, former_chunk, extra
):
    ctx = active_context()
    if stack.training:
        raise ValueError("Context parallelism requires model.eval()")
    n = z.shape[-2]
    ctx.stats["pairformer_calls"] += 1
    ctx.stats["last_pair_rows"] = ctx.extent(n)[2]

    def prepare():
        valid = z.new_ones(z.shape[:-1]) if mask is None else mask
        return (
            None if s is None else ctx.shard(s, -2, n),
            ctx.shard(z, -3, n),
            ctx.shard(valid, -2, n),
            None if extra is None else ctx.shard(extra, -2, n),
        )

    s, z, mask, extra = ctx.local(prepare)
    for index, block in enumerate(stack.blocks):
        if index in stack._adapter_before_block:
            adapter = stack.adapters[stack._adapter_before_block[index]]
            s, z = ctx.local(lambda: adapter(s, z))
        update = _triangle_mult(block.tri_mul_out, z, mask, ctx, n)
        z = ctx.local(lambda: z + update)
        update = _triangle_mult(block.tri_mul_in, z, mask, ctx, n)
        z = ctx.local(lambda: z + update)
        if hasattr(block, "tri_att_start"):
            update = _triangle_attention(
                block.tri_att_start, z, mask, ctx, n, chunk_size
            )
            z = ctx.local(lambda: z + update)
            z = ctx.transpose_pair(z, n)
            transposed_mask = ctx.transpose_pair(mask.unsqueeze(-1), n).squeeze(-1)
            update = _triangle_attention(
                block.tri_att_end, z, transposed_mask, ctx, n, chunk_size
            )
            z = ctx.local(lambda: z + update)
            z = ctx.transpose_pair(z, n)
        transition_chunk = (
            mixer_chunk if index < stack.n_pairmixer_blocks else former_chunk
        )
        if transition_chunk is None:
            transition_chunk = chunk_size
        z = ctx.local(lambda: z + block.pair_transition(z, chunk_size=transition_chunk))
        if block.c_s > 0:
            update = _token_attention(
                block.attention_pair_bias, s, None, z, ctx, n, extra=extra
            )
            s = ctx.local(lambda: s + update)
            s = ctx.local(lambda: s + block.single_transition(s))
    if stack._output_adapter_name is not None:
        s, z = ctx.local(lambda: stack.adapters[stack._output_adapter_name](s, z))
    return None if s is None else ctx.gather(s, -2, n), ctx.gather(z, -3, n)


def diffusion_parallel(stack, a, s, z, fusion, projections, extra):
    ctx = active_context()
    if stack.training:
        raise ValueError("Context parallelism requires model.eval()")
    n = a.shape[-2]
    ctx.stats["diffusion_calls"] += 1
    ctx.stats["last_diffusion_queries"] = ctx.extent(n)[2]

    def prepare():
        return (
            ctx.shard(a, -2, n),
            ctx.shard(s, -2, n),
            ctx.shard(z, -2 if fusion else -3, n),
            None if extra is None else ctx.shard(extra, -2, n),
        )

    a, s, z, extra = ctx.local(prepare)
    for index, block in enumerate(stack.blocks):
        bias = ctx.local(
            lambda: (
                None
                if projections is None
                else ctx.shard(projections.blocks[index], -2, n)
            )
        )
        update = _token_attention(
            block.attention_pair_bias,
            a,
            s,
            z,
            ctx,
            n,
            fusion=fusion,
            bias=bias,
            extra=extra,
        )
        a = ctx.local(lambda: a + update)
        a = ctx.local(lambda: a + block.conditioned_transition_block(a=a, s=s))
    return ctx.gather(a, -2, n)
