# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test fp32 accumulator fix for ssm_state across chunks.

Validates that the chunk_fwd_o_kernel produces correct results (no NaN,
matches reference) for long sequences where fp16 accumulation would overflow.
Specifically targets batch_size=100 and batch_size=256 with seq_len=8192.
"""

import math
import random

import pytest
import torch
import torch.nn.functional as F

import vllm_xpu_kernels._xpu_C  # noqa: F401


NUM_K_HEADS = 16
NUM_V_HEADS = 32
HEAD_K_DIM = 128
HEAD_V_DIM = 128
WIDTH = 4
TP_SIZE = 1
ACTIVATION = "silu"
NUM_ACTUAL_TOKENS = 8192
CACHE_BATCH_SIZE = 300


def ref_gdn_attention(
    core_attn_out,
    z,
    projected_states_qkvz,
    projected_states_ba,
    num_k_heads,
    num_v_heads,
    head_k_dim,
    head_v_dim,
    conv_state,
    ssm_state,
    conv_weights,
    conv_bias,
    activation,
    A_log,
    dt_bias,
    num_prefills,
    num_decodes,
    has_initial_state,
    non_spec_query_start_loc,
    non_spec_state_indices_tensor,
    num_actual_tokens,
    tp_size,
):
    eps = 0.000001
    scale = 1.0 / math.sqrt(head_k_dim)
    dtype = projected_states_qkvz.dtype
    batch_size = non_spec_query_start_loc.shape[0] - 1

    split_arg_list_ba = [
        num_v_heads // num_k_heads,
        num_v_heads // num_k_heads,
    ]
    projected_states_ba = projected_states_ba.reshape(
        num_actual_tokens, num_k_heads // tp_size,
        (2 * num_v_heads // num_k_heads))
    (b, a) = torch.split(projected_states_ba, split_arg_list_ba, dim=-1)
    b = b.reshape(num_actual_tokens, num_v_heads // tp_size)
    a = a.reshape(num_actual_tokens, num_v_heads // tp_size)

    split_arg_list_qkvz = [
        head_k_dim,
        head_k_dim,
        num_v_heads // num_k_heads * head_v_dim,
        num_v_heads // num_k_heads * head_v_dim,
    ]
    projected_states_qkvz = projected_states_qkvz.reshape(
        num_actual_tokens, num_k_heads // tp_size,
        (2 * head_k_dim + 2 * num_v_heads // num_k_heads * head_v_dim))
    (q_split, k_split, v_split, z_split) = torch.split(
        projected_states_qkvz, split_arg_list_qkvz, dim=-1)
    q_split = q_split.reshape(num_actual_tokens,
                              num_k_heads // tp_size * head_k_dim)
    k_split = k_split.reshape(num_actual_tokens,
                              num_k_heads // tp_size * head_k_dim)
    v_split = v_split.reshape(
        num_actual_tokens,
        num_k_heads // tp_size * num_v_heads // num_k_heads * head_v_dim)
    qkv = torch.cat((q_split, k_split, v_split), dim=-1).reshape(
        num_actual_tokens, num_k_heads // tp_size *
        (2 * head_k_dim + num_v_heads // num_k_heads * head_v_dim))
    qkv_elems_size = qkv.shape[-1]
    z.copy_(
        z_split.reshape(num_actual_tokens, num_v_heads // tp_size, head_v_dim))

    A_log_exp = -torch.exp(A_log)
    softplus = torch.nn.Softplus(beta=1.0, threshold=20.0)

    for batch in range(batch_size):
        if has_initial_state[batch]:
            conv_state_batch = conv_state[non_spec_state_indices_tensor[batch]]
        else:
            conv_state_batch = torch.zeros_like(conv_state[0])

        batch_start_id = non_spec_query_start_loc[batch]
        batch_end_id = non_spec_query_start_loc[batch + 1]
        batch_num_tokens = batch_end_id - batch_start_id

        qkv_batch = qkv[batch_start_id:batch_end_id]
        qkv_conv_input = torch.cat([conv_state_batch, qkv_batch], dim=0)
        conv_state[non_spec_state_indices_tensor[batch]] = qkv_conv_input[
            batch_num_tokens:]

        qkv_conv_input = qkv_conv_input.transpose(0, 1).unsqueeze(0)

        qkv_conv_out = F.conv1d(qkv_conv_input.to(torch.float32),
                                conv_weights.unsqueeze(1).to(torch.float32),
                                conv_bias.to(torch.float32)
                                if conv_bias is not None else None,
                                padding=0,
                                groups=qkv_elems_size)
        qkv_conv_out = (qkv_conv_out if activation is None else
                        F.silu(qkv_conv_out)).to(dtype=dtype)
        qkv_conv_out = qkv_conv_out.transpose(-2, -1).reshape(
            batch_num_tokens, qkv_elems_size)

        split_arg_list_qkv = [
            num_k_heads // tp_size * head_k_dim,
            num_k_heads // tp_size * head_k_dim,
            num_k_heads // tp_size * num_v_heads // num_k_heads * head_v_dim,
        ]
        (q_out, k_out, v_out) = torch.split(qkv_conv_out,
                                            split_arg_list_qkv,
                                            dim=-1)
        q_out = q_out.reshape(batch_num_tokens, num_k_heads // tp_size,
                              head_k_dim)
        k_out = k_out.reshape(batch_num_tokens, num_k_heads // tp_size,
                              head_k_dim)
        v_out = v_out.reshape(batch_num_tokens, num_v_heads // tp_size,
                              head_v_dim)

        if has_initial_state[batch]:
            ssm_state_batch = ssm_state[non_spec_state_indices_tensor[
                batch]]  # [num_v_heads // tp_size, head_v_dim, head_k_dim]
        else:
            ssm_state_batch = torch.zeros_like(ssm_state[0])

        for token_id in range(batch_num_tokens):
            b_t = b[batch_start_id + token_id].to(torch.float32)
            beta_t = torch.sigmoid(b_t)
            a_t = a[batch_start_id + token_id].to(torch.float32)
            g_t = torch.exp(A_log_exp * softplus(a_t + dt_bias))

            q_t = q_out[token_id].to(torch.float32)
            k_t = k_out[token_id].to(torch.float32)
            v_t = v_out[token_id].to(torch.float32)

            q_t = q_t / torch.sqrt(
                torch.sum(q_t * q_t, dim=-1) + eps).unsqueeze(-1)
            k_t = k_t / torch.sqrt(
                torch.sum(k_t * k_t, dim=-1) + eps).unsqueeze(-1)
            q_t *= scale

            q_t = torch.repeat_interleave(
                q_t, repeats=num_v_heads // num_k_heads, dim=0)
            k_t = torch.repeat_interleave(
                k_t, repeats=num_v_heads // num_k_heads, dim=0)

            ssm_state_batch *= g_t.unsqueeze(-1).unsqueeze(-1)
            kv_mem_t = (ssm_state_batch * k_t.unsqueeze(1)).sum(dim=-1)
            delta_t = (v_t - kv_mem_t) * beta_t.unsqueeze(-1)
            ssm_state_batch += k_t.unsqueeze(1) * delta_t.unsqueeze(2)

            core_attn_out[batch_start_id + token_id] = (
                ssm_state_batch * q_t.unsqueeze(1)).sum(dim=-1).to(dtype)

        ssm_state[non_spec_state_indices_tensor[batch]] = ssm_state_batch.to(
            ssm_state.dtype)


def simple_random_distribute(N, batch_size):
    distribution = torch.ones([batch_size])
    for i in range(N - batch_size):
        selected_idx = random.randint(0, batch_size - 1)
        distribution[selected_idx] += 1
    return distribution


def run_gdn_test(batch_size, num_actual_tokens, dtype, ssm_state_is_fp32):
    """Run gdn_attention kernel vs reference and return results."""
    device = "xpu"
    random.seed(42)
    torch.manual_seed(42)

    num_k_heads = NUM_K_HEADS
    num_v_heads = NUM_V_HEADS
    head_k_dim = HEAD_K_DIM
    head_v_dim = HEAD_V_DIM
    tp_size = TP_SIZE
    width = WIDTH
    activation = ACTIVATION

    ssm_state_dtype = torch.float32 if ssm_state_is_fp32 else dtype

    if batch_size > num_actual_tokens:
        batch_size = num_actual_tokens

    # All prefill mode (long sequences)
    num_prefills = batch_size
    num_decodes = 0

    mixed_qkvz_size = num_k_heads // tp_size * (
        2 * head_k_dim + 2 * head_v_dim * num_v_heads // num_k_heads)
    mixed_ba_size = num_k_heads // tp_size * (2 * num_v_heads // num_k_heads)

    projected_states_qkvz = torch.randn(
        (num_actual_tokens, mixed_qkvz_size), dtype=dtype, device=device)
    projected_states_ba = torch.randn(
        (num_actual_tokens, mixed_ba_size), dtype=dtype, device=device)

    mixed_qkv_size = num_k_heads // tp_size * (
        2 * head_k_dim + head_v_dim * num_v_heads // num_k_heads)
    conv_state = torch.randn(
        (CACHE_BATCH_SIZE, width - 1, mixed_qkv_size),
        dtype=dtype, device=device)
    ref_conv_state = conv_state.clone()

    ssm_state = torch.randn(
        (CACHE_BATCH_SIZE, num_v_heads // tp_size, head_v_dim, head_k_dim),
        dtype=ssm_state_dtype, device=device)
    ref_ssm_state = ssm_state.clone()

    conv_weights = torch.randn(
        (mixed_qkv_size, width), dtype=dtype, device=device)
    conv_bias = torch.randn((mixed_qkv_size), dtype=dtype, device=device)

    A_log = torch.randn(
        (num_v_heads // tp_size), dtype=torch.float32, device=device)
    dt_bias = torch.randn(
        (num_v_heads // tp_size), dtype=dtype, device=device)

    prefill_batches = simple_random_distribute(
        num_actual_tokens - num_decodes, batch_size - num_decodes)
    token_batches = torch.cat(
        [torch.ones([num_decodes]), prefill_batches]).to(device)
    perm = torch.randperm(token_batches.size(0)).to(device)
    shuffled_tensor = token_batches[perm]
    non_spec_query_start_loc = torch.cat([
        torch.zeros([1], device=device),
        torch.cumsum(shuffled_tensor, dim=0)
    ]).to(torch.int32)
    has_initial_state = perm >= num_decodes
    non_spec_state_indices_tensor = torch.tensor(
        random.sample(range(CACHE_BATCH_SIZE), batch_size),
        device=device, dtype=torch.int32)

    core_attn_out = torch.zeros(
        (num_actual_tokens, num_v_heads // tp_size, head_v_dim),
        dtype=dtype, device=device)
    z = torch.empty_like(core_attn_out)

    # Run kernel under test
    torch.ops._xpu_C.gdn_attention(
        core_attn_out, z,
        projected_states_qkvz.clone(),
        projected_states_ba.clone(),
        num_k_heads, num_v_heads, head_k_dim, head_v_dim,
        conv_state=conv_state, ssm_state=ssm_state,
        conv_weights=conv_weights, conv_bias=conv_bias,
        activation=activation,
        A_log=A_log, dt_bias=dt_bias,
        num_prefills=num_prefills, num_decodes=num_decodes,
        has_initial_state=has_initial_state,
        non_spec_query_start_loc=non_spec_query_start_loc,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        num_actual_tokens=num_actual_tokens, tp_size=tp_size,
        reorder_input=False)

    # Run reference
    ref_core_attn_out = torch.zeros_like(core_attn_out)
    ref_z = torch.empty_like(core_attn_out)

    ref_gdn_attention(
        ref_core_attn_out, ref_z,
        projected_states_qkvz.clone(),
        projected_states_ba.clone(),
        num_k_heads, num_v_heads, head_k_dim, head_v_dim,
        conv_state=ref_conv_state, ssm_state=ref_ssm_state,
        conv_weights=conv_weights, conv_bias=conv_bias,
        activation=activation,
        A_log=A_log, dt_bias=dt_bias,
        num_prefills=num_prefills, num_decodes=num_decodes,
        has_initial_state=has_initial_state,
        non_spec_query_start_loc=non_spec_query_start_loc,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        num_actual_tokens=num_actual_tokens, tp_size=tp_size)

    return {
        "core_attn_out": core_attn_out,
        "ref_core_attn_out": ref_core_attn_out,
        "ssm_state": ssm_state,
        "ref_ssm_state": ref_ssm_state,
        "conv_state": conv_state,
        "ref_conv_state": ref_conv_state,
        "z": z,
        "ref_z": ref_z,
        "non_spec_state_indices_tensor": non_spec_state_indices_tensor,
        "batch_size": batch_size,
    }


@pytest.mark.parametrize("batch_size", [100, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("ssm_state_is_fp32", [False])
@torch.inference_mode()
def test_gdn_fp32_accumulator_no_nan(batch_size, dtype, ssm_state_is_fp32):
    """Verify no NaN in kernel output for long sequences with large batch."""
    results = run_gdn_test(
        batch_size=batch_size,
        num_actual_tokens=NUM_ACTUAL_TOKENS,
        dtype=dtype,
        ssm_state_is_fp32=ssm_state_is_fp32,
    )

    core_attn_out = results["core_attn_out"]
    ssm_state = results["ssm_state"]
    indices = results["non_spec_state_indices_tensor"]

    nan_in_output = torch.isnan(core_attn_out).any().item()
    assert not nan_in_output, (
        f"NaN detected in core_attn_out! "
        f"batch_size={batch_size}, dtype={dtype}, "
        f"num_nan={torch.isnan(core_attn_out).sum().item()}"
    )

    for i in range(results["batch_size"]):
        state_id = indices[i]
        nan_in_state = torch.isnan(ssm_state[state_id]).any().item()
        assert not nan_in_state, (
            f"NaN in ssm_state[{state_id}] for batch {i}! "
            f"batch_size={batch_size}, dtype={dtype}"
        )


@pytest.mark.parametrize("batch_size", [100, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("ssm_state_is_fp32", [False])
@torch.inference_mode()
def test_gdn_fp32_accumulator_accuracy(batch_size, dtype, ssm_state_is_fp32):
    """Verify kernel output matches reference within tolerance."""
    results = run_gdn_test(
        batch_size=batch_size,
        num_actual_tokens=NUM_ACTUAL_TOKENS,
        dtype=dtype,
        ssm_state_is_fp32=ssm_state_is_fp32,
    )

    atol = 5e-2
    rtol = 5e-2

    # Check z (should always match since it's just a reshape/copy)
    torch.testing.assert_close(
        results["z"], results["ref_z"], atol=atol, rtol=rtol)

    # Check core_attn_out
    core_out = results["core_attn_out"]
    ref_out = results["ref_core_attn_out"]
    assert not torch.isnan(core_out).any(), "NaN in core_attn_out"
    torch.testing.assert_close(core_out, ref_out, atol=atol, rtol=rtol)

    # Check ssm_state
    indices = results["non_spec_state_indices_tensor"]
    for i in range(results["batch_size"]):
        state_id = indices[i]
        assert not torch.isnan(results["ssm_state"][state_id]).any(), \
            f"NaN in ssm_state[{state_id}]"
        torch.testing.assert_close(
            results["ssm_state"][state_id],
            results["ref_ssm_state"][state_id],
            atol=atol, rtol=rtol)


if __name__ == "__main__":
    print("=" * 60)
    print("GDN FP32 Accumulator Test")
    print(f"seq_len={NUM_ACTUAL_TOKENS}, heads=k{NUM_K_HEADS}/v{NUM_V_HEADS}, "
          f"dim=k{HEAD_K_DIM}/v{HEAD_V_DIM}")
    print("=" * 60)

    for dtype in [torch.float16, torch.bfloat16]:
        for batch_size in [100, 256]:
            print(f"\n--- batch_size={batch_size}, dtype={dtype} ---")
            try:
                results = run_gdn_test(
                    batch_size=batch_size,
                    num_actual_tokens=NUM_ACTUAL_TOKENS,
                    dtype=dtype,
                    ssm_state_is_fp32=False,
                )
                core_out = results["core_attn_out"]
                ref_out = results["ref_core_attn_out"]
                indices = results["non_spec_state_indices_tensor"]

                nan_out = torch.isnan(core_out).any().item()
                nan_ref = torch.isnan(ref_out).any().item()

                print(f"  NaN in kernel output: {nan_out}")
                print(f"  NaN in reference:     {nan_ref}")

                if not nan_out and not nan_ref:
                    diff = (core_out.float() - ref_out.float()).abs()
                    print(f"  core_attn_out max_diff: {diff.max().item():.6f}")
                    print(f"  core_attn_out mean_diff: {diff.mean().item():.6f}")

                    # Check ssm_state
                    ssm_nan = False
                    ssm_max_diff = 0.0
                    for i in range(results["batch_size"]):
                        sid = indices[i]
                        if torch.isnan(results["ssm_state"][sid]).any():
                            ssm_nan = True
                            break
                        d = (results["ssm_state"][sid].float() -
                             results["ref_ssm_state"][sid].float()).abs().max()
                        ssm_max_diff = max(ssm_max_diff, d.item())

                    print(f"  NaN in ssm_state:     {ssm_nan}")
                    print(f"  ssm_state max_diff:   {ssm_max_diff:.6f}")

                    ok = (not nan_out and not ssm_nan and
                          diff.max().item() < 5e-2 and ssm_max_diff < 5e-2)
                    print(f"  Result: {'PASS' if ok else 'FAIL'}")
                else:
                    print(f"  Result: FAIL (NaN detected)")

            except Exception as e:
                print(f"  Result: ERROR - {e}")

    print("\n" + "=" * 60)
    print("Done.")
