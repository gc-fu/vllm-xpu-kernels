#include <sycl/sycl.hpp>
#include <torch/all.h>

#include "utils.h"
#include "dispatch_utils.h"

#include "causal_conv1d.hpp"
#include "gated_delta_rule.hpp"
#ifdef VLLM_XPU_ENABLE_XE2
  #include "xe_2/chunk_causal_conv1d_xe2.hpp"
  #include "xe_2/chunk_gated_delta_rule_xe2.h"
#endif

// Forward declaration
void gdn_attention(
    torch::Tensor& core_attn_out,
    torch::Tensor& z,
    const torch::Tensor& projected_states_qkvz,
    const torch::Tensor& projected_states_ba,
    const int64_t num_k_heads,
    const int64_t num_v_heads,
    const int64_t head_k_dim,
    const int64_t head_v_dim,
    torch::Tensor& conv_state,
    torch::Tensor& ssm_state,
    const torch::Tensor& conv_weights,
    const std::optional<torch::Tensor>& conv_bias,
    const std::string& activation,
    const torch::Tensor& A_log,
    const torch::Tensor& dt_bias,
    const int64_t num_prefills,
    const int64_t num_decodes,
    const std::optional<torch::Tensor>& has_initial_state,
    const torch::Tensor& non_spec_query_start_loc,
    const torch::Tensor& non_spec_state_indices_tensor,
    const int64_t num_actual_tokens,
    const int64_t tp_size);

// ============================================================
// Qwen3.5 rearrangement kernel:
// Converts separate [qkv, z, b, a] projections to
// GQA-interleaved [projected_states_qkvz, projected_states_ba]
// format expected by the causal_conv1d kernel.
// ============================================================
namespace gdn {

template <typename T>
struct rearrange_qwen35_kernel {
 public:
  static constexpr int group_size = 256;

  rearrange_qwen35_kernel(
      T* qkvz_out,
      T* ba_out,
      const T* mixed_qkv,
      const T* z_in,
      const T* b_in,
      const T* a_in,
      const int num_tokens,
      const int num_k_heads,
      const int head_k_dim,
      const int num_v_heads,
      const int head_v_dim,
      const int q_dim_total,
      const int k_dim_total,
      const int v_dim_total,
      const int z_dim_total,
      const int v_per_group,
      const int qkvz_group_dim,
      const int qkvz_out_stride,
      const int ba_out_stride)
      : qkvz_out(qkvz_out),
        ba_out(ba_out),
        mixed_qkv(mixed_qkv),
        z_in(z_in),
        b_in(b_in),
        a_in(a_in),
        num_tokens(num_tokens),
        num_k_heads(num_k_heads),
        head_k_dim(head_k_dim),
        num_v_heads(num_v_heads),
        head_v_dim(head_v_dim),
        q_dim_total(q_dim_total),
        k_dim_total(k_dim_total),
        v_dim_total(v_dim_total),
        z_dim_total(z_dim_total),
        v_per_group(v_per_group),
        qkvz_group_dim(qkvz_group_dim),
        qkvz_out_stride(qkvz_out_stride),
        ba_out_stride(ba_out_stride) {}

  void operator()(sycl::nd_item<2> item) const {
    const int token_id = item.get_group(0);
    const int local_id = item.get_local_linear_id();
    const int group_id = item.get_group(1);
    const int global_elem = group_id * group_size + local_id;

    if (token_id >= num_tokens) return;

    const int total_qkvz_elems = qkvz_out_stride;
    const int total_ba_elems = ba_out_stride;
    const int total_work = total_qkvz_elems + total_ba_elems;

    if (global_elem >= total_work) return;

    if (global_elem < total_qkvz_elems) {
      // Rearrange into qkvz interleaved format
      // Per k_head group: [q(head_k_dim), k(head_k_dim),
      //   v(v_per_group*head_v_dim), z(v_per_group*head_v_dim)]
      const int kg = global_elem / qkvz_group_dim;
      const int dim_in_group = global_elem % qkvz_group_dim;

      T val;
      if (dim_in_group < head_k_dim) {
        // q region
        val = mixed_qkv[token_id * (q_dim_total + k_dim_total + v_dim_total) +
                         kg * head_k_dim + dim_in_group];
      } else if (dim_in_group < 2 * head_k_dim) {
        // k region
        val = mixed_qkv[token_id * (q_dim_total + k_dim_total + v_dim_total) +
                         q_dim_total + kg * head_k_dim +
                         (dim_in_group - head_k_dim)];
      } else if (dim_in_group < 2 * head_k_dim + v_per_group * head_v_dim) {
        // v region
        int v_offset = dim_in_group - 2 * head_k_dim;
        val = mixed_qkv[token_id * (q_dim_total + k_dim_total + v_dim_total) +
                         q_dim_total + k_dim_total +
                         kg * v_per_group * head_v_dim + v_offset];
      } else {
        // z region
        int z_offset =
            dim_in_group - (2 * head_k_dim + v_per_group * head_v_dim);
        val = z_in[token_id * z_dim_total +
                   kg * v_per_group * head_v_dim + z_offset];
      }
      qkvz_out[token_id * qkvz_out_stride + global_elem] = val;
    } else {
      // Rearrange into ba interleaved format
      // Per k_head group: [b(v_per_group), a(v_per_group)]
      int ba_elem = global_elem - total_qkvz_elems;
      int kg = ba_elem / (2 * v_per_group);
      int dim_in_group = ba_elem % (2 * v_per_group);

      T val;
      if (dim_in_group < v_per_group) {
        val = b_in[token_id * num_v_heads + kg * v_per_group + dim_in_group];
      } else {
        val = a_in[token_id * num_v_heads +
                   kg * v_per_group + (dim_in_group - v_per_group)];
      }
      ba_out[token_id * ba_out_stride + ba_elem] = val;
    }
  }

 private:
  T* qkvz_out;
  T* ba_out;
  const T* mixed_qkv;
  const T* z_in;
  const T* b_in;
  const T* a_in;
  const int num_tokens;
  const int num_k_heads;
  const int head_k_dim;
  const int num_v_heads;
  const int head_v_dim;
  const int q_dim_total;
  const int k_dim_total;
  const int v_dim_total;
  const int z_dim_total;
  const int v_per_group;
  const int qkvz_group_dim;
  const int qkvz_out_stride;
  const int ba_out_stride;
};

template <typename T>
void rearrange_qwen35_launch(
    sycl::queue& queue,
    T* qkvz_out,
    T* ba_out,
    const T* mixed_qkv,
    const T* z_in,
    const T* b_in,
    const T* a_in,
    const int num_tokens,
    const int num_k_heads,
    const int head_k_dim,
    const int num_v_heads,
    const int head_v_dim) {
  const int v_per_group = num_v_heads / num_k_heads;
  const int q_dim_total = num_k_heads * head_k_dim;
  const int k_dim_total = num_k_heads * head_k_dim;
  const int v_dim_total = num_v_heads * head_v_dim;
  const int z_dim_total = num_v_heads * head_v_dim;
  const int qkvz_group_dim =
      2 * head_k_dim + 2 * v_per_group * head_v_dim;
  const int qkvz_out_stride = num_k_heads * qkvz_group_dim;
  const int ba_out_stride = 2 * num_v_heads;

  constexpr int group_size = rearrange_qwen35_kernel<T>::group_size;
  const int total_work = qkvz_out_stride + ba_out_stride;
  const int num_groups = (total_work + group_size - 1) / group_size;

  sycl::range<2> local(1, group_size);
  sycl::range<2> global(num_tokens, num_groups);
  auto nd_range = sycl::nd_range<2>(global * local, local);

  queue.submit([&](sycl::handler& cgh) {
    rearrange_qwen35_kernel<T> task(
        qkvz_out, ba_out, mixed_qkv, z_in, b_in, a_in,
        num_tokens, num_k_heads, head_k_dim, num_v_heads, head_v_dim,
        q_dim_total, k_dim_total, v_dim_total, z_dim_total,
        v_per_group, qkvz_group_dim, qkvz_out_stride, ba_out_stride);
    cgh.parallel_for(nd_range, task);
  });
}

}  // namespace gdn

void gdn_attention_qwen3_5(
    torch::Tensor&
        core_attn_out,  // [total_seqlen, num_v_heads / tp_size, head_v_dim]
    torch::Tensor& z_out,  // [total_seqlen, num_v_heads / tp_size, head_v_dim]
    const torch::Tensor&
        mixed_qkv,  // [total_seqlen, (q_dim + k_dim + v_dim) / tp_size]
    const torch::Tensor&
        z_in,  // [total_seqlen, v_dim / tp_size]
    const torch::Tensor&
        b_in,  // [total_seqlen, num_v_heads / tp_size]
    const torch::Tensor&
        a_in,  // [total_seqlen, num_v_heads / tp_size]
    const int64_t num_k_heads,
    const int64_t num_v_heads,
    const int64_t head_k_dim,
    const int64_t head_v_dim,
    torch::Tensor& conv_state,
    torch::Tensor& ssm_state,
    const torch::Tensor& conv_weights,
    const std::optional<torch::Tensor>& conv_bias,
    const std::string& activation,
    const torch::Tensor& A_log,
    const torch::Tensor& dt_bias,
    const int64_t num_prefills,
    const int64_t num_decodes,
    const std::optional<torch::Tensor>& has_initial_state,
    const torch::Tensor& non_spec_query_start_loc,
    const torch::Tensor& non_spec_state_indices_tensor,
    const int64_t num_actual_tokens,
    const int64_t tp_size) {
  // Validate inputs
  TORCH_CHECK(core_attn_out.is_contiguous());
  TORCH_CHECK(z_out.is_contiguous());
  TORCH_CHECK(mixed_qkv.is_contiguous());
  TORCH_CHECK(z_in.is_contiguous());
  TORCH_CHECK(b_in.is_contiguous());
  TORCH_CHECK(a_in.is_contiguous());

  auto& queue = vllm::xpu::vllmGetQueue();
  auto dtype = mixed_qkv.dtype();
  auto device = mixed_qkv.device();

  const int64_t nk = num_k_heads / tp_size;
  const int64_t nv = num_v_heads / tp_size;
  const int64_t v_per_group = num_v_heads / num_k_heads;
  const int64_t qkvz_dim =
      nk * (2 * head_k_dim + 2 * v_per_group * head_v_dim);
  const int64_t ba_dim = 2 * nv;

  // Allocate interleaved buffers
  torch::Tensor projected_states_qkvz = torch::empty(
      {num_actual_tokens, qkvz_dim},
      torch::dtype(dtype).device(device).requires_grad(false));
  torch::Tensor projected_states_ba = torch::empty(
      {num_actual_tokens, ba_dim},
      torch::dtype(dtype).device(device).requires_grad(false));

  // Launch rearrangement kernel
  if (dtype == at::kBFloat16) {
    using scalar_t = sycl::ext::oneapi::bfloat16;
    gdn::rearrange_qwen35_launch<scalar_t>(
        queue,
        reinterpret_cast<scalar_t*>(projected_states_qkvz.data_ptr()),
        reinterpret_cast<scalar_t*>(projected_states_ba.data_ptr()),
        reinterpret_cast<scalar_t*>(mixed_qkv.data_ptr()),
        reinterpret_cast<scalar_t*>(z_in.data_ptr()),
        reinterpret_cast<scalar_t*>(b_in.data_ptr()),
        reinterpret_cast<scalar_t*>(a_in.data_ptr()),
        num_actual_tokens, nk, head_k_dim, nv, head_v_dim);
  } else if (dtype == at::kHalf) {
    using scalar_t = sycl::half;
    gdn::rearrange_qwen35_launch<scalar_t>(
        queue,
        reinterpret_cast<scalar_t*>(projected_states_qkvz.data_ptr()),
        reinterpret_cast<scalar_t*>(projected_states_ba.data_ptr()),
        reinterpret_cast<scalar_t*>(mixed_qkv.data_ptr()),
        reinterpret_cast<scalar_t*>(z_in.data_ptr()),
        reinterpret_cast<scalar_t*>(b_in.data_ptr()),
        reinterpret_cast<scalar_t*>(a_in.data_ptr()),
        num_actual_tokens, nk, head_k_dim, nv, head_v_dim);
  } else {
    using scalar_t = float;
    gdn::rearrange_qwen35_launch<scalar_t>(
        queue,
        reinterpret_cast<scalar_t*>(projected_states_qkvz.data_ptr()),
        reinterpret_cast<scalar_t*>(projected_states_ba.data_ptr()),
        reinterpret_cast<scalar_t*>(mixed_qkv.data_ptr()),
        reinterpret_cast<scalar_t*>(z_in.data_ptr()),
        reinterpret_cast<scalar_t*>(b_in.data_ptr()),
        reinterpret_cast<scalar_t*>(a_in.data_ptr()),
        num_actual_tokens, nk, head_k_dim, nv, head_v_dim);
  }

  // Now delegate to the original gdn_attention with interleaved format
  gdn_attention(
      core_attn_out,
      z_out,
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
      tp_size);
}

void gdn_attention(
    torch::Tensor&
        core_attn_out,  // [total_seqlen, num_v_heads / tp_size, head_v_dim]
    torch::Tensor& z,   // [total_seqlen, num_v_heads / tp_size, head_v_dim]
    const torch::Tensor&
        projected_states_qkvz,  // [total_seqlen, num_k_heads / tp_size * (2 *
                                // head_k_dim + 2 * head_v_dim * num_v_heads /
                                // num_k_heads)]
    const torch::Tensor&
        projected_states_ba,  // [total_seqlen, num_k_heads / tp_size * (2 *
                              // num_v_heads / num_k_heads)]
    const int64_t num_k_heads,
    const int64_t num_v_heads,
    const int64_t head_k_dim,
    const int64_t head_v_dim,
    torch::Tensor&
        conv_state,  // [cache_batch_size, width - 1, num_k_heads / tp_size * (2
                     // * head_k_dim + head_v_dim * num_v_heads / num_k_heads)]
    torch::Tensor& ssm_state,  // [cache_batch_size, num_v_heads / tp_size,
                               // head_v_dim, head_k_dim]
    const torch::Tensor&
        conv_weights,  // [num_k_heads / tp_size * (2 * head_k_dim + head_v_dim
                       // * num_v_heads / num_k_heads), width]
    const std::optional<torch::Tensor>&
        conv_bias,  // [num_k_heads / tp_size * (2 * head_k_dim + head_v_dim *
                    // num_v_heads / num_k_heads)] or None
    const std::string& activation,
    const torch::Tensor& A_log,    // [num_v_heads / tp_size]
    const torch::Tensor& dt_bias,  // [num_v_heads / tp_size]
    const int64_t num_prefills,
    const int64_t num_decodes,
    const std::optional<torch::Tensor>&
        has_initial_state,                               // [batch_size] or None
    const torch::Tensor& non_spec_query_start_loc,       // [batch_size + 1]
    const torch::Tensor& non_spec_state_indices_tensor,  // [batch_size]
    const int64_t num_actual_tokens,
    const int64_t tp_size) {
  TORCH_CHECK(
      core_attn_out.is_contiguous(), "core_attn_out must be contiguous");
  TORCH_CHECK(z.is_contiguous(), "z must be contiguous");
  TORCH_CHECK(
      projected_states_qkvz.is_contiguous(),
      "projected_states_qkvz must be contiguous");
  TORCH_CHECK(
      projected_states_ba.is_contiguous(),
      "projected_states_ba must be contiguous");
  TORCH_CHECK(
      conv_state[0].is_contiguous(),
      "conv_state of each batch must be contiguous");
  TORCH_CHECK(
      ssm_state[0].is_contiguous(),
      "ssm_state of each batch must be contiguous");
  TORCH_CHECK(conv_weights.is_contiguous(), "conv_weights must be contiguous");
  TORCH_CHECK(A_log.is_contiguous(), "A_log must be contiguous");
  TORCH_CHECK(dt_bias.is_contiguous(), "dt_bias must be contiguous");
  TORCH_CHECK(
      non_spec_query_start_loc.is_contiguous(),
      "non_spec_query_start_loc must be contiguous");
  TORCH_CHECK(
      non_spec_state_indices_tensor.is_contiguous(),
      "non_spec_state_indices_tensor must be contiguous");

  // check core_attn_out shape
  TORCH_CHECK(core_attn_out.size(0) == num_actual_tokens);
  TORCH_CHECK(core_attn_out.size(1) == num_v_heads / tp_size);
  TORCH_CHECK(core_attn_out.size(2) == head_v_dim);

  // check z shape
  TORCH_CHECK(z.size(0) == core_attn_out.size(0));
  TORCH_CHECK(z.size(1) == core_attn_out.size(1));
  TORCH_CHECK(z.size(2) == core_attn_out.size(2));

  // check projected_states_qkvz shape
  TORCH_CHECK(projected_states_qkvz.size(0) == num_actual_tokens);
  TORCH_CHECK(
      projected_states_qkvz.size(1) ==
      num_k_heads / tp_size *
          (2 * head_k_dim + 2 * head_v_dim * num_v_heads / num_k_heads));

  // check projected_states_ba shape
  TORCH_CHECK(projected_states_ba.size(0) == num_actual_tokens);
  TORCH_CHECK(projected_states_ba.size(1) == 2 * num_v_heads / tp_size);

  auto& queue = vllm::xpu::vllmGetQueue();
  auto dtype = projected_states_qkvz.dtype();
  auto device = projected_states_qkvz.device();
  gdn::ActMode act_mode;

  if (activation == "silu") {
    act_mode = gdn::ActMode::silu;
  } else if (activation == "swish") {
    act_mode = gdn::ActMode::swish;
  } else {
    TORCH_CHECK(false);
  }
  const int pad_slot_id = -1;

#define NATIVE_LAUNCHER                                           \
  do {                                                            \
    torch::Tensor q = torch::empty(                               \
        {num_actual_tokens, num_k_heads / tp_size, head_k_dim},   \
        torch::dtype(dtype).device(device).requires_grad(false)); \
    torch::Tensor k = torch::empty(                               \
        {num_actual_tokens, num_k_heads / tp_size, head_k_dim},   \
        torch::dtype(dtype).device(device).requires_grad(false)); \
    torch::Tensor v = torch::empty(                               \
        {num_actual_tokens, num_v_heads / tp_size, head_v_dim},   \
        torch::dtype(dtype).device(device).requires_grad(false)); \
    torch::Tensor b = torch::empty(                               \
        {num_actual_tokens, num_v_heads / tp_size},               \
        torch::dtype(dtype).device(device).requires_grad(false)); \
    torch::Tensor a = torch::empty(                               \
        {num_actual_tokens, num_v_heads / tp_size},               \
        torch::dtype(dtype).device(device).requires_grad(false)); \
    gdn::causal_conv1d(                                           \
        queue,                                                    \
        q,                                                        \
        k,                                                        \
        v,                                                        \
        z,                                                        \
        b,                                                        \
        a,                                                        \
        projected_states_qkvz,                                    \
        projected_states_ba,                                      \
        conv_weights,                                             \
        conv_bias,                                                \
        conv_state,                                               \
        non_spec_query_start_loc,                                 \
        non_spec_state_indices_tensor,                            \
        has_initial_state,                                        \
        act_mode,                                                 \
        pad_slot_id,                                              \
        num_prefills,                                             \
        num_decodes);                                             \
    gdn::gated_delta_rule(                                        \
        queue,                                                    \
        core_attn_out,                                            \
        q,                                                        \
        k,                                                        \
        v,                                                        \
        b,                                                        \
        a,                                                        \
        A_log,                                                    \
        dt_bias,                                                  \
        ssm_state,                                                \
        non_spec_query_start_loc,                                 \
        non_spec_state_indices_tensor,                            \
        has_initial_state,                                        \
        num_prefills,                                             \
        num_decodes);                                             \
  } while (0)

#ifdef VLLM_XPU_ENABLE_XE2
  if (num_prefills > 0) {
    int batch_size = non_spec_query_start_loc.size(0) - 1;
    int padding_size = batch_size * (gdn::chunk_size_xe2 - 1);

    torch::Tensor q = torch::zeros(
        {num_actual_tokens + padding_size, num_k_heads / tp_size, head_k_dim},
        torch::dtype(dtype).device(device).requires_grad(false));
    torch::Tensor k = torch::zeros(
        {num_actual_tokens + padding_size, num_k_heads / tp_size, head_k_dim},
        torch::dtype(dtype).device(device).requires_grad(false));
    torch::Tensor v = torch::zeros(
        {num_actual_tokens + padding_size, num_v_heads / tp_size, head_v_dim},
        torch::dtype(dtype).device(device).requires_grad(false));
    torch::Tensor b = torch::zeros(
        {num_v_heads / tp_size, num_actual_tokens + padding_size},
        torch::dtype(torch::kFloat32).device(device).requires_grad(false));
    torch::Tensor a = torch::zeros(
        {num_v_heads / tp_size, num_actual_tokens + padding_size},
        torch::dtype(torch::kFloat32).device(device).requires_grad(false));

    gdn::chunk_causal_conv1d_xe2(
        queue,
        q,
        k,
        v,
        z,
        b,
        a,
        projected_states_qkvz,
        projected_states_ba,
        conv_weights,
        conv_bias,
        conv_state,
        non_spec_query_start_loc,
        non_spec_state_indices_tensor,
        has_initial_state,
        act_mode,
        pad_slot_id,
        num_prefills,
        num_decodes);

    chunk_gated_delta_rule_xe2(
        queue,
        core_attn_out,
        q,
        k,
        v,
        b,
        a,
        A_log,
        dt_bias,
        ssm_state,
        non_spec_query_start_loc,
        non_spec_state_indices_tensor,
        has_initial_state,
        num_prefills,
        num_decodes);
  } else {
    NATIVE_LAUNCHER;
  }
#else
  NATIVE_LAUNCHER;
#endif
#undef NATIVE_LAUNCHER
}