#pragma once

#include <torch/types.h>

namespace mfa {

std::vector<at::Tensor> flash_attention_forward(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v);

} // namespace mfa