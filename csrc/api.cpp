#include "mfa/api.h"
#include <torch/python.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "MiniFlashAttention";
    m.def("flash_attention_v2", &mfa::flash_attention_v2, "Forward pass");
}
