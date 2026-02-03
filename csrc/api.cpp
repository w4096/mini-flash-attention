#include "mfa/api.h"
#include <torch/python.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "mini flash attention";
    m.def("mini_flash_attention_forward", &mfa::flash_attention_forward, "Forward pass");
}
