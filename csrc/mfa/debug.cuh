#pragma once

#include <stdio.h>

namespace mfa {

template<typename Tensor>
__device__ void print(const char* message, const Tensor& tensor, int rows, int cols) {
    printf("%s:\n", message);
    float sum = 0.0f;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {

            printf("%4.2f ", static_cast<float>(tensor(i, j)));
            sum += static_cast<float>(tensor(i, j));
        }
        printf("\n");
    }
    printf("Sum: %f\n\n", sum);
}

template<typename Tensor>
__device__ void print(const char* message, const Tensor& tensor, int cols) {
    printf("%s:\n", message);
    float sum = 0.0f;
    for (int j = 0; j < cols; j++) {
        printf("%4.2f ", static_cast<float>(tensor(j)));
        sum += static_cast<float>(tensor(j));
    }
    printf("\n");
    printf("Sum: %f\n\n", sum);
}

} // namespace mfa
