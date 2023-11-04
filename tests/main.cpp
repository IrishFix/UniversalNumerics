#include <UniversalNumerics/Tensor.h>
#include <string>
#include <iostream>

int main() {
    Tensor<double> Test({5}, 10);
    std::cout << std::to_string(Test({5})) << std::endl;
    return 0;
}