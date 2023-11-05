#include <UniversalNumerics/Tensor.h>
#include <string>
#include <iostream>

int main() {
    Tensor<double> Test({5,2});
    Test({0}) = 10;
    std::cout << std::to_string(Test[0].at(0)) << std::endl;
    return 0;
}