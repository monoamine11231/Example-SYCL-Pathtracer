#include <iostream>
#include <string>
#include "include/VariantContainer.hpp"


class A {};

int main() {
    using var = std::variant<int, A, std::string>;

    VariantContainer<var> container{};
    container.push_back(12);
    container.push_back(22);
    container.push_back(32);
    container.push_back(42);

    container.push_back(std::string("aa"));
    container.push_back(std::string("ba"));
    container.push_back(std::string("ca"));
    container.push_back(std::string("da"));
    container.push_back(std::string("ea"));
    container.push_back(std::string("fa"));

    for (auto it = container.begin<std::string>(); it != container.end<std::string>(); it++) {
        std::cout << *it << std::endl;
    }

    return 0;
}