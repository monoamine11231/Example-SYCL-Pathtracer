#include <iostream>
#include <string>
#include "include/variantcontainer.h"


class A {};

int main() {
    using var = std::variant<int, std::string>;

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

    container.forEach([](auto a){ std::cout << a << std::endl; });

    std::cout << std::endl << std::endl;

    for (int i = 0; i < 10; i++) {
        container.use_at(i, [](auto a) { std::cout << a << std::endl;});
    }
    return 0;
}