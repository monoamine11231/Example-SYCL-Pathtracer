#include <iostream>
#include <optional>
#include <string>

#include "include/variantcontainer.h"

int main() {
  using var = std::variant<int, float>;

  VariantContainer<var> container{};
  container.push_back(12);
  container.push_back(22);
  container.push_back(32);
  container.push_back(42);

  container.push_back(12.2f);
  container.push_back(22.5f);
  container.push_back(32.22f);
  container.push_back(42.555f);

  float sum = 0.0f;
  auto a = [&sum](const auto &a) { sum += a; };

  container.forEach(a);

  std::cout << sum << std::endl;
  return 0;
}