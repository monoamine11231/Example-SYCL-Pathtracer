#include <array>
#include <type_traits>
#include <variant>
#include <vector>

/* BIG cred to https://github.com/nnaumenko for this compile time solution */
/* Note: Nested if-else is needed for this to work */
/*  Returns the index of a type inside the given `std::variant`
    Example:
        `variant_index<decltype(std::variant<float, int>), float>`  -> 0
        `variant_index<decltype(std::variant<float, int>), int>     -> 1
        `variant_index<decltype(std::variant<float, int>), char>`   -> 2
    Note that on the last entry the char is not inside the given `std::variant`
    so it returns the number of elements inside the variant instead  */
template <typename V, typename T, std::size_t I = 0>
static constexpr std::size_t variant_index() {
  if constexpr (I >= std::variant_size_v<V>) {
    return std::variant_size_v<V>;
  } else {
    if constexpr (std::is_same_v<std::variant_alternative_t<I, V>, T>) {
      return I;
    } else {
      return variant_index<V, T, I + 1>();
    }
  }
}

/*  Single object `std::variant` unwrap with generic lambda. If same lambda
   needs to be operated on all items of `VariantContainer` then
   `VariantContainer::forEach` should be used instead. */
template <typename F, typename V, std::size_t I = 0>
static void variant_unwrap(F&& func, V& obj) {
  if constexpr (I < std::variant_size_v<V>) {
    if (obj.index() == I) {
      func(std::get<I>(obj));
    } else {
      variant_unwrap<F, V, I + 1>(std::forward<F>(func), obj);
    }
  } else {
    __builtin_unreachable();
  }
}

/*  Raise out of bounds exception if given type not in variant and return the
    type index in variant otherwise */
template <typename V, typename T>
static constexpr std::size_t assert_in_variant() {
  /* Check if type is defined in the container variant */
  constexpr std::size_t variant_size = std::variant_size_v<V>;
  constexpr std::size_t T_index = variant_index<V, T>();
  if constexpr (T_index >= variant_size) {
    throw std::out_of_range("Unexpected variant type given");
  }

  return T_index;
}

/*  A variant iterator that automatically fetches and returns the value of
    the variant from a vector variant iterator. Works by wrapping a given
    iterator and overwriting the comparison, increment and dereferecing
    operators with respect to the `std::variant` and the choosen unwrapping
    type to be used with `std::get<T>()`.  */
template <std::size_t T, typename VARIANT>
struct VariantIterator {
 public:
  VariantIterator(typename std::vector<VARIANT>::const_iterator it) : it_(it){};

  auto operator*() const { return std::get<T>(*(this->it_)); }

  void operator++(int a) { this->it_++; }

  bool operator==(const VariantIterator& v_it) const { return this->it_ == v_it.it_; }

  bool operator!=(const VariantIterator& v_it) const {
    return !(this->it_ == v_it.it_);
  }

 private:
  /* The target vector iterator */
  typename std::vector<VARIANT>::const_iterator it_;
};

/*  Dynamic polymorphism with virtual functions is not allowed in SYCL and I had
    a terrible experience with trying to implement `std::visit` for
   `std::variant` with a function pointer-free approach. The solution is to
   implement a container that holds a `std::variant` type and to allow all types
   declared inside the given `std::variant` to be pushed or poped from the
   container. Each type inside the given `std::variant` has each own
   `std::vector` attached. All this is done to decrease the number of switch and
   if-else statements (branches) later */
template <typename VARIANT>
class VariantContainer {
 private:
  /* Holding an individual vector for each type */
  std::array<std::vector<VARIANT>, std::variant_size_v<VARIANT>> data_;
  std::vector<std::pair<std::size_t, std::size_t>> index_map_;

 public:
  VariantContainer() {}

  template <typename T>
  auto begin() const {
    constexpr std::size_t T_index = assert_in_variant<VARIANT, T>();
    auto&& beg_it = this->data_.at(T_index).begin();

    return VariantIterator<T_index, VARIANT>(beg_it);
  }

  template <typename T>
  auto end() const {
    constexpr std::size_t T_index = assert_in_variant<VARIANT, T>();
    auto&& end_it = this->data_.at(T_index).end();

    return VariantIterator<T_index, VARIANT>(end_it);
  }

  /*  Because `std::get<T>` in `variant_iterator` has to accept a constexpr
      index we cannot simply use a for loop through to go through all types
      in the given variant and call the function we want. (NOTE that `F func`
      is a lambda function and not a function pointer because SYCL does not
      support that.) So instead of having to write a for loop for each type
      in the given variant manually, we let the compiler do that for us :)

      Example with given variant: `std::variant<std::string, double, int>`:
          `container.forEach(lambda)` expands at compile time to:

          for(auto it = variant_iterator<std::string>; it != end; i++) {
              lambda(*it);
          }

          for(auto it = variant_iterator<double>; it != end; i++) {
              lambda(*it);
          }

          for(auto it = variant_iterator<int>; it != end; i++) {
              lambda(*it);
          }                                                               */
  template <typename F, std::size_t I = 0>
  void forEach(F&& func) const {
    if constexpr (I < std::variant_size_v<VARIANT>) {
      auto&& begin_it = this->data_.at(I).begin();
      auto&& end_it = this->data_.at(I).end();

      auto it = VariantIterator<I, VARIANT>(begin_it);
      auto end = VariantIterator<I, VARIANT>(end_it);
      for (; it != end; it++) {
        func(*it);
      }
      forEach<F, I + 1>(std::forward<F>(func));
    } else {
      return;
    }
  }

  template <typename T>
  T at(std::size_t index) const {
    std::size_t T_index = assert_in_variant<VARIANT, T>();

    /* Return the data from the corresponding vector */
    return this->data_.at(T_index).at(index);
  }

  template <typename F>
  void use_at(std::size_t index, F&& func) const {
    std::pair<std::size_t, std::size_t> index_map = this->index_map_.at(index);
    VARIANT& obj = this->data_.at(index_map.first).at(index_map.second);
    variant_unwrap<F, VARIANT>(std::forward<F>(func), obj);
  }

  template <typename T>
  void push_back(T value) {
    std::size_t T_index = assert_in_variant<VARIANT, T>();

    /* Insert the value */
    std::size_t target_size = this->data_.at(T_index).size();
    std::pair<std::size_t, std::size_t> index_pair(T_index, target_size);
    this->index_map_.push_back(index_pair);
    this->data_.at(T_index).push_back(value);
  }
};