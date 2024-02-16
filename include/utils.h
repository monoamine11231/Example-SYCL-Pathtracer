#ifndef PATHTRACER_INCLUDE_UTILS_H_
#define PATHTRACER_INCLUDE_UTILS_H_

#include <array>
#include <sycl/sycl.hpp>
#include <type_traits>
#include <variant>
#include <vector>

const int kStackVectorCapacity = 32;

namespace vecutils {
template <typename T, std::size_t N>
inline sycl::vec<T, N> Lerp(const sycl::vec<T, N>& a, const sycl::vec<T, N>& b,
                            float r) {
  return a + (b - a) * r;
}

/* Calculates 2 orthogonal vectors parallel to the plane with the normal vector
 * `n` */
template <typename T>
void PlaneVectors(const sycl::vec<T, 3>& n, sycl::vec<T, 3>& u,
                  sycl::vec<T, 3>& v) {
  sycl::vec<T, 3> up = std::fabs(n.z) < 0.999
                           ? sycl::vec<T, 3>{T(0), T(0), T(1)}
                           : sycl::vec<T, 3>{T(1), T(0), T(0)};
  u = sycl::normalize(sycl::cross(up, n));
  v = sycl::cross(n, u);
}
};  // namespace vecutils

namespace miscutils {
int ShadowFactor(float a);
};  // namespace miscutils

namespace containerutils {

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
      func(*std::get_if<I>(obj));
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
    __builtin_unreachable();
  }

  return T_index;
}

/* No except vector that uses the stack and thus cannot be expanded */
template <typename T, std::size_t N>
class StackVector {
 private:
  /* Uninitialized aligned storage for `T` */
  std::aligned_storage_t<sizeof(T), alignof(T)> data_[N];
  std::size_t size_ = 0;

 public:
  SYCL_EXTERNAL StackVector() : size_(0){};

  /* Returns false on unsuccessful operations */
  SYCL_EXTERNAL bool push_back_if(T value) noexcept {
    if (this->size_ >= N) return false;

    this->at(this->size_++) = value;
    return true;
  }

  SYCL_EXTERNAL bool pop_if() noexcept {
    if (this->size_ <= 0) return false;

    --this->size_;
    return true;
  }

  SYCL_EXTERNAL const T& at(std::size_t index) const noexcept {
    return *std::launder(reinterpret_cast<const T*>(&this->data_[index]));
  }
  SYCL_EXTERNAL const T& operator[](std::size_t index) const noexcept {
    return *std::launder(reinterpret_cast<const T*>(&this->data_[index]));
  }

  SYCL_EXTERNAL T& at(std::size_t index) noexcept {
    return *std::launder(reinterpret_cast<T*>(&this->data_[index]));
  }
  SYCL_EXTERNAL T& operator[](std::size_t index) noexcept {
    return *std::launder(reinterpret_cast<T*>(&this->data_[index]));
  }

  SYCL_EXTERNAL std::size_t size() const noexcept { return this->size_; }
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
  StackVector<VARIANT, kStackVectorCapacity>
      data_[std::variant_size_v<VARIANT>];
  // std::vector<std::pair<std::size_t, std::size_t>> index_map_;

 public:
  SYCL_EXTERNAL VariantContainer() : data_{} {};

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
  SYCL_EXTERNAL void forEach(F&& func) const {
    if constexpr (I < std::variant_size_v<VARIANT>) {
      for (std::size_t i = 0; i < this->data_[I].size(); i++) {
        /* We use noexcept `std::get_if` because we can guarantee that variant
         * indexes out of bounds are not possible */
        func(*std::get_if<I>(&(this->data_[I].at(i))));
      }
      forEach<F, I + 1>(std::forward<F>(func));
    } else {
      return;
    }
  }

  template <typename T>
  SYCL_EXTERNAL T at(std::size_t index) const {
    std::size_t T_index = assert_in_variant<VARIANT, T>();

    /* Return the data from the corresponding vector */
    return this->data_[T_index].at(index);
  }

  template <typename F>
  SYCL_EXTERNAL void use_at(std::size_t index, F&& func) const {
    std::pair<std::size_t, std::size_t> index_map = this->index_map_.at(index);
    VARIANT& obj = this->data_[index_map.first].at(index_map.second);
    variant_unwrap<F, VARIANT>(std::forward<F>(func), obj);
  }

  template <typename T>
  SYCL_EXTERNAL void push_back(T value) {
    std::size_t T_index = assert_in_variant<VARIANT, T>();

    /* Insert the value */
    std::size_t target_size = this->data_[T_index].size();
    // std::pair<std::size_t, std::size_t> index_pair(T_index, target_size);
    // this->index_map_.push_back_if(index_pair);
    this->data_[T_index].push_back_if(value);
  }
};
}  // namespace containerutils
#endif