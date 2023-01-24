#include <vector>
#include <array>
#include <variant>


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
        if constexpr (std::is_same_v<std::variant_alternative_t<I,V>, T>) {
            return I;
        } else {
            return variant_index<V, T, I + 1>();
        }
    }
}

/*  Raise out of bounds exception if given type not in variant and return the 
    type index in variant otherwise */
template <typename V, typename T>
static constexpr std::size_t assert_in_variant() {
    /* Check if type is defined in the container variant */
    constexpr std::size_t variant_size  = std::variant_size_v<V>;
    constexpr std::size_t T_index       = variant_index<V, T>();
    if constexpr(T_index >= variant_size)    {
        throw std::out_of_range("Unexpected variant type given");
    }

    return T_index;
}


/*  A variant iterator that automatically fetches and returns the value of
    the variant from a vector variant iterator. Works by wrapping a given
    iterator and overwriting the comparison, increment and dereferecing 
    operators with respect to the `std::variant` and the choosen unwrapping
    type to be used with `std::get<T>()`.  */
template<std::size_t T, typename VARIANT>
struct variant_iterator {
public:
    variant_iterator(typename std::vector<VARIANT>::iterator it) : _it(it) {};

    auto operator*() {
        return std::get<T>(*(this->_it));
    }

    void operator++(int a) {
        this->_it++;
    }

    bool operator==(const variant_iterator& v_it) {
        return this->_it==v_it._it;
    }

    bool operator!=(const variant_iterator& v_it) {
        return !(this->_it==v_it._it);
    }

private:
    /* The target vector iterator */
    typename std::vector<VARIANT>::iterator _it;
};


/*  Dynamic polymorphism with virtual functions is not allowed in SYCL and I had
    a terrible experience with trying to implement `std::visit` for `std::variant`
    with a function pointer-free approach. The solution is to implement a container
    that holds a `std::variant` type and to allow all types declared inside the
    given `std::variant` to be pushed or poped from the container. Each type inside
    the given `std::variant` has each own `std::vector` attached. All this is done
    to decrease the number of switch and if-else statements (branches) later */
template<typename VARIANT>
class VariantContainer {
private:
    /* Holding an individual vector for each type */
    std::array<std::vector<VARIANT>, std::variant_size_v<VARIANT>> _data;
public:
    VariantContainer() {}

    template <typename T>
    auto begin() {
        constexpr std::size_t T_index = assert_in_variant<VARIANT, T>();
        auto&& beg_it = this->_data.at(T_index).begin();

        return variant_iterator<T_index, VARIANT>(beg_it);
    }

    template <typename T>
    auto end() {
        constexpr std::size_t T_index = assert_in_variant<VARIANT, T>();
        auto&& end_it = this->_data.at(T_index).end();

        return variant_iterator<T_index, VARIANT>(end_it); 
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
    void forEach(F&& func) {
        if constexpr(I < std::variant_size_v<VARIANT>) {
            auto&& begin_it = this->_data.at(I).begin();
            auto&& end_it = this->_data.at(I).end();

            auto it = variant_iterator<I, VARIANT>(begin_it);
            auto end = variant_iterator<I, VARIANT>(end_it);
            for (; it != end; it++) {
                func(*it);
            }
            forEach<F, I+1>(std::forward<F>(func));
        } else {
            return;
        }
    }

    template <typename T>
    T at(std::size_t index) {
        std::size_t T_index = assert_in_variant<VARIANT, T>();

        /* Return the data from the corresponding vector */
        return this->_data.at(T_index).at(index);
    }

    template <typename T>
    void push_back(T value) {
        std::size_t T_index = assert_in_variant<VARIANT, T>();

        /* Insert the value */
        this->_data.at(T_index).push_back(value);
    }

    template <typename T>
    void pop_back(T value) {
        std::size_t T_index = assert_in_variant<VARIANT, T>();

        /* Remove the last value */
        this->_data.at(T_index).pop_back(value);
    }
};