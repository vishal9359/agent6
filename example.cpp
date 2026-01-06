// Example C++ file for testing the agent

namespace Math {
    int add(int a, int b) {
        return a + b;
    }
    
    int multiply(int a, int b) {
        return a * b;
    }
}

class Calculator {
public:
    int calculate(int x, int y) {
        if (x > 0) {
            return Math::add(x, y);
        } else {
            return Math::multiply(x, y);
        }
    }
    
    int process(int value) {
        int result = 0;
        while (value > 0) {
            result = calculate(result, value);
            value--;
        }
        return result;
    }
};

int main() {
    Calculator calc;
    int result = calc.process(10);
    return result;
}

