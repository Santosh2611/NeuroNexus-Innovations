import unittest  # Importing the unittest module for testing

# Functions for basic arithmetic operations
def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def divide(x, y):
    if y != 0:
        return x / y  # Performing division and returning the result
    else:
        raise ValueError("Division by zero is not allowed.")  # Raising a ValueError for division by zero

# Dictionary mapping operation symbols to corresponding functions
operations = {
    '+': add,
    '-': subtract,
    '*': multiply,
    '/': divide
}

# Function to calculate the result based on user input
def calculate(num1, num2, operation):
    if operation in operations:  # Checking if the specified operation is valid
        return operations[operation](num1, num2)  # Using the corresponding function to perform the calculation
    else:
        raise ValueError("Invalid operation selected.")  # Raising a ValueError for an invalid operation

# Test class for unit testing the calculator functions
class TestCalculator(unittest.TestCase):

    # Test for addition function
    def test_addition(self):
        self.assertEqual(calculate(5, 3, '+'), 8)  # Asserting that the addition function produces the expected result

    # Test for subtraction function
    def test_subtraction(self):
        self.assertEqual(calculate(5, 3, '-'), 2)  # Asserting that the subtraction function produces the expected result

    # Test for multiplication function
    def test_multiplication(self):
        self.assertEqual(calculate(5, 3, '*'), 15)  # Asserting that the multiplication function produces the expected result

    # Test for division function
    def test_division(self):
        self.assertEqual(calculate(6, 3, '/'), 2)  # Asserting that the division function produces the expected result
        with self.assertRaises(ValueError):  # Testing division by zero error
            calculate(6, 0, '/')  # Making sure that dividing by zero raises a ValueError

    # Test for invalid operation
    def test_invalid_operation(self):
        with self.assertRaises(ValueError):  # Testing invalid operation
            calculate(5, 3, '%')  # Making sure that an invalid operation raises a ValueError

# Running the unit tests
if __name__ == '__main__':
    unittest.main()  # Executing the unit tests when the script is run directly
