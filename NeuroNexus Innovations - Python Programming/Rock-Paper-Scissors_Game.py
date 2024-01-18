import unittest  # Importing the unittest module for writing and running tests
from unittest.mock import patch  # Allowing us to mock the input function for testing
import random  # Importing the random module to generate computer choice

# The functions that we want to test
def get_user_choice(user_input):
    """
    Function to get user's choice for rock, paper, or scissors.
    Keeps prompting for input until a valid choice is entered.
    """
    user_choice = user_input.lower()
    while user_choice not in ["rock", "paper", "scissors"]:
        user_input = input("Invalid choice. Please choose rock, paper, or scissors: ")
        user_choice = user_input.lower()
    return user_choice

def get_computer_choice():
    """
    Function to generate the computer's choice of rock, paper, or scissors.
    """
    return random.choice(["rock", "paper", "scissors"])

def determine_winner(user_choice, computer_choice):
    """
    Function to determine the winner based on user's and computer's choices.
    Compares the choices and returns the result.
    """
    outcomes = {
        ("rock", "scissors"): "You win!",
        ("scissors", "paper"): "You win!",
        ("paper", "rock"): "You win!",
        ("scissors", "rock"): "Computer wins!",
        ("paper", "scissors"): "Computer wins!",
        ("rock", "paper"): "Computer wins!"
    }
    if user_choice == computer_choice:
        return "It's a tie!"
    else:
        return outcomes[(user_choice, computer_choice)]

class TestRockPaperScissors(unittest.TestCase):

    @patch('builtins.input', side_effect=['rock'])
    def test_get_user_choice(self, mock_input):
        """
        Test method for get_user_choice function.
        It uses the patched input function to simulate user input for testing.
        """
        self.assertEqual(get_user_choice(mock_input()), 'rock')

    def test_get_computer_choice(self):
        """
        Test method for get_computer_choice function.
        It checks if the generated computer choice is one of rock, paper, or scissors.
        """
        self.assertIn(get_computer_choice(), ['rock', 'paper', 'scissors'])

    def test_determine_winner(self):
        """
        Test method for determine_winner function.
        It tests the outcome of the game for different combinations of user and computer choices.
        """
        self.assertEqual(determine_winner('rock', 'rock'), "It's a tie!")
        self.assertEqual(determine_winner('rock', 'scissors'), "You win!")
        self.assertEqual(determine_winner('rock', 'paper'), "Computer wins!")

if __name__ == '__main__':
    unittest.main()  # Running the tests when the script is executed directly
