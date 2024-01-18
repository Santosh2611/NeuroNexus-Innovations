import secrets  # Import the 'secrets' module to generate cryptographically strong random numbers suitable for managing data such as passwords.

import pyperclip  # Import the 'pyperclip' module to provide a cross-platform way to copy and paste text to the clipboard.

import string  # Import the 'string' module to access various string constants.

# Define a dictionary that contains different character sets used to generate the password.
CHARACTER_SETS = {
    'uppercase': string.ascii_uppercase,  # Uppercase letters
    'lowercase': string.ascii_lowercase,  # Lowercase letters
    'digits': string.digits,  # Numeric digits
    'punctuation': string.punctuation  # Punctuation characters
}

# Function to generate a password of specified length using the provided character sets.
def generate_password(length, **kwargs):
    characters = ''.join(CHARACTER_SETS[set_name] for set_name, include in kwargs.items() if include)  # Concatenate selected character sets based on user input
    password = ''.join(secrets.choice(characters) for _ in range(length))  # Generate the password using random characters from the concatenated character set
    return password

# Function to get the desired length of the password from the user
def get_desired_length():
    while True:
        try:
            length = int(input("Enter the desired length of the password: "))  # Prompt the user to input the desired length of the password
            if length <= 0:  # Check if the input is a positive integer
                print("Length must be a positive integer.")  # Display an error message if the input is not valid
            else:
                return length  # Return the valid input length
        except ValueError:
            print("Invalid input. Please enter a valid integer.")  # Display an error message if the input is not a valid integer

# Main function to interact with the user, generate the password, and allow copying it to the clipboard
def main():
    length = get_desired_length()  # Call the 'get_desired_length' function to obtain the desired length of the password from the user
    password = generate_password(length, uppercase=True, lowercase=True, digits=True, punctuation=True)  # Generate the password with selected character sets based on user input
    
    print(f"Your generated password is: {password}")  # Display the generated password to the user
    
    copy_to_clipboard = input("Do you want to copy the password to clipboard? (yes/no): ").strip().lower()  # Prompt the user to decide whether to copy the password to the clipboard
    if copy_to_clipboard == "yes":  # Check the user's decision
        pyperclip.copy(password)  # Copy the generated password to the clipboard using the 'pyperclip' module
        print("Password copied to clipboard!")  # Inform the user that the password has been copied to the clipboard
    else:
        print("Password not copied to clipboard.")  # Inform the user that the password has not been copied to the clipboard
        
if __name__ == "__main__":
    main()  # Call the 'main' function if the script is executed directly
