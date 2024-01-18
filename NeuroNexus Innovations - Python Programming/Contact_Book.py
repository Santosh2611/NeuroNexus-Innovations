# Import the necessary modules
import sqlite3
import tkinter as tk
from tkinter import messagebox

# Define the ContactBookApp class
class ContactBookApp:
    def __init__(self, root):
        # Initialize the ContactBookApp with the root window
        self.root = root
        self.root.title("Contact Book")  # Set the title of the root window
        self.create_ui()  # Create the user interface
        self.conn = sqlite3.connect('contact_book.db')  # Establish connection to the SQLite database
        self.c = self.conn.cursor()  # Create a cursor object for executing SQL commands
        self.create_table()  # Create the "contacts" table in the database
        self.root.protocol("WM_DELETE_WINDOW", self.close_connection)  # Handle window closing event

    # Method to create the user interface
    def create_ui(self):
        entries = ['name', 'phone', 'email', 'address']
        for entry in entries:
            setattr(self, f"{entry}_entry", tk.Entry(self.root))  # Create Entry widgets for name, phone, email, and address
            getattr(self, f"{entry}_entry").pack()  # Pack the Entry widgets into the root window

        button_texts = ['Add Contact', 'View Contacts', 'Search Contact', 'Update Contact', 'Delete Contact']
        commands = [self.add_contact, self.view_contacts, self.search_contact, self.update_contact, self.delete_contact]
        for button_text, command in zip(button_texts, commands):
            button = tk.Button(self.root, text=button_text, command=command)  # Create buttons for different contact operations
            button.pack()  # Pack the buttons into the root window

    # Method to create the "contacts" table in the database
    def create_table(self):
        self.c.execute('''CREATE TABLE IF NOT EXISTS contacts (
                        name TEXT PRIMARY KEY,
                        phone_number TEXT,
                        email TEXT,
                        address TEXT)''')
        self.conn.commit()  # Commit the changes to the database

    # Method to add a new contact to the database
    def add_contact(self):
        if self.conn:  # Check if the database connection is open
            name, phone_number, email, address = self.get_entry_values()  # Get the values from the Entry widgets
            if name and phone_number:  # Check if name and phone number are provided
                self.c.execute("INSERT INTO contacts (name, phone_number, email, address) VALUES (?, ?, ?, ?)", (name, phone_number, email, address))  # Insert the contact details into the database
                self.conn.commit()  # Commit the changes to the database
                self.show_message("Success", "Contact added successfully.")  # Show success message
            else:
                self.show_message("Error", "Name and phone number are required.")  # Show error message
        else:
            self.show_message("Error", "Database connection is closed.")  # Show error message

    # Method to view all contacts in the database
    def view_contacts(self):
        if self.conn:  # Check if the database connection is open
            contact_list = self.c.execute("SELECT name, phone_number FROM contacts").fetchall()  # Retrieve the list of contacts
            self.display_results(contact_list, "Contacts", "No contacts found.")  # Display the contact list
        else:
            self.show_message("Error", "Database connection is closed.")  # Show error message

    # Method to search for a contact based on the name
    def search_contact(self):
        keyword = self.name_entry.get()  # Get the search keyword from the name Entry widget
        result = self.c.execute("SELECT name, phone_number FROM contacts WHERE name LIKE ?", ('%' + keyword + '%',)).fetchall()  # Search for matching contacts
        self.display_results(result, "Search Results", "No matching contacts found.")  # Display the search results

    # Method to update the contact details
    def update_contact(self):
        name, phone_number, email, address = self.get_entry_values()  # Get the updated values from the Entry widgets
        self.c.execute("UPDATE contacts SET phone_number=?, email=?, address=? WHERE name=?", (phone_number, email, address, name))  # Update the contact details
        self.conn.commit()  # Commit the changes to the database
        self.show_message("Update Successful", "Contact updated successfully.")  # Show success message

    # Method to delete a contact from the database
    def delete_contact(self):
        name = self.name_entry.get()  # Get the name of the contact to be deleted
        self.c.execute("DELETE FROM contacts WHERE name=?", (name,))  # Delete the contact from the database
        self.conn.commit()  # Commit the changes to the database
        self.show_message("Deletion Successful", "Contact deleted successfully.")  # Show success message

    # Method to get the values from the Entry widgets
    def get_entry_values(self):
        return (self.name_entry.get(), self.phone_entry.get(), self.email_entry.get(), self.address_entry.get())

    # Method to display the search or view results
    def display_results(self, result, success_message, no_match_message):
        if result:
            result_str = "\n".join([f"{name}: {phone}" for name, phone in result])  # Format the results as a string
            self.show_message(success_message, result_str)  # Show the success message with the results
        else:
            self.show_message("No Match", no_match_message)  # Show no match message

    # Method to show a message dialog
    def show_message(self, title, message):
        messagebox.showinfo(title, message)  # Show an info message dialog

    # Method to close the database connection
    def close_connection(self):
        self.conn.close()  # Close the database connection

# Create the root window
root = tk.Tk()
app = ContactBookApp(root)  # Create an instance of ContactBookApp
root.mainloop()  # Run the main loop for the GUI
