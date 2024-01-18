import tkinter as tk          # Importing the tkinter module for GUI
from tkinter import messagebox  # Importing messagebox from tkinter

class Task:
    def __init__(self, description, status="incomplete", due_date=None, priority=None, category=None):
        # Initialize Task object with provided attributes
        self.description = description
        self.status = status
        self.due_date = due_date
        self.priority = priority
        self.category = category

    def mark_as_complete(self):
        # Method to mark the task as complete
        self.status = "complete"

class ToDoList:
    def __init__(self):
        # Initialize ToDoList object with an empty list of tasks
        self.tasks = []

    def add_task(self, task):
        # Add a new task to the list of tasks
        self.tasks.append(task)

    def remove_task(self, index):
        # Remove task at the specified index from the list
        if 0 <= index < len(self.tasks):
            del self.tasks[index]
        else:
            messagebox.showerror("Error", "Invalid task number.")

    def display_tasks(self):
        # Generate a formatted string representation of all tasks in the list
        if self.tasks:
            tasks_info = "\n".join([f"{i}. [{task.status}] {task.description}" for i, task in enumerate(self.tasks, 1)])
            return f"{tasks_info}\n"
        else:
            return "No tasks in the list.\n"

    def save_to_file(self, filename):
        # Save the tasks to a file
        try:
            with open(filename, 'w') as file:
                for task in self.tasks:
                    file.write(f"{task.description},{task.status}\n")
            messagebox.showinfo("Success", "To-do list saved to file.")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving to file: {e}")

    def load_from_file(self, filename):
        # Load tasks from a file and add them to the list
        try:
            with open(filename, 'r') as file:
                for line in file:
                    description, status = line.strip().split(',')
                    task = Task(description, status)
                    self.add_task(task)
            messagebox.showinfo("Success", "To-do list loaded from file.")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading from file: {e}")

def create_entry_button_window(window_title, button_text, button_command, *args):
    # Function to create a window with an entry and a button
    window = tk.Toplevel()
    window.title(window_title)
    
    label = tk.Label(window, text=args[0])
    label.pack()

    entry = tk.Entry(window)
    entry.pack()

    button = tk.Button(window, text=button_text, command=lambda: button_command(args[1], entry))
    button.pack()

    return window

# Functions handling user actions

def add_task(todo_list, entry_description):
    # Add a new task to the ToDoList
    description = entry_description.get()
    
    if description:
        task = Task(description)
        todo_list.add_task(task)
        messagebox.showinfo("Success", "Task added successfully.")
    else:
        messagebox.showerror("Error", "Description cannot be empty.")

def remove_task(todo_list, entry_index):
    # Remove a task from the ToDoList
    index = int(entry_index.get()) - 1  # Convert to zero-based index
    todo_list.remove_task(index)
    messagebox.showinfo("Success", "Task removed successfully.")

def mark_task_complete(todo_list, entry_index):
    # Mark a task as complete
    index = int(entry_index.get()) - 1  # Convert to zero-based index
    task = todo_list.tasks[index]
    task.mark_as_complete()
    messagebox.showinfo("Success", f"The task '{task.description}' has been marked as complete.")

def display_tasks(todo_list):
    # Display all tasks in a new window
    window = tk.Toplevel()
    window.title("Display Tasks")

    tasks_label = tk.Label(window, text=todo_list.display_tasks())
    tasks_label.pack()

def main():
    # Main application function
    todo_list = ToDoList()

    root = tk.Tk()
    root.title("To-Do List Application")

    add_button = tk.Button(root, text="Add Task",
                           command=lambda: create_entry_button_window(
                               "Add New Task", "Add", add_task, "Description:", todo_list, entry_description))
    add_button.pack()

    remove_button = tk.Button(root, text="Remove Task",
                              command=lambda: create_entry_button_window(
                                  "Remove Task", "Remove", remove_task, "Task Index:", todo_list, entry_index))
    remove_button.pack()

    complete_button = tk.Button(root, text="Mark Task as Complete",
                                command=lambda: create_entry_button_window(
                                    "Mark Task as Complete", "Mark Complete", mark_task_complete, "Task Index:", todo_list, entry_index))
    complete_button.pack()

    display_button = tk.Button(root, text="Display Tasks", command=lambda: display_tasks(todo_list))
    display_button.pack()

    save_exit_button = tk.Button(root, text="Save and Exit", command=lambda: (todo_list.save_to_file("todo_list.txt"), root.quit()))
    save_exit_button.pack()

    entry_description = tk.Entry(root)
    entry_description.pack()

    entry_index = tk.Entry(root)
    entry_index.pack()

    root.mainloop()

if __name__ == "__main__":
    main()
