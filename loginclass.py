import tkinter as tk


class MainApplication:
    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)
        tk.Label(text="Select", width="300", height="2").pack()
        tk.Label(text="").pack()
        tk.Button(text="Login", width="30", height="2", command=self.login).pack()
        tk.Label(text="").pack()
        tk.Button(text="Register", width="30", height="2").pack()
        self.frame.pack()

    def login(self):
        self.loginWindow = tk.Toplevel(self.master)
        self.loginWindow.title("Login")
        self.loginWindow.geometry("300x250")
        self.app = LoginApplication(self.loginWindow)


class LoginApplication:
    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)
        tk.Label(text="Please enter details below to login").pack()

        tk.Button(text="test").pack()
        self.frame.pack()


def start_screen():
    root = tk.Tk()
    root.geometry("300x250")
    root.title("Start Screen")
    app = MainApplication(root)

    root.mainloop()


if __name__ == '__main__':
    start_screen()

