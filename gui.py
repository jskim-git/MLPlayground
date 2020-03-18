import tkinter as tk


class Demo1:
    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)
        self.button1 = tk.Button(self.frame, text='New Window', width=25, command=self.new_window)
        self.button1.pack()
        self.frame.pack()

    def new_window(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app = Demo2(self.newWindow)


class Demo2:
    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)
        self.quitButton = tk.Button(self.frame, text='Quit', width=25, command=self.close_windows)
        self.quitButton.pack()
        self.frame.pack()

    def close_windows(self):
        self.master.destroy()


class MainApplication(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.name = tk.Label(master, text="Name").place(x=30, y=50)
        self.name_entry = tk.Entry(master).place(x=85, y=50)
        self.submit = tk.Button(self)
        self.QUIT = tk.Button(self)
        self.pack()
        self.createWidgets()

    def createWidgets(self):
        self.QUIT["text"] = "QUIT"
        self.QUIT["fg"] = "red"
        self.QUIT["command"] = self.quit

        self.submit["text"] = "Submit"
        self.submit["activebackground"] = "green"
        self.submit["activeforeground"] = "blue"

        self.QUIT.pack({"side": "left"})
        self.submit.pack({"side": "left"})


def main():
    root = tk.Tk()
    app = Demo1(root)
    root.mainloop()


if __name__ == '__main__':
    main()
