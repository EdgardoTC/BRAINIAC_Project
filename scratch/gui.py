import sys

try:
    import tkinter as tk
    from tkinter import messagebox, filedialog
except ImportError:
    print("Tkinter is not installed. Please install it to use the GUI.")
    sys.exit(1)

REQUIRED = [
    "numpy", "nibabel", "scipy", "torch", "sklearn", "matplotlib", "joblib", "PIL"
]

missing = []
for pkg in REQUIRED:
    try:
        __import__(pkg if pkg != "PIL" else "PIL.Image")
    except ImportError:
        missing.append(pkg)

if missing:
    root = tk.Tk()
    root.withdraw()
    msg = (
        f"The following packages are missing:\n\n{', '.join(missing)}\n\n"
        "Would you like to install them now?"
    )
    if messagebox.askyesno("Missing Packages", msg):
        import subprocess

        cmd = [sys.executable, "-m", "pip", "install"] + missing

        installing = tk.Toplevel()
        installing.title("Installing Packages")
        tk.Label(installing, text="Installing packages, please wait...").pack(padx=20, pady=20)
        installing.update()

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            installing.destroy()
            if result.returncode == 0:
                messagebox.showinfo("Success", "Packages installed! Please RESTART the program.")
            else:
                messagebox.showerror("Install Error", f"Error during install:\n\n{result.stderr}")
        except Exception as e:
            installing.destroy()
            messagebox.showerror("Install Error", f"Installation failed: {e}")
    else:
        messagebox.showinfo("Aborted", "Cannot continue without required packages.")
    sys.exit(1)

from threading import Thread
from main import main
from PIL import Image, ImageTk

class PipelineGUI:
    def __init__(self, master):
        self.master = master
        master.title("BRAINIAC Pipeline")
        self.gp1_path = tk.StringVar()
        self.gp2_path = tk.StringVar()
        self.preprocessed_dir = tk.StringVar()

        tk.Label(master, text="Group 1 Dir:").grid(row=0, column=0, sticky="e")
        tk.Entry(master, textvariable=self.gp1_path, width=40).grid(row=0, column=1)
        tk.Button(master, text="Browse", command=self.browse_gp1).grid(row=0, column=2)

        tk.Label(master, text="Group 2 Dir:").grid(row=1, column=0, sticky="e")
        tk.Entry(master, textvariable=self.gp2_path, width=40).grid(row=1, column=1)
        tk.Button(master, text="Browse", command=self.browse_gp2).grid(row=1, column=2)

        tk.Label(master, text="Output Dir:").grid(row=2, column=0, sticky="e")
        tk.Entry(master, textvariable=self.preprocessed_dir, width=40).grid(row=2, column=1)
        tk.Button(master, text="Browse", command=self.browse_pre).grid(row=2, column=2)

        self.progress = tk.Label(master, text="", fg="green")
        self.progress.grid(row=4, column=0, columnspan=3)

        tk.Button(master, text="Run Pipeline", command=self.run_pipeline).grid(row=5, column=1, pady=10)

    def browse_gp1(self):
        self.gp1_path.set(filedialog.askdirectory())

    def browse_gp2(self):
        self.gp2_path.set(filedialog.askdirectory())

    def browse_pre(self):
        self.preprocessed_dir.set(filedialog.askdirectory())

    def run_pipeline(self):
        gp1 = self.gp1_path.get()
        gp2 = self.gp2_path.get()
        pre = self.preprocessed_dir.get()
        if not all([gp1, gp2, pre]):
            messagebox.showerror("Error", "Please select all directories.")
            return

        def worker():
            try:
                self.progress.config(text="Processing...")
                main(gp1, gp2, pre)
                self.progress.config(text="Done!")
                # Schedule the image display in the main thread
                self.master.after(0, self.display_images)
            except Exception as e:
                messagebox.showerror("Pipeline Error", str(e))
                self.progress.config(text="Failed.")

        Thread(target=worker).start()

    def display_images(self):
        img_win = tk.Toplevel(self.master)
        img_win.title("Results")
        images = []
        for fname in ["roc_curve.png", "probability_distribution.png"]:
            try:
                img = Image.open(fname)
                img = img.resize((400, 300))  # adjust as needed
                photo = ImageTk.PhotoImage(img)
                images.append(photo)
                label = tk.Label(img_win, image=photo)
                label.pack(padx=10, pady=10)
                tk.Label(img_win, text=fname).pack()
            except Exception as e:
                tk.Label(img_win, text=f"Could not load {fname}: {e}").pack()
        img_win.images = images  # prevent garbage collection

if __name__ == "__main__":
    root = tk.Tk()
    app = PipelineGUI(root)
    root.mainloop()
