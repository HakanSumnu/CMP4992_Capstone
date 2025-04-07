import sys
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import random

# Data structures
current_path = []  # Active path being tracked
ball_paths = {}    # Dictionary of completed paths
path_colors = []   # Store different colors for past paths

# Predefined colors for up to N paths
COLORS = ["red", "blue", "green", "orange", "purple", "brown", "cyan", "magenta"]

# UI Class
class BallPathUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Dodgeball Path Visualizer")

        # Layout frame
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel (Ball side visual panel)
        self.left_canvas = tk.Canvas(self.main_frame, width=100, bg="white", highlightthickness=1, highlightbackground="black")
        self.left_canvas.pack(side=tk.LEFT, fill=tk.Y)
        self.left_canvas.create_text(50, 20, text="Ball Side", font=("Helvetica", 10, "bold"))

        # Center panel (Matplotlib plot)
        self.center_frame = ttk.Frame(self.main_frame)
        self.center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Ball Trajectories (XZ Plane)")
        self.ax.set_xlabel("X Position (cm)")
        self.ax.set_ylabel("Z Position (cm)")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.center_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Right panel (Robot visualization)
        self.right_canvas = tk.Canvas(self.main_frame, width=100, bg="white", highlightthickness=1, highlightbackground="black")
        self.right_canvas.pack(side=tk.RIGHT, fill=tk.Y)
        self.right_canvas.create_text(50, 20, text="Robot", font=("Helvetica", 10, "bold"))
        self.draw_robot_icon()

        # Controls
        control_frame = ttk.Frame(root)
        control_frame.pack(fill=tk.X)

        self.new_path_btn = ttk.Button(control_frame, text="New Path", command=self.store_current_path)
        self.new_path_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.clear_btn = ttk.Button(control_frame, text="Clear All", command=self.clear_all_paths)
        self.clear_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Start update loop
        self.update_plot_loop()

    def draw_robot_icon(self):
        # Simple robot drawing
        c = self.right_canvas
        c.create_oval(30, 40, 70, 80, fill="black")  # Head
        c.create_rectangle(40, 80, 60, 100, fill="black")  # Body
        c.create_line(30, 60, 20, 40, fill="black", width=3)  # Left antenna
        c.create_line(70, 60, 80, 40, fill="black", width=3)  # Right antenna
        c.create_rectangle(25, 100, 35, 120, fill="black")  # Left leg
        c.create_rectangle(65, 100, 75, 120, fill="black")  # Right leg
        c.create_rectangle(20, 80, 30, 100, fill="black")  # Left arm
        c.create_rectangle(70, 80, 80, 100, fill="black")  # Right arm

    def store_current_path(self):
        if current_path:
            key = datetime.now().strftime("session_%H%M%S")
            ball_paths[key] = current_path.copy()
            path_colors.append(COLORS[len(path_colors) % len(COLORS)])
            current_path.clear()
            print(f"Stored path as {key}")

    def clear_all_paths(self):
        ball_paths.clear()
        path_colors.clear()
        current_path.clear()
        print("Cleared all paths")

    def update_plot_loop(self):
        self.ax.clear()
        self.ax.set_title("Ball Trajectories (XZ Plane)")
        self.ax.set_xlabel("X Position (cm)")
        self.ax.set_ylabel("Z Position (cm)")

        # Plot all stored paths
        for idx, (key, path) in enumerate(ball_paths.items()):
            if len(path) >= 2:
                xs = [p[0] for p in path]
                zs = [p[2] for p in path]
                color = path_colors[idx % len(path_colors)]
                self.ax.plot(xs, zs, marker='o', linestyle='-', color=color, label=key)

        # Plot current path
        if len(current_path) >= 2:
            xs = [p[0] for p in current_path]
            zs = [p[2] for p in current_path]
            self.ax.plot(xs, zs, marker='x', linestyle='--', color='black', label="Current Path")

        if self.ax.get_legend_handles_labels()[0]:
            self.ax.legend()
        self.canvas.draw()

        self.simulate_live_data()

        self.root.after(100, self.update_plot_loop)

    def simulate_live_data(self):
        # Simulate a new point being added (replace with OpenCV integration)
        if random.random() < 0.5:
            x = current_path[-1][0] + random.uniform(-0.5, 0.5) if current_path else 0
            y = -1.8
            z = current_path[-1][2] - 1 if current_path else 0
            current_path.append([x, y, z])


# Run the UI
if __name__ == "__main__":
    root = tk.Tk()
    app = BallPathUI(root)
    root.mainloop()