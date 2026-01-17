import customtkinter
import onnxruntime as ort
import cv2
import numpy as np
import pyautogui
from mss import mss
import time
from tkinter import filedialog
import threading
import pygame
import json
from pynput import keyboard
import win32api
import win32con

# Set appearance and theme
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")


# Mouse control functions
def move_mouse(dx, dy):
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, dx, dy, 0, 0)


def left_click():
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


class ONNXLoaderApp(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("360x720+200+100")
        self.overrideredirect(True)
        self.config(background="#000001")
        self.attributes("-transparentcolor", "#000001")
        self.attributes("-alpha", 0.78)
        self.attributes("-topmost", True)

        # Frame
        self.frame = customtkinter.CTkFrame(
            self, corner_radius=22, fg_color="#4B0082", bg_color="#000001"
        )
        self.frame.pack(fill="both", expand=True)
        self.frame.bind("<Button-1>", self._start_drag)
        self.frame.bind("<B1-Motion>", self._drag_motion)

        # Close button
        customtkinter.CTkButton(
            self.frame,
            text="X",
            command=self.destroy,
            width=36,
            height=36,
            corner_radius=18,
            fg_color="#800080",
        ).pack(anchor="ne", padx=12, pady=12)

        # Title
        customtkinter.CTkLabel(
            self.frame,
            text="Silicate's Universal Aim Assist",
            font=("Arial", 18, "bold"),
            text_color="white",
        ).pack(pady=15)

        # Load ONNX button
        customtkinter.CTkButton(
            self.frame, text="Load ONNX Model", command=self.load_onnx, fg_color="#9370DB", height=40
        ).pack(pady=8)

        # Aim toggle
        self.aim_toggle = customtkinter.CTkSwitch(
            self.frame, text="Enable Aim Assist", command=self.toggle_aim, progress_color="#800080"
        )
        self.aim_toggle.pack(pady=8)

        # Confidence slider
        self.conf_label = customtkinter.CTkLabel(self.frame, text="Confidence: 0.45")
        self.conf_label.pack()
        self.conf_slider = customtkinter.CTkSlider(self.frame, from_=0.1, to=0.95, command=self.update_conf)
        self.conf_slider.set(0.45)
        self.conf_slider.pack()

        # Smoothing slider
        self.smooth_label = customtkinter.CTkLabel(self.frame, text="Smoothing: 0.38")
        self.smooth_label.pack()
        self.smooth_slider = customtkinter.CTkSlider(self.frame, from_=0.05, to=0.85, command=self.update_smooth)
        self.smooth_slider.set(0.38)
        self.smooth_slider.pack()

        # Bone selection
        self.bone_combo = customtkinter.CTkComboBox(
            self.frame, values=["Center", "Head", "Upper Body", "Lower Body", "Random"]
        )
        self.bone_combo.set("Head")
        self.bone_combo.pack(pady=6)

        # Random humanize toggle
        self.random_toggle = customtkinter.CTkSwitch(self.frame, text="Humanize (Random Offset)")
        self.random_toggle.pack()

        # Triggerbot toggle
        self.trigger_toggle = customtkinter.CTkSwitch(self.frame, text="Triggerbot")
        self.trigger_toggle.pack()

        # FOV entry
        self.fov_entry = customtkinter.CTkEntry(self.frame)
        self.fov_entry.insert(0, "640")
        self.fov_entry.pack(pady=8)

        # FPS label
        self.fps_label = customtkinter.CTkLabel(self.frame, text="FPS: 0.0")
        self.fps_label.pack(pady=8)

        # Model and control flags
        self.model = None
        self.running = False
        self.fps = 0.0
        self.current_target = None  # <-- added to track locked target

        # Hotkey listener
        self.start_hotkey_listener()

        # Update FPS periodically
        self.after(500, self.update_fps)

    # Drag functions
    def _start_drag(self, e):
        self.dx = e.x
        self.dy = e.y

    def _drag_motion(self, e):
        self.geometry(f"+{self.winfo_pointerx() - self.dx}+{self.winfo_pointery() - self.dy}")

    # Slider updates
    def update_conf(self, v):
        self.conf_label.configure(text=f"Confidence: {v:.2f}")

    def update_smooth(self, v):
        self.smooth_label.configure(text=f"Smoothing: {v:.2f}")

    # Load ONNX model
    def load_onnx(self):
        path = filedialog.askopenfilename(filetypes=[("ONNX", "*.onnx")])
        if path:
            self.model = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    # Toggle aim assist
    def toggle_aim(self):
        self.running = self.aim_toggle.get()
        if self.running:
            threading.Thread(target=self.aim_loop, daemon=True).start()

    # Main aim loop
    def aim_loop(self):
        while self.running:
            start = time.time()
            try:
                sw, sh = pyautogui.size()
                fov = int(self.fov_entry.get())
                l = (sw - fov) // 2
                t = (sh - fov) // 2

                with mss() as sct:
                    img = np.array(sct.grab({"left": l, "top": t, "width": fov, "height": fov}))[:, :, :3]
                    inp = self.model.get_inputs()[0]
                    _, _, h, w = inp.shape
                    img = cv2.resize(img, (w, h))
                    img = np.expand_dims(img.transpose(2, 0, 1) / 255.0, 0).astype(np.float32)
                    outs = self.model.run(None, {inp.name: img})[0]

                if outs.ndim == 3:
                    outs = outs[0].T

                dets = outs[outs[:, 4] > self.conf_slider.get()]

                target_found = False

                if len(dets):
                    # If we already have a target, try to track it
                    if self.current_target is not None:
                        # Find the detection closest to current target
                        distances = np.linalg.norm(dets[:, :2] - np.array(self.current_target[:2]), axis=1)
                        min_idx = np.argmin(distances)
                        if distances[min_idx] < 100:  # max distance to consider same target
                            cx, cy, bw, bh = dets[min_idx][:4].astype(int)
                            target_found = True
                        else:
                            # Target disappeared
                            self.current_target = None

                    # If no current target, lock onto the first detection
                    if self.current_target is None:
                        cx, cy, bw, bh = dets[0][:4].astype(int)
                        self.current_target = [cx, cy, bw, bh]
                        target_found = True

                    if target_found:
                        # Update current_target for next frame
                        self.current_target = [cx, cy, bw, bh]

                        # Bone adjustment
                        bone = self.bone_combo.get()
                        if bone == "Head":
                            cy -= int(bh * 0.3)
                        elif bone == "Lower Body":
                            cy += int(bh * 0.2)
                        elif bone == "Random":
                            cy += np.random.randint(-bh // 4, bh // 4)

                        tx, ty = l + cx, t + cy
                        mx, my = pyautogui.position()
                        smooth = self.smooth_slider.get()
                        dx = int((tx - mx) * smooth)
                        dy = int((ty - my) * smooth)

                        if abs(dx) > 1 or abs(dy) > 1:
                            move_mouse(dx, dy)

                        if self.trigger_toggle.get():
                            left_click()
                else:
                    self.current_target = None  # no detections

            except Exception:
                pass

            self.fps = 1 / (time.time() - start + 1e-6)
            time.sleep(0.005)

    # FPS display
    def update_fps(self):
        self.fps_label.configure(text=f"FPS: {self.fps:.1f}")
        self.after(500, self.update_fps)

    # Hotkey listener (F1 toggle)
    def start_hotkey_listener(self):
        def on_press(key):
            if key == keyboard.Key.f1:
                self.aim_toggle.invoke()

        keyboard.Listener(on_press=on_press).start()


if __name__ == "__main__":
    app = ONNXLoaderApp()
    app.mainloop()
