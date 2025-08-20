import numpy as np
import cv2
import json
from pathlib import Path
from typing import Optional
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class Object_detection_graphical_user_interface:
    def __init__(self, window):
        self.gui = window
        self.gui.title("[Object detection (amount, titles, positions of tables) in 1 page PDF file]")
        self.gui.configure(bg="#888888")
        self.gui.iconphoto(False, ImageTk.PhotoImage(file="table_detector_in_PDF_file_icon.png"))
        try:
            self.gui.state("zoomed")
        except tk.TclError:
            try:
                self.gui.attributes("-zoomed", True)
            except tk.TclError:
                self.gui.update_idletasks()
                width = self.gui.winfo_screenwidth()
                height = self.gui.winfo_screenheight()
                self.gui.geometry(f"{width}x{height}+0+0")

        top_panel = tk.Frame(window, bg="#888888", padx=10, pady=10)
        top_panel.pack(side=tk.TOP, fill=tk.X)
        button_border = tk.Frame(top_panel, bg="black")
        button_border.pack(side=tk.LEFT, padx=10, pady=5)
        self.button_load_1_page_PDF_file = tk.Button(
            button_border,
            text="[Load 1 page PDF file with tables]",
            font=("Arial", 16, "bold"),
            bg="#ffffff", fg="#000000",
            bd=0, relief="flat", padx=20, pady=12,
            activebackground="#f2f2f2", activeforeground="#000000",
            cursor="hand2"
        )
        self.button_load_1_page_PDF_file.pack(padx=4, pady=4)

        content = tk.PanedWindow(
            window, orient=tk.HORIZONTAL,
            sashwidth=6, 
            bg="#888888", 
            bd=0, 
            relief="flat",
            sashrelief="flat", 
            sashpad=2
        )
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_part = tk.Frame(content, bg="#888888")
        right_part = tk.Frame(content, bg="#888888")
        content.add(left_part)
        content.add(right_part)

        def place_sash(_=None):
            width = content.winfo_width()
            content.sash_place(0, int(width * 0.65), 0)

        content.bind("<Configure>", place_sash)
        self.gui.after(50, place_sash)

        self.image_title = tk.Label(
            left_part, 
            text="[1 page PDF file with detected tables]",
            font=("Arial", 10, "bold"),
            bg="#ffffff", fg="#000000"
        )
        self.image_title.pack(anchor="w")

        left_border = tk.Frame(left_part, bg="black")
        left_border.pack(fill=tk.BOTH, expand=True, pady=(4, 8))

        self.canvas = tk.Canvas(left_border, bg="#ffffff", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        right_panel = ttk.Notebook(right_part)
        right_panel.pack(fill=tk.BOTH, expand=True)

        self.tab_detection_information = ttk.Frame(right_panel)
        self.tab_JSON_object = ttk.Frame(right_panel)
        right_panel.add(self.tab_detection_information, text="[Information about detection]")
        right_panel.add(self.tab_JSON_object, text="[JSON object]")

        detection_information_border = tk.Frame(self.tab_detection_information, bg="black")
        detection_information_border.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.text_detection_information = tk.Text(
            detection_information_border, height=10, wrap="word",
            bg="#ffffff", fg="#111111", relief="flat", bd=0,
            font=("Consolas", 10)
        )
        self.text_detection_information.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        JSON_object_border = tk.Frame(self.tab_JSON_object, bg="black")
        JSON_object_border.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.text_JSON_object = tk.Text(
            JSON_object_border, height=10, wrap="none",
            bg="#ffffff", fg="#111111", relief="flat", bd=0,
            font=("Consolas", 10)
        )
        self.text_JSON_object.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.photo: Optional[ImageTk.PhotoImage] = None

    def image_to_pil(self, image_or_path):
        if isinstance(image_or_path, (str, Path)):
            return Image.open(image_or_path)
        if isinstance(image_or_path, Image.Image):
            return image_or_path
        if isinstance(image_or_path, np.ndarray):
            array = image_or_path
            if array.ndim == 2:
                return Image.fromarray(array)
            if array.ndim == 3 and array.shape[2] in (3, 4):
                try:
                    if array.shape[2] == 3:
                        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
                        return Image.fromarray(array)
                    if array.shape[2] == 4:
                        array = cv2.cvtColor(array, cv2.COLOR_BGRA2RGBA)
                        return Image.fromarray(array)
                except Exception:
                    if array.shape[2] == 3:
                        return Image.fromarray(array[..., ::-1])
                    if array.shape[2] == 4:
                        rgb = array[..., :3][..., ::-1]
                        a = array[..., 3:]
                        return Image.fromarray(np.concatenate([rgb, a], axis=2))
            raise TypeError(f"Unsupported ndarray shape for image: {array.shape}")
        raise TypeError(f"Function set_image expects str/Path, PIL.Image.Image or numpy.ndarray, got {type(image_or_path)}")

    def set_image(self, image_or_path):
        image = self.image_to_pil(image_or_path)
        self.gui.update_idletasks()
        canvas_width = max(800, self.canvas.winfo_width() or 1000)
        canvas_height = max(500, self.canvas.winfo_height() or 700)
        image = image.copy()
        image.thumbnail((canvas_width, canvas_height))
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(10, 10, anchor=tk.NW, image=self.photo)

    def set_detection_information(self, detection_information):
        self.text_detection_information.delete("1.0", tk.END)
        self.text_detection_information.insert(tk.END, detection_information or "")

    def set_JSON_object(self, JSON_object):
        self.text_JSON_object.delete("1.0", tk.END)
        try:
            pretty = json.dumps(JSON_object if JSON_object is not None else {}, ensure_ascii=False, indent=2)
        except Exception:
            pretty = str(JSON_object)
        self.text_JSON_object.insert(tk.END, pretty)