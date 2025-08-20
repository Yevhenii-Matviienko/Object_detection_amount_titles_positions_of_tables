import tkinter as tk
from tkinter import filedialog, messagebox
from graphical_user_interface import Object_detection_graphical_user_interface
from object_detector import process_1_page_PDF_file 

class Controller:
    def __init__(self, window):
        self.gui = Object_detection_graphical_user_interface(window)
        self.gui.button_load_1_page_PDF_file.configure(
            command=self.handler_button_click_load_1_page_PDF_file
        )

    def handler_button_click_load_1_page_PDF_file(self):
        path_to_1_page_PDF_file = filedialog.askopenfilename(
            title="[Select 1 page PDF file with tables]",
            filetypes=[("PDF files", "*.pdf")],
        )
        if path_to_1_page_PDF_file:
            self.load_1_page_PDF_file(path_to_1_page_PDF_file)

    def load_1_page_PDF_file(self, path_to_1_page_PDF_file):
        try:
            result = process_1_page_PDF_file(path_to_1_page_PDF_file)
            image = None
            information_about_detection = ""
            JSON_object = {}
            if isinstance(result, dict):
                image = (
                    result.get("image_pil")
                    or result.get("image")
                    or result.get("image_path")
                )
                information_about_detection = result.get("information_about_detection", "")
                JSON_object = result.get("JSON_object", {})
            elif isinstance(result, tuple):
                if len(result) >= 1:
                    image = result[0]
                if len(result) >= 2:
                    information_about_detection = result[1] if result[1] is not None else ""
                if len(result) >= 3:
                    JSON_object = result[2] if result[2] is not None else {}
            else:
                image = result
            self.gui.set_image(image)
            self.gui.set_detection_information(information_about_detection)
            self.gui.set_JSON_object(JSON_object)
        except Exception as exception:
            messagebox.showerror("Error", f"[Failed to load PDF file]\n[{exception}]")
