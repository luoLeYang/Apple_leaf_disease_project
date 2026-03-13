# GUI/AppleLeafGUI.py
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from model.AppleLeafModel import AppleLeafModel

class AppleLeafGUI:
    def __init__(self):
        self.model_obj = AppleLeafModel()
        self.model_obj.load_model()  # load trained model directly
        self.img_size = self.model_obj.img_size
        self.class_names = self.model_obj.class_names

        self.root = tk.Tk()
        self.root.title("Apple Leaf Disease Detector")

        tk.Button(self.root, text="Upload Image & Predict", command=self.load_and_predict).pack(pady=10)

        self.canvas = tk.Canvas(self.root, width=300, height=300)
        self.canvas.pack()

        self.result_label = tk.Label(self.root, text="", font=("Arial", 14))
        self.result_label.pack()

        self.tk_image = None
        self.root.mainloop()

    def load_and_predict(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_path:
            return
    
        try:
            img = Image.open(file_path).convert("RGB")
            img_resized = img.resize(self.img_size)
    
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # 增加 batch 维度
    
            predicted_label, confidence, healthy_score, diseased_score = self.model_obj.predict_single_image(img_array)
    
            self.tk_image = ImageTk.PhotoImage(img_resized)
            self.canvas.delete("all")
            self.canvas.create_image(150, 150, image=self.tk_image)
    
            self.result_label.config(
                text=(
                    f"Predicted: {self.class_names[predicted_label]} ({confidence:.2f})\n"
                    f"Healthy score: {healthy_score:.2f} | Diseased score: {diseased_score:.2f}"
                )
            )
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
