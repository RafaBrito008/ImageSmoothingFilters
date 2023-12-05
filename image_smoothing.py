import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageFilter
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import ndimage
from scipy.fft import fft2, fftshift


class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Procesador de Imágenes")
        self.image_path = None
        self.setup_ui()

    def setup_ui(self):
        self.frame = tk.Frame(self.root)
        self.frame.pack(padx=10, pady=10)

        self.button_load = tk.Button(
            self.frame, text="Cargar Imagen", command=self.load_image
        )
        self.button_load.pack(side=tk.TOP, pady=5)

        self.canvas = tk.Canvas(self.root, width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        self.root.quit()
        self.root.destroy()

    def load_image(self):
        self.image_path = filedialog.askopenfilename(
            filetypes=[("Archivos de imagen", "*.jpg;*.png;*.jpeg")]
        )
        if self.image_path:
            self.process_image()

    def process_image(self):
        img = cv2.imread(self.image_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Generar ruido y asegurarse de que está en el tipo de datos correcto
        noise = np.random.normal(0, 25, img_gray.shape).astype(np.uint8)

        # Añadir ruido a la imagen en escala de grises
        img_noisy = cv2.add(img_gray, noise)

        filtro_promedio = np.ones((3, 3)) / 9
        filtro_mediana = cv2.getGaussianKernel(3, 1)
        filtro_mediana = np.outer(filtro_mediana, filtro_mediana)

        img_promedio = cv2.filter2D(img_noisy, -1, filtro_promedio)
        img_mediana = cv2.medianBlur(img_noisy, 3)
        img_gaussiano = cv2.filter2D(img_noisy, -1, filtro_mediana)

        img_fft = fftshift(fft2(img_noisy))
        img_fft_abs = np.log(1 + np.abs(img_fft))

        fig, axs = plt.subplots(2, 3, figsize=(10, 7))

        # Mostrar la imagen original en color
        axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title("Imagen Original")
        axs[0, 0].axis("off")

        axs[0, 1].imshow(img_noisy, cmap="gray")
        axs[0, 1].set_title("Imagen con Ruido Sal y Pimienta")
        axs[0, 1].axis("off")

        axs[0, 2].imshow(img_promedio, cmap="gray")
        axs[0, 2].set_title("Filtro Promedio")
        axs[0, 2].axis("off")

        axs[1, 0].imshow(img_mediana, cmap="gray")
        axs[1, 0].set_title("Filtro Mediana")
        axs[1, 0].axis("off")

        axs[1, 1].imshow(img_gaussiano, cmap="gray")
        axs[1, 1].set_title("Filtro Gaussiano")
        axs[1, 1].axis("off")

        axs[1, 2].imshow(np.abs(img_fft_abs), cmap="gray")
        axs[1, 2].set_title("Transformada de Fourier")
        axs[1, 2].axis("off")

        for ax in axs.flat:
            ax.label_outer()

        canvas = FigureCanvasTkAgg(fig, master=self.canvas)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
