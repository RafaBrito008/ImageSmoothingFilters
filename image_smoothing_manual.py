import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.fft import fft2, fftshift


class SmoothingFilters:
    @staticmethod
    def apply_average_filter(image, kernel_size):
        """
        Aplica un filtro promedio a la imagen proporcionada utilizando un kernel cuadrado del tamaño especificado.
        El filtro promedio reemplaza cada píxel con el promedio de los píxeles de su vecindario.

        :param image: Imagen de entrada como una matriz NumPy.
        :param kernel_size: Tamaño del lado del kernel cuadrado.
        :return: Imagen filtrada.
        """
        # Agrega píxeles alrededor de los bordes para permitir el cálculo del vecindario para los píxeles de borde.
        padded_image = cv2.copyMakeBorder(
            image,
            kernel_size // 2,
            kernel_size // 2,
            kernel_size // 2,
            kernel_size // 2,
            cv2.BORDER_REFLECT,
        )

        # Inicializa la imagen de salida con ceros, del mismo tamaño que la imagen de entrada.
        output_image = np.zeros_like(image)

        # Itera sobre cada píxel de la imagen.
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                # Extrae la región del vecindario del píxel actual basado en el tamaño del kernel.
                kernel_region = padded_image[i : i + kernel_size, j : j + kernel_size]
                # Calcula el promedio de los píxeles del vecindario y asigna el resultado al píxel correspondiente.
                output_image[i, j] = np.mean(kernel_region)

        return output_image

    @staticmethod
    def apply_median_filter(image, kernel_size):
        """
        Aplica un filtro de mediana a la imagen utilizando un kernel cuadrado del tamaño especificado.
        El filtro de mediana reemplaza cada píxel por la mediana de los píxeles en su vecindario.

        :param image: Imagen de entrada como una matriz NumPy.
        :param kernel_size: Tamaño del lado del kernel cuadrado.
        :return: Imagen filtrada.
        """
        # Agrega píxeles alrededor de los bordes de la misma manera que en el filtro promedio.
        padded_image = cv2.copyMakeBorder(
            image,
            kernel_size // 2,
            kernel_size // 2,
            kernel_size // 2,
            kernel_size // 2,
            cv2.BORDER_REFLECT,
        )

        # Inicializa la imagen de salida con ceros.
        output_image = np.zeros_like(image)

        # Itera sobre cada píxel de la imagen.
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                # Extrae la región del vecindario del píxel actual.
                kernel_region = padded_image[i : i + kernel_size, j : j + kernel_size]
                # Calcula la mediana de los píxeles del vecindario y asigna el resultado al píxel correspondiente.
                output_image[i, j] = np.median(kernel_region)

        return output_image

    @staticmethod
    def generate_gaussian_kernel(kernel_size, sigma):
        """
        Genera un kernel gaussiano que se utilizará para el filtrado gaussiano.

        :param kernel_size: Tamaño del lado del kernel cuadrado.
        :param sigma: Desviación estándar de la distribución gaussiana.
        :return: Kernel gaussiano como una matriz NumPy.
        """
        # Crea un rango de valores de acuerdo al tamaño del kernel para calcular el kernel gaussiano.
        ax = np.linspace(-(kernel_size - 1) / 2.0, (kernel_size - 1) / 2.0, kernel_size)
        # Crea una cuadrícula 2D de valores desde la matriz lineal.
        xx, yy = np.meshgrid(ax, ax)
        # Aplica la fórmula gaussiana a cada elemento de la cuadrícula.
        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
        # Normaliza el kernel para que la suma de todos sus elementos sea igual a 1.
        return kernel / np.sum(kernel)

    @staticmethod
    def apply_gaussian_filter(image, kernel_size, sigma):
        """
        Aplica un filtro gaussiano a la imagen proporcionada.

        :param image: Imagen de entrada como una matriz NumPy.
        :param kernel_size: Tamaño del lado del kernel cuadrado.
        :param sigma: Desviación estándar del kernel gaussiano.
        :return: Imagen filtrada.
        """
        # Genera el kernel gaussiano utilizando la función anterior.
        kernel = SmoothingFilters.generate_gaussian_kernel(kernel_size, sigma)
        # Agrega píxeles alrededor de los bordes de la imagen.
        padded_image = cv2.copyMakeBorder(
            image,
            kernel_size // 2,
            kernel_size // 2,
            kernel_size // 2,
            kernel_size // 2,
            cv2.BORDER_REFLECT,
        )

        # Inicializa la imagen de salida con ceros.
        output_image = np.zeros_like(image)

        # Aplica el kernel gaussiano a cada píxel de la imagen.
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                # Extrae la región correspondiente al vecindario del píxel.
                kernel_region = padded_image[i : i + kernel_size, j : j + kernel_size]
                # Realiza la operación de convolución: multiplica el kernel por los píxeles del vecindario y suma los resultados.
                output_image[i, j] = np.sum(kernel_region * kernel)

        return output_image


# Clase principal de la aplicación de procesamiento de imágenes
class ImageProcessorApp:
    def __init__(self, root):
        # Constructor de la clase
        self.root = root  # Referencia a la ventana principal de Tkinter
        self.root.title("Procesador de Imágenes")  # Título de la ventana
        self.image_path = None  # Ruta de la imagen a procesar
        self.setup_ui()  # Inicializar la interfaz de usuario

    def setup_ui(self):
        # Configuración de la interfaz de usuario
        self.frame = tk.Frame(self.root)  # Crear un marco en la ventana principal
        self.frame.pack(padx=10, pady=10)  # Empaquetar el marco con un poco de espacio

        # Botón para cargar imágenes
        self.button_load = tk.Button(
            self.frame, text="Cargar Imagen", command=self.load_image
        )
        self.button_load.pack(side=tk.TOP, pady=5)  # Posicionar el botón

        # Crear un lienzo para mostrar imágenes
        self.canvas = tk.Canvas(self.root, width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)  # Empaquetar el lienzo

        # Vincular el evento de cierre de la ventana
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        # Método para manejar el cierre de la ventana
        self.root.quit()  # Terminar el bucle principal
        self.root.destroy()  # Destruir la ventana

    def load_image(self):
        # Método para cargar una imagen
        self.image_path = filedialog.askopenfilename(
            filetypes=[("Archivos de imagen", "*.jpg;*.png;*.jpeg")]
        )  # Mostrar un cuadro de diálogo para elegir un archivo
        if self.image_path:
            self.process_image()  # Si se selecciona una imagen, procesarla

    def process_image(self):
        # Parámetros configurables para el procesamiento de la imagen
        noise_intensity = 25  # Intensidad del ruido a añadir. Cuanto mayor es el valor, más intenso es el ruido.
        filter_size = 3  # Tamaño de los filtros. Define el área que afectará cada filtro (3x3 en este caso).
        gaussian_sigma = 1  # Desviación estándar para el filtro gaussiano. Afecta la "suavidad" del filtro.

        # Cargar la imagen y convertirla a escala de grises
        img = cv2.imread(self.image_path)  # Leer la imagen desde la ruta especificada
        img_gray = cv2.cvtColor(
            img, cv2.COLOR_BGR2GRAY
        )  # Convertir la imagen a escala de grises para simplificar el procesamiento

        # Añadir ruido tipo 'sal y pimienta'
        # El ruido se genera como una matriz de valores aleatorios siguiendo una distribución normal
        noise = np.random.normal(0, noise_intensity, img_gray.shape).astype(np.uint8)
        img_noisy = cv2.add(
            img_gray, noise
        )  # Sumar el ruido a la imagen en escala de grises

        # Aplicar filtros a la imagen
        # Aplicar filtro promedio
        img_promedio = SmoothingFilters.apply_average_filter(img_noisy, filter_size)

        # Aplicar filtro de mediana
        img_mediana = SmoothingFilters.apply_median_filter(img_noisy, filter_size)

        # Aplicar filtro gaussiano
        img_gaussiano = SmoothingFilters.apply_gaussian_filter(
            img_noisy, filter_size, gaussian_sigma
        )

        # Realizar transformada de Fourier y calcular espectro
        # La transformada de Fourier convierte la imagen del dominio del espacio al dominio de la frecuencia
        img_fft = fftshift(
            fft2(img_noisy)
        )  # Calcula la transformada de Fourier y centra el espectro
        img_fft_abs = np.log(
            1 + np.abs(img_fft)
        )  # Usa el logaritmo para mejorar la visualización del espectro de frecuencias

        # Configuración de la visualización de resultados
        fig, axs = plt.subplots(
            2, 3, figsize=(10, 7)
        )  # Crear una figura con 6 subplots (2 filas, 3 columnas)

        # Mostrar las imágenes procesadas en los subplots
        # Imagen original en color (se convierte de BGR a RGB para mostrar correctamente con Matplotlib)
        axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title("Imagen Original")
        axs[0, 0].axis("off")

        # Imagen con ruido en escala de grises
        axs[0, 1].imshow(img_noisy, cmap="gray")
        axs[0, 1].set_title("Imagen con Ruido Sal y Pimienta")
        axs[0, 1].axis("off")

        # Imagen con filtro promedio aplicado
        axs[0, 2].imshow(img_promedio, cmap="gray")
        axs[0, 2].set_title("Filtro Promedio")
        axs[0, 2].axis("off")

        # Imagen con filtro de mediana aplicado
        axs[1, 0].imshow(img_mediana, cmap="gray")
        axs[1, 0].set_title("Filtro Mediana")
        axs[1, 0].axis("off")

        # Imagen con filtro gaussiano aplicado
        axs[1, 1].imshow(img_gaussiano, cmap="gray")
        axs[1, 1].set_title("Filtro Gaussiano")
        axs[1, 1].axis("off")

        # Espectro de la Transformada de Fourier de la imagen con ruido
        axs[1, 2].imshow(np.abs(img_fft_abs), cmap="gray")
        axs[1, 2].set_title("Transformada de Fourier")
        axs[1, 2].axis("off")

        # Ajustar etiquetas y mostrar figura en el lienzo de Tkinter
        for ax in axs.flat:
            ax.label_outer()

        canvas = FigureCanvasTkAgg(
            fig, master=self.canvas
        )  # Integrar la figura de Matplotlib en el lienzo de Tkinter
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        canvas.draw()


# Bloque principal para ejecutar la aplicación
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
