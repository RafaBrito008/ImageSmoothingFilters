import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.fft import fft2, fftshift
import os


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


class FourierTransforms:
    @staticmethod
    def pad_to_power_of_two(signal):
        """
        Aumenta el tamaño del array al siguiente número que sea potencia de 2.
        """
        original_len = len(signal)
        new_len = 2**np.ceil(np.log2(original_len)).astype(int)
        return np.pad(signal, (0, new_len - original_len), mode='constant', constant_values=0)

    @staticmethod
    def fft1d(signal):
        """
        Calcula la Transformada Rápida de Fourier (FFT) de un array 1D.
        """
        signal = FourierTransforms.pad_to_power_of_two(signal)
        N = len(signal)

        if N <= 1:
            return signal

        even = FourierTransforms.fft1d(signal[0::2])
        odd = FourierTransforms.fft1d(signal[1::2])

        T = [np.exp(-2j * np.pi * k / N) * odd[k % (N//2)] for k in range(N)]
        return [even[k % (N//2)] + T[k] for k in range(N)] + [even[k % (N//2)] - T[k] for k in range(N)]

    @staticmethod
    def fft2d(image):
        """
        Calcula la Transformada Rápida de Fourier (FFT) de un array 2D.
        """
        h, w = image.shape
        padded_image = np.pad(image, ((0, h - image.shape[0]), (0, w - image.shape[1])), mode='constant', constant_values=0)

        fft_rows = np.array([FourierTransforms.fft1d(row) for row in padded_image])
        fft_image = np.array([FourierTransforms.fft1d(fft_rows[:, col]) for col in range(w)])
        return fft_image.T

    @staticmethod
    def shift_fft_quadrants(fft_image):
        """
        Reorganiza los cuadrantes de la imagen de la FFT para que las bajas frecuencias estén en el centro del arreglo.

        :param fft_image: Resultado de la FFT 2D.
        :return: Imagen FFT con cuadrantes reorganizados.
        """
        N, M = fft_image.shape
        # Dividir la imagen en 4 cuadrantes y reorganizarlos
        top_left = fft_image[: N // 2, : M // 2]
        top_right = fft_image[: N // 2, M // 2 :]
        bottom_left = fft_image[N // 2 :, : M // 2]
        bottom_right = fft_image[N // 2 :, M // 2 :]

        # Intercambiar cuadrantes diagonales
        new_top_left = bottom_right
        new_top_right = bottom_left
        new_bottom_left = top_right
        new_bottom_right = top_left

        # Combinar los cuadrantes reorganizados en una nueva imagen
        top_half = np.concatenate((new_top_left, new_top_right), axis=1)
        bottom_half = np.concatenate((new_bottom_left, new_bottom_right), axis=1)
        return np.concatenate((top_half, bottom_half), axis=0)

    @staticmethod
    def calculate_magnitude_spectrum(fft_image):
        """
        Calcula la magnitud del espectro de frecuencias de la imagen FFT.

        :param fft_image: Resultado de la FFT 2D.
        :return: Magnitud del espectro de frecuencias.
        """
        # Calcular la magnitud del espectro
        magnitude_spectrum = np.abs(fft_image)
        # Escalar el espectro para mejorar la visualización
        magnitude_spectrum = np.log1p(magnitude_spectrum)
        return magnitude_spectrum


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

    def save_image(self, image, image_name):
        """
        Guarda una imagen procesada en el disco duro.

        :param image: Imagen procesada a guardar.
        :param image_name: Nombre bajo el cual se guardará la imagen.
        """
        # Obtener el nombre base de la imagen original
        original_name = os.path.splitext(os.path.basename(self.image_path))[0]
        
        # Crear un nombre de carpeta con '_Processed' al final
        folder_name = f"{original_name}_Processed"

        # Verificar si la carpeta existe. Si no, crearla.
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Construir la ruta completa del archivo donde se guardará la imagen
        filename = os.path.join(folder_name, f"{original_name}_{image_name}.jpg")

        # Guardar la imagen en el formato deseado
        cv2.imwrite(filename, image)

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
        fft_image = FourierTransforms.fft2d(img_noisy)
        shifted_fft = FourierTransforms.shift_fft_quadrants(fft_image)
        magnitude_spectrum = FourierTransforms.calculate_magnitude_spectrum(shifted_fft)

                # Guardar las imágenes procesadas
        self.save_image(img_noisy, "noisy")
        self.save_image(img_promedio, "average_filtered")
        self.save_image(img_mediana, "median_filtered")
        self.save_image(img_gaussiano, "gaussian_filtered")

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

        # Mostrar el espectro de la Transformada de Fourier
        axs[1, 2].imshow(magnitude_spectrum, cmap="gray")
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
