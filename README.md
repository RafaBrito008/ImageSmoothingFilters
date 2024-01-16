## Overview
This Python application, built using Tkinter, OpenCV, and Matplotlib, provides an interface for applying various smoothing filters to images. It includes functionality for adding noise to images and applying average, median, and Gaussian filters. Additionally, it features Fourier transform visualization for frequency analysis.

## Features
- Load and display images using a graphical user interface.
- Add salt-and-pepper noise to images.
- Apply average, median, and Gaussian smoothing filters.
- Display the Fourier transform of noisy images.
- Save processed images to disk.

![smoothing_filters](https://github.com/RafaBrito008/ImageSmoothingFilters/assets/94416107/09742555-4429-4535-b903-7347d163a824)


## Requirements
- Python 3.x
- Tkinter
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- SciPy

## Installation
Ensure Python 3.x and the above-listed libraries are installed. No additional installation is required for the application itself.

## Usage
Run the script to launch the GUI. Use the "Cargar Imagen" button to load an image. Processed images will be displayed within the application, and saved in a folder named after the original image with "_Processed" appended.

## Note
The application is designed for educational and demonstration purposes, focusing on image processing techniques. It may require modifications for advanced or specific use cases.
