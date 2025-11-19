#Resource
#https://scikit-image.org/docs/stable/auto_examples/transform/plot_radon_transform.html

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon_sart, resize
from PIL import Image

#Open image and turn to gray alongside resize to 541 x 541 
im = Image.open(r"C:\Users\Amy\Documents\sinogram\DiskSquareTrianglesRectangle.PNG").convert('L')
image_array = np.array(im)
image_resized = resize(image_array, (541, 541), mode='reflect', anti_aliasing=True)

# Projection angles
theta = np.linspace(0., 180., 180, endpoint=False)  

def compute_sinogram(image, theta):
    return radon(image, theta=theta)

def reconstruct_sart(sinogram, theta):
    return iradon_sart(sinogram, theta=theta)

def plot_results(original, sinogram, reconstruction, title):
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title(f'Original {title}')
    axes[1].imshow(sinogram, cmap='gray', aspect='auto')
    axes[1].set_title(f'Sinogram ({title})')
    axes[2].imshow(reconstruction, cmap='gray')
    axes[2].set_title(f'Reconstruction ({title})')
    
    error = reconstruction - original
    axes[3].imshow(error, cmap='gray')
    axes[3].set_title(f'SART Reconstruction Error: {np.sqrt(np.mean(error**2)):.3g}')
    plt.tight_layout()
    plt.show()

# Compute sinogram and reconstruction
sinogram = compute_sinogram(image_resized, theta)
reconstruction = reconstruct_sart(sinogram, theta)

# Plot results
plot_results(image_resized, sinogram, reconstruction, "Shapes")