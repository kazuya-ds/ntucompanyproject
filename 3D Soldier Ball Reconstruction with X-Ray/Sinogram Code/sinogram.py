#Resource
#https://scikit-image.org/docs/stable/auto_examples/transform/plot_radon_transform.html

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon_sart
from skimage.draw import disk, rectangle, polygon


# Parameters
image_size = 128
theta = np.linspace(0., 180., 180, endpoint=False)  # Projection angles

#Make shapes
def generate_disk(radius=30, center=None):
    if center is None:
        center = (image_size // 2, image_size // 2)
    img = np.zeros((image_size, image_size))
    rr, cc = disk(center, radius, shape=img.shape)
    img[rr, cc] = 1
    return img

def generate_square(size=40, center=None):
    if center is None:
        center = (image_size // 2, image_size // 2)
    img = np.zeros((image_size, image_size))
    start = (center[0] - size // 2, center[1] - size // 2)
    end = (start[0] + size, start[1] + size)
    rr, cc = rectangle(start, end=end, shape=img.shape)
    img[rr, cc] = 1
    return img

def generate_rectangle(width=30, height=50, center=None):
    if center is None:
        center = (image_size // 2, image_size // 2)
    img = np.zeros((image_size, image_size))
    start = (center[0] - height // 2, center[1] - width // 2)
    end = (start[0] + height, start[1] + width)
    rr, cc = rectangle(start, end=end, shape=img.shape)
    img[rr, cc] = 1
    return img

def generate_triangle(base=40, height=40, center=None):
    if center is None:
        center = (image_size // 2, image_size // 2)
    img = np.zeros((image_size, image_size))
    vertices = np.array([
        [center[0], center[1] - base // 2],
        [center[0], center[1] + base // 2],
        [center[0] + height, center[1]]
    ])
    rr, cc = polygon(vertices[:, 0], vertices[:, 1], img.shape)
    img[rr, cc] = 1
    return img

disk_img = generate_disk(radius=30)
square_img = generate_square(size=50)
rectangle_img = generate_rectangle(width=30, height=50)
triangle_img = generate_triangle(base=50, height=60)

#Find sinogram using radon
def compute_sinogram(image, theta):
    return radon(image, theta=theta, circle=True)

sinogram_disk = compute_sinogram(disk_img, theta)
sinogram_square = compute_sinogram(square_img, theta)
sinogram_rectangle = compute_sinogram(rectangle_img, theta)
sinogram_triangle = compute_sinogram(triangle_img, theta)

#use skimage sart to reconstruct
def reconstruct_sart(sinogram, theta):
    return iradon_sart(sinogram, theta=theta)

recon_disk = reconstruct_sart(sinogram_disk, theta)
recon_square = reconstruct_sart(sinogram_square, theta)
recon_rectangle = reconstruct_sart(sinogram_rectangle, theta)
recon_triangle = reconstruct_sart(sinogram_triangle, theta)

#Plots
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
    axes[3].set_title(f'SART Reconstruction Error: ' f'{np.sqrt(np.mean(error**2)):.3g}')
    plt.tight_layout()
    plt.show()

plot_results(disk_img, sinogram_disk, recon_disk, "Disk")
plot_results(square_img, sinogram_square, recon_square, "Square")
plot_results(rectangle_img, sinogram_rectangle, recon_rectangle, "Rectangle")
plot_results(triangle_img, sinogram_triangle, recon_triangle, "Triangle")