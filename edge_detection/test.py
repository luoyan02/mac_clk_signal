import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# Define the 6x6 input image (random grayscale values between 0 and 255)
image = np.array([
    [600, 0, 0, 0, 0, 600],
    [600, 0, 0, 0, 0, 600],
    [600, 0, 0, 0, 0, 600],
    [600, 0, 0, 0, 0, 600],
    [600, 0, 0, 0, 0, 600],
    [600, 0, 0, 0, 0, 600]
], dtype=np.float32)

# Define a single edge detection kernel
kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

# Apply the edge detection kernel to the image
edges = convolve2d(image, kernel, mode='valid', boundary='fill', fillvalue=0)

# Normalize the edge-detected image for better visualization (scale to 0-255)
edges_normalized = (edges - edges.min()) / (edges.max() - edges.min()) * 255

# Save the original image as a separate file
plt.imshow(image, cmap='gray', interpolation='nearest')
plt.title("Original Image")
plt.axis('off')
plt.savefig("original_image.png", bbox_inches='tight')  # Save original image as PNG
plt.clf()  # Clear the figure to avoid overlap

# Save the edge-detected image as a separate file
plt.imshow(edges_normalized, cmap='gray', interpolation='nearest')
plt.title("Edge Detected Image")
plt.axis('off')
plt.savefig("edge_detected_image.png", bbox_inches='tight')  # Save edge-detected image as PNG
plt.clf()  # Clear the figure to avoid overlap

print("Images saved successfully.")
