import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize


def visualize_ssim(image1_path, image2_path):
  # Load images
  img1 = img_as_float(io.imread(image1_path, as_gray=True))
  img2 = img_as_float(io.imread(image2_path, as_gray=True))

  if img1.shape != img2.shape:
    img2 = resize(img2, img1.shape, anti_aliasing=True)

  data_range = max(img1.max() - img1.min(), img2.max() - img2.min())
  ssim_value, ssim_map = ssim(img1, img2, full=True, data_range=data_range)

  fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

  ax1.imshow(img1, cmap='gray')
  ax1.set_title('Image 1')
  ax1.axis('off')

  ax2.imshow(img2, cmap='gray')
  ax2.set_title('Image 2')
  ax2.axis('off')

  im = ax3.imshow(ssim_map, cmap='viridis')
  ax3.set_title(f'SSIM Map\nSSIM Value: {ssim_value:.4f}')
  ax3.axis('off')

  plt.colorbar(im, ax=ax3)

  plt.tight_layout()
  plt.savefig('ssim_visualization.png')
  plt.close()

  return ssim_value


if __name__ == "__main__":
  ssim_value = visualize_ssim('745439676.jpeg',
                              'photo_2024-08-05_10-51-54.jpg')
  print(f"SSIM visualization saved as 'ssim_visualization.png'")
  print(f"SSIM Value: {ssim_value:.4f}")
