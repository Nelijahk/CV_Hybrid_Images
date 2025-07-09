from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from MyHybridImages import myHybridImages
import cv2


def progressively_downsample(image: np.ndarray, num_levels: int = 5, scale_factor: float = 0.5) -> np.ndarray:
    """
    Create a progressively down-sampled version of the image in a single composite image.

    :param image: Input hybrid image in RGB format.
    :param num_levels: Number of downsampling levels.
    :param scale_factor: Factor by which the image is reduced in each step.
    :return: Concatenated image showing all levels of downsampling.
    """
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    downsampled_images = [image_bgr]  # Start with the original image

    for _ in range(num_levels - 1):
        h, w = downsampled_images[-1].shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)

        if new_h < 1 or new_w < 1:
            break  # Stop if the image becomes too small

        resized = cv2.resize(downsampled_images[-1], (new_w, new_h), interpolation=cv2.INTER_AREA)
        downsampled_images.append(resized)

    # Pad images to the same height
    max_height = downsampled_images[0].shape[0]
    downsampled_images = [cv2.copyMakeBorder(img, 0, max_height - img.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                          for img in downsampled_images]

    # Concatenate images horizontally
    final_image = np.hstack(downsampled_images)
    filename = f"hybrid.png"
    cv2.imwrite(filename, final_image)  # Save current image
    print(f"Saved: {filename}")

    return final_image



low_img = np.array(Image.open("hybrid-images/data/king_upd.jpg"))
high_img = np.array(Image.open("hybrid-images/data/clown_upd.jpg"))

# low_img = cv2.cvtColor(low_img, cv2.COLOR_BGR2GRAY)
# high_img = cv2.cvtColor(high_img, cv2.COLOR_BGR2GRAY)

print(low_img.shape, high_img.shape)

hybrid = myHybridImages(low_img, 4.0, high_img, 20.0)

progressively_downsample(hybrid)

plt.imshow(hybrid, cmap="gray")
plt.title("Hybrid Image")
plt.show()