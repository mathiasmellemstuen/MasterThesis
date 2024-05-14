import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np

def color_gradient_min_red_max_blue(min_value, max_value, current_value):

    gradient_value = (current_value - min_value) / (max_value - min_value)

    return (int(255 * (1.0 - gradient_value)), 0, int(255 * gradient_value))

def color_gradient_min_red_max_blue_inverse(r, b):
    return (b / 255)

# Showing the color gradient in the image
if __name__ == "__main__":
    image = np.zeros((100, 300, 3), dtype=np.uint8)

    # Constructing a image from the gradient
    for i in range(300):
        color = color_gradient_min_red_max_blue(0, 300, i)
        image[:, i, 0] = color[0]
        image[:, i, 1] = color[1]
        image[:, i, 2] = color[2]

    # Saving the gradient to file
    matplotlib.image.imsave("data/figures/r_to_b_gradient.png", image)

    # Plotting the gradient
    plt.figure()
    plt.imshow(image)
    plt.show()
