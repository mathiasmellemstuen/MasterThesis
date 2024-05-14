from geometric_shape_generator.geometric_shapes import GeometricShape
from geometric_shape_generator.color_mode import ColorMode

import numpy as np
import cv2

def create_right_triangle(image: np.ndarray, center_position: tuple[int, int], size: int, color: tuple[int, ...]): 
    """
    Fills an image with a right-triangle with certain parameters.

    Parameters
    ----------
    image : np.ndarray
        The current image that the triangle should be drawed on. 
    center_position: tuple[int, int]
        The triangles center position
    size: int
        The size of the triangle
    color: tuple[int, ...]
        The fill color of the triangle
    """
    top_left = (center_position[0] - size, center_position[1] - size)
    bottom_left = (top_left[0], top_left[1] + 2 * size)
    bottom_right = (top_left[0] + 2 * size, top_left[1])

    vertices = np.array([top_left, bottom_left, bottom_right])
    vertices = [vertices.reshape((-1, 1, 2))]

    cv2.polylines(image, vertices, True, color, thickness=1)
    cv2.fillPoly(image, vertices, color)

def create_equiangular_triangle(image: np.ndarray, center_position: tuple[int, int], size: int, color: tuple[int, ...]): 
    """
    Fills an image with an equiangular-triangle with certain parameters.

    Parameters
    ----------
    image : np.ndarray
        The current image that the triangle should be drawed on. 
    center_position: tuple[int, int]
        The triangles center position
    size: int
        The size of the triangle
    color: tuple[int, ...]
        The fill color of the triangle
    """
    top = (center_position[0], center_position[1] + size)
    bottom_left = (center_position[0] - size, center_position[1] - size)
    bottom_right = (center_position[0] + size, center_position[1] - size)

    vertices = np.array([top, bottom_left, bottom_right])
    vertices = [vertices.reshape((-1, 1, 2))]

    cv2.polylines(image, vertices, True, color, thickness=1)
    cv2.fillPoly(image, vertices, color)

def create_rectangle(image: np.ndarray, center_position: tuple[int, int], size: tuple[int, int], color: tuple[int, ...]):
        """
        Fills an image with a rectangle with certain parameters.

        Parameters
        ----------
        image : np.ndarray
            The current image that the rectangle should be drawed on. 
        center_position: tuple[int, int]
            The triangles center position
        size: tuple[int, int]
            The size of the rectangle (width and height)
        color: tuple[int, ...]
            The fill color of the rectangle
        """

        odd_modifier = 1 if not size[0] % 2 == 0 else 0

        size_0 = int(size[0] / 2)
        size_1 = int(size[1] / 2)

        top_left = (center_position[0] - size_0 - odd_modifier, center_position[1] + size_1 + odd_modifier)
        bottom_right = (center_position[0] + size_1, center_position[1] - size_1)

        cv2.rectangle(image, top_left, bottom_right, color, -1)


def create_random_angle_steps(steps: int, irregularity: float): 
    angles = []
    lower_boundary = (2 * np.pi / steps) - irregularity
    upper_boundary = (2 * np.pi / steps) + irregularity

    cumulative_sum = 0

    for i in range(steps): 
        angle = np.random.uniform(lower_boundary, upper_boundary)
        angles.append(angle)
        cumulative_sum += angle

    cumulative_sum = cumulative_sum / (2 * np.pi)
    
    for i in range(steps): 
        angles[i] = angles[i] / cumulative_sum

    return angles

def clip(value, low, high):

    return min(high, max(value, low))

def create_random_polygon(image: np.ndarray, center_position: tuple[int, int] = (64, 64), color: tuple[int, ...] = 255, average_radius: float = 1.0, irregularity: float = 0.5, spikiness: float = 0.5, num_vertices: int = 5):
    """
        Generating a image of a random polygon with certain parameters. Code inspired by answer on stackoverflow post: https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon

        Parameters
        ----------
        image : np.ndarray
            The current image that the rectangle should be drawed on. 
        center_position: tuple[int, int]
            The polygons center position
        color: tuple[int, ...]
            The fill color of the polygon
        average_radius: float
            The average radius (distance of each generated vertex to the center of the cercumference) used to generate points with normal distribution
        irregularity: float
            Variance of the spacing of the angles between the consecutive vertices
        spikiness: float
            Variance of the distance of each vertex to the center of the cercumference
        num_vertices: int
            The number of vertices in the polygon
    """
    
    irregularity = irregularity * 2 * np.pi / num_vertices
    spikiness = spikiness * average_radius

    angle_steps = create_random_angle_steps(num_vertices, irregularity)

    points = []

    angle = np.random.uniform(0, 2 * np.pi)

    for i in range(num_vertices): 
        radius = clip(np.random.normal(average_radius, spikiness), 0, 2 * average_radius)
        point = (center_position[0] + radius * np.cos(angle), center_position[1] + radius * np.sin(angle))
        points.append(point)
        angle += angle_steps[i]
    
    points = np.array(points, dtype=np.int32)

    cv2.fillPoly(image, [points], color)

def generate_geometric_shape(center_position: tuple[int, int] = (64, 64), image_size: tuple[int, int] = (128, 128), size: tuple[int, ...] = 25, color: tuple[int, ...] = (255, 255, 255), random_polygon_num_vertices: int = 5, random_polygon_irregularity: float = 0.5, random_polygon_spikiness: float = 0.5, random_polygon_average_radius: float = 1, shape_type: GeometricShape = GeometricShape.CIRCLE, color_mode: ColorMode = ColorMode.BINARY):
    """
    Generating an image with a simple geometric shape. Can generate: 
    - Circle
    - Rectangle
    - Right triangle
    - Equiangular triangle

    The position, size and color can be assigned. The color and size can be assigned at random with various constraints. 

    Parameters
    ----------
    center_position : tuple[int, int]
        The center position of the figure.
    image_size : tuple[int, int]
        The size of the image (width and height)
    size : tuple[int, ...]
        The size of the figure. Assign a tuple[int, int] (width, height) if the shape is a rectangle. Else, assign a single integer for any other shape type. 
    color : tuple[int, ...]
        The color of the figure. Assign a single integer if grayscale color mode is selected. Else, assign a tuple[int, int, int] with RGB values. 
    shape_type : GeometricShape
        Determines the shape type that should be generated. Can be any value in the GeomtricShape enum. 
    color_mode : ColorMode
        Determines the color-mode that should be generated. Can be any value in the ColorMode enum. 

    Returns
    -------
    np.ndarray
        Image with final generated geometric shape.
    """

    three_channels = False if color_mode == ColorMode.BINARY or color_mode == ColorMode.GRAYSCALE else True

    # Creating numpy array with size (width, height) for binary or grayscale images and (width, height, 3) for rgb images
    image = np.zeros(image_size if not three_channels else (image_size[0], image_size[1], 3), np.uint8)

    color = 1 if color_mode == ColorMode.BINARY else color

    if shape_type == GeometricShape.CIRCLE: 
        cv2.circle(image, center_position, size, color=color, thickness=-1)
    elif shape_type == GeometricShape.RECTANGLE:

        # Changing size to a tuple if it is a single value. This will make the rectangle a square in this special case
        if(type(size) == int or type(size) == float or type(size) == np.int64): 
            size = (size, size)

        create_rectangle(image, center_position, size, color)
    elif shape_type == GeometricShape.RIGHT_TRIANGLE: 
        create_right_triangle(image, center_position, size, color)
    elif shape_type == GeometricShape.EQUIANGULAR_TRIANGLE: 
        create_equiangular_triangle(image, center_position, size, color)
    elif shape_type == GeometricShape.RANDOM_POLYGON:
        create_random_polygon(image, center_position, color, random_polygon_average_radius, random_polygon_irregularity, random_polygon_spikiness, random_polygon_num_vertices)

    return image
