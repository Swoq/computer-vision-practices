import numpy as np
import cv2
import math


def draw_point_matrix(matrix, offset_x=256, offset_y=256):
    img = np.ones((512, 512, 3), dtype=np.uint8) * 255
    for i in range(-1, matrix.shape[1] - 1):
        cv2.line(img,
                 (int(matrix[0, i] + offset_x), int(matrix[1, i] + offset_y)),
                 (int(matrix[0, i + 1] + offset_x), int(matrix[1, i + 1] + offset_y)),
                 (0, 0, 0), 1)
    cv2.imshow("", img)
    cv2.waitKey(500)


def rotate_matrix(angle):
    return np.array([[math.cos(math.radians(angle)), -math.sin(math.radians(angle)), 0],
                     [math.sin(math.radians(angle)), math.cos(math.radians(angle)), 0],
                     [0, 0, 1]])


def scale_matrix(cof):
    return np.array([[cof, 0, 0],
                     [0, cof, 0],
                     [0, 0, 1]])


def shear_matrix(cof):
    return np.array([[1, cof, 0],
                     [cof, 1, 0],
                     [0, 0, 1]])


def translate_matrix(cof):
    return np.array([[1, 0, cof],
                     [0, 1, cof],
                     [0, 0, 1]])


def create_2d_figure(outer_radius, edges):
    return_figure = np.ones((3, edges), dtype=np.float32)
    for i in range(0, edges):
        angle = i * 360 / edges
        return_figure[0, i] = outer_radius * math.cos(math.radians(angle))
        return_figure[1, i] = outer_radius * math.sin(math.radians(angle))
    return return_figure


def draw_figure_transformation(fig, trans_func, trans_param):
    temp = fig.copy()
    for i in range(10):
        temp = np.dot(trans_func(trans_param), temp)
        print("Figure after applying ", trans_func, "\n", temp)
        draw_point_matrix(temp)


if __name__ == '__main__':
    # CREATE PENTAGON MATRIX
    figure = create_2d_figure(128, 5)

    # ROTATE
    draw_figure_transformation(figure, rotate_matrix, 10)

    # SCALE
    draw_figure_transformation(figure, scale_matrix, 1.1)

    # SHIFT
    draw_figure_transformation(figure, translate_matrix, -10)

    # SHEAR
    draw_figure_transformation(figure, shear_matrix, 0.2)
    draw_figure_transformation(figure, shear_matrix, -0.2)
