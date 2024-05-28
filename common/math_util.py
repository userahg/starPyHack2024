import math

import numpy
import numpy as np
import numpy.typing as npt


def least_squares_fit(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.ArrayLike:
    """
    Does a least squares fit for a system of equations Ax=b. Compute the projection p of b onto the column space of a
    and solve for the x that multiples a to produce p.

    :param a:
    :param b:
    :return:
    """
    at = np.transpose(a)
    at_a = np.matmul(at, a)
    at_a_inv = np.linalg.inv(at_a)
    P = np.matmul(at_a_inv, at)
    x_hat = np.matmul(P, b)
    return x_hat


def rotate_vector(x: float, y: float, angle: float, x_orig: float = 0.0, y_orig: float = 0.0) -> (float, float):
    """
    Rotate a vector in 2D about a specified origin.

    :param x: x coordinate of vector to rotate, required
    :type x: float
    :param y: y coordinate of vector to rotate, required
    :type y: float
    :param angle: angle to rotate, required
    :type angle: float
    :param x_orig: x coordinate for rotation axis, optional (default is 0.0)
    :type x_orig: float
    :param y_orig: y coordinate for rotation axis, optional (default is 0.0)
    :return: Rotated coordinates with origin at (x_orig, x_orig)
    """
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    v_old = np.array([x - x_orig, y - y_orig])
    v_new = np.matmul(rotation_matrix, v_old)
    return v_new[0], v_new[1]


def rotate_vector_3d(x: float, y: float, z: float,
                     angle: float,
                     axis_of_rotation: int = 0,
                     x_orig: float = 0.0, y_orig: float = 0.0, z_orig: float = 0.0) -> (float, float, float):
    axes = [0, 1, 2]
    if axis_of_rotation not in axes:
        raise ValueError(f"axis_of_rotation must be one of f{axes}. {axis_of_rotation} is not found in {axes}")
    r_x = numpy.array([[1.0, 0.0, 0.0],
                       [0.0, np.cos(angle), -np.sin(angle)],
                       [0.0, np.sin(angle), np.cos(angle)]])
    r_y = numpy.array([[np.cos(angle), 0.0, np.sin(angle)],
                       [0.0, 1.0, 0.0],
                       [-np.sin(angle), 0.0, np.cos(angle)]])
    r_z = numpy.array([[np.cos(angle), -np.sin(angle), 0.0],
                       [np.sin(angle), np.cos(angle), 0.0],
                       [0.0, 0.0, 1.0]])
    v_old = np.array([x - x_orig, y - y_orig, z - z_orig])
    if axis_of_rotation == 0:
        v_new = np.matmul(r_x, v_old)
    elif axis_of_rotation == 1:
        v_new = np.matmul(r_y, v_old)
    else:
        v_new = np.matmul(r_z, v_old)
    return v_new[0], v_new[1], v_new[2]


def euclidean_distance(x_1: npt.ArrayLike, x_0: npt.ArrayLike) -> float:
    r = x_1 - x_0
    r = np.sqrt(np.square(r).sum())
    return r


def floor(x: float, digits: int = 0) -> float:
    leading = float(math.floor(x))
    trailing = x - leading
    a = 10 ** digits
    temp = trailing * a
    temp_floor = math.floor(temp)
    x_floor = temp_floor / a
    x_floor = round(leading + x_floor, digits)
    return x_floor


def ceiling(x: float, digits: int = 0) -> float:
    leading = float(math.floor(x))
    trailing = x - leading
    a = 10 ** digits
    temp = trailing * a
    temp_ceiling = math.ceil(temp)
    x_ceiling = temp_ceiling / a
    x_ceiling = round(leading + x_ceiling, digits)
    return x_ceiling
