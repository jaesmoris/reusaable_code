"""Flood filling algorithms"""
import numpy as np


def flood_fill4_rec(field: np.ndarray, x: int, y: int,
                    old_value: int | float,
                    new_value: int | float) -> None:
    """Implements the recursive flood filling algorithm
    with a 4-connection.

    Args:
        field (np.ndarray): image
        x (int): row index of the current position
        y (int): column index of the current position
        old_value (int | float): value wanted to be changed
        new_value (int | float): new value for the cells with old values
    """
    # Checking if x and y are valid indices
    if x < 0 or x >= len(field[0]) or y < 0 or y >= len(field):
        return

    # Checking if the current position equals the old value
    if field[y][x] != old_value:
        return
    # Set the current position to the new value
    field[y][x] = new_value

    # Attempt to fill the neighboring positions
    flood_fill4_rec(field, x + 1, y, old_value, new_value)
    flood_fill4_rec(field, x - 1, y, old_value, new_value)
    flood_fill4_rec(field, x, y + 1, old_value, new_value)
    flood_fill4_rec(field, x, y - 1, old_value, new_value)


def flood_fill4_iter(field: np.ndarray, x: int, y: int,
                     old_value: int | float,
                     new_value: int | float) -> None:
    """Implements the iterative flood filling algorithm with a 4-connection
    using a stack, for saving all the pixels to be changed, and a matrix of
    visits, to avoid inserting again a pixel.

    Args:
        field (np.ndarray): image
        x (int): row index of the current position
        y (int): column index of the current position
        old_value (int | float): value wanted to be changed
        new_value (int | float): new value for the cells with old values
    """
    # Checking if x and y are valid indices
    if x < 0 or x >= len(field[0]) or y < 0 or y >= len(field):
        return

    # Definition of the stack and the matrix of visits
    stack = [[x, y]]
    visits = np.zeros(field.shape, dtype=int)

    # While stack not void
    while stack:
        i, j = stack.pop()
        visits[i][j] = 1

        # Checking if the current position equals the old value
        if field[i][j] != old_value:
            continue

        # Set the current position to the new value
        field[i][j] = new_value

        # Attempt to fill the neighboring positions
        if i + 1 < field.shape[0]:
            if visits[i + 1][j] == 0:
                stack.append([i + 1, j])
        if i > 1:
            if visits[i - 1][j] == 0:
                stack.append([i - 1, j])
        if j + 1 < field.shape[1]:
            if visits[i][j + 1] == 0:
                stack.append([i, j + 1])
        if j > 1:
            if visits[i][j - 1] == 0:
                stack.append([i, j - 1])
