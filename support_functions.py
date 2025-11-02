
from __future__ import annotations
from typing import Tuple, List
import numpy as np

def floor(a: float, n: int):
    return np.floor(a * 10**n) / 10**n
def ceil(a: float, n: int):
    return np.ceil(a * 10**n) / 10**n
def euclidean_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    # Odległość między punktami
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

def read_points(path) -> List[Tuple[float, float]]:
    points = []
    with open(path, 'r', encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            x_str, y_str = line.split(';')
            x, y = float(x_str.replace(',', '.')), float(y_str.replace(',', '.'))
            points.append((x, y))
    return points