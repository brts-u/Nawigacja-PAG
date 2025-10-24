"""
Moduł z klasą grafu sieci drogowej oraz funkcjami do tworzenia go z pliku shapefile SKJZ (jezdni) w strukturze BDOT10k.

Autor: Bartosz Urbanek 331931
Data: 2025-10-17
"""
from __future__ import annotations
from typing import List, Tuple, Dict

import numpy as np
import geopandas as gpd
from scipy.spatial import KDTree
import matplotlib.pyplot as plt


def euclidean_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    # Odległość między punktami
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)
def approx(a: float, b: float, tol: float = 0.10) -> bool:
    # Przybliżenie
    return abs(a - b) < tol
def point_approx(coord1: Tuple[float, float], coord2: Tuple[float, float], tol: float = 0.10) -> bool:
    # Przybliżenie dla punktów (krotek 2D)
    return approx(coord1[0], coord2[0], tol) and approx(coord1[1], coord2[1], tol)
def node_id(x: float, y: float) -> str:
    # ID węzła na podstawie współrzędnych (może lepiej robić jakoś inaczej?)
    return f'({x} {y})'

# Klasa węzła grafu
class Node:
    def __init__(self, x: float, y: float):
        self.id: str = node_id(x, y)
        self.x: float = x
        self.y: float = y
        self.edges: List[Edge] = []

    # Zwraca listę sąsiadów i drogi do nich, sąsiady wielopowiązane rozróżnialne przez ID krawędzi
    def get_neighbours(self) -> List[Tuple[Node, Edge]]:
        return [(edge.get_other_node(self), edge) for edge in self.edges]
        # Teoretycznie powinien być blok Try-Except, ale to by oznaczało, że graf został błędnie zbudowany

    def heuristic(self, other: Node) -> float:
        return euclidean_distance((self.x, self.y), (other.x, other.y))

# Prędkości przypisane do klas dróg w km/h
speed = {   # Okazuje się, że nie da się tego zmapować tak 1:1, więc wybrałem takie, które mają największy sens
    "A": 140,   # Autostrada
    "S": 120,   # Droga ekspresowa
    "GP": 90,   # Droga główna przyspieszona
    "G": 70,    # Droga główna
    "Z": 50,    # Droga zbiorcza
    "L": 40,    # Droga lokalna
    "I": 30     # Droga dojazdowa
}

class Edge:
    def __init__(self, id: str, start_node: Node, end_node: Node, length: float, classification: str, one_way: bool = False):
        self.id = id
        self.start_node = start_node
        self.end_node = end_node
        self.length = length
        self.classification = classification
        self.one_way = one_way

    def get_other_node(self, current_node: Node) -> Node:
        if current_node == self.start_node:
            return self.end_node
        elif current_node == self.end_node:
            return self.start_node
        else:
            raise ValueError("Node not part of this edge")

    # Koszt przejazdu krawędzi w sekundach
    def cost(self) -> float:
        return self.length * 3.6 / speed.get(self.classification)

class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self._shapefile_path: str | List[str] | None = None
        self._kdtree = None
        self._node_coords: List[Tuple[float, float]] = []

    def add_node(self, node: Node):
        self.nodes[node.id] = node
        self._node_coords.append((node.x, node.y))
        self._kdtree = None  # Invalidate KDTree

    def build_kdtree(self):
        if not self._kdtree:
            self._kdtree = KDTree(self._node_coords)

    def add_edge(self, edge: Edge):
        self.edges[edge.id] = edge
        edge.start_node.edges.append(edge)
        if not edge.one_way:
            edge.end_node.edges.append(edge)

    def add_feature(self, feature: Dict):
        coords = feature["geometry"]["coordinates"]
        start_coord = coords[0]
        end_coord = coords[-1]

        # TODO: zakłada, że końce się idealnie pokrywają - powinno wykorzystywać funkcję point_approx z tol = 0.1
        start_node = self.nodes.get(node_id(start_coord[0], start_coord[1]))
        if not start_node:
            start_node = Node(start_coord[0], start_coord[1])
            self.add_node(start_node)
        end_node = self.nodes.get(node_id(end_coord[0], end_coord[1]))
        if not end_node:
            end_node = Node(end_coord[0], end_coord[1])
            self.add_node(end_node)

        # Długość polilinii w metrach
        # Shapely.length może szybsza metoda? Nie sprawdzałem, ale chyba nie da się tu nic zoptymalizować, może tylko for-loop?
        length = 0.0
        for i in range(len(coords) - 1):
            length += euclidean_distance((coords[i][0], coords[i][1]), (coords[i + 1][0], coords[i + 1][1]))

        classification = feature["properties"]["klasaDrogi"]
        # TODO: obsługa dróg jednokierunkowych (?)

        edge = Edge(feature["properties"]["idIIP_BT_I"], start_node, end_node, length, classification)
        self.add_edge(edge)

    def find_nearest_node(self, coordinates: Tuple[float, float]) -> Node:
        self.build_kdtree()
        _, idx = self._kdtree.query(coordinates)
        nearest_coord = self._node_coords[idx]
        return self.nodes[node_id(nearest_coord[0], nearest_coord[1])]

    def reconstruct_shp_path(self, path: List[Edge], output_path: str):
        if self._shapefile_path is None:
            print("Brak ścieżki pliku shapefile. Nie można zrekonstruować trasy")
            return
        if isinstance(self._shapefile_path, list):
            gdf = gpd.read_file(self._shapefile_path[0])
            for p in self._shapefile_path[1:]:
                gdf = gdf._append(gpd.read_file(p), ignore_index=True)
        else:
            gdf = gpd.read_file(self._shapefile_path)
        edge_ids = {edge.id for edge in path}
        gdf_filtered = gdf[gdf["idIIP_BT_I"].isin(edge_ids)]
        gdf_filtered.to_file(output_path, driver='ESRI Shapefile')
        print(f"Zrekonstruowana trasa zapisana do {output_path}")


def draw_graph(G: Graph):
    # Rysowanie dla dużych grafów jest bardzo, bardzo wolne i pewnie nieczytelne, szczególnie rysowanie węzłów,
    # można rysować same krawędzie, ale i tak obrazek będzie jedynie do potwierdzenia, że "coś" zrobił
    fig, ax = plt.subplots(figsize=(10, 10))

    for edge in G.edges.values():
        x_coords = [edge.start_node.x, edge.end_node.x]
        y_coords = [edge.start_node.y, edge.end_node.y]
        ax.plot(x_coords, y_coords, color='blue', linewidth=0.5)

    for node in G.nodes.values():
        ax.scatter(node.x, node.y, color='k', s=5)

    ax.set_title("Wizualizacja grafu")
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")

def build_graph_from_shapefile(file_path: str | List[str]) -> Graph:
    if isinstance(file_path, list):
        gdf = gpd.read_file(file_path[0])
        for p in file_path[1:]:
            gdf = gdf._append(gpd.read_file(p), ignore_index=True)
    else:
        gdf = gpd.read_file(file_path)
    G = Graph()
    G.shapefile_path = file_path

    # Iteracja po obiektach wczytanych do GeoDataFrame
    iterator = gdf.iterfeatures(drop_id=True)
    while True:
        try:
            feature = next(iterator)
        except StopIteration:
            break
        G.add_feature(feature)
    G.build_kdtree()
    return G

# Demo
if __name__ == "__main__":
    shp_file_paths = ["kujawsko_pomorskie_m_Torun/L4_1_BDOT10k__OT_SKJZ_L.shp",
                      "kujawsko_pomorskie_pow_torunski/L4_2_BDOT10k__OT_SKJZ_L.shp"]
    test_path = "..\\test_shp.shp"

    graph = build_graph_from_shapefile(test_path)
    print(f"Graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges.")

    draw_graph(graph)
    plt.show()
