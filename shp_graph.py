"""
Moduł z klasą graf oraz funkcjami do tworzenia grafu z pliku shapefile.

Autor: Bartosz Urbanek, 331931
Data: 2025-10-17
"""
from __future__ import annotations
from typing import List, Tuple

import numpy as np
import geopandas as gpd


shp_file_paths = ["kujawsko_pomorskie_m_Torun/L4_1_BDOT10k__OT_SKJZ_L.shp", "kujawsko_pomorskie_pow_torunski/L4_2_BDOT10k__OT_SKJZ_L.shp"]

test_path = "test_shp.shp"

# gdf = gpd.read_file(shp_file_paths[0])._append(gpd.read_file(shp_file_paths[1]), ignore_index=True)
# iterator = gdf.iterfeatures(drop_id=True)
# while True:
#     try:
#         feature = next(iterator)
#     except StopIteration:
#         break
#     print(feature["properties"]["idIIP_BT_I"] ,feature["properties"]["klasaDrogi"], feature["geometry"]["coordinates"][0], feature["geometry"]["coordinates"][-1])

def euclidean_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)
def approx(a: float, b: float, tol: float = 0.10) -> bool:
    return abs(a - b) < tol
def point_approx(coord1: Tuple[float, float], coord2: Tuple[float, float], tol: float = 0.10) -> bool:
    return approx(coord1[0], coord2[0], tol) and approx(coord1[1], coord2[1], tol)
def node_id(x: float, y: float) -> str:
    return f'({x} {y})'

class Node:
    def __init__(self, x: float, y: float):
        self.id = node_id(x, y)
        self.x = x
        self.y = y
        self.edges = []

    def add_edge(self, edge: 'Edge'):
        self.edges.append(edge)

    def get_neighbours(self) -> List[Tuple['Node', 'Edge']]:
        return [(edge.get_other_node(), edge) for edge in self.edges]
        # Teoretycznie powinien być blok Try-Except, ale to by oznaczało, że graf jest błędnie zbudowany

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

    def cost(self) -> float:
        return self.length * 3600 / speed.get(self.classification)

class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def add_node(self, node: Node):
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge):
        self.edges[edge.id] = edge
        edge.start_node.add_edge(edge)
        if not edge.one_way:
            edge.end_node.add_edge(edge)

    def add_feature(self, feature: dict):
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

        # Shapely.length może szybsza metoda? Nie sprawdzałem, ale chyba nie da się tu nic zoptymalizować, może tylko for-loop?
        length = 0.0
        for i in range(len(coords) - 1):
            length += euclidean_distance((coords[i][0], coords[i][1]), (coords[i + 1][0], coords[i + 1][1]))

        classification = feature["properties"]["klasaDrogi"]
        # TODO: obsługa dróg jednokierunkowych (?)

        edge = Edge(feature["properties"]["idIIP_BT_I"], start_node, end_node, length, classification)
        self.add_edge(edge)

    # TODO: implement
    def find_nearest_node(self, coordinates: Tuple[float, float]) -> Node:
        pass

def build_graph_from_shapefile(shp_path: str) -> Graph:
    gdf = gpd.read_file(shp_path)
    graph = Graph()
    iterator = gdf.iterfeatures(drop_id=True)
    while True:
        try:
            feature = next(iterator)
        except StopIteration:
            break
        graph.add_feature(feature)
    return graph

if __name__ == "__main__":
    graph = build_graph_from_shapefile(test_path)
    print(f"Graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges.")