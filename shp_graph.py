"""
Moduł z klasą grafu sieci drogowej oraz funkcjami do tworzenia go z pliku shapefile SKJZ (jezdni) w strukturze BDOT10k.

Autor: Bartosz Urbanek 331931
Data: 2025-10-17
"""
from __future__ import annotations

from typing import List, Tuple, Dict

from support_functions import *

import time
import random
import numpy as np
import geopandas as gpd
from komiwojazer import opt_route_edges
from scipy.spatial import cKDTree
import pickle
import heapq

# Klasa węzła grafu
class Node:
    def __init__(self, x: float, y: float):
        self.id: int = -1
        self.x: float = x
        self.y: float = y
        self.edges: List[Edge] = []

    # Zwraca listę sąsiadów i drogi do nich, sąsiady wielopowiązane rozróżnialne przez ID krawędzi
    def get_neighbours(self) -> List[Tuple[Node, Edge]]:
        return [(edge.get_other_node(self), edge) for edge in self.edges]
        # Teoretycznie powinien być blok Try-Except, ale to by oznaczało, że graf został błędnie zbudowany

    def heuristic(self, other: Node) -> float:
        return euclidean_distance((self.x, self.y), (other.x, other.y)) * kiloseconds_per_hour / (speed_kmh['autostrada'])

# Prędkości przypisane do klas dróg w km/h
speed_kmh = {   # Okazuje się, że nie da się tego zmapować tak 1:1, więc wybrałem takie, które mają największy sens
    "autostrada": 140,   # Autostrada
    "droga ekspresowa": 120,   # Droga ekspresowa
    "droga główna ruchu przyśpieszonego": 90,   # Droga główna przyspieszona
    "droga główna": 70,    # Droga główna
    "droga zbiorcza": 50,    # Droga zbiorcza
    "droga lokalna": 40,    # Droga lokalna
    "droga dojazdowa": 30, # Droga dojazdowa
    "droga wewnętrzna": 20  # Droga wewnętrzna
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
        return self.length * kiloseconds_per_hour / speed_kmh.get(self.classification)

class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self._shapefile_path: str | List[str] | None = None
        self._kdtree = None
        self._node_coords: List[Tuple[float, float]] = [] # Potencjalnie niepotrzebne, można wyjmować z Dict[Node]

    def add_node(self, node: Node):
        self.nodes[node.id] = node
        self._node_coords.append((node.x, node.y))
        self._kdtree = None  # Invalidate KDTree

    def build_kdtree(self):
        if not self._kdtree:
            self._kdtree = cKDTree(self._node_coords)

    def add_edge(self, edge: Edge):
        self.edges[edge.id] = edge
        edge.start_node.edges.append(edge)
        if not edge.one_way:
            edge.end_node.edges.append(edge)

    def add_feature(self, feature: Dict, manager: NodeManager):
        coords = feature["geometry"]["coordinates"] # z modułu geometry pobiera tylko coordinates
        start_coord = coords[0]
        end_coord = coords[-1]

        start_node = manager.manage_new_node(start_coord[0], start_coord[1])
        if start_node not in self.nodes:
            self.add_node(start_node)
        end_node = manager.manage_new_node(end_coord[0], end_coord[1])
        if end_node not in self.nodes:
            self.add_node(end_node)

        # Długość polilinii w metrach
        # Shapely.length może szybsza metoda? Nie sprawdzałem, ale chyba nie da się tu nic zoptymalizować, może tylko for-loop?
        length = 0.0
        for i in range(len(coords) - 1):
            length += euclidean_distance((coords[i][0], coords[i][1]), (coords[i + 1][0], coords[i + 1][1]))

        classification = feature["properties"]["KLASA_DROG"]
        # TODO: obsługa dróg jednokierunkowych (?)

        edge = Edge(feature["properties"]["LOKALNYID"], start_node, end_node, length, classification)
        self.add_edge(edge)

    def find_nearest_node(self, coordinates: Tuple[float, float]) -> Node:
        self.build_kdtree()
        _, idx = self._kdtree.query(coordinates)
        nearest_coord = self._node_coords[idx]
        for node in self.nodes.values():
            if (node.x, node.y) == nearest_coord:
                return node
    
    def reconstruct_path(self, p:Dict, start:Node, end:Node)-> List[Node]:
        path_nodes = []
        current_id = end.id
        
        while current_id != -1:
            path_nodes.append(self.nodes[current_id])
            current_id = p[current_id]
        path_nodes.reverse() #sciezka węzłów

        return path_nodes
     
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

    def matrixes(self, points:List[Tuple[float, float]]) -> str:
        """
        Tworzy macierze kosztów i tras między wszytskimi punkami i zapisuje je do pickla
        :param points: lista punktów (x, y)
        :return: ścieżka do pliku .pkl z macierzą kosztów 'cost_matrix', macierzą tras 'route_matrix' i listą indeksów 'mat_ids'
        """
        n = len(points)
        cost_matrix = np.zeros((n,n)) #macierz nxn wypełniona zerami
        routes_matrix = [[None for _ in range(n)] for _ in range(n)] 
        mat_ids =[] # trzyma mi który punkt jest po kolei w macierzy

        for i in range(n):
            mat_ids.append(points[i])
            for j in range(n):
                if i ==j :
                    cost_matrix[i, j]=0
                    routes_matrix[i][j] = []
                else: 
                    route = convert_nodes_to_edges(self.A_star(points[i], points[j]))
                    # zamiast zapisywać obiekty — zapisujemy tylko ID 
                    safe_route = [(edge.start_node.id, edge.end_node.id) for edge in route]
                    routes_matrix[i][j] = safe_route
                    #routes_matrix[i][j] = route
                    cost_matrix[i,j] =calculate_route_cost(route)

        pickle_path = "matrix_data2.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump({
                "cost_matrix": cost_matrix,
                "routes_matrix": routes_matrix, #zawiera pełną listę krawędzi (par ID węzłów)
                "mat_ids": mat_ids
            }, f)
        
        return pickle_path

    def dijkstra(self, startpoint:Tuple[float, float], endpoint:Tuple[float, float]) -> List[Node]:
        startpoint = self.find_nearest_node(startpoint)
        endpoint = self.find_nearest_node(endpoint)

        S = {} #wierzchołki przejrzane
        Q = {} #wierzchołki nieodiwedzone
        p = {} #poprzednicy
        d = {} #aktualna długość ścieżek nieskonczonosc
        for n in self.nodes.values():
            Q[n.id] = n
            d[n.id] = np.inf
            p[n.id]=-1

        d[startpoint.id]=0
        
        while Q: 
            #szukam wezla o najmniejszym dystansie
            min_dist = np.inf
            curr_id = None

            for node_id in Q:
                dist = d[node_id]
                if dist< min_dist:
                    min_dist = dist
                    curr_id = node_id

            #przerzucam wezel
            current = Q.pop(curr_id) 
            if current in S: continue #zabezpiecznie przed zduplikowanymi wierzcholkami
            S[curr_id] = current

            if curr_id == endpoint.id:
                break
                
            #szukam sasiadow i ustalam dla nich koszt
            for neighbour, edge in current.get_neighbours():
                if neighbour.id not in S:
                    new_dist = d[curr_id] + edge.cost()
                    if new_dist < d[neighbour.id]:
                        d[neighbour.id] = new_dist
                        p[neighbour.id] = curr_id
        
        route = self.reconstruct_path(p, startpoint, endpoint) #przemienia p na sciezke 
        return route

    def A_star(self, startpoint:Tuple[float, float], endpoint:Tuple[float, float]) -> List[Node]:
        # Find nearest nodes to the given coordinates
        startpoint = self.find_nearest_node(startpoint)
        endpoint = self.find_nearest_node(endpoint)

        # Initialize data structures
        S = {}  # visited nodes
        p = {}  # predecessors
        d = {}  # distances from start (g-score)

        # Initialize all nodes
        for n in self.nodes.values():
            d[n.id] = np.inf
            p[n.id] = -1

        d[startpoint.id] = 0

        # Priority queue: (f_score, node_id)
        # f_score = g_score + heuristic
        open_set = [(0 + startpoint.heuristic(endpoint), startpoint.id)]
        open_set_ids = {startpoint.id}  # Track what's in the queue

        while open_set:
            # Get node with lowest f_score
            current_f, curr_id = heapq.heappop(open_set)
            open_set_ids.discard(curr_id)

            # Skip if already visited
            if curr_id in S:
                continue

            # Mark as visited
            current = self.nodes[curr_id]
            S[curr_id] = current

            # Check if we reached the goal
            if curr_id == endpoint.id:
                break

            # Explore neighbors
            for neighbour, edge in current.get_neighbours():
                if neighbour.id not in S:
                    # Calculate new distance (g-score)
                    new_dist = d[curr_id] + edge.cost()

                    # If this path is better
                    if new_dist < d[neighbour.id]:
                        d[neighbour.id] = new_dist
                        p[neighbour.id] = curr_id

                        # Calculate f_score for priority queue
                        f_score = new_dist + neighbour.heuristic(endpoint)

                        # Add to open set
                        if neighbour.id not in open_set_ids:
                            heapq.heappush(open_set, (f_score, neighbour.id))
                            open_set_ids.add(neighbour.id)

        # Reconstruct and return the path (same format as your Dijkstra)
        route = self.reconstruct_path(p, startpoint, endpoint)
        return route


class NodeManager:
    def __init__(self):
        self.node_indexes = {}
        self.new_node_id = 0
        self.tolerance = 1 # miejsca po przecinku, ujemne też działają

    def manage_new_node(self, x: float, y: float) -> Node:
        if f'({round(x, self.tolerance)}, {round(y, self.tolerance)})' in self.node_indexes:
            return self.node_indexes[f'({round(x, self.tolerance)}, {round(y, self.tolerance)})']
        
        node = Node(x, y)
        node.id = self.new_node_id
        self.new_node_id += 1
        keyUU = f'({ceil(x, self.tolerance)}, {ceil(y, self.tolerance)})'
        keyUD = f'({ceil(x, self.tolerance)}, {floor(y, self.tolerance)})'
        keyDU = f'({floor(x, self.tolerance)}, {ceil(y, self.tolerance)})'
        keyDD = f'({floor(x, self.tolerance)}, {floor(y, self.tolerance)})'
        self.node_indexes[keyUU] = node
        self.node_indexes[keyUD] = node
        self.node_indexes[keyDU] = node
        self.node_indexes[keyDD] = node
        return node

def build_graph_from_shapefile(file_path: str | List[str]) -> Graph:
    if isinstance(file_path, list):
        gdf = gpd.read_file(file_path[0])
        for p in file_path[1:]:
            gdf = gdf._append(gpd.read_file(p), ignore_index=True)
    else:
        gdf = gpd.read_file(file_path)
    G = Graph()
    NM = NodeManager()
    G._shapefile_path = file_path

    # Iteracja po obiektach wczytanych do GeoDataFrame
    iterator = gdf.iterfeatures(drop_id=True)
    while True:
        try:
            feature = next(iterator)
        except StopIteration:
            break
        G.add_feature(feature, NM)
    G.build_kdtree()
    return G

def convert_nodes_to_edges(path_nodes: List[Node])->List[Edge]: #konwertuje sciezke nodow na edgy
    path_edges =[] #sciezka edgy

    for i in range(len(path_nodes)-1):
        n1=path_nodes[i]
        n2=path_nodes[i+1]
        for edge in n1.edges: #sprawdzam wszystkie edge dla noda
            if edge.get_other_node(n1) ==n2: #szukam krawedzi dla dobrego konca
                path_edges.append(edge)
                break
    return path_edges

def calculate_route_cost(route:List[Edge]):
    cost =0
    for e in route:
        cost += e.cost()
    return cost

# Demo
if __name__ == "__main__":
    shp_file_paths = [r"..\SHP_SKJZ\PL.PZGiK.994.BDOT10k.0463__OT_SKJZ_L.shp", r"..\SHP_SKJZ\PL.PZGiK.994.BDOT10k.0415__OT_SKJZ_L.shp"]
    json_pts_path = r"..\punkty\25.geojson"
    #test_path = r"C:\aszkola\5 sem\pag\projekt1\dane\testowe.shp"
    test_path = r"..\test_shp.shp"
    rozjechane = r"C:\Users\burb2\Desktop\Pliki Studia\PAG\rozjechane_drogi.shp"
    points_path = r"punkty2.txt"
    points2_path =r"punkty2.txt"

    graph = build_graph_from_shapefile(shp_file_paths)
    print(f"Graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges.")

    # points = read_points(points_path) # tutaj trzeba wczytac te 40 pkt
    points = read_json_points(json_pts_path)
    # points2 = random.sample(points, len(points)) # wybiera wszystkie
    points2 = range(len(points)) # wybiera wszystkie
    pickle_file = graph.matrixes(points) #a tu sie beda cala noc liczyly
    # points2 = read_points(points2_path) # tutaj wczytac punkty ktore sb wybralam

    final, pts = opt_route_edges(points2, pickle_file)

    # saved_matrixes = "matrix_data.pkl"
    #TODO: wgrane jest 14 punktów -> trzebby zrobić na końcowm jakieś validate
    # selected_ids = [0,3, 6, 7, 10,13]
    # final, pts = opt_route_edges(selected_ids, pickle_file)
   
    from drawing_plt import draw_pts_connection
    draw_pts_connection(graph, final,pts)


