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
import random

def floor(a: float, n: int):
    return np.floor(a * 10**n) / 10**n
def ceil(a: float, n: int):
    return np.ceil(a * 10**n) / 10**n
def euclidean_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    # Odległość między punktami
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)
def approx(a: float, b: float, tol: float = 0.10) -> bool:
    # Przybliżenie
    return abs(a - b) < tol

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
        self._node_coords: List[Tuple[float, float]] = [] # Potencjalnie niepotrzebne, można wyjmować z Dict[Node]

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

        classification = feature["properties"]["klasaDrogi"]
        # TODO: obsługa dróg jednokierunkowych (?)

        edge = Edge(feature["properties"]["idIIP_BT_I"], start_node, end_node, length, classification)
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

    def matrixes(self, points:List[Tuple[float, float]]):
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
                    route = convert_nodes_to_edges(self.dijkstra(points[i], points[j]))
                    routes_matrix[i][j] = route
                    #KOSZT WYRAZAM JAKO SUMA LEGTH?!
                    cost_matrix[i,j] =calculate_route_cost(route)
        
        return cost_matrix, routes_matrix, mat_ids

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
                if d[node_id] < min_dist:
                    min_dist = d[node_id]
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
                    new_dist = d[curr_id] + edge.length
                    if new_dist < d[neighbour.id]:
                        d[neighbour.id] = new_dist
                        p[neighbour.id] = curr_id
        
        route = self.reconstruct_path(p, startpoint, endpoint) #przemienia p na sciezke 
        return route

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


def draw_path (graph:Graph, route: List[Node]):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    draw_graph(graph)

    Xcoords = [r.x for r in route]
    Ycoords = [r.y for r in route]
    plt.plot(Xcoords, Ycoords, color='r', marker ='o')
   
    plt.title("Ścieżka znaleziona przez Dijkstrę")

def draw_point(point: Tuple[float, float]):
    x, y = point
    plt.scatter(x, y, color='red', s=50, zorder=5)

def read_points(path)-> List[Tuple[float, float]]:
    points =[]
    with open(path, 'r', encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            
            if not line:
                continue
            
            x_str, y_str = line.split(';')
            x, y = float(x_str.replace(',','.')), float(y_str.replace(',','.'))
            points.append((x,y))
    return points

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
        cost += e.length
    return cost

def rand_route_points(points: List[Tuple[float, float]], route_mat:np.ndarray, 
                      cost_mat:np.ndarray, mat_ids: List[Tuple[float, float]]):

    n = len(points)
    ids = [mat_ids.index(p) for p in points] # konwertuje wybrane punkty na indeksy w macierzach
    random.shuffle(ids)
  
    total_cost = 0
    full_route = []
    
    for i in range (n-1):
        a, b = ids[i], ids[i +1]
        #ia = point_to_index[a]
        #ib = point_to_index[b]
        total_cost += cost_mat[a, b]
        full_route.extend(route_mat[a][b])
    
    return full_route, total_cost, ids


# Demo
if __name__ == "__main__":
    shp_file_paths = ["..\\kujawsko_pomorskie_m_Torun/L4_1_BDOT10k__OT_SKJZ_L.shp",
                      "..\\kujawsko_pomorskie_pow_torunski/L4_2_BDOT10k__OT_SKJZ_L.shp"]
    #test_path = r"C:\aszkola\5 sem\pag\projekt1\dane\testowe.shp"
    test_path = r"..\test_shp.shp"
    rozjechane = r"C:\Users\burb2\Desktop\Pliki Studia\PAG\rozjechane_drogi.shp"
    points_path = r"C:\aszkola\5 sem\pag\projekt1\PUNKTY.txt"
    points2_path =r"C:\aszkola\5 sem\pag\projekt1\Nawigacja-PAG\punkty2.txt"

    graph = build_graph_from_shapefile(rozjechane)
    print(f"Graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges.")
    #
    # points = read_points(points_path)
    # cost_matrix, routes_matrix, mat_ids = graph.matrixes(points)
    # print(cost_matrix)
    # points2 = read_points(points2_path)
    # full_route, total_cost, ids = rand_route_points(points2, routes_matrix, cost_matrix, mat_ids)
    #
    # print(total_cost)
    
    

    # nodes = list(graph.nodes.values())
    # route = graph.dijkstra(nodes[0], nodes[5])
    #
    # draw_path(graph, route)

    #print(f"Node 3 {graph.nodes} nodes and {len(graph.edges)} edges."))

    draw_graph(graph)
    my_point = (473600, 571800)
    nearest = graph.find_nearest_node(my_point)
    draw_point(my_point)
    draw_point((nearest.x, nearest.y))
    plt.show()
