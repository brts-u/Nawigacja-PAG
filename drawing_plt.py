from matplotlib import pyplot as plt
from matplotlib import cm
import time
import random
from shp_graph import *

def draw_graph(G: Graph, /, nodeless: bool = False):
    # Rysowanie dla dużych grafów jest bardzo, bardzo wolne i pewnie nieczytelne, szczególnie rysowanie węzłów,
    # można rysować same krawędzie, ale i tak obrazek będzie jedynie do potwierdzenia, że "coś" zrobił
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('equal')

    for edge in G.edges.values():
        x_coords = [edge.start_node.x, edge.end_node.x]
        y_coords = [edge.start_node.y, edge.end_node.y]
        ax.plot(x_coords, y_coords, color='blue', linewidth=0.5)

    #if not nodeless:
      #  for node in G.nodes.values():
      #      ax.scatter(node.x, node.y, color='k', s=5)

    ax.set_title("Wizualizacja grafu")
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")

def draw_path(graph: Graph, route: List[Node]):

    draw_graph(graph, nodeless=True)

    Xcoords = [r.x for r in route]
    Ycoords = [r.y for r in route]
    plt.plot(Xcoords, Ycoords, color='r')  # , marker ='o')

    plt.title("Najkrótsza ścieżka")
    plt.show()

def draw_point(point: Tuple[float, float]):
    x, y = point
    plt.scatter(x, y, color='red', s=50, zorder=5)


def draw_pts_connection(graph: Graph, edges, coords=List[Tuple[float, float]]):
    draw_graph(graph)

    cmap = cm.get_cmap('autumn')
    colors = cmap([i / len(edges) for i in range(len(edges))])

    for edge, color in zip(edges, colors):
        start = graph.nodes[edge[0]]
        end = graph.nodes[edge[1]]
        x_coords = [start.x, end.x]
        y_coords = [start.y, end.y]
        plt.plot(x_coords, y_coords, color=color)  # linia

    ptsXcoords = [c[0] for c in coords]
    ptsYcoords = [c[1] for c in coords]

    plt.plot(ptsXcoords, ptsYcoords, color = 'black', marker='.', linestyle='None')  # punkty

    plt.title("Ścieżka travelling salesman")
    # plt.show()

# Demo
if __name__ == "__main__":

    test_path = [r'..\SHP_SKJZ\PL.PZGiK.994.BDOT10k.0463__OT_SKJZ_L.shp', r'..\SHP_SKJZ\PL.PZGiK.994.BDOT10k.0415__OT_SKJZ_L.shp']

    # WAŻNE !!!! jeśli chcesz porównać trasę z Google Maps lub inną nawigacją,
    # pomocne będzie zmienienie wyników na jednostki kątowe
    wspolrzedne_polskie = False

    t1 = time.time()

    graph = build_graph_from_shapefile(test_path)

    t2 = time.time()
    print(f"Budowanie grafu zajęło {round(t2-t1, 3)} s")

    point1 = graph.nodes[random.choice(list(graph.nodes.keys()))]
    point2 = graph.nodes[random.choice(list(graph.nodes.keys()))]
    while point2 == point1:
        point2 = graph.nodes[random.choice(list(graph.nodes.keys()))]

    if not wspolrzedne_polskie:
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:2180", "EPSG:4326", always_xy=True)
        lat1, lon1 = transformer.transform(point1.x, point1.y)
        lat2, lon2 = transformer.transform(point2.x, point2.y)
        print(f"Wylosowane punkty ({lon1}, {lat1}), ({lon2}, {lat2})")

    route = graph.A_star((point1.x, point1.y), (point2.x, point2.y))
    t3 = time.time()

    print(f"Obliczanie trasy zajęło {round(t3-t2, 3)} s")

    draw_path(graph, route)
    t4 = time.time()
    print(f"Rysowanie trasy zajęło {round(t4-t3, 3)} s")
    plt.show()