import shp_graph as sg
import drawing_plt
import support_functions
import komiwojazer as k
import time
from matplotlib import pyplot as plt


def doma_run():

    shp_file_paths =[r"C:\aszkola\5 sem\pag\projekt1\PLIKISHP\PL.PZGiK.994.BDOT10k.0415__OT_SKJZ_L.shp", r"C:\aszkola\5 sem\pag\projekt1\PLIKISHP\PL.PZGiK.994.BDOT10k.0463__OT_SKJZ_L.shp"]
    
    
    json_pts_path=r"C:\aszkola\5 sem\pag\projekt1\dane\40.geojson"
    pickle_file ="matrix_data2.pkl"
    
    points2 = [7, 6, 12, 39, 11]

def nopickle():
    start=time.time()
    shp_file_paths =[r"C:\aszkola\5 sem\pag\projekt1\PLIKISHP\PL.PZGiK.994.BDOT10k.0463__OT_SKJZ_L.shp"]
    json_pts_path=r"C:\aszkola\5 sem\pag\projekt1\dane\40.geojson"
    
    graph = sg.build_graph_from_shapefile(shp_file_paths)
    points = support_functions.read_json_points(json_pts_path)

    mats = graph.matrixes(points)
    sel_ids = [7, 6, 12, 39, 11]

    final, pts = k.opt_route_edges(sel_ids, mats)
    drawing_plt.draw_pts_connection(graph, final,pts)

    end=time.time()
    print(f"Czas wykonania przy obliczaniu macierzy: {end - start:.3f} s")
    
    plt.show()

    
def withpickle():
    start=time.time()
    shp_file_paths =[r"C:\aszkola\5 sem\pag\projekt1\PLIKISHP\PL.PZGiK.994.BDOT10k.0463__OT_SKJZ_L.shp"]
        
    graph = sg.build_graph_from_shapefile(shp_file_paths)
    mats = "matrix_data_torun.pkl"
    sel_ids = [7, 6, 12, 39, 11]

    final, pts = k.opt_route_edges(sel_ids, mats)
    drawing_plt.draw_pts_connection(graph, final,pts)

    end=time.time()
    print(f"Czas wykonania z plikiem pickle: {end - start:.3f} s")
    plt.show()


def best_route():
    pass

def route_directed():
    shp_path = r"C:\aszkola\5 sem\pag\projekt1\PLIKISHP\kierunkowe\kierunkowe_fragment.shp"

    P1 = (473606.5, 571734.5)
    P2 = (473518.7, 571728.1)

    graph = sg.build_graph_from_shapefile(shp_path)

    r = graph.A_star(P2, P1)
    drawing_plt.draw_path(graph, r)

if __name__ == "__main__":
    #nopickle()
    withpickle()




