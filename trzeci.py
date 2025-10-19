import arcpy
import csv

arcpy.env.workspace = r"C:\aszkola\5 sem\pag\projekt1\dane"
cursor =arcpy.SearchCursor('testowe.shp')

vertex=[] #id, x, y
edges=[] #  id , id_from, id_to, id_jezdni
vertex_id =0
edge_id=0

vertex_path = r"C:\aszkola\5 sem\pag\projekt1\dane\vertexes.txt"
edge_path = r"C:\aszkola\5 sem\pag\projekt1\dane\edges.txt"


def find_vertex_id(x, y):
    #Zwraca id istniejącego wierzchołka lub 0 jeśli brak
    x, y = round(x, 2), round(y, 2)
    for v in vertex:
        if v[1]==x and v[2]==y:
            return v[0]
    return 0

with arcpy.da.SearchCursor('testowe.shp', ['OID@','SHAPE@']) as cursor:
    for oid, shape in cursor:
        start = shape.firstPoint
        end = shape.lastPoint

        sX, sY = round(start.X, 2), round(start.Y, 2)
        eX, eY = round(end.X, 2), round(end.Y, 2)

        #wierzcholek poczatkowy
        id_from = find_vertex_id(sX, sY)
        if id_from ==0:
            vertex_id = vertex_id +1
            id_from = vertex_id
            vertex.append((id_from, sX, sY))

        id_to = find_vertex_id(eX, eY)
        if id_to ==0:
            vertex_id = vertex_id +1
            id_to = vertex_id
            vertex.append((id_to, eX, eY))

        #krawędź
        edge_id += 1
        edges.append((edge_id, id_from, id_to, oid))


with open(vertex_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["vertex_id", "x", "y"])  # nagłówek
    writer.writerows(vertex)

with open(edge_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["edge_id", "id_from", "id_to", "id_jezdni"])  # nagłówek
    writer.writerows(edges)






