from shp_graph import *




# Demo
if __name__ == "__main__":
    shp_file_paths = ["kujawsko_pomorskie_m_Torun/L4_1_BDOT10k__OT_SKJZ_L.shp",
                      "kujawsko_pomorskie_pow_torunski/L4_2_BDOT10k__OT_SKJZ_L.shp"]
    #test_path = r"C:\aszkola\5 sem\pag\projekt1\dane\testowe.shp"
    test_path = r"C:\aszkola\5 sem\pag\projekt1\DROGIWYB\drogiwyb.shp"
    points_path = r"C:\aszkola\5 sem\pag\projekt1\PUNKTY.txt"
    points2_path =r"C:\aszkola\5 sem\pag\projekt1\Nawigacja-PAG\punkty2.txt"

    graph = build_graph_from_shapefile(test_path)
    print(f"Graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges.")

    XY_coords = []
    for edge in graph.edges.values():
        XY_coords.append([edge.start_node.x, edge.start_node.y])
        XY_coords.append([edge.end_node.x, edge.end_node.y])

       # x_coords = [edge.start_node.x, edge.end_node.x]
       # y_coords = [edge.start_node.y, edge.end_node.y]
    map = folium.Map(location=[53.9, 18.4], zoom_start=10)
    map.save("navigation.html")
    folium.PolyLine(
    locations=XY_coords,
    color="#FF0000",
    weight=5,
    marker = 'o'
    ).add_to(map)
        
    