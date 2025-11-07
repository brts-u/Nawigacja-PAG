from typing import List, Tuple, Dict
import numpy as np
import random
import pickle

def rand_route_points(points: List[Tuple[float, float]], route_mat:np.ndarray, cost_mat:np.ndarray, mat_ids: List[Tuple[float, float]] ):
    
    #points: List[Tuple[float, float]], route_mat:np.ndarray, cost_mat:np.ndarray, mat_ids: List[Tuple[float, float]]):
    #jeśli pętla to na koniec trzeba dodac punkt który był pierwszy
    # pierwszy punkt jest startowy!

    n = len(points)
    ids = [mat_ids.index(p) for p in points] # konwertuje wybrane punkty na indeksy w macierzach
    
    cheapest =np.inf
    cheapest_route =[]
    cheapest_ids =[]

    for _ in range(10000): #losuje x=10 razy i sprawdzam ktora jest
        shuf_ids = ids.copy()
        random.shuffle(shuf_ids)

        total_cost = 0
        #costs = []
        full_route = []
    
        for i in range (n-1):
            a, b = shuf_ids[i], shuf_ids[i +1]
            total_cost += cost_mat[a, b]
            #costs.append(cost_mat[a, b])
            full_route.extend(route_mat[a][b])
            
        if total_cost<cheapest:
            cheapest=total_cost
            cheapest_route = full_route
            cheapest_ids = shuf_ids.copy()
        
            print(cheapest)

    # print(f"typ: {type(cheapest_ids)}")
    # print(cheapest_ids)
    return cheapest_route, cheapest, cheapest_ids

def optymalization(cheapest_ids: List[int], cost_mat: np.ndarray):

    def route_cost(route):
        total =0
        for i in range(len(route)-1):
            total += cost_mat[route[i], route[i+1]]
        return total

    opt_route = cheapest_ids.copy()
    opt_cost =route_cost(cheapest_ids)
    print(f"cheapest opt: {opt_cost}")
    improved = 1

    while improved:
        improved -= 1
        for i in range(len(opt_route)-1):
            for j in range(i+1, len(opt_route)):
                test_route = opt_route.copy()
                test_route[i], test_route[j] = test_route[j], test_route[i]
                new_cost = route_cost(test_route)
                if new_cost < opt_cost:
                    print(f"Lepsza trasa po zamianie {i}<->{j}: {new_cost:.3f}")
                    opt_cost = new_cost
                    opt_route = test_route
                    improved += 1
                    break
            if improved:
                break
    print(f"Ostateczny koszt po optymalizacji: {opt_cost:.3f}")
    print(f"Kolejność punktów (indeksy): {opt_route}")
    return opt_route, opt_cost

def opt_route_edges(selected_ids=List[int], matdata="matrix_data.pkl" ):
    #route_mat:np.ndarray, cost_mat:np.ndarray, mat_ids: List[Tuple[float, float]]

    with open(matdata, "rb") as f:
        data = pickle.load(f)

    cost_mat = data["cost_matrix"]
    route_mat = data["routes_matrix"]
    mat_ids = data["mat_ids"]

    selected_points = [mat_ids[i] for i in selected_ids]

    _, _, cheapest_ids = rand_route_points(selected_points, route_mat, cost_mat, mat_ids)
    opt_route, opt_cost = optymalization(cheapest_ids, cost_mat)
    
    opt_edges =[]
    for i in range(len(opt_route)-1 ):
        a, b = opt_route[i], opt_route[i+1]
        opt_edges.extend(route_mat[a][b])
    
    opt_edges.extend(route_mat[opt_route[-1]][opt_route[0]]) #powrót do pkt startowego
    pts_coords = [mat_ids[i] for i in opt_route] # wspolrzedne punktow
    
    return opt_edges, pts_coords


