import time
from typing import List, Tuple, Dict
import numpy as np
import random
import pickle

def cycle_cost(route: List[int], cost_mat: np.ndarray) -> float:
    total_cost = 0
    n = len(route)
    for i in range(n):
        a = route[i]
        b = route[(i + 1) % n]  # wrap around to the start
        total_cost += cost_mat[a, b]
    return total_cost

def route_from_list(list_of_ids: List[int], route_mat: np.ndarray) -> List[Tuple[int, int]]:
    full_route = []
    n = len(list_of_ids)
    for i in range(n - 1):
        a = list_of_ids[i]
        b = list_of_ids[i + 1]
        full_route.extend(route_mat[a][b])
    # Dodaj powrót do punktu startowego
    full_route.extend(route_mat[list_of_ids[-1]][list_of_ids[0]])
    return full_route

def set_cycle_to_beginning(list_of_ids: List[int], start_id: int = 0) -> List[int]:
    if start_id not in list_of_ids:
        raise ValueError("Start_id not in list_of_ids")

    start_index = list_of_ids.index(start_id)
    return list_of_ids[start_index:] + list_of_ids[:start_index]

def rand_route_points(points: List[Tuple[float, float]], route_mat:np.ndarray, cost_mat:np.ndarray, mat_ids: List[Tuple[float, float]] ):
    #points: List[Tuple[float, float]], route_mat:np.ndarray, cost_mat:np.ndarray, mat_ids: List[Tuple[float, float]]):
    #jeśli pętla to na koniec trzeba dodac punkt który był pierwszy
    # pierwszy punkt jest startowy!

    n = len(points)
    ids = [mat_ids.index(p) for p in points] # konwertuje wybrane punkty na indeksy w macierzach
    
    cheapest =np.inf
    cheapest_route =[]
    cheapest_ids =[]

    for _ in range(100): #losuje x=10 razy i sprawdzam ktora jest
        shuf_ids = ids.copy()
        random.shuffle(shuf_ids)

        total_cost = 0
        #costs = []
    
        total_cost = cycle_cost(shuf_ids, cost_mat)
        if total_cost<cheapest:
            cheapest=total_cost
            cheapest_route = route_from_list(shuf_ids, route_mat) # zapisujemy liste id Nodeów, dopiero gdy znajdziemy najtańszą trasę
            cheapest_ids = shuf_ids.copy()

            print(cheapest)

    return cheapest_route, cheapest, cheapest_ids

def optymalization(cheapest_ids: List[int], cost_mat: np.ndarray):

    opt_route = cheapest_ids.copy()
    opt_cost =cycle_cost(cheapest_ids, cost_mat)
    print(f"cheapest opt: {opt_cost}")
    improved = 10

    while improved:
        improved -= 1
        for i in range(len(opt_route)-1):
            for j in range(i+1, len(opt_route)):
                test_route = opt_route.copy()
                test_route[i], test_route[j] = test_route[j], test_route[i]
                new_cost = cycle_cost(test_route, cost_mat)
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

def simulated_annealing(cheapest_ids: List[int], cost_mat: np.ndarray) -> Tuple[List[int], float]:
    current_route = cheapest_ids.copy()
    current_cost = cycle_cost(current_route, cost_mat)
    best_route = current_route.copy()
    best_cost = current_cost

    T = 1000.0  # Initial temperature
    T_min = 1e-8
    alpha = 0.999  # Cooling rate

    while T > T_min:
        i, j = random.sample(range(len(current_route)), 2)
        new_route = current_route.copy()
        new_route[i], new_route[j] = new_route[j], new_route[i]
        new_cost = cycle_cost(new_route, cost_mat)

        delta_cost = new_cost - current_cost
        if delta_cost < 0 or random.uniform(0, 1) < np.exp(-delta_cost / T):
            current_route = new_route
            current_cost = new_cost

            if current_cost < best_cost:
                best_route = current_route
                best_cost = current_cost

        T *= alpha

    return best_route, best_cost

def opt_route_edges(selected_ids: List[int], matdata="matrix_data.pkl" ):
    #route_mat:np.ndarray, cost_mat:np.ndarray, mat_ids: List[Tuple[float, float]]

    with open(matdata, "rb") as f:
        data = pickle.load(f)

    cost_mat = data["cost_matrix"]
    route_mat = data["routes_matrix"]
    mat_ids = data["mat_ids"]

    selected_points = [mat_ids[i] for i in selected_ids]

    _, _, cheapest_ids = rand_route_points(selected_points, route_mat, cost_mat, mat_ids)
    # opt_route, opt_cost = optymalization(cheapest_ids, cost_mat)
    opt_route, opt_cost = simulated_annealing(cheapest_ids, cost_mat)
    # set_cycle_to_beginning(opt_route) # to nic nie robi, ja nie rozumiem Doma jak ty to shuffle'ujesz
                                        # serio nie umiem tego odtworzyć, żeby się zawsze rysowało od pierwszego punku

    opt_edges =[]
    for i in range(len(opt_route)-1 ):
        a, b = opt_route[i], opt_route[i+1]
        opt_edges.extend(route_mat[a][b])
    
    opt_edges.extend(route_mat[opt_route[-1]][opt_route[0]]) #powrót do pkt startowego
    pts_coords = [mat_ids[i] for i in opt_route] # wspolrzedne punktow
    
    return opt_edges, pts_coords


