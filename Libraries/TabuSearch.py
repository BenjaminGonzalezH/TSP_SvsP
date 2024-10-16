########## Libraries ##########
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

########## Functions TS ##########

def ObjFun(Solution, DistanceMatrix):
    """
    ObjFun (function)
        Input: Permutation vector and
        distance matrix.
        Output: Value in objective function.
        Description: Calculates the objective
        function using permutation vector and
        distance matrix.
        The TSP objective function is the sum
        of all edge's cost.
    """
    return sum(DistanceMatrix[Solution[i] - 1][Solution[i + 1] - 1] for i in range(len(Solution) - 1)) + \
            DistanceMatrix[Solution[-1] - 1][Solution[0] - 1]

def first_solution(AmountNodes):
    """
    first_solution (function)
        Input: Number of nodes.
        Output: Permutation vector.
        Description: Generates first solution.
    """
    # Random permutation node.
    Vector = np.random.permutation(np.arange(1, AmountNodes + 1))
    return Vector

def two_opt_swap(Solution, i, j):
    """
    two_opt_swap (function)
        Input: Solution, and two indices i and j.
        Output: New solution with 2-opt swap applied.
        Description: Reverses the tour between indices i and j.
    """
    new_solution = np.copy(Solution)
    new_solution[i:j+1] = np.flip(Solution[i:j+1])  # Reverse the segment
    return new_solution

def get_neighbors_2opt(Solution):
    """
    get_neighbors_2opt (function)
        Input: Reference solution.
        Output: List of neighbor solutions (2-opt neighborhood).
        Description: Generates a neighborhood using 2-opt swaps.
    """
    neighbors = []
    n = len(Solution)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if j - i == 1:  # Cambiar sólo si i y j no son adyacentes
                continue
            neighbor = np.concatenate((Solution[:i], Solution[i:j + 1][::-1], Solution[j + 1:]))
            neighbors.append(neighbor)
    return neighbors

def best_neighbor(Neighborhood, DistanceMatrix, TabuList):
    """
    get_neighbors (function)
        Input: Neighborhood, Distance Matrix and
        Tabu List.
        Output: Neighbor with best improvement 
        in objective function.
        Description: Evaluates Neighborhood
        of permutation solutions.
    """
    # Best solution and value in obj. function.
    Best = None
    Best_f = float('inf')

    # Searching.
    for candidate in Neighborhood:
        # Take candidate.
        Candidate_f = ObjFun(candidate, DistanceMatrix)

        # Conditions for change.
        if (Best_f > Candidate_f and candidate not in TabuList):
            Best = candidate
            Best_f = Candidate_f
    
    return Best

def TabuSearch(DistanceMatrix, AmountNodes, MaxIterations=100, TabuSize=10,
               minErrorInten=0.001):
    """
    TabuSearch (function)
        Input: Distance Matrix (TSP instance), Total number of
        nodes, Max iterations for algorithm, size of
        tabu list, minimal error for intensification and amount
        of solutions for intensification.
        Output: Unique solutions (Best found).
        Description: Implementation of Tabu Search with 2-opt
        for TSP.
    """ 
    BestSolution = first_solution(AmountNodes)
    CurrentSolution = BestSolution
    tabu_list = set()

    for _ in range(MaxIterations):
        Neighborhood = get_neighbors_2opt(CurrentSolution)
        BestNeighbor = best_neighbor(Neighborhood, DistanceMatrix, tabu_list)

        if BestNeighbor is None:
            CurrentSolution = first_solution(AmountNodes)
            continue
        
        CurrentSolution_f = ObjFun(CurrentSolution, DistanceMatrix)
        BestNeighbor_f = ObjFun(BestNeighbor, DistanceMatrix)

        # Check for intensification
        if abs(BestNeighbor_f - CurrentSolution_f) < minErrorInten:
            continue  # No improvement

        # Update tabu list
        if len(tabu_list) >= TabuSize:
            tabu_list = set(list(tabu_list)[-TabuSize:])
        tabu_list.add(tuple(CurrentSolution))

        CurrentSolution = BestNeighbor
        if BestNeighbor_f < ObjFun(BestSolution, DistanceMatrix):
            BestSolution = BestNeighbor

    # Return the best solution found.
    return BestSolution


########## Versiones convergencia ##########
def TabuSearch_Con(DistanceMatrix, AmountNodes, MaxIterations=100, TabuSize=10,
               minErrorInten=0.001):
    """
    TabuSearch (function)
        Input: Distance Matrix (TSP instance), Total number of
        nodes, Max iterations for algorithm, size of
        tabu list, minimal error for intensification and amount
        of solutions for intensification.
        Output: Unique solutions (Best found).
        Description: Implementation of Tabu Search with 2-opt
        for TSP.
    """ 
    # Setting initial variables.
    BestSolution = first_solution(AmountNodes)
    CurrentSolution = BestSolution
    tabu_list = []
    bests_n = []
    best_sol = []

    # Until Max Iterations (Stop Criteria)
    for _ in range(MaxIterations):
        # Creating Neighborhood with 2-opt swaps.
        Neighborhood = get_neighbors_2opt(CurrentSolution)

        # Search the best neighbor.
        BestNeighbor = best_neighbor(Neighborhood, DistanceMatrix, tabu_list)

        # Intensification criteria: Minimal improvement.
        BestSolution_f = ObjFun(BestSolution, DistanceMatrix)
        CurrentSolution_f = ObjFun(BestNeighbor, DistanceMatrix)
        if abs(BestSolution_f - CurrentSolution_f) < minErrorInten:
            Neighborhood = get_neighbors_2opt(CurrentSolution)
            BestNeighbor = best_neighbor(Neighborhood, DistanceMatrix, tabu_list)

        # Diversification criteria: No improvement.
        if BestNeighbor is None:
            CurrentSolution = first_solution(AmountNodes)
            Neighborhood = get_neighbors_2opt(CurrentSolution)
            BestNeighbor = best_neighbor(Neighborhood, DistanceMatrix, tabu_list)

        # Adding to tabu list and change current solution.
        CurrentSolution = BestNeighbor
        tabu_list.append(CurrentSolution)
        if len(tabu_list) > TabuSize:
            tabu_list.pop(0)

        # Change best solution.
        if ObjFun(BestNeighbor, DistanceMatrix) < ObjFun(BestSolution, DistanceMatrix):
            BestSolution = BestNeighbor
        
        best_sol.append(ObjFun(BestSolution, DistanceMatrix))
        bests_n.append(ObjFun(BestNeighbor, DistanceMatrix))

    # Return the best solution found.
    return bests_n, best_sol
