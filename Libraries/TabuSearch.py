########## Libraries ##########
import numpy as np

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
    return np.random.permutation(np.arange(1, AmountNodes + 1))

def two_opt_swap(Solution, i, j, DistanceMatrix):
    """
    two_opt_swap (function)
        Input: Solution, two indices i and j, and the DistanceMatrix.
        Output: New solution with 2-opt swap applied and updated cost.
        Description: Reverses the tour between indices i and j
        and calculates the change in the objective function.
    """
    n = len(Solution)
    
    # Cost before the swap
    before_swap = (DistanceMatrix[Solution[i - 1] - 1][Solution[i] - 1] +
                   DistanceMatrix[Solution[j] - 1][Solution[(j + 1) % n] - 1])
    
    # Cost after the swap
    after_swap = (DistanceMatrix[Solution[i - 1] - 1][Solution[j] - 1] +
                  DistanceMatrix[Solution[i] - 1][Solution[(j + 1) % n] - 1])
    
    # Reverse the segment between i and j
    new_solution = np.copy(Solution)
    new_solution[i:j+1] = np.flip(Solution[i:j+1])  # Reverse the segment
    
    # Calculate the change in objective function
    cost_change = after_swap - before_swap
    return new_solution, cost_change

def get_neighbors_2opt(Solution, DistanceMatrix):
    neighbors = []
    n = len(Solution)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if j - i == 1:  # Cambiar sólo si i y j no son adyacentes
                continue
            if j == n - 1:  # Evitar el último índice para j
                neighbor, cost_change = two_opt_swap(Solution, i, j, DistanceMatrix)
                neighbors.append((neighbor, cost_change))
            else:  # Aquí j puede ser el penúltimo índice
                neighbor, cost_change = two_opt_swap(Solution, i, j, DistanceMatrix)
                neighbors.append((neighbor, cost_change))
    return neighbors


def best_neighbor(Neighborhood, TabuList, CurrentCost):
    Best = None
    Best_f = float('inf')

    for candidate, cost_change in Neighborhood:
        Candidate_f = CurrentCost + cost_change

        if Candidate_f < Best_f and tuple(candidate) not in TabuList:
            Best = candidate
            Best_f = Candidate_f
            
            # Rompe el ciclo si encuentras un vecino mejor
            if Best_f == 0:  
                break  
    return Best, Best_f

def TabuSearch(DistanceMatrix, AmountNodes, MaxIterations=100, TabuSize=10,
               minErrorInten=0.001):
    BestSolution = first_solution(AmountNodes)
    CurrentSolution = BestSolution
    CurrentCost = ObjFun(CurrentSolution, DistanceMatrix)
    tabu_list = set()

    for _ in range(MaxIterations):
        Neighborhood = get_neighbors_2opt(CurrentSolution, DistanceMatrix)
        BestNeighbor, BestNeighbor_f = best_neighbor(Neighborhood, tabu_list, CurrentCost)

        if BestNeighbor is None:
            CurrentSolution = first_solution(AmountNodes)
            CurrentCost = ObjFun(CurrentSolution, DistanceMatrix)
            continue
        
        # Check for intensification
        if abs(BestNeighbor_f - CurrentCost) < minErrorInten:
            continue  # No improvement

        # Update tabu list
        if len(tabu_list) >= TabuSize:
            tabu_list = set(list(tabu_list)[-TabuSize:])
        tabu_list.add(tuple(CurrentSolution))

        CurrentSolution = BestNeighbor
        CurrentCost = BestNeighbor_f
        if BestNeighbor_f < ObjFun(BestSolution, DistanceMatrix):
            BestSolution = BestNeighbor

    return BestSolution
