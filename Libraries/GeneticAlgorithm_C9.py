########## Libraries ##########
import numpy as np
import sys
import os
import random
import copy

########## Own files ##########
# Path from the workspace.
sys.path.append(os.path.join(os.path.dirname(__file__), 'Libraries'))
import GAoperators
from TabuSearch import ObjFun

########## First one ##########

def GAe_PMX_swap(Pop_size, DistanceMatrix, AmountNodes,
                 MaxOfCalls, Crossover_rate, 
                 Mutation_rate):
    ### initialize and evaluate.
    pop, pop_l = GAoperators.initialize_population_C9(Pop_size, AmountNodes)
    pop_eval = GAoperators.EvaluateC9(pop, DistanceMatrix, Pop_size)

    ### initialize registration.
    calls = Pop_size * Pop_size
    Generations = []
    Generations.append(copy.deepcopy(pop_l))
    Best = GAoperators.obtain_minimal_C9(pop_eval)

    # Precompute seen chromosomes as a set of tuples for fast lookup
    seen = set(tuple(individual[0].tolist()) for row in pop_eval for individual in row)

    ### Iterations.
    while calls < MaxOfCalls:
        for i in range(Pop_size):
            for j in range(Pop_size):
                # Obtain neighbors.
                neighbors = GAoperators.select_neighbors(pop_eval, i, j)

                # Tournament selection
                parent1 = GAoperators.random_selection(neighbors)
                parent2 = GAoperators.tournament_selection(neighbors, 2)

                # Crossover
                if random.random() < Crossover_rate:
                    child1, child2 = GAoperators.PMX(parent1[0], parent2[0])

                    # Mutation
                    m_child1 = GAoperators.swap_mutation(child1, Mutation_rate)
                    m_child2 = GAoperators.swap_mutation(child2, Mutation_rate)

                    # Evaluation.
                    m_child1 = (m_child1, ObjFun(m_child1, DistanceMatrix))
                    m_child2 = (m_child2, ObjFun(m_child2, DistanceMatrix))
                    calls += 2

                    if calls > MaxOfCalls:
                        break

                    # Replacement with best selection
                    child_selected = min([m_child1, m_child2], key=lambda x: x[1])

                    # Convert child to tuple for fast duplicate check
                    child_tuple = tuple(child_selected[0].tolist())

                    # Check if child is already in the population
                    if child_tuple not in seen:
                        # Insert the child and add to 'seen'
                        direcciones = [(-1, 0), (1, 0), (0, -1), (0, 1),  # arriba, abajo, izquierda, derecha
                                    (-1, -1), (-1, 1), (1, -1), (1, 1)]  # diagonales

                        # Iterate over directions and check if child is better than any neighbor
                        for df, dc in direcciones:
                            nuevo_fila, nuevo_col = i + df, j + dc
                            if 0 <= nuevo_fila < len(pop_eval) and 0 <= nuevo_col < len(pop_eval[0]):
                                if child_selected[1] <= pop_eval[nuevo_fila][nuevo_col][1]:
                                    pop_eval[nuevo_fila][nuevo_col] = child_selected
                                    index_lineal = nuevo_fila * len(pop_eval[0]) + nuevo_col
                                    pop_l[index_lineal] = child_selected[0]
                                    seen.add(child_tuple)  # Add the child to 'seen'
                                    break  # Exit after replacing

                    # Update best solution found so far
                    if Best[1] > child_selected[1]:
                        Best = child_selected

        Generations.append(copy.deepcopy(pop_l))
                    
    return Generations, Best

########## Second one ##########

def GAe_OX_invertion(Pop_size, DistanceMatrix, AmountNodes,
                 MaxOfCalls, Crossover_rate, 
                 Mutation_rate):
    ### initialize and evaluate.
    pop, pop_l = GAoperators.initialize_population_C9(Pop_size, AmountNodes)
    pop_eval = GAoperators.EvaluateC9(pop, DistanceMatrix, Pop_size)

    ### initialize registration.
    calls = Pop_size * Pop_size
    Generations = []
    Generations.append(copy.deepcopy(pop_l))
    Best = GAoperators.obtain_minimal_C9(pop_eval)

    # Precompute seen chromosomes as a set of tuples for fast lookup
    seen = set(tuple(individual[0].tolist()) for row in pop_eval for individual in row)

    ### Iterations.
    while calls < MaxOfCalls:
        for i in range(Pop_size):
            for j in range(Pop_size):
                # Obtain neighbors.
                neighbors = GAoperators.select_neighbors(pop_eval, i, j)

                # Tournament selection
                parent1 = GAoperators.random_selection(neighbors)
                parent2 = GAoperators.tournament_selection(neighbors, 2)

                # Crossover
                if random.random() < Crossover_rate:
                    child1, child2 = GAoperators.OX(parent1[0], parent2[0])

                    # Mutation
                    m_child1 = GAoperators.scramble_mutation(child1, Mutation_rate)
                    m_child2 = GAoperators.scramble_mutation(child2, Mutation_rate)

                    # Evaluation.
                    m_child1 = (m_child1, ObjFun(m_child1, DistanceMatrix))
                    m_child2 = (m_child2, ObjFun(m_child2, DistanceMatrix))
                    calls += 2

                    if calls > MaxOfCalls:
                        break

                    # Replacement with best selection
                    child_selected = min([m_child1, m_child2], key=lambda x: x[1])

                    # Convert child to tuple for fast duplicate check
                    child_tuple = tuple(child_selected[0])

                    # Check if child is already in the population
                    if child_tuple not in seen:
                        # Insert the child and add to 'seen'
                        direcciones = [(-1, 0), (1, 0), (0, -1), (0, 1),  # arriba, abajo, izquierda, derecha
                                    (-1, -1), (-1, 1), (1, -1), (1, 1)]  # diagonales

                        # Iterate over directions and check if child is better than any neighbor
                        for df, dc in direcciones:
                            nuevo_fila, nuevo_col = i + df, j + dc
                            if 0 <= nuevo_fila < len(pop_eval) and 0 <= nuevo_col < len(pop_eval[0]):
                                if child_selected[1] <= pop_eval[nuevo_fila][nuevo_col][1]:
                                    pop_eval[nuevo_fila][nuevo_col] = child_selected
                                    index_lineal = nuevo_fila * len(pop_eval[0]) + nuevo_col
                                    pop_l[index_lineal] = child_selected[0]
                                    seen.add(child_tuple)  # Add the child to 'seen'
                                    break  # Exit after replacing

                    # Update best solution found so far
                    if Best[1] > child_selected[1]:
                        Best = child_selected

        Generations.append(copy.deepcopy(pop_l))
                    
    return Generations, Best

########## Third one ##########

def GAe_PBX_scramble(Pop_size, DistanceMatrix, AmountNodes,
                 MaxOfCalls, Crossover_rate, 
                 Mutation_rate):
    ### initialize and evaluate.
    pop, pop_l = GAoperators.initialize_population_C9(Pop_size, AmountNodes)
    pop_eval = GAoperators.EvaluateC9(pop, DistanceMatrix, Pop_size)

    ### initialize registration.
    calls = Pop_size * Pop_size
    Generations = []
    Generations.append(copy.deepcopy(pop_l))
    Best = GAoperators.obtain_minimal_C9(pop_eval)

    # Precompute seen chromosomes as a set of tuples for fast lookup
    seen = set(tuple(individual[0].tolist()) for row in pop_eval for individual in row)

    ### Iterations.
    while calls < MaxOfCalls:
        for i in range(Pop_size):
            for j in range(Pop_size):
                # Obtain neighbors.
                neighbors = GAoperators.select_neighbors(pop_eval, i, j)

                # Tournament selection
                parent1 = GAoperators.random_selection(neighbors)
                parent2 = GAoperators.tournament_selection(neighbors, 2)

                # Crossover
                if random.random() < Crossover_rate:
                    child1, child2 = GAoperators.PBX(parent1[0], parent2[0])

                    # Mutation
                    m_child1 = GAoperators.scramble_mutation(child1, Mutation_rate)
                    m_child2 = GAoperators.scramble_mutation(child2, Mutation_rate)

                    # Evaluation.
                    m_child1 = (m_child1, ObjFun(m_child1, DistanceMatrix))
                    m_child2 = (m_child2, ObjFun(m_child2, DistanceMatrix))
                    calls += 2

                    if calls > MaxOfCalls:
                        break

                    # Replacement with best selection
                    child_selected = min([m_child1, m_child2], key=lambda x: x[1])

                    # Convert child to tuple for fast duplicate check
                    child_tuple = tuple(child_selected[0])

                    # Check if child is already in the population
                    if child_tuple not in seen:
                        # Insert the child and add to 'seen'
                        direcciones = [(-1, 0), (1, 0), (0, -1), (0, 1),  # arriba, abajo, izquierda, derecha
                                    (-1, -1), (-1, 1), (1, -1), (1, 1)]  # diagonales

                        # Iterate over directions and check if child is better than any neighbor
                        for df, dc in direcciones:
                            nuevo_fila, nuevo_col = i + df, j + dc
                            if 0 <= nuevo_fila < len(pop_eval) and 0 <= nuevo_col < len(pop_eval[0]):
                                if child_selected[1] <= pop_eval[nuevo_fila][nuevo_col][1]:
                                    pop_eval[nuevo_fila][nuevo_col] = child_selected
                                    index_lineal = nuevo_fila * len(pop_eval[0]) + nuevo_col
                                    pop_l[index_lineal] = child_selected[0]
                                    seen.add(child_tuple)  # Add the child to 'seen'
                                    break  # Exit after replacing

                    # Update best solution found so far
                    if Best[1] > child_selected[1]:
                        Best = child_selected

        Generations.append(copy.deepcopy(pop_l))
                    
    return Generations, Best