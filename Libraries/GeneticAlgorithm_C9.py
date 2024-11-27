########## Libraries ##########
import numpy as np
import sys
import os
import random

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
    pop = GAoperators.initialize_population_C9(Pop_size, AmountNodes)
    pop_eval = GAoperators.EvaluateC9(pop, DistanceMatrix, Pop_size)

    ### initialize registration.
    calls = Pop_size*Pop_size
    Generations = []
    Generations.append(pop_eval)
    Best = GAoperators.obtain_minimal_C9(pop_eval)

    ### Iterations.
    while(calls < MaxOfCalls):
        
        for i in range(Pop_size):
            for j in range(Pop_size):
                # Obtain neighbors.
                neighbors = GAoperators.select_neighbors(pop_eval, i, j)

                # Tournamente selection.
                parent1 = GAoperators.random_selection(neighbors)
                parent2 = GAoperators.tournament_selection(neighbors, 2)

                # crossover
                if(random.random() < Crossover_rate):
                    child1, child2 = GAoperators.PMX(parent1[0], parent2[0])

                    # mutation.
                    m_child1 = GAoperators.swap_mutation(child1,Mutation_rate)
                    m_child2 = GAoperators.swap_mutation(child2,Mutation_rate)

                    # Evaluation.
                    m_child1 = (m_child1, ObjFun(m_child1,DistanceMatrix))
                    m_child2 = (m_child2, ObjFun(m_child2,DistanceMatrix))
                    calls = calls + 2

                    if(calls > MaxOfCalls):
                        break

                    # Replacement.
                    child_selected = min([m_child1, m_child2], key=lambda x: x[1])
                    pop_eval = GAoperators.replace_populationC9(pop_eval, child_selected, i, j)
                    
                    # Updating.
                    Generations.append(pop_eval)
                    if(Best[1] > child_selected[1]):
                        Best = child_selected
                    
    return Generations, Best

########## Second one ##########

def GAe_OX_invertion(Pop_size, DistanceMatrix, AmountNodes,
                 MaxOfCalls, Crossover_rate, 
                 Mutation_rate):
    ### initialize and evaluate.
    pop = GAoperators.initialize_population_C9(Pop_size, AmountNodes)
    pop_eval = GAoperators.EvaluateC9(pop, DistanceMatrix, Pop_size)

    ### initialize registration.
    calls = Pop_size*Pop_size
    Generations = []
    Generations.append(pop_eval)
    Best = GAoperators.obtain_minimal_C9(pop_eval)

    ### Iterations.
    while(calls < MaxOfCalls):
        
        for i in range(Pop_size):
            for j in range(Pop_size):
                # Obtain neighbors.
                neighbors = GAoperators.select_neighbors(pop_eval, i, j)

                # Tournamente selection.
                parent1 = GAoperators.random_selection(neighbors)
                parent2 = GAoperators.tournament_selection(neighbors, 2)

                # crossover
                if(random.random() < Crossover_rate):
                    child1, child2 = GAoperators.OX(parent1[0], parent2[0])

                    # mutation.
                    m_child1 = GAoperators.inversion_mutation(child1,Mutation_rate)
                    m_child2 = GAoperators.inversion_mutation(child2,Mutation_rate)

                    # Evaluation.
                    m_child1 = (m_child1, ObjFun(m_child1,DistanceMatrix))
                    m_child2 = (m_child2, ObjFun(m_child2,DistanceMatrix))
                    calls = calls + 2

                    if(calls > MaxOfCalls):
                        break

                    # Replacement.
                    child_selected = min([m_child1, m_child2], key=lambda x: x[1])
                    pop_eval = GAoperators.replace_populationC9(pop_eval, child_selected, i, j)
                    
                    # Updating.
                    Generations.append(pop_eval)
                    if(Best[1] > child_selected[1]):
                        Best = child_selected
                    
    return Generations, Best

########## Third one ##########

def GAe_PBX_scramble(Pop_size, DistanceMatrix, AmountNodes,
                 MaxOfCalls, Crossover_rate, 
                 Mutation_rate):
    ### initialize and evaluate.
    pop = GAoperators.initialize_population_C9(Pop_size, AmountNodes)
    pop_eval = GAoperators.EvaluateC9(pop, DistanceMatrix, Pop_size)

    ### initialize registration.
    calls = Pop_size*Pop_size
    Generations = []
    Generations.append(pop_eval)
    Best = GAoperators.obtain_minimal_C9(pop_eval)

    ### Iterations.
    while(calls < MaxOfCalls):
        
        for i in range(Pop_size):
            for j in range(Pop_size):
                # Obtain neighbors.
                neighbors = GAoperators.select_neighbors(pop_eval, i, j)

                # Tournamente selection.
                parent1 = GAoperators.random_selection(neighbors)
                parent2 = GAoperators.tournament_selection(neighbors, 2)

                # crossover
                if(random.random() < Crossover_rate):
                    child1, child2 = GAoperators.PBX(parent1[0], parent2[0])

                    # mutation.
                    m_child1 = GAoperators.scramble_mutation(child1,Mutation_rate)
                    m_child2 = GAoperators.scramble_mutation(child2,Mutation_rate)

                    # Evaluation.
                    m_child1 = (m_child1, ObjFun(m_child1,DistanceMatrix))
                    m_child2 = (m_child2, ObjFun(m_child2,DistanceMatrix))
                    calls = calls + 2

                    if(calls > MaxOfCalls):
                        break

                    # Replacement.
                    child_selected = min([m_child1, m_child2], key=lambda x: x[1])
                    pop_eval = GAoperators.replace_populationC9(pop_eval, child_selected, i, j)
                    
                    # Updating.
                    Generations.append(pop_eval)
                    if(Best[1] > child_selected[1]):
                        Best = child_selected
                    
    return Generations, Best