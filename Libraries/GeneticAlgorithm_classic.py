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

def GAc_PMX_swap(Pop_size, DistanceMatrix, AmountNodes,
                 MaxOfCalls, tournament_size,
                 Crossover_rate, Mutation_rate):
    
    ### initialize and evaluate.
    pop = GAoperators.initialize_population(Pop_size, AmountNodes)
    pop_eval = GAoperators.Evaluate(pop, DistanceMatrix)

    ### initialize registration.
    calls = Pop_size
    Generations = []
    Generations.append(pop_eval)
    Best = Best = min(pop_eval, key=lambda x: x[1])

    ### Iterations.
    while(calls < MaxOfCalls):
        # Tournamente selection.
        parent1 = GAoperators.tournament_selection(pop_eval, tournament_size)
        parent2 = GAoperators.tournament_selection(pop_eval, tournament_size)

        # crossover.
        if(random.random() < Crossover_rate):
            child1, child2 = GAoperators.PMX(parent1[0], parent2[0])

            # mutation.
            m_child1 = GAoperators.swap_mutation(child1,Mutation_rate)
            m_child2 = GAoperators.swap_mutation(child2,Mutation_rate)

            # Evaluation.
            m_child1 = (m_child1, ObjFun(m_child1,DistanceMatrix))
            m_child2 = (m_child2, ObjFun(m_child2,DistanceMatrix))
            calls = calls + 2

            # Reduction.
            pop_eval.append(m_child1)
            pop_eval.append(m_child2)
            pop_eval = GAoperators.reduce_population(pop_eval, Pop_size)
            Generations.append(pop_eval)

            # Update best fitness.
            current_best = min(pop_eval, key=lambda x: x[1])  # Get the best individual in the current population
            if current_best[1] < Best[1]:  # If the new best is better
                Best = current_best  # Update Best


    return Best, Generations

########## Second one ##########

def GAc_OX_invertion(Pop_size, DistanceMatrix, AmountNodes,
                 MaxOfCalls, tournament_size,
                 Crossover_rate, Mutation_rate):
    
    ### initialize and evaluate.
    pop = GAoperators.initialize_population(Pop_size, AmountNodes)
    pop_eval = GAoperators.Evaluate(pop, DistanceMatrix)

    ### initialize registration.
    calls = Pop_size
    Generations = []
    Generations.append(pop_eval)
    Best = Best = min(pop_eval, key=lambda x: x[1])

    ### Iterations.
    while(calls < MaxOfCalls):
        # Tournamente selection.
        parent1 = GAoperators.tournament_selection(pop_eval, tournament_size)
        parent2 = GAoperators.tournament_selection(pop_eval, tournament_size)

        # crossover.
        if(random.random() < Crossover_rate):
            child1, child2 = GAoperators.OX(parent1[0], parent2[0])

            # mutation.
            m_child1 = GAoperators.inversion_mutation(child1,Mutation_rate)
            m_child2 = GAoperators.inversion_mutation(child2,Mutation_rate)

            # Evaluation.
            m_child1 = (m_child1, ObjFun(m_child1,DistanceMatrix))
            m_child2 = (m_child2, ObjFun(m_child2,DistanceMatrix))
            calls = calls + 2

            # Reduction.
            pop_eval.append(m_child1)
            pop_eval.append(m_child2)
            pop_eval = GAoperators.reduce_population(pop_eval, Pop_size)
            Generations.append(pop_eval)

            # Update best fitness.
            current_best = min(pop_eval, key=lambda x: x[1])  # Get the best individual in the current population
            if current_best[1] < Best[1]:  # If the new best is better
                Best = current_best  # Update Best


    return Best, Generations
            
########## Third one ##########

def GAc_PBX_scramble(Pop_size, DistanceMatrix, AmountNodes,
                 MaxOfCalls, tournament_size,
                 Crossover_rate, Mutation_rate):
    
    ### initialize and evaluate.
    pop = GAoperators.initialize_population(Pop_size, AmountNodes)
    pop_eval = GAoperators.Evaluate(pop, DistanceMatrix)

    ### initialize registration.
    calls = Pop_size
    Generations = []
    Generations.append(pop_eval)
    Best = Best = min(pop_eval, key=lambda x: x[1])

    ### Iterations.
    while(calls < MaxOfCalls):
        # Tournamente selection.
        parent1 = GAoperators.tournament_selection(pop_eval, tournament_size)
        parent2 = GAoperators.tournament_selection(pop_eval, tournament_size)

        # crossover.
        if(random.random() < Crossover_rate):
            child1, child2 = GAoperators.OX(parent1[0], parent2[0])

            # mutation.
            m_child1 = GAoperators.inversion_mutation(child1,Mutation_rate)
            m_child2 = GAoperators.inversion_mutation(child2,Mutation_rate)

            # Evaluation.
            m_child1 = (m_child1, ObjFun(m_child1,DistanceMatrix))
            m_child2 = (m_child2, ObjFun(m_child2,DistanceMatrix))
            calls = calls + 2

            # Reduction.
            pop_eval.append(m_child1)
            pop_eval.append(m_child2)
            pop_eval = GAoperators.reduce_population(pop_eval, Pop_size)
            Generations.append(pop_eval)

            # Update best fitness.
            current_best = min(pop_eval, key=lambda x: x[1])  # Get the best individual in the current population
            if current_best[1] < Best[1]:  # If the new best is better
                Best = current_best  # Update Best


    return Best, Generations