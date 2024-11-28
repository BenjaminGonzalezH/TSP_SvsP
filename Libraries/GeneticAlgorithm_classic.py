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
    
    # Inicialización y evaluación de la población
    pop = GAoperators.initialize_population(Pop_size, AmountNodes)
    pop_eval = GAoperators.Evaluate(pop, DistanceMatrix)

    # Registro de las generaciones y la mejor solución
    calls = Pop_size
    Generations = []
    Generations.append(pop_eval)
    Best = min(pop_eval, key=lambda x: x[1])

    # Crear un conjunto con las representaciones de los cromosomas como tuplas
    pop_eval_set = set(tuple(par[0]) for par in pop_eval)

    # Iteraciones del algoritmo genético
    while calls < MaxOfCalls:
        childs = []
        
        # Selección, cruce y mutación
        for i in range(Pop_size):
            parent1 = GAoperators.tournament_selection(pop_eval, tournament_size)
            parent2 = GAoperators.tournament_selection(pop_eval, tournament_size)

            if random.random() < Crossover_rate:
                child1, child2 = GAoperators.PMX(parent1[0], parent2[0])

                # Mutación
                m_child1 = GAoperators.swap_mutation(child1, Mutation_rate)
                m_child2 = GAoperators.swap_mutation(child2, Mutation_rate)

                # Evaluación de los hijos
                m_child1 = (m_child1, ObjFun(m_child1, DistanceMatrix))
                m_child2 = (m_child2, ObjFun(m_child2, DistanceMatrix))
                calls += 2

                # Agregar los hijos solo si no están duplicados
                if tuple(m_child1[0]) not in pop_eval_set:
                    childs.append(m_child1)
                    pop_eval_set.add(tuple(m_child1[0]))

                if tuple(m_child2[0]) not in pop_eval_set:
                    childs.append(m_child2)
                    pop_eval_set.add(tuple(m_child2[0]))

        # Integración de los nuevos hijos a la población
        pop_eval.extend(childs)

        # Reducción de población (sin necesidad de recorrer toda la población)
        pop_eval = GAoperators.reduce_population(pop_eval, Pop_size)
        
        # Actualizar la mejor solución
        current_best = min(pop_eval, key=lambda x: x[1])
        if current_best[1] < Best[1]:
            Best = current_best

        # Registrar las generaciones
        Generations.append(pop_eval)

    return Best, Generations

########## Second one ##########

def GAc_OX_invertion(Pop_size, DistanceMatrix, AmountNodes,
                     MaxOfCalls, tournament_size,
                     Crossover_rate, Mutation_rate):
    
    # Inicialización y evaluación de la población
    pop = GAoperators.initialize_population(Pop_size, AmountNodes)
    pop_eval = GAoperators.Evaluate(pop, DistanceMatrix)

    # Registro de las generaciones y la mejor solución
    calls = Pop_size
    Generations = []
    Generations.append(pop_eval)
    Best = min(pop_eval, key=lambda x: x[1])

    # Crear un conjunto con las representaciones de los cromosomas como tuplas
    pop_eval_set = set(tuple(par[0]) for par in pop_eval)

    # Iteraciones del algoritmo genético
    while calls < MaxOfCalls:
        childs = []
        
        # Selección, cruce y mutación
        for i in range(Pop_size):
            parent1 = GAoperators.tournament_selection(pop_eval, tournament_size)
            parent2 = GAoperators.tournament_selection(pop_eval, tournament_size)

            if random.random() < Crossover_rate:
                child1, child2 = GAoperators.OX(parent1[0], parent2[0])

                # Mutación
                m_child1 = GAoperators.inversion_mutation(child1, Mutation_rate)
                m_child2 = GAoperators.inversion_mutation(child2, Mutation_rate)

                # Evaluación de los hijos
                m_child1 = (m_child1, ObjFun(m_child1, DistanceMatrix))
                m_child2 = (m_child2, ObjFun(m_child2, DistanceMatrix))
                calls += 2

                # Agregar los hijos solo si no están duplicados
                if tuple(m_child1[0]) not in pop_eval_set:
                    childs.append(m_child1)
                    pop_eval_set.add(tuple(m_child1[0]))

                if tuple(m_child2[0]) not in pop_eval_set:
                    childs.append(m_child2)
                    pop_eval_set.add(tuple(m_child2[0]))

        # Integración de los nuevos hijos a la población
        pop_eval.extend(childs)

        # Reducción de población (sin necesidad de recorrer toda la población)
        pop_eval = GAoperators.reduce_population(pop_eval, Pop_size)
        
        # Actualizar la mejor solución
        current_best = min(pop_eval, key=lambda x: x[1])
        if current_best[1] < Best[1]:
            Best = current_best

        # Registrar las generaciones
        Generations.append(pop_eval)

    return Best, Generations
            
########## Third one ##########

def GAc_PBX_scramble(Pop_size, DistanceMatrix, AmountNodes,
                 MaxOfCalls, tournament_size,
                 Crossover_rate, Mutation_rate):
    
    # Inicialización y evaluación de la población
    pop = GAoperators.initialize_population(Pop_size, AmountNodes)
    pop_eval = GAoperators.Evaluate(pop, DistanceMatrix)

    # Registro de las generaciones y la mejor solución
    calls = Pop_size
    Generations = []
    Generations.append(pop_eval)
    Best = min(pop_eval, key=lambda x: x[1])

    # Crear un conjunto con las representaciones de los cromosomas como tuplas
    pop_eval_set = set(tuple(par[0]) for par in pop_eval)

    # Iteraciones del algoritmo genético
    while calls < MaxOfCalls:
        childs = []
        
        # Selección, cruce y mutación
        for i in range(Pop_size):
            parent1 = GAoperators.tournament_selection(pop_eval, tournament_size)
            parent2 = GAoperators.tournament_selection(pop_eval, tournament_size)

            if random.random() < Crossover_rate:
                child1, child2 = GAoperators.PBX(parent1[0], parent2[0])

                # Mutación
                m_child1 = GAoperators.scramble_mutation(child1, Mutation_rate)
                m_child2 = GAoperators.scramble_mutation(child2, Mutation_rate)

                # Evaluación de los hijos
                m_child1 = (m_child1, ObjFun(m_child1, DistanceMatrix))
                m_child2 = (m_child2, ObjFun(m_child2, DistanceMatrix))
                calls += 2

                # Agregar los hijos solo si no están duplicados
                if tuple(m_child1[0]) not in pop_eval_set:
                    childs.append(m_child1)
                    pop_eval_set.add(tuple(m_child1[0]))

                if tuple(m_child2[0]) not in pop_eval_set:
                    childs.append(m_child2)
                    pop_eval_set.add(tuple(m_child2[0]))

        # Integración de los nuevos hijos a la población
        pop_eval.extend(childs)

        # Reducción de población (sin necesidad de recorrer toda la población)
        pop_eval = GAoperators.reduce_population(pop_eval, Pop_size)
        
        # Actualizar la mejor solución
        current_best = min(pop_eval, key=lambda x: x[1])
        if current_best[1] < Best[1]:
            Best = current_best

        # Registrar las generaciones
        Generations.append(pop_eval)

    return Best, Generations