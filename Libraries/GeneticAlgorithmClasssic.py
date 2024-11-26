########## Libraries ##########
import sys
import os
import random

# Path from the workspace.
sys.path.append(os.path.join(os.path.dirname(__file__), 'Libraries'))
import GeneticAlgorithmOp as GAop
from TabuSearch import (
    ObjFun
)

def GeneticAlgorithm_Classic(Pop_Size, AmountNodes, DistanceMatrix, 
                             crossover_rate, mutation_rate, mutation_proportion,
                             MaxOfCalls, num_contestant):
    
    # Initialice and evaluate.
    Pop = GAop.initialize_population(Pop_Size, AmountNodes)
    Pop_eval = GAop.Evaluate(Pop, DistanceMatrix)
    calls = len(Pop_eval)
    
    # Initialice list of generations.
    Generations = []
    Generations.append(Pop_eval)

    # Best.
    best = min(Pop_eval, key=lambda x: x[1])

    while(calls < MaxOfCalls):
        parent1 = GAop.tournament_selection(Pop, num_contestant)
        parent2 = GAop.tournament_selection(Pop, num_contestant)

        # Crossover.
        if(random.random() < crossover_rate):
            child1, child2 = GAop.PMX(parent1, parent2)

            # Mutation.
            child1 = GAop.swap_mutation(child1, mutation_rate/100, mutation_proportion)
            child2 = GAop.swap_mutation(child2, mutation_rate/100, mutation_proportion)

            child1 = (child1, ObjFun(child1,DistanceMatrix))
            child2 = (child2, ObjFun(child2,DistanceMatrix))

            calls = calls + 2

            # Agregar los hijos a la población.
            Pop_eval.extend([child1, child2])

            # Actualizar el mejor individuo.
            current_best = min([child1, child2], key=lambda x: x[1])
            if current_best[1] < best[1]:
                best = current_best

        # Reducción de la población: Selección de supervivencia.
        Pop_eval = GAop.reduce_population(Pop_eval, Pop_Size)

        # Actualizar población.
        Pop = [ind[0] for ind in Pop_eval]

        # Guardar la población actual en las generaciones.
        Generations.append(Pop_eval)

    return best, Generations
