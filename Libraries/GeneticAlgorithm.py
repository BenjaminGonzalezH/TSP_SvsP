########## Libraries ##########
import numpy as np
import sys
import os
import random

########## Own files ##########
# Path from the workspace.
sys.path.append(os.path.join(os.path.dirname(__file__), 'Libraries'))
from TabuSearch import (
    ObjFun,
    first_solution
)

########## Functions GA ##########

def initialize_population(pop_size, AmountNodes):
    """
    initialize_population(function)
        Input:
            - pop_size: Amount of solutions to manage.
            - AmountNodes: Amount of cities.
        Output:
            - Population: first population of solutions.

        Description: Function that creates the first population
        of solutions.
    """
    Population = []
    for _ in range(pop_size):
        Population.append(first_solution(AmountNodes))
    return Population

def Evaluate(population, DistanceMatrix):
    """
    Evaluate(function)
        Input:
            - population: Collection of solution.
            - DistanceMatrix: Matrix of distance (each pair of cities).
        Output:
            - evaluated_population: Pair of solution and fitness.

        Description: Evaluate the first population.
    """
    evaluated_population = [(individual, ObjFun(individual, DistanceMatrix)) for individual in population]
    return evaluated_population

def tournament_selection(population, tournament_size=3, num_winners = 5):
    """
    tournament_selection(funcion)
        Input:
            - population: Collection of solution.
            - tournament_size: amount of solution that are gonna be
            in the tournament.
            - num_winners: Amount of solutions that are gonna be in the next
            step.
        Output:
            - winners: selected solutions.
        
        Description: Selection by tournament method.
    """
    # Winners array and a element to
    # avoid repeated selections.
    winners = []
    selected_individuals = set()

    # Main Iteration.
    while len(winners) < num_winners:
        # Select candidates for the tournament.
        tournament = random.sample([ind for ind in population if tuple(ind[0]) not in selected_individuals], tournament_size)

        # Initialize variables for the winner in
        # this iteration.
        winner = None
        winner_fitness = float('inf')

        # compare candidates.
        for competitor , competitor_fitness in tournament:
            if competitor_fitness < winner_fitness:
                winner_fitness = competitor_fitness
                winner = competitor

        # Add in winners and selected individuals.
        winners.append((winner,winner_fitness))
        selected_individuals.add(tuple(winner))

    # Output.
    return winners

def crossover(parent1, parent2):
    """
    crossover(function)
        Input:
            - parent1 and parent2: Two solutions for crossover.
        Output:
            - offspring1 and offspring2: Childrens generates of
            crossovers.

        Description: crossover function.
    """
    # Initialize childs.
    size = len(parent1)
    offspring1, offspring2 = [None]*size, [None]*size

    # Take two index for crossing.
    p1, p2 = sorted(random.sample(range(size), 2))

    # Copy segment of parents to sons.
    offspring1[p1:p2] = parent1[p1:p2]
    offspring2[p1:p2] = parent2[p1:p2]

    # Avoid bad copy in childrens.
    def map_genes(offspring, parent, start, end):
        for i in range(start, end):
            if parent[i] not in offspring:
                j = i
                while start <= j < end:
                    j = np.where(parent == offspring[j])[0][0]
                offspring[j] = parent[i]
        return offspring

    offspring1 = map_genes(offspring1, parent2, p1, p2)
    offspring2 = map_genes(offspring2, parent1, p1, p2)

    # Fill empty spaces.
    for i in range(size):
        if offspring1[i] is None:
            offspring1[i] = parent2[i]
        if offspring2[i] is None:
            offspring2[i] = parent1[i]

    # Output.
    return np.array(offspring1), np.array(offspring2)
 
def mutation(individual):
    """
    mutation(function)
        Input:
            - Individual: Solution.
        Output:
            - Same individual mutated by invertion.
    """
    # Take two random potitions.
    pos1, pos2 = sorted(random.sample(range(len(individual)), 2))
    
    # Invert segment.
    individual[pos1:pos2+1] = individual[pos1:pos2+1][::-1]
    
    return individual

def genetic_algorithm(DistanceMatrix, AmountNodes, pop_size=50, MaxOFcalls=8000):
    """
    genetic_algorithm(function)
        Input:
            - DistanceMatrix: Matrix of distance (each pair of cities).
            - AmountNodes: Amount of cities.
            - pop_size: Amount of solutions to manage.
            - MaxOFcalls: Maximum of calls of objective function.
        Output:
            - evaluated_population: Last population obtained in
            iterations.
            - best_solution: best solution obtained.
    """
    # Initialize population.
    population = initialize_population(pop_size, AmountNodes)
    evaluated_population = Evaluate(population, DistanceMatrix)     # Obtain fitness.
    num_of_calls = len(evaluated_population)                        # Num of calls update.
    best_solution = min(evaluated_population, key=lambda x: x[1])   # Best solution.
    
    # Main Loop.
    while num_of_calls < MaxOFcalls:
        # Tournament Selection.
        # Using 2% size of the population and just select the
        # half of population.
        selected_parents = tournament_selection(evaluated_population,
                                                tournament_size=2,
                                                num_winners=(pop_size//2))
        
        # Create descendants.
        offspring_population = []
        for i in range(0, len(selected_parents) - 1, 2):
            # Crossovers.
            parent1, parent2 = selected_parents[i][0], selected_parents[i + 1][0]
            offspring1, offspring2 = crossover(parent1, parent2)
            
            # Mutation.
            offspring1 = mutation(offspring1)
            offspring2 = mutation(offspring2)
            
            # Evaluate childern with objective function.
            offspring1_fitness = ObjFun(offspring1, DistanceMatrix)
            offspring2_fitness = ObjFun(offspring2, DistanceMatrix)
            # Add.
            offspring_population.extend([(offspring1, offspring1_fitness), (offspring2, offspring2_fitness)])
            
            # Update Num of calls.
            num_of_calls += 2

            # Stop criteria.
            if num_of_calls >= MaxOFcalls:
                break
        
        # Add children in the main population.
        evaluated_population.extend(offspring_population)
        
        # Limit population to pop_size.
        evaluated_population = sorted(evaluated_population, key=lambda x: x[1])[:pop_size]
        
        # Update best solution.
        current_best = min(evaluated_population, key=lambda x: x[1])
        if current_best[1] < best_solution[1]:
            best_solution = current_best
    
    # Output.
    return evaluated_population, best_solution
