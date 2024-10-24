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

########## Functions MA ##########

# Función para inicializar la población
def initialize_population(pop_size, AmountNodes):
    Population = []
    for _ in range(pop_size):
        Population.append(first_solution(AmountNodes))
    return Population

# Función para ordenar la población.
def Evaluate(population, DistanceMatrix):
    evaluated_population = [(individual, ObjFun(individual, DistanceMatrix)) for individual in population]
    return evaluated_population

# Función por selección en torneo
def tournament_selection(population, tournament_size=3, num_winners = 5):
    winners = []
    selected_individuals = set()  # Para asegurar que no se repitan ganadores

    while len(winners) < num_winners:
        tournament = random.sample([ind for ind in population if tuple(ind[0]) not in selected_individuals], tournament_size)

        winner = None
        winner_fitness = float('inf')

        for competitor , competitor_fitness in tournament:
            if competitor_fitness < winner_fitness:
                winner_fitness = competitor_fitness
                winner = competitor

        # Añadir el ganador a la lista de ganadores y al conjunto para evitar duplicados
        winners.append((winner,winner_fitness))
        selected_individuals.add(tuple(winner))  # Usamos tuple para que sea "hashable" y se pueda añadir al set

    return winners

# Función de recombinación.
def crossover(parent1, parent2):
    size = len(parent1)
    offspring1, offspring2 = [None]*size, [None]*size

    # Elegir dos puntos de cruce al azar
    p1, p2 = sorted(random.sample(range(size), 2))

    # Copiar el segmento del primer padre al primer hijo y viceversa
    offspring1[p1:p2] = parent1[p1:p2]
    offspring2[p1:p2] = parent2[p1:p2]

    # Mapeo para asegurar que los descendientes no tengan duplicados
    def map_genes(offspring, parent, start, end):
        for i in range(start, end):
            if parent[i] not in offspring:
                j = i
                while start <= j < end:
                    # Usar np.where para encontrar el índice en lugar de .index()
                    j = np.where(parent == offspring[j])[0][0]
                offspring[j] = parent[i]
        return offspring

    # Completar con los genes restantes
    offspring1 = map_genes(offspring1, parent2, p1, p2)
    offspring2 = map_genes(offspring2, parent1, p1, p2)

    # Rellenar los vacíos
    for i in range(size):
        if offspring1[i] is None:
            offspring1[i] = parent2[i]
        if offspring2[i] is None:
            offspring2[i] = parent1[i]

    return np.array(offspring1), np.array(offspring2)
 
# Función de mutación
def mutation(individual):
    # Elegir dos posiciones al azar
    pos1, pos2 = sorted(random.sample(range(len(individual)), 2))
    
    # Invertir el segmento entre las dos posiciones
    individual[pos1:pos2+1] = individual[pos1:pos2+1][::-1]
    
    return individual


# Función principal del algoritmo memético con control de llamadas a la función objetivo
def genetic_algorithm(DistanceMatrix, AmountNodes, pop_size=50, MaxOFcalls=8000):
    # Inicializar población
    population = initialize_population(pop_size, AmountNodes)
    
    # Evaluar la población inicial
    evaluated_population = Evaluate(population, DistanceMatrix)
    
    # Contador de llamadas a la función objetivo
    num_of_calls = len(evaluated_population)
    
    # Almacenar la mejor solución inicial
    best_solution = min(evaluated_population, key=lambda x: x[1])
    
    # Bucle principal del algoritmo genético
    while num_of_calls < MaxOFcalls:
        # Selección por torneo
        selected_parents = tournament_selection(evaluated_population, tournament_size=3, num_winners=2)
        
        # Cruce (recombinación)
        offspring1, offspring2 = crossover(selected_parents[0][0], selected_parents[1][0])
        
        # Aplicar mutación con cierta probabilidad (ej. 0.1)
        if random.random() < 0.1:
            offspring1 = mutation(offspring1)
        if random.random() < 0.1:
            offspring2 = mutation(offspring2)
        
        # Evaluar descendencia
        offspring1_fitness = ObjFun(offspring1, DistanceMatrix)
        offspring2_fitness = ObjFun(offspring2, DistanceMatrix)
        
        # Añadir descendencia a la población
        evaluated_population.extend([(offspring1, offspring1_fitness), (offspring2, offspring2_fitness)])
        
        # Incrementar el contador de llamadas a la función objetivo
        num_of_calls += 2
        
        # Mantener la población limitada al tamaño máximo eliminando las peores soluciones
        evaluated_population = sorted(evaluated_population, key=lambda x: x[1])[:pop_size]
        
        # Actualizar la mejor solución si es necesario
        current_best = min(evaluated_population, key=lambda x: x[1])
        if current_best[1] < best_solution[1]:
            best_solution = current_best
    
    return best_solution
