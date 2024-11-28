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

########## AUX function ##########


def fill_remaining_genes(offspring, parent):
    pos = 0
    for gene in parent:
        if gene not in offspring:
            while offspring[pos] is not None:
                pos += 1
            offspring[pos] = gene

########## Initialize and evaluate ##########

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

def initialize_population_C9(pop_size, AmountNodes):
    Population = [[None] * pop_size for _ in range(pop_size)]
    Population_l = []
    for i in range(pop_size):
        for j in range(pop_size):
            Population[i][j] = first_solution(AmountNodes)
            Population_l.append(Population[i][j])
    return Population, Population_l


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

def EvaluateC9(population, DistanceMatrix, pop_size):
    for i in range(pop_size):
        for j in range(pop_size):
            population[i][j] = (population[i][j], ObjFun(population[i][j], DistanceMatrix))

    return population

########## Selection ##########

def tournament_selection(population, tournament_size=3):
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
    tournament = random.sample(population, tournament_size)
        
    # Check if tournament has candidates to avoid 'NoneType' error
    winner, winner_fitness = min(tournament, key=lambda x: x[1])

    return (winner, winner_fitness)

def random_selection(population):
    choice = random.choice(population)
    return choice

def select_neighbors(population, i, j):
    neighbors = []
    filas = len(population)
    cols = len(population[0])
    
    # Direcciones de los vecinos (arriba, abajo, izquierda, derecha, y las diagonales)
    direcciones = [(-1, 0), (1, 0), (0, -1), (0, 1),  # arriba, abajo, izquierda, derecha
                   (-1, -1), (-1, 1), (1, -1), (1, 1)]  # diagonales
    
    # Comprobamos los vecinos y agregamos los que están dentro de los límites de la matriz
    for df, dc in direcciones:
        nuevo_fila, nuevo_col = i + df, j + dc
        if 0 <= nuevo_fila < filas and 0 <= nuevo_col < cols:
            neighbors.append(population[nuevo_fila][nuevo_col])
    
    return neighbors

########## Crossover ##########

def PMX(parent1, parent2):
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
 
def OX(parent1, parent2):
    """
    Ordered Crossover (OX) implementation.
    
    Args:
        parent1: List representing the first parent chromosome.
        parent2: List representing the second parent chromosome.
    
    Returns:
        offspring1, offspring2: Two offspring chromosomes.
    """
    size = len(parent1)

    # Elegir dos puntos de corte aleatorios
    p1, p2 = sorted(random.sample(range(size), 2))

    # Inicializar los hijos con None
    offspring1, offspring2 = [None] * size, [None] * size

    # Copiar el segmento seleccionado de los padres a los hijos
    offspring1[p1:p2] = parent1[p1:p2]
    offspring2[p1:p2] = parent2[p1:p2]

    # Completar el resto de los genes del otro padre en orden relativo
    def fill_genes(offspring, parent):
        pos = p2
        for gene in parent:
            if gene not in offspring:
                if pos >= size:
                    pos = 0
                offspring[pos] = gene
                pos += 1

    fill_genes(offspring1, parent2)
    fill_genes(offspring2, parent1)

    return offspring1, offspring2

def PBX(parent1, parent2):
    """
    Position-Based Crossover (PBX) implementation.
    
    Args:
        parent1: List representing the first parent chromosome.
        parent2: List representing the second parent chromosome.
    
    Returns:
        offspring1, offspring2: Two offspring chromosomes.
    """
    size = len(parent1)

    # Elegir posiciones aleatorias
    selected_positions = sorted(random.sample(range(size), size // 2))

    # Inicializar los hijos con None
    offspring1 = [None] * size
    offspring2 = [None] * size

    # Copiar los genes en las posiciones seleccionadas
    for pos in selected_positions:
        offspring1[pos] = parent1[pos]
        offspring2[pos] = parent2[pos]

    fill_remaining_genes(offspring1, parent2)
    fill_remaining_genes(offspring2, parent1)

    return offspring1, offspring2

########## Mutation ##########

def swap_mutation(chromosome, mutation_rate):
    """
    Realiza mutación por intercambio en un cromosoma.

    Args:
        chromosome: Lista que representa una permutación.
        mutation_rate: Probabilidad de mutación.

    Returns:
        Mutated chromosome.
    """
    mutation_rate = mutation_rate / 100.0
    if random.random() < mutation_rate:
        # Seleccionar dos posiciones aleatorias
        i, j = random.sample(range(len(chromosome)), 2)
        # Intercambiar los elementos
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    return chromosome

def inversion_mutation(chromosome, mutation_rate):
    """
    Realiza mutación por inversión en un cromosoma.

    Args:
        chromosome: Lista que representa una permutación.
        mutation_rate: Probabilidad de mutación.

    Returns:
        Mutated chromosome.
    """
    mutation_rate = mutation_rate / 100.0
    if random.random() < mutation_rate:
        # Seleccionar dos posiciones aleatorias
        start, end = sorted(random.sample(range(len(chromosome)), 2))
        # Invertir el segmento
        chromosome[start:end] = reversed(chromosome[start:end])
    return chromosome

def scramble_mutation(chromosome, mutation_rate):
    """
    Realiza mutación por mezcla en un cromosoma.

    Args:
        chromosome: Lista que representa una permutación.
        mutation_rate: Probabilidad de mutación.

    Returns:
        Mutated chromosome.
    """
    mutation_rate = mutation_rate / 100.0
    if random.random() < mutation_rate:
        # Seleccionar dos posiciones aleatorias
        start, end = sorted(random.sample(range(len(chromosome)), 2))
        # Mezclar aleatoriamente el segmento
        segment = chromosome[start:end]
        random.shuffle(segment)
        chromosome[start:end] = segment
    return chromosome

########## Eliminate ##########

def reduce_population(population, max_population_size):
    """
    Elimina las soluciones con mayor fitness para mantener el tamaño de la población.

    Args:
        population: Lista de cromosomas (soluciones).
        max_population_size: Tamaño máximo de la población después de la reducción.

    Returns:
        reduced_population: Lista de cromosomas reducida.
        reduced_fitnesses: Lista de fitness correspondiente a los cromosomas restantes.
    """
    # Ordenar por fitness (menor es mejor)
    population.sort(key=lambda x: x[1])

    # Seleccionar los mejores individuos según max_population_size
    reduced_combined = population[:max_population_size]

    return reduced_combined

def replace_populationC9(population, child, i, j):
    filas = len(population)
    cols = len(population[0])

    # Direcciones de los vecinos (arriba, abajo, izquierda, derecha, y las diagonales)
    direcciones = [(-1, 0), (1, 0), (0, -1), (0, 1),  # arriba, abajo, izquierda, derecha
                   (-1, -1), (-1, 1), (1, -1), (1, 1)]  # diagonales

    # Conjunto para verificar duplicados de manera eficiente
    seen = set(tuple(individual[0].tolist()) for row in population for individual in row)
    
    # Convirtiendo el cromosoma del hijo a tupla solo una vez
    child_tuple = tuple(child[0].tolist())

    # Comprobamos los vecinos y agregamos los que están dentro de los límites de la matriz
    for df, dc in direcciones:
        nuevo_fila, nuevo_col = i + df, j + dc
        if 0 <= nuevo_fila < filas and 0 <= nuevo_col < cols:
            # Si el cromosoma hijo es mejor (menor fitness) que el de la población actual
            if child[1] <= population[nuevo_fila][nuevo_col][1]:
                # Solo agregar si el cromosoma del hijo no está ya en 'seen'
                if child_tuple not in seen:
                    population[nuevo_fila][nuevo_col] = child
                    seen.add(child_tuple)  # Añadir el cromosoma al conjunto

    return population

def obtain_minimal_C9(matriz):
    minimo = ([], float('inf'))
    
    for fila in matriz:
        for par in fila:
            segundo_elemento = par[1]
            if segundo_elemento < minimo[1]:
                minimo = par
    
    return minimo