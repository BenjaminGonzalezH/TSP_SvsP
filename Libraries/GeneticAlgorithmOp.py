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

########## AUX Functions ##########

def fill_with_parent_genes_optimized(offspring, parent, p1, p2):
    """
    Optimized function to fill the offspring with genes from the other parent.
    """
    size = len(offspring)
    # Track the genes already present in the offspring.
    present_genes = set(offspring[p1:p2])

    # Fill the rest of the offspring.
    pos = p2
    for gene in parent:
        if gene not in present_genes:
            if pos >= size:
                pos = 0
            offspring[pos] = gene
            pos += 1

def fill_positions(offspring, parent, selected_positions):
    """
    Fill the empty positions in the offspring with genes from the other parent.

    Args:
        offspring: Partially filled offspring chromosome.
        parent: The other parent chromosome.
        selected_positions: List of positions already filled in the offspring.
    """
    size = len(parent)
    current_pos = 0

    for gene in parent:
        if gene not in offspring:  # Añadir solo genes que no estén en el hijo
            while current_pos in selected_positions:  # Saltar posiciones seleccionadas
                current_pos += 1
            offspring[current_pos] = gene
            current_pos += 1

########## Operators Functions ##########

### Initialization and evaluation.
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

### Selection.

def tournament_selection(population, num_contestants):
    """
    tournament_selection(funcion)
        Input:
            - population: Collection of solution.
            - num_contestants: amount of solution that are gonna be
            in the tournament.
        Output:
            - winner: selected solution.
        
        Description: Selection by tournament method.
    """
    contestant = random.sample(population, num_contestants)
    winner = (min(contestant, key=lambda x: x[1]))
    return winner

def random_selection(population):
    """
    tournament_selection(funcion)
        Input:
            - population: Collection of solution.
        Output:
            - winner: selected solution.
        
        Description: Selection by random selection.
    """
    winner = random.choice(population)
    return winner

### Crossover.

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
    size = len(parent1)
    offspring1, offspring2 = [None] * size, [None] * size

    # Select two indices for the crossover segment
    p1, p2 = sorted(random.sample(range(size), 2))

    # Copy crossover segment from parents to offspring
    offspring1[p1:p2] = parent1[p1:p2]
    offspring2[p1:p2] = parent2[p1:p2]

    # Mapping dictionaries for resolving conflicts
    mapping1 = {parent1[i]: parent2[i] for i in range(p1, p2)}
    mapping2 = {parent2[i]: parent1[i] for i in range(p1, p2)}

    # Function to resolve conflicts using the mapping
    def resolve_mapping(value, mapping):
        while value in mapping:
            value = mapping[value]
        return value

    # Fill remaining positions in offspring
    for i in range(size):
        if offspring1[i] is None:
            offspring1[i] = resolve_mapping(parent2[i], mapping1)
        if offspring2[i] is None:
            offspring2[i] = resolve_mapping(parent1[i], mapping2)

    return np.array(offspring1), np.array(offspring2)

def OX(parent1, parent2):
    """
    Optimized Ordered Crossover (OX).
    """
    size = len(parent1)
    offspring1, offspring2 = [None] * size, [None] * size

    # Step 1: Choose two random cut points.
    p1, p2 = sorted(random.sample(range(size), 2))

    # Step 2: Copy the segment from parents to offspring.
    offspring1[p1:p2] = parent1[p1:p2]
    offspring2[p1:p2] = parent2[p1:p2]

    # Step 3: Fill the rest of the genes from the other parent.
    fill_with_parent_genes_optimized(offspring1, parent2, p1, p2)
    fill_with_parent_genes_optimized(offspring2, parent1, p1, p2)

    return np.array(offspring1), np.array(offspring2)

def PBX(parent1, parent2):
    """
    Position-Based Crossover (PBX).
    
    Args:
        parent1: List representing the first parent chromosome.
        parent2: List representing the second parent chromosome.

    Returns:
        offspring1, offspring2: Two offspring chromosomes.
    """
    size = len(parent1)
    
    # Paso 1: Seleccionar posiciones aleatorias en parent1
    num_positions = random.randint(1, size // 2)  # Número de posiciones aleatorias
    selected_positions = sorted(random.sample(range(size), num_positions))

    # Paso 2: Crear hijos inicializando como listas vacías
    offspring1 = [None] * size
    offspring2 = [None] * size

    # Paso 3: Copiar los genes en las posiciones seleccionadas
    for pos in selected_positions:
        offspring1[pos] = parent1[pos]
        offspring2[pos] = parent2[pos]

    # Paso 4: Completar los hijos con los genes del otro padre respetando el orden
    fill_positions(offspring1, parent2, selected_positions)
    fill_positions(offspring2, parent1, selected_positions)

    return offspring1, offspring2

### Mutation.

def swap_mutation(chromosome, mutation_rate, proportion_mutation):
    """
    Realiza mutación por intercambio en un cromosoma de permutación.
    
    Args:
        chromosome: Lista que representa una permutación.
        mutation_rate: Probabilidad de mutación por gen.
    
    Returns:
        Mutated chromosome.
    """
    size = len(chromosome)
    num_swaps = max(1, (size * proportion_mutation)//100)

    for _ in range(num_swaps):
        if random.random() < mutation_rate:
            i = random.randint(0, size - 1)
            j = random.randint(0, size - 1)
            chromosome[i], chromosome[j] = chromosome[j], chromosome[i]  # Intercambia

    return chromosome

def inversion_mutation(chromosome, mutation_rate, proportion_mutation):
    """
    Realiza mutación por inversión en un cromosoma de permutación.
    
    Args:
        chromosome: Lista que representa una permutación.
        mutation_rate: Probabilidad de mutación.
    
    Returns:
        Mutated chromosome.
    """
    size = len(chromosome)
    num_mutations = max(1, (size * proportion_mutation)//100)  # Asegurar que afecte al menos a un segmento.

    for _ in range(num_mutations):
        if random.random() < mutation_rate:
            start, end = sorted(random.sample(range(size), 2))
            chromosome[start:end] = reversed(chromosome[start:end])  # Invierte el segmento

    return chromosome

def scramble_mutation(chromosome, mutation_rate, proportion_mutation):
    """
    Realiza mutación por mezcla en un cromosoma de permutación.
    
    Args:
        chromosome: Lista que representa una permutación.
        mutation_rate: Probabilidad de mutación.
    
    Returns:
        Mutated chromosome.
    """
    size = len(chromosome)
    num_mutations = max(1, (size * proportion_mutation)//100)  # Asegurar que afecte al menos a un segmento.

    for _ in range(num_mutations):
        if random.random() < mutation_rate:
            start, end = sorted(random.sample(range(size), 2))
            segment = chromosome[start:end]
            random.shuffle(segment)  # Mezcla el segmento
            chromosome[start:end] = segment

    return chromosome


### Eliminación

def reduce_population(Pop_eval, Pop_Size):
    """
    Reduce la población evaluada al tamaño máximo permitido (Pop_Size).
    
    Args:
        Pop_eval: Lista de tuplas [(cromosoma, fitness), ...].
        Pop_Size: Tamaño máximo de la población.

    Returns:
        Lista reducida de tuplas [(cromosoma, fitness), ...].
    """
    # Ordenar la población evaluada por fitness (de menor a mayor)
    sorted_population = sorted(Pop_eval, key=lambda x: x[1])

    # Seleccionar los mejores individuos hasta el tamaño permitido
    reduced_population = sorted_population[:Pop_Size]

    return reduced_population