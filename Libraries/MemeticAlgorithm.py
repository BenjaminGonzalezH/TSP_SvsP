########## Libraries ##########
import numpy as np
import sys
import os

########## Own files ##########
# Path from the workspace.
sys.path.append(os.path.join(os.path.dirname(__file__), 'Libraries'))
from TabuSearch import (
    ObjFun,
    first_solution,
    two_opt_swap,
    get_neighbors_2opt
)

########## Functions MA ##########

# Función para inicializar la población
def initialize_population(pop_size, AmountNodes):

def tournament_selection(population, DistanceMatrix, tournament_size=3):

# Función de recombinación o cruce
def crossover(parent1, parent2):

# Función de mutación
def mutation(solution, DistanceMatrix):

# Búsqueda local con contador de evaluaciones
def local_search(solution, DistanceMatrix, calls_counter):

# Función principal del algoritmo memético con control de llamadas a la función objetivo
def memetic_algorithm(DistanceMatrix, AmountNodes, pop_size=50, MaxOFcalls=8000):
