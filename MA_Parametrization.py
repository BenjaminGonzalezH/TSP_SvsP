########## Libraries ##########
import sys
import os
import optuna
import json
import csv
import time

########## Globals ##########
"""
    Path_Instances 
        Path of TSP instances inside of
        the cumputer.
    Paath_OPT
        Path of Instances's optimal solutions in
        Path_Instances.
    output_directory
        Path where the results of the parametrization
        is going to be.
"""
Path_Instances = "Instances/Parametrizacion"
Path_OPT = "Optimals/Parametrizacion/Optimals.txt"
output_directory = 'Results/Parametrization'
best_MA_params_file = 'best_MA_params.txt'
trials_MA_file = 'trials_MA.csv'

########## Own files ##########
# Path from the workspace.
sys.path.append(os.path.join(os.path.dirname(__file__), 'Libraries'))
from ReadTSP import ReadTsp # type: ignore
from ReadTSP import ReadTSP_optTour # type: ignore
from TabuSearch import ObjFun  # type: ignore
from MemeticAlgorithm import memetic_algorithm # type: ignore

########## Secundary functions ##########
def Read_Content(filenames_Ins, filenames_Opt):
    """
        Read_Content (function)
            Input: File names of TSP instances and
            optimal tour associated to TSP instances.
            Output: Distance Matrix and optimal solution
            permutation vector.
            Description: Read content.
    """
    # Instances and OPT_tour.
    Instances = []

    # Reading files.
    for file in filenames_Ins:
        Instances.append(ReadTsp(file))

    OPT_Instances = ReadTSP_optTour(filenames_Opt)

    return Instances, OPT_Instances

########## Procedure ##########
# Obtain TSP's instances.
Content_Instances = os.listdir(Path_Instances)
files_Instances = []
for file in Content_Instances:
    if(os.path.isfile(os.path.join(Path_Instances,file))):
        files_Instances.append(Path_Instances+"/"+file)

# Obtain TSP Instances and optimal tour
# corresponding to each one.
Instances, Opt_Instances = Read_Content(files_Instances, Path_OPT)

# Using best parameters to obtain solutions.
n = len(Instances)
results = []

start_time = time.time()
#for Instance, opt_value in zip(Instances, Opt_Instances):
# Llamar a GLS (o TabuSearch) utilizando los mejores parámetros cargados.
result = memetic_algorithm(Instances[1], len(Instances[1]), pop_size=50, MaxOFcalls=8000)
        
# Calcular el valor de la función objetivo para la solución obtenida
obj_value = ObjFun(result, Instances[1])

# Calcular el error respecto al valor óptimo
error = (obj_value - Opt_Instances[1]) / Opt_Instances[1]
        
# Guardar el resultado y el error
results.append((obj_value, error))

end_time = time.time()
        
# Imprimir el valor de la función objetivo para la solución obtenida.
print(f"Objective Value: {obj_value}, Error: {error}")
execution_time = end_time - start_time
print(f"Tiempo de ejecución: {execution_time} segundos")