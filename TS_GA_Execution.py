########## Libraries ##########
import sys
import os
import json

########## Globals ##########
"""
    Path_Instances 
        Path of TSP instances in your
    max_calls_obj_func (global variable)
        Minimum of calls for end parametrization.
    obj_func_calls (global variable)
        Counter of every time that
        the objective function is called.
"""
Path_Instances = "Instances/Experimental"
Path_Params = 'Results/Parametrization/best_GA_params.txt'
Path_OPT = "Optimals/Experimental/Optimals.txt"
output_directory = 'Results/Experimentals'


########## Own files ##########
# Path from the workspace.
sys.path.append(os.path.join(os.path.dirname(__file__), 'Libraries'))
from ReadTSP import ReadTsp # type: ignore
from ReadTSP import ReadTSP_optTour # type: ignore
from TabuSearch import ObjFun  # type: ignore
from TabuSearch import TabuSearch  # type: ignore
from TabuSearch import TabuSearch_Con  # type: ignore
from GeneticAlgorithm import genetic_algorithm # type: ignore

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

def load_best_params(file_path):
    """
    load_best_params (function)
        Input: File path to the best parameters.
        Output: Dictionary with best parameters.
        Description: Reads the best parameters from a JSON or text file.
    """
    with open(file_path, 'r') as file:
        params = json.load(file)
    return params

def write_results(file_path, results):
    """
    write_results (function)
        Input: File path to save the results and a list of results.
        Output: None.
        Description: Writes the results and corresponding errors to a text file.
    """
    with open(file_path, 'w') as file:
        for result in results:
            # Descomponer el resultado en el valor de la función objetivo y el error
            obj_value, error = result
            file.write(f"Objective Value: {obj_value}, Error: {error}\n")

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

# Params.
best_params = load_best_params(Path_Params)
results_file_path = output_directory + '/genetic_algorithm_results_318_4000000.txt'

# Using best parameters to obtain solutions.
n = len(Instances)
results = []
#for Instance, opt_value in zip(Instances, Opt_Instances):
for i in range(11):
        best_pop, result = genetic_algorithm(Instances[1], len(Instances[1]), 
                     pop_size=best_params["POP_SIZE"],
                     MaxOFcalls = 4000000,
                     T_size=best_params["T_SIZE"])
        
        # Calcular el valor de la función objetivo para la solución obtenida
        #obj_value = ObjFun(result, Instances[0])

        # Calcular el error respecto al valor óptimo
        #error = (obj_value - Opt_Instances[0]) / Opt_Instances[0]
        error = (result[1] - Opt_Instances[1]) / Opt_Instances[1]
        
        # Guardar el resultado y el error
        #results.append((obj_value, error))
        results.append((result[1], error))
        
        # Imprimir el valor de la función objetivo para la solución obtenida.
        #print(f"Objective Value: {obj_value}, Error: {error}")
        print(f"Objective Value: {result[1]}, Error: {error}")

# Escribir los resultados en un archivo
write_results(results_file_path, results)