########## Libraries ##########
import sys
import os
import time

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
Path_Instances = "Instances/Parametrizacion"
Path_Params = 'Results/Parameters/best_TS_params.txt'
Path_OPT = "Optimals/Parametrizacion/Optimals.txt"
output_directory = 'Results/Experimentals'

########## Own files ##########
# Path from the workspace.
sys.path.append(os.path.join(os.path.dirname(__file__), 'Libraries'))
from ReadTSP import ReadTsp # type: ignore
from ReadTSP import ReadTSP_optTour # type: ignore
from TabuSearch import ObjFun  # type: ignore
from TabuSearch import TabuSearch  # type: ignore

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
result = TabuSearch(Instances[4], len(Instances[4]), MaxIterations=10, TabuSize=100,
               minErrorInten=0.001)
        
# Calcular el valor de la función objetivo para la solución obtenida
obj_value = ObjFun(result, Instances[4])

# Calcular el error respecto al valor óptimo
error = (obj_value - Opt_Instances[4]) / Opt_Instances[4]
        
# Guardar el resultado y el error
results.append((obj_value, error))

end_time = time.time()
        
# Imprimir el valor de la función objetivo para la solución obtenida.
print(f"Objective Value: {obj_value}, Error: {error}")
execution_time = end_time - start_time
print(f"Tiempo de ejecución: {execution_time} segundos")