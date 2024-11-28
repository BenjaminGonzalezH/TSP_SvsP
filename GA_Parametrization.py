########## Libraries ##########
import sys
import numpy as np
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
best_GA_params_file = 'best_GAc_PBX_scramble_params.txt'
trials_GA_file = 'trials_GA_PBX_scr.csv'

########## Own files ##########
# Path from the workspace.
sys.path.append(os.path.join(os.path.dirname(__file__), 'Libraries'))
from ReadTSP import ReadTsp # type: ignore
from ReadTSP import ReadTSP_optTour # type: ignore
from GeneticAlgorithm_classic import GAc_PMX_swap # type: ignore
from GeneticAlgorithm_classic import GAc_OX_invertion # type: ignore
from GeneticAlgorithm_classic import GAc_PBX_scramble # type: ignore

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

def Parametrization_GA(trial, Instances, Opt_Instances):
    """
    Parametrization_TS (Function)
        Input: trial (parameters for evaluation) and
        list of distance matrices (one per instance).
        Output: Best trial of parameters.
        Description: Perform the parameterization procedure with
        every trial.
    """
    # Define parameter intervals using Optuna's suggest methods.
    pop_size = trial.suggest_int('POP_SIZE', 10, 150)
    T_size = trial.suggest_int('T_SIZE', 2, 5)
    crossover_rate = trial.suggest_int('C_RATE', 70, 95)
    mutation_rate = trial.suggest_int('M_RATE', 1, 5)

    # Initialize total normalized error.
    total_normalized_error = 0
    num_instances = len(Instances)

    for i in range(num_instances):
        # Run Tabu Search with the parameters from Optuna
        _ , population = GAc_PBX_scramble(pop_size, Instances[i], len(Instances[i]),
                 80000, T_size, crossover_rate, mutation_rate)
        
        # Evaluate the solution quality and calculate normalized error
        nor_error_list = []
        for solution in population[-1]:
            normalized_error = abs(solution[1] - Opt_Instances[i]) / Opt_Instances[i]
            nor_error_list.append(normalized_error)
        median_error = np.median(nor_error_list)

        # Sum the normalized error
        total_normalized_error += median_error

    # Return the average normalized error across all instances
    return total_normalized_error / num_instances

def Parametrization_GA_capsule(Instances, Opt_Instances):
    """
    Parametrization_capsule (Function)
        Encapsulates other inputs from the main
        parameterization function for optimization purposes.
    """
    return lambda trial: Parametrization_GA(trial, Instances, Opt_Instances)

def save_GA_study_txt(study, output_dir, best_params_file, trials_file):
    """
        save_TS_study_txt (function)
            Input: 
                - study: Optuna study object for Tabu Search.
                - output_dir: Directory where the files will be saved.
                - best_params_file: Filename for the best parameters.
                - trials_file: Filename for the trials data.
            Description:
                Saves the best parameters and trials of a Tabu Search Optuna study
                in the specified directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    best_params_path = os.path.join(output_dir, best_params_file)
    trials_path = os.path.join(output_dir, trials_file)

    # Save the best parameters in a readable text file (JSON format)
    with open(best_params_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)

    # Save the trials data in a CSV file
    with open(trials_path, 'w', newline='') as csvfile:
        fieldnames = ['trial_number', 'value', 'params']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for trial in study.trials:
            writer.writerow({
                'trial_number': trial.number,
                'value': trial.value,
                'params': trial.params
            })


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

# Parametrization genetic algorithm.
study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(),  # Using TPE as the sampler
    pruner=optuna.pruners.MedianPruner()   # Using Median Pruner for early stopping
)
study.optimize(Parametrization_GA_capsule(Instances, Opt_Instances), n_trials=11, n_jobs=-1)
best_params = study.best_params
print('Best parameters:', best_params)
save_GA_study_txt(study, output_directory ,best_GA_params_file, trials_GA_file)