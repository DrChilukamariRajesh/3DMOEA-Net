import numpy as np
import random as rn
from numpy.random import rand
from numpy.random import randint


# Population initialiation
def init_pop(pop_size, ch_len):
    return np.random.randint(0, 2, (pop_size, ch_len))
    
def init_obj(pop_size):
    return np.zeros((pop_size, 3))  #dice loss, number of parameters, number of FLOPs


# parent selection
def find_optimal_individual(i, j, scores, params, flops, z):
    if abs(scores[i]-scores[j]) > (z/10):
        return i if scores[i] < scores[j] else j
    elif abs(params[i]-params[j]) > z:
        return i if params[i] < params[j] else j
    elif abs(flops[i]-flops[j]) > z:
        return i if flops[i] < flops[j] else j
    else:
        return np.random.choice([i,j])
    
def parent_selection(pop, objective_values, z=2, k=3):
    selection_ix = randint(pop.shape[0])
    for ix in randint(0, pop.shape[0], k-1):
        selection_ix = find_optimal_individual(ix, selection_ix, objective_values[:,0], objective_values[:,1], objective_values[:,2], z)
    # print(selection_ix)
    return pop[selection_ix]


# crossover operation
def crossover(p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = randint(1, p1.shape[0]-2)
        # perform crossover
        c1 = np.append(p1[:pt], p2[pt:])
        c2 = np.append(p2[:pt], p1[pt:])
    return [c1, c2]
 
    
# mutation operation
def mutation(bitstring, r_mut):
    for i in range(bitstring.shape[0]):
        # check for a mutation
        if rand() < r_mut:
            # flip the bit
            bitstring[i] = 1 - bitstring[i]
    return bitstring



# Crowding distance calculation
def crowding_calculation(objective_values):
     
    pop_size = len(objective_values[:, 0])
    objective_value_number = len(objective_values[0, :])
    matrix_for_crowding = np.zeros((pop_size, objective_value_number))
    normalize_objective_values = (objective_values - objective_values.min(0))/objective_values.ptp(0) # normalizing objective values
    for i in range(objective_value_number):
        crowding_results = np.zeros(pop_size)
        crowding_results[0] = 1 #extreme point has the max crowding distance
        crowding_results[pop_size - 1] = 1 #extreme point has the max crowding distance
        sorting_normalize_objective_values = np.sort(normalize_objective_values[:, i])
        sorting_normalized_values_index = np.argsort(normalize_objective_values[:, i])
        #crowding distance calculation
        crowding_results[1:pop_size-1] = (sorting_normalize_objective_values[2:pop_size] - sorting_normalize_objective_values[i])
        re_sorting = np.argsort(sorting_normalized_values_index) # resorting to the original or
        matrix_for_crowding[:, i] = crowding_results[re_sorting]
 
    crowding_distance = np.sum(matrix_for_crowding, axis=1) # crowding distance of each solution
 
    return crowding_distance

def remove_using_crowding(objective_values, number_solutions_needed):
   
    pop_index = np.arange(objective_values.shape[0])
    crowding_distance = crowding_calculation(objective_values)  #calculating crowding distances
    selected_pop_index = np.zeros((number_solutions_needed))
    selected_objective_values = np.zeros((number_solutions_needed, len(objective_values[0, :])))
     
    for i in range(number_solutions_needed):
        pop_size = pop_index.shape[0]
        solution_1 = rn.randint(0, pop_size - 1)
        solution_2 = rn.randint(0, pop_size - 1)
        if crowding_distance[solution_1] >= crowding_distance[solution_2]:
            #solution1 is better than solution2
            selected_pop_index[i] = pop_index[solution_1]
            selected_objective_values[i, :] = objective_values[solution_1, :]
            pop_index = np.delete(pop_index, (solution_1), axis=0) # remove the solution 1
            objective_values = np.delete(objective_values, (solution_1), axis=0) # remove the finess of solution1
            crowding_distance = np.delete(crowding_distance, (solution_1), axis=0) # remove the crowding distance  of solution1
        else:
            #solution2 is better than solution1
            selected_pop_index[i] = pop_index[solution_2]
            selected_objective_values[i, :] = objective_values[solution_2, :]
            pop_index = np.delete(pop_index, (solution_2), axis=0) # remove the selected solution2
            objective_values = np.delete(objective_values, (solution_2), axis=0) # remove the finess of solution2
            crowding_distance = np.delete(crowding_distance, (solution_2), axis=0) # remove the crowding distance of solution2
            
    selected_pop_index = np.asarray(selected_pop_index, dtype=int) # Convert the data to integer      
    return (selected_pop_index)

# Pareto front
def pareto_front_finding(objective_values, pop_index):
 
    pop_size = objective_values.shape[0]
    pareto_front = np.ones(pop_size, dtype=bool) # initially assume all solutions are in pareto front by using "1"
    for i in range(pop_size):
        for j in range(pop_size):
                if all(objective_values[j] <= objective_values[i]) and any(objective_values[j] < objective_values[i]):  
                    pareto_front[i] = 0 # i is not in pareto front because j dominates i
                    break #no more comparison is needed to find out which one is dominated
                    
    return pop_index[pareto_front]      # non-dominated solutions index nums (minimum solu.s)

# Selection operation
def selection(pop, objective_vals, pop_size):
        
    pop_index_0 = np.arange(pop.shape[0])
    pop_index = np.arange(pop.shape[0])
    objective_values = objective_vals[:,:2]
    pareto_front_index = []
     
    while len(pareto_front_index) < pop_size:
        new_pareto_front = pareto_front_finding(objective_values[pop_index_0, :], pop_index_0)  #gives non-dominated sol.s
        total_pareto_size = len(pareto_front_index) + len(new_pareto_front)
        #check the pareto front, if larger than pop_size, remove some solutions using 
        if total_pareto_size > pop_size:
            number_solutions_needed = pop_size - len(pareto_front_index)
            selected_solutions = (remove_using_crowding(objective_values[new_pareto_front], number_solutions_needed))
            new_pareto_front = new_pareto_front[selected_solutions]
            
        pareto_front_index = np.hstack((pareto_front_index, new_pareto_front)) # add to pareto front
        remaining_index = set(pop_index) - set(pareto_front_index)
        pop_index_0 = np.array(list(remaining_index))
        
    new_pop = pop[pareto_front_index.astype(int)]
    new_objectives = objective_vals[pareto_front_index.astype(int)]
    
    return new_pop, new_objectives
