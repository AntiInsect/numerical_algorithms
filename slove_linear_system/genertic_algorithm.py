import random
from tqdm import tqdm
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np


def initialization(variables, populations, bound):
    return [random.sample(range(1, bound), variables) for i in range(populations)]

def selection(chromosomes, fitness, populations):
    get_pop = lambda pop1, pop2: pop1 if fitness[pop1] >= fitness[pop2] else pop2
    new_population = [get_pop(*random.sample(range(1,populations+1), 2)) for i in range(populations)]
    return [chromosomes[i-1] for i in new_population]

def crossover(chromosomes, variables, populations):
    R = np.random.uniform(0, 1, populations)
    # crossover rate = 0.7
    cross_pairs = list(combinations([idx for idx, val in enumerate(R) if val < 0.7], 2))
    get_crossover = lambda i, crossover_point: list(chromosomes[i[0]][:crossover_point]) + \
                                               list(chromosomes[i[1]][crossover_point:])
    for i in cross_pairs:
        chromosomes[i[0]] = get_crossover(i, random.randint(1, variables + 1))
    return chromosomes

def mutation(chromosomes, variables, populations, bound):
    # mutation_rate = 0.1
    number_of_mutations = int(variables * populations * 0.1)
    mutations_points = np.random.randint(1, variables * populations - 1, number_of_mutations)
    mutations_values = np.random.randint(1, bound, number_of_mutations)

    for i in range(number_of_mutations):
        row = mutations_points[i] // variables
        col = mutations_points[i] % variables - 1
        chromosomes[row][col] = mutations_values[i]
    return chromosomes

def calculate_fitnesses(chromosomes, obj_fn):
    fitness = []
    fitness_dict = dict()
    for idx, i in enumerate(chromosomes):
        obj = obj_fn(*i)
        if obj == 0:
            return i, True
        fitness_score = 1 / (obj)
        fitness_dict[idx+1] = fitness_score
    return fitness_dict, False


if __name__ == '__main__':
    random.seed(666)
    obj_fn = lambda a,b,c,d,e: a+(2*b)+(3*c)+(4*d)+(5*e) - 60
    variables, populations, bound = 5, 10, 60

    fitness_history = []
    obj_fn_history = []

    chromosomes = initialization(variables, populations, bound)

    iters = 0
    while True:
        iters += 1
        
        fitness, complete = calculate_fitnesses(chromosomes, obj_fn)

        fitness_history.append(1 if complete else max(fitness.values()))
        obj_fn_history.append(0 if complete else min(obj_fn(*i) for i in chromosomes) )
        
        if complete:
            solution = fitness
            break

        selections = selection(chromosomes, fitness, populations)
        cross = crossover(selections, variables, populations)
        chromosomes = mutation(cross, variables, populations, bound)

    print(f"Total Generations: {iters}")
    print(f"Fitness Value: {fitness_history[-1]}")
    print(f"Objective Function Value: {obj_fn_history[-1]}\n")
    print(f"Solution: {solution}")

    plt.figure(figsize=(8, 9))
    plt.subplot(1, 2, 1)
    plt.title('Fitness values')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.plot(range(1, len(fitness_history)+1), fitness_history, color='blue')
    
    plt.subplot(1, 2, 2)
    plt.title('Objective values')
    plt.xlabel('Generations')
    plt.ylabel('Objective Value')
    plt.plot(range(1, len(obj_fn_history)+1), obj_fn_history, color='green')
    plt.show()

