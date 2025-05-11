############################## IMPORT #######################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from queue import PriorityQueue
import seaborn as sns

############################## INPUTS #########################################
population_size = 10
reproduction_rate = 0.3  # lambda
probability_to_improve = 0.99
alpha = 0.2
initial_lifetime_gen_0 = 50
generations = {} # dictionary that take track of the individuals per generation
                 # key: number of the generation, value: list of individuals of that generation

# list of values for each parameter used for evaluating how the system changes varying them
initial_pop_sizes = np.random.randint(10, 50, 60)
rates = np.random.uniform(0.4, 1.2, 60)
# rate values chosen in order to be equally comparable to the probability to improvement ones
prob_improves = np.random.uniform(0.1, 0.9, 60)
alphas = np.random.uniform(0.1, 0.9, 15)
initial_lifetimes = np.random.randint(10, 50, 15)

# construction of the dataframes: the first dataframe consider the first adopted approach in which the reproduction
# rate is chosen in order to be global (i.e., all the individuals have the same reproduction rate; the second dataframe)
# consider the second approach in which the reproduction rate is singular and randomly chosen for each individual
columns = ['initial population size', 'reproduction rate', 'probability to improve', 'alpha', 'LF for gen 0',
           'total #individuals', 'time to extinction', 'avg #children', 'avg LF', 'avg # individuals per gen']
main_df = pd.DataFrame(columns=columns)
ind_columns = ['initial population size', 'probability to improve', 'alpha', 'LF for gen 0',
               'total #individuals', 'time to extinction', 'avg #children', 'avg LF', 'avg # individuals per gen']
ind_main_df = pd.DataFrame(columns=ind_columns)
seed = 42
random.seed(seed)
#################################### CLASS INDIVIDUAL #########################
class Individual:
    def __init__(self, idx, generation):
        self.idx = idx
        self.generation = generation
        self.reproduction_rate = 0.0
        self.lifetime = 0
        self.children = []
        self.parent = None

    def __str__(self):
        return f'{(self.idx, self.generation)}'


########################### INITIALIZATION ###############################
def initialize_population(population_size,initial_lifetime_gen_0):
    population_alive = {}
    count = 0
    generations[count] = []
    for num in range(population_size):
        ind = Individual(num, 0)
        ind.lifetime = initial_lifetime_gen_0
        population_alive[ind] = []
        generations[count].append(ind)  # the first generation is fulfilled at the initialization time

    return population_alive # -> return a dictionary of the type: Individual: [list of children]

######################## METHODS TRIGGERED BY THE FES #####################
def death(time, fes, population, queue):
    # schedule death event: the individual is simply taken out from the queue, so it is not processed anymore
    # the event is triggered at the end of the lifetime of the individual
    if queue:
        individual = queue.pop(0)
        # print(f'queue.pop(0) = {individual}')


def born(time, fes, population, reproduction_rate, probability_to_improve, queue, alpha):
    individuals = [ind for ind in population.keys()]
    if individuals:  # if there is at least one individual alive (i.e., not dead)
        selected_individual = random.choice(individuals) # select randomly one individual
        selected_individual.reproduction_rate = reproduction_rate # set the GLOBAL reproduction rate
        fes.put((selected_individual.lifetime, 'death')) # schedule its death
        # let the selected individual create children
        new_gen = selected_individual.generation + 1 # set the number of the new generation of children adding + 1 to the generation of the parent
        if new_gen not in generations.keys():  # if this generation wasn't present until now (i.e., it is the first individual of this new generation)
            generations[new_gen] = [] # add the number of the generation to the dictionary of the generations
        parent_lifetime = selected_individual.lifetime # retrieve the lifetime of the parent (i.e., the selected individual)
        if len(generations[new_gen]) != 0:  # if there are other individual of the same generation
            index = max([ind.idx for ind in list(generations[new_gen])]) + 1
            # take the highest index (the last born individual) of the individuals born in that generation
        else:  # instead, if it is the first of its generation
            index = 0 # se its index to 0
        child = Individual(index, new_gen) # create the child as instance of teh class Individual
        child.parent = selected_individual # set the parent of the child
        generations[new_gen].append(child) # add the child to its generation in the dictionary of the generations
        improve = np.random.choice([True, False], p=[probability_to_improve, 1 - probability_to_improve])
        # choose randomly if there would be an improvement in the new child or not
        if improve == True: # set the child's lifetime
            child.lifetime = random.uniform(parent_lifetime, (parent_lifetime * (1 + alpha)))
        else:  # if not improve
            child.lifetime = random.uniform(0, parent_lifetime)
        if selected_individual.children:  # if the individual has children
            if child not in selected_individual.children: # if it is not already present
                selected_individual.children.append(child) # add the new child
                population[selected_individual].append(child) # add the child to the dictionary of the population
                population[child] = []
        else:  # if the individual does not have children, add its first child
            selected_individual.children.append(child)
            population[selected_individual].append(child)
            population[child] = []

        if selected_individual.lifetime > time: # if the selected individual (i.e., the parent) is already alive
            born_time = np.random.poisson(reproduction_rate) # schedule the birth of its next child
            fes.put((time + born_time, 'born'))
        else: # otherwise, if the lifetime is expired
            queue.append(selected_individual) # add the individual to the queue and schedule its death event
            fes.put((selected_individual.lifetime, 'death'))

# the following method is exactly the same as before, but in this second approach the reproduction rate is individual
# and not global, which means that each individual has an individual rate associated to them, which is chosen among
# a list of rates
# the comments here are missing to avoid redundances
def born_individual_rate(time, fes, population, probability_to_improve, queue, alpha):
    individuals = [ind for ind in population.keys()]
    if individuals:
        selected_individual = random.choice(individuals)
        repr_rate = random.choice([0.1, 0.3, 0.6]) # here the set of the INDIVIDUAL reproduction rate
        selected_individual.reproduction_rate = repr_rate
        fes.put((selected_individual.lifetime, 'death'))
        new_gen = selected_individual.generation + 1
        if new_gen not in generations.keys():
            generations[new_gen] = []
        parent_lifetime = selected_individual.lifetime
        if len(generations[new_gen]) != 0:
            index = max([ind.idx for ind in list(generations[new_gen])]) + 1
        else:
            index = 0
        child = Individual(index, new_gen)
        child.parent = selected_individual
        generations[new_gen].append(child)
        improve = np.random.choice([True, False], p=[probability_to_improve, 1 - probability_to_improve])
        if improve == True:
            child.lifetime = random.uniform(parent_lifetime, (parent_lifetime * (1 + alpha)))
        else:  # not improve
            child.lifetime = random.uniform(0, parent_lifetime)
        if selected_individual.children:
            if child not in selected_individual.children:
                selected_individual.children.append(child)
                population[selected_individual].append(child)
                population[child] = []
        else:
            selected_individual.children.append(child)
            population[selected_individual].append(child)
            population[child] = []

        if selected_individual.lifetime > time:
            born_time = np.random.poisson(selected_individual.reproduction_rate)
            fes.put((time + born_time, 'born'))
        else:
            queue.append(selected_individual)
            fes.put((selected_individual.lifetime, 'death'))

################# METHODS TO PRINT THE STRUCTURE OF THE POPULATION ##########
def print_recursive(entity, indentation=""):
    print(f"{indentation}{entity}")

    if entity.children:
        for child in entity.children:
            print_recursive(child, indentation + "\t")


def print_population(population):
    for key, liste in population.items():
        if key.generation == 0:
            print(key)
            for individual in liste:
                print_recursive(individual, "\t")

################################ SIMULATION METHOD ###########################
# FIRST APPROACH:
def simulate(population_size, alpha, reproduction_rate, initial_lifetime_gen_0, probability_to_improve):
    # the population is initialized
    population = initialize_population(population_size, initial_lifetime_gen_0)
    # the FES is initialized
    time = 0
    queue = []
    fes = PriorityQueue()
    # the first event of birth is scheduled
    fes.put((0, 'born'))

    while not fes.empty():
        time, event_type = fes.get()
        # print((time, event_type))
        if event_type == 'born':
            born(time, fes, population, reproduction_rate, probability_to_improve, queue, alpha)
            if len(population) == 0: # if no individual alive
                break  # stop simulation
        elif event_type == 'death':
            death(time, fes, population, queue)
    # print(f'simulation ended in time {time}')
    # print(f'TOT INDIVIDUALS: len(population) = {len(population)}')
    # print(f'queue_size = {len(queue)}')

    # empty lists for the storing of the values
    lengths_ind_gen = []
    children_per_ind_gen = []
    lifetime_per_ind_gen = []
    avg_children_per_gen = []
    avg_lifetime_per_gen = []

    children_per_individual = []
    lifetime_per_individual = []

    # fulfilling of the lists -> the meaning is more understandable when creating the new record
    for gen, individuals in generations.items():
        # print(gen)
        lengths_ind_gen.append(len(individuals)) # for each generation, compute the number of individuals of this generation
        for individ in individuals:
            # print(f'\t{individ}\t\t{len(individ.children)}\t\t{individ.lifetime}')
            children_per_ind_gen.append(len(individ.children)) # for each individual of this generation, compute the number of its children
            # print(f'')
            lifetime_per_ind_gen.append(individ.lifetime) # for each individual of this generation, compute its lifetime
        avg_children_per_gen.append(np.mean(children_per_ind_gen)) # take the mean number of children considering all the individuals of a single generation
        avg_lifetime_per_gen.append(np.mean(lifetime_per_ind_gen)) # take the mean lifetime considering all the individuals of a single generation
    # print(f'mean # children considering all the generations = {np.mean(avg_children_per_gen)}')
    # print(f'mean lifetime considering all the generations = {np.mean(avg_lifetime_per_gen)}')
    # print(f'mean # individuals considering all the generations = {np.mean(lengths_ind_gen)}')
    # print('\n')

    new_record = {
        'initial population size': population_size,
        'reproduction rate': reproduction_rate,
        'probability to improve': probability_to_improve,
        'alpha': alpha,
        'LF for gen 0': initial_lifetime_gen_0,
        'total #individuals': len(population),
        'time to extinction': time,
        'avg #children': np.mean(avg_children_per_gen), # mean number of children considering all the generations
        'avg LF': np.mean(avg_lifetime_per_gen), # mean lifetime considering all the generations
        'avg # individuals per gen': np.mean(lengths_ind_gen) # mean number of individuals considering all the generations
    }
    main_df.loc[len(main_df)] = new_record # insert the record in the main dataframe
    return main_df # return the dataframe

# SECOND APPORACH:
# the following method is exactly the same as before but the reproduction rate is not treated as fixed (i.e., GLOBAL)
# parameter: it is, instead, characteristic of the individual and chosen randomly and differently for each individual
# the comments are missing to avoid redundance
def simulate_with_individual_rate(population_size, alpha, initial_lifetime_gen_0, probability_to_improve):
    population = initialize_population(population_size, initial_lifetime_gen_0)
    time = 0
    queue = []
    fes = PriorityQueue()
    fes.put((0, 'born'))

    while not fes.empty():
        time, event_type = fes.get()
        # print((time, event_type))
        if event_type == 'born':
            born_individual_rate(time, fes, population, probability_to_improve, queue, alpha)
            if len(population) == 0:
                break
        elif event_type == 'death':
            death(time, fes, population, queue)
    # print(f'simulation ended in time {time}')
    # print(f'TOT INDIVIDUALS: len(population) = {len(population)}')
    # print(f'queue_size = {len(queue)}')
    lengths_ind_gen = []
    children_per_ind_gen = []
    lifetime_per_ind_gen = []
    avg_children_per_gen = []
    avg_lifetime_per_gen = []

    children_per_individual = []
    lifetime_per_individual = []

    for gen, individuals in generations.items():
        # print(gen)
        lengths_ind_gen.append(len(individuals))
        for individ in individuals:
            # print(f'\t{individ}\t{len(individ.children)}\t{individ.reproduction_rate}')
            children_per_ind_gen.append(len(individ.children))
            lifetime_per_ind_gen.append(individ.lifetime)
        avg_children_per_gen.append(np.mean(children_per_ind_gen))
        avg_lifetime_per_gen.append(np.mean(lifetime_per_ind_gen))
    # print(f'mean # children considering all the generations = {np.mean(avg_children_per_gen)}')
    # print(f'mean lifetime considering all the generations = {np.mean(avg_lifetime_per_gen)}')
    # print(f'mean # individuals per generation = {np.mean(lengths_ind_gen)}')
    # print('\n')
    for indi in population.keys():
        children_per_individual.append(len(indi.children))
        lifetime_per_individual.append(indi.lifetime)
    # print(f'avg # children per this population = {np.mean(children_per_individual)}') # to be reviewed
    # print(f'avg lifetime per this population = {np.mean(lifetime_per_individual)}')

    new_record = {
        'initial population size': population_size,
        'probability to improve': probability_to_improve,
        'alpha': alpha,
        'LF for gen 0': initial_lifetime_gen_0,
        'total #individuals': len(population),
        'time to extinction': time,
        'avg #children': np.mean(avg_children_per_gen),
        'avg LF': np.mean(avg_lifetime_per_gen),
        'avg # individuals per gen': np.mean(lengths_ind_gen)
    }
    ind_main_df.loc[len(ind_main_df)] = new_record
    return ind_main_df

############################## TUNING OF THE PARAMETERS ###################
for rreproduction_rate in rates:
    repr_df = simulate(10, 0.2, rreproduction_rate, 50, 0.99)

for pprobability_to_improve in prob_improves:
    impr_df = simulate(10, 0.2, 0.3, 50, pprobability_to_improve)

for ppopulation_size in initial_pop_sizes:
    pop_df = simulate(ppopulation_size, 0.2, 0.3, 50, 0.99)

for ind_pprobability_to_improve in prob_improves:
    ind_impr_df = simulate_with_individual_rate(10, 0.2, 50, ind_pprobability_to_improve)

for ind_ppopulation_size in initial_pop_sizes:
    ind_pop_df = simulate_with_individual_rate(ind_ppopulation_size, 0.2, 50, 0.99)

###################àà########### OUTPUTS ########################################
plt.figure(figsize=(23, 35))
# 1: Correlation: Extinction Time vs. Reproduction Rate
filtered_data = repr_df[repr_df['reproduction rate'] != 0.3]
# let's skip all the line associated with rate = 0.3 because we will see this case of reproduction rate fixed to 0.3 later on
sorted_data = filtered_data.sort_values(by='reproduction rate')
extinction_time = sorted_data['time to extinction']
reproduction_rates = sorted_data['reproduction rate']
plt.subplot(7, 1, 1)
sns.set(style="whitegrid")
sns.regplot(x='reproduction rate', y='time to extinction',
            data=sorted_data[['reproduction rate', 'time to extinction']])
plt.title('Correlation: Extinction Time vs. Reproduction Rate')
plt.xlabel('Reproduction Rate')
plt.ylabel('Extinction Time')
plt.yticks(np.linspace(min(extinction_time), max(extinction_time), 10))
plt.xticks(np.linspace(min(reproduction_rates), max(reproduction_rates), 10))

# 2. Correlation: Avg Lifetime Time vs. Reproduction Rate
avg_lf = sorted_data['avg LF']
plt.subplot(7, 1, 2)
sns.set(style="whitegrid")
sns.regplot(x='reproduction rate', y='avg LF', data=sorted_data[['reproduction rate', 'avg LF']])
plt.title('Correlation: Avg Lifetime Time vs. Reproduction Rate')
plt.xlabel('Reproduction Rate')
plt.ylabel('Avg Lifetime')
plt.yticks(np.linspace(min(avg_lf), max(avg_lf), 10))
plt.xticks(np.linspace(min(reproduction_rates), max(reproduction_rates), 10))

# 3. Correlation: Extinction Time vs. Probability To Improve
# FIRST APPROACH: GLOBAL RATE
filter_df = impr_df[impr_df['probability to improve'] != 0.99]
# let's skip all the line associated with probability = 0.99 because we will see this case of improvement probability fixed to 0.99 later on
sort_df = filter_df.sort_values(by='probability to improve')
probability_to_improves = sort_df['probability to improve']
i_extinction_time = sort_df['time to extinction']
plt.subplot(7, 1, 3)
sns.set(style="whitegrid")
sns.regplot(x='probability to improve', y='time to extinction',
            data=sort_df[['probability to improve', 'time to extinction']], label='With global reproduction rate')
plt.title('Correlation: Extinction Time vs. Probability To Improve')
plt.xlabel('Probability To Improve')
plt.ylabel('Extinction Time')
plt.yticks(np.linspace(min(i_extinction_time), max(i_extinction_time), 10))
plt.xticks(np.linspace(min(probability_to_improves), max(probability_to_improves), 10))

# SECOND APPROACH: INDIVIDUAL RATE
ind_filter_df = ind_impr_df[ind_impr_df['probability to improve'] != 0.99]
ind_sort_df = ind_filter_df.sort_values(by='probability to improve')
ind_probability_to_improves = ind_sort_df['probability to improve']
ind_extinction_time = ind_sort_df['time to extinction']
sns.set(style="whitegrid")
sns.regplot(x='probability to improve', y='time to extinction',
            data=ind_sort_df[['probability to improve', 'time to extinction']],
            label='With individual reproduction rate')
plt.legend()

# 4. Correlation: Avg Lifetime Time vs. Probability To Improve
# FIRST APPROACH: GLOBAL RATE
avg_lf = sort_df['avg LF']
plt.subplot(7, 1, 4)
sns.set(style="whitegrid")
sns.regplot(x='probability to improve', y='avg LF', data=sort_df[['probability to improve', 'avg LF']],
            label='With global reproduction rate')
plt.title('Correlation: Avg Lifetime Time vs. Probability To Improve')
plt.xlabel('Probability To Improve')
plt.ylabel('Avg Lifetime')
plt.yticks(np.linspace(min(avg_lf), max(avg_lf), 10))
plt.xticks(np.linspace(min(probability_to_improves), max(probability_to_improves), 10))

# SECOND APPROACH: INDIVIDUAL RATE
ind_avg_lf = ind_sort_df['avg LF']
sns.set(style="whitegrid")
sns.regplot(x='probability to improve', y='avg LF', data=ind_sort_df[['probability to improve', 'avg LF']],
            label='With individual reproduction rate')
plt.legend()

# 5. Correlation: Extinction Time vs. Initial Population Size
# FIRST APPROACH: GLOBAL RATE
filter_also_this = main_df[main_df['initial population size'] != 10]
# let's skip all the lines associated with size = 10 because we will see this case of initial population size fixed to 10 later on
sorted_also_this = filter_also_this.sort_values(by='initial population size')
initial_pop = sorted_also_this['initial population size']
p_extinction_time = sorted_also_this['time to extinction']
plt.subplot(7, 1, 5)
sns.set(style="whitegrid")
sns.regplot(x='initial population size', y='time to extinction',
            data=sorted_also_this[['initial population size', 'time to extinction']],
            label='With global reproduction rate')
plt.title('Correlation: Extinction Time vs. Initial Population Size')
plt.xlabel('Initial Population Size')
plt.ylabel('Extinction Time')
plt.xticks(np.linspace(min(initial_pop), max(initial_pop), 10))
plt.yticks(np.linspace(min(p_extinction_time), max(p_extinction_time), 10))

# SECOND APPROACH: INDIVIDUAL RATE
ind_filter_also_this = ind_main_df[ind_main_df['initial population size'] != 10]
ind_sorted_also_this = ind_filter_also_this.sort_values(by='initial population size')
ind_initial_pop = ind_sorted_also_this['initial population size']
ind_p_extinction_time = ind_sorted_also_this['time to extinction']
sns.set(style="whitegrid")
sns.regplot(x='initial population size', y='time to extinction',
            data=ind_sorted_also_this[['initial population size', 'time to extinction']],
            label='With individual reproduction rate')
plt.legend()

# 6. Correlation: Extinction Time vs. Avg # of Children
# FIRST APPROACH: GLOBAL RATE
sorted_main_df = main_df.sort_values(by='avg #children')
a_extinction_time = sorted_main_df['time to extinction']
avg_children = sorted_main_df['avg #children']
plt.subplot(7, 1, 6)
sns.set(style="whitegrid")
sns.regplot(x='avg #children', y='time to extinction', data=sorted_main_df[['avg #children', 'time to extinction']],
            label='With global reproduction rate')
plt.title('Correlation: Extinction Time vs. Avg # of Children')
plt.xlabel('Avg # of Children')
plt.ylabel('Extinction Time')
plt.yticks(np.linspace(min(a_extinction_time), max(p_extinction_time), 10))
plt.xticks(np.linspace(min(avg_children), max(avg_children), 10))

# SECOND APPROACH: INDIVIDUAL RATE
ind_sorted_main_df = ind_main_df.sort_values(by='avg #children')
ind_a_extinction_time = ind_sorted_main_df['time to extinction']
ind_avg_children = ind_sorted_main_df['avg #children']
sns.set(style="whitegrid")
sns.regplot(x='avg #children', y='time to extinction', data=ind_sorted_main_df[['avg #children', 'time to extinction']],
            label='With individual reproduction rate')
# plt.xlim(min(ind_avg_children), max(ind_avg_children))
plt.legend()

# 7. Correlation: Time To Extinction vs. Total # Individuals
# FIRST APPROACH: GLOBAL RATE
plt.subplot(7, 1, 7)
sns.set(style="whitegrid")
tot_ind = sorted_main_df['total #individuals']
extinctio_time = sorted_main_df['time to extinction']
sns.regplot(x='total #individuals', y='time to extinction',
            data=sorted_main_df[['total #individuals', 'time to extinction']], label='With global reproduction rate')
plt.title('Correlation: Extinction Time vs. Total # Individuals')
plt.ylabel('Extinction Time')
plt.xlabel('Total # Individuals')
plt.yticks(np.linspace(min(extinctio_time), max(extinctio_time), 10))
plt.xticks(np.linspace(min(tot_ind), max(tot_ind), 10))

# SECOND APPROACH: INDIVIDUAL RATE

sns.set(style="whitegrid")
ind_tot_ind = ind_sorted_main_df['total #individuals']
ind_extinctio_time = ind_sorted_main_df['time to extinction']
sns.regplot(x='total #individuals', y='time to extinction',
            data=ind_sorted_main_df[['total #individuals', 'time to extinction']],
            label='With individual reproduction rate')
plt.legend()

plt.tight_layout()
plt.show()

# 8. Correlation matrices
# note: alpha and LF for gen 0 are not considered because fixed
# FIRST APPROACH: GLOBAL RATE
selected_columns = sorted_main_df[['initial population size', 'reproduction rate',
                                   'probability to improve', 'total #individuals',
                                   'time to extinction', 'avg #children', 'avg LF',
                                   'avg # individuals per gen']]
correlation_matrix = selected_columns.corr()
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap with global reproduction rate')

# SECOND APPROACH: INDIVIDUAL RATE
ind_selected_columns = ind_sorted_main_df[['initial population size',
                                           'probability to improve', 'total #individuals',
                                           'time to extinction', 'avg #children', 'avg LF',
                                           'avg # individuals per gen']]
ind_correlation_matrix = ind_selected_columns.corr()
plt.subplot(1, 2, 2)
sns.heatmap(ind_correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

plt.tight_layout()
plt.title('Correlation Heatmap with individual reproduction rate')
plt.show()