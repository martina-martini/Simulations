############################## IMPORT #######################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from queue import PriorityQueue
import seaborn as sns
from scipy.stats import t

############################## INPUTS #########################################
INITIAL_POPULATION_SIZE = 10
PROBABILITY_TO_IMPROVE = 0.5
ALPHA = 0.2

STANDARD_REPR_RATE = 1/365
WINNER_SURVIVAL_RATIO = 0.9
LOOSER_SURVIVAL_RATIO = 0.2
WARRIORS_RATIO = 0.7
HOURS_IN_A_DAY = 24
ADDITIONAL_LF_RATIO = 0.2
ADDITIONAL_SURVIVAL_LF = 0.1

W_INITIAL_LF = 15*365
E_INITIAL_LF = 20*365
D_INITIAL_LF = 6*365
F_INITIAL_LF = 4*365
M_INITIAL_LF = 5*365

W_STRONG_RATIO = 0.9
E_STRONG_RATIO = 0.8
D_STRONG_RATIO = 0.5
F_STRONG_RATIO = 0.4
M_STRONG_RATIO = 0.1

W_SPEED = 56
E_SPEED = 120
D_SPEED = 51
F_SPEED = 51
M_SPEED = 13

W_FEMALE_RATIO = 0.5
E_FEMALE_RATIO = 0.47
D_FEMALE_RATIO = 0.47
F_FEMALE_RATIO = 0.46
M_FEMALE_RATIO = 0.46

WOOD_AREA = 250
NUM_AREAS = 5

CONFIDENCE_LEVEL = .80
ACC_ACCEPTANCE = .85
batches = 4
acc_sizes = 0.0
acc_ch = 0.0

class Individual:
    def __init__(self, idx, generation, specie):
        self.idx = idx
        self.generation = generation
        self.reproduction_rate = 0.0
        self.lifetime = 0
        self.children = []
        self.parent = None
        self.weak = False
        self.strong = False
        self.is_fighting = False
        self.specie = specie
        self.area = None
        self.death_time = 0.0
        self.gender = None

    def __str__(self):
        return f'{self.specie.name} {(self.idx, self.generation)}'

class Area:
    def __init__(self, aidx, individuals_in_group, length):
        self.aidx = aidx
        self.max_ind_per_area = 0
        self.specie_in_there = None
        self.length = length
        self.individuals_in_there = 0

    def __str__(self):
        return f'Area {self.aidx} (max ind allowed = {self.max_ind_per_area})'

class Specie:
    def __init__(self, sidx, serie):
        self.sidx = sidx
        self.serie = serie
        self.name = None
        self.area = None
        self.reproduction_rate = 0.0
        self.in_fight = False
        self.individuals=[]
        self.speed = 0.0
        self.female_ratio = 0.0

    def __str__(self):
        if self.serie == 3:
          abc = 'A'
        elif self.serie == 2:
          abc = 'B'
        elif self.serie == 1:
          abc = 'C'
        return f'{self.name} (specie {self.sidx}) in {self.area}, serie {abc}'


########################### INITIALIZATION ###############################
def initialize_population(idx_specie, serie, name, ecosystem, repr_rate, strong_ind_rate, initial_lf, speed, female_ratio):
    population_alive = {}
    generations = {}
    count = 0
    generations[count] = []
    s = Specie(idx_specie, serie)
    s.name = name
    s.speed = speed
    s.reproduction_rate = repr_rate
    s.area = ecosystem
    s.female_ratio = female_ratio
    ecosystem.specie_in_there = s
    for num in range(INITIAL_POPULATION_SIZE):
        ind = Individual(num, 0, s)
        ind.lifetime = initial_lf
        ind.area = s.area
        female = np.random.choice([True, False], p=[female_ratio, 1 - female_ratio])
        if female == True:
          ind.gender = 'f'
        else:
          ind.gender = 'm'
        strong = np.random.choice([True, False], p=[strong_ind_rate, 1 - strong_ind_rate])
        if strong == True:
          ind.strong = True
        else:
          ind.weak = True
        ecosystem.individuals_in_there += 1
        s.individuals.append(ind)
        population_alive[ind] = []
        generations[count].append(ind)  # the first generation is fulfilled at the initialization time
    return s, generations, population_alive # -> return a dictionary of the type: Individual: [list of children]

def initialize_ecosystem(area_idx, individuals_in_group, length):
    ecosystem = Area(area_idx, individuals_in_group, length/NUM_AREAS)
    ecosystem.max_ind_per_area = round(length / NUM_AREAS * individuals_in_group)
    return ecosystem
# imagine a mean wood of 250 km squared -> taking a system in 1 dimension I have 250 \ 5 areas = 50 km per area

def death(time, fes, queue, all_population):
    if queue:
      individual = queue.pop(0)
      # print(f'{individual} dies')
      s = individual.specie
      a = s.area
      # if individual in a.individuals_in_there:
      #   a.individuals_in_there.remove(individual)
      a.individuals_in_there -= 1
      if individual in s.individuals:
        s.individuals.remove(individual)
      if individual in all_population.keys():
        del all_population[individual]


def born(alpha, probability_to_improve, time, fes, queue, all_species, all_gens, all_population, attack_warriors, defend_warriors, pop_to_print, areas):
    individuals = [ind for ind in all_population.keys() if ind.is_fighting == False and ind.gender == 'f']
    if len(individuals) != 0:
      selected_individual = random.choice(individuals)
      # print(selected_individual)
      the_specie = selected_individual.specie
      female_ratio = the_specie.female_ratio
      if selected_individual not in queue:
        queue.append(selected_individual)
        fes.put((selected_individual.lifetime, 'death'))
        selected_individual.death_time = selected_individual.lifetime
      new_gen = selected_individual.generation + 1
      if new_gen not in all_gens[the_specie].keys():
        all_gens[the_specie][new_gen] = []
      if len(all_gens[the_specie][new_gen]) != 0:
        index = max([ind.idx for ind in list(all_gens[the_specie][new_gen])]) + 1
      else:
        index = 0
      env = the_specie.area
      if env.individuals_in_there <= env.max_ind_per_area:
        parent = selected_individual
        child = Individual(index, new_gen, parent.specie)
        child.parent = parent
        female = np.random.choice([True, False], p=[female_ratio, 1 - female_ratio])
        if female == True:
          child.gender = 'f'
        else:
          child.gender = 'm'
        child.reproduction_rate = the_specie.reproduction_rate
        if parent.strong == True:
          child.strong = True
        elif parent.weak == True:
          child.weak = True
        improve = np.random.choice([True, False], p=[probability_to_improve, 1 - probability_to_improve])
        if improve == True:
          child.lifetime = random.uniform(parent.lifetime, (parent.lifetime + (1 + alpha)))
        else:
          child.lifetime = random.uniform(0, parent.lifetime)
        if parent.children:
          if child not in parent.children:
            parent.children.append(child)
            all_population[parent].append(child)
            all_population[child] = []
            pop_to_print[parent].append(child)
            pop_to_print[child] = []
        else:
          parent.children.append(child)
          all_population[parent].append(child)
          pop_to_print[parent].append(child)
        if child not in queue:
          queue.append(child)
          fes.put((time + child.lifetime, 'death'))
          child.death_time = time + child.lifetime
        all_gens[the_specie][new_gen].append(child)
        the_specie.individuals.append(child)
        env.individuals_in_there += 1

        if parent.lifetime > time:
          born_time = np.random.poisson(parent.reproduction_rate)
          fes.put((time + born_time, 'born'))
        else:
          if parent not in queue:
            queue.append(parent)
            fes.put((parent.lifetime, 'death'))
            parent.death_time = parent.lifetime
      else: # overpopulation
        # print(f'{the_specie.name} OVERPOPULATED -> FIGHT since in {env} there are {env.individuals_in_there} individuals')
        # print(f'at time {time} the {the_specie.name} starts moving')
        destination = random.choice([a for a in areas if a != the_specie.area])
        specie_to_fight = destination.specie_in_there
        # print(f'{the_specie.name} vs {specie_to_fight.name}')
        mobility_time = (destination.length * abs(destination.aidx - the_specie.area.aidx)) / the_specie.speed / HOURS_IN_A_DAY
        warriors_specie = random.sample(the_specie.individuals, int(len(the_specie.individuals) * WARRIORS_RATIO))
        warriors_specie_to_fight = random.sample(specie_to_fight.individuals, int(len(specie_to_fight.individuals) * WARRIORS_RATIO))
        attack_warriors.extend(warriors_specie)
        defend_warriors.extend(warriors_specie_to_fight)
        # print(f'{len(attack_warriors)} attack vs {len(defend_warriors)} defend')
        fes.put((time + mobility_time, 'fight'))
        # print(f'at time {time + mobility_time} they arrive and start fighting')

def fight(fes, time, queue, attack_warriors, defend_warriors): # no fighting time
    fighting_time = 7 #gg
    for ind_a in attack_warriors:
      ind_a.is_fighting = True
      attack_warriors[0].specie.area.individuals_in_there -= 1
    for ind_d in defend_warriors:
      ind_d.is_fighting = True
    if attack_warriors and defend_warriors:
      if attack_warriors[0].specie.serie > defend_warriors[0].specie.serie: # if the specie who attacks have serie > the defend warriors' serie
        # attack warriors win
        # print(f'attack_warriors {attack_warriors[0].specie.name} wins')
        winner_survivals = random.sample([a for a in attack_warriors if a.strong == True], int(len([a for a in attack_warriors if a.strong == True]) * WINNER_SURVIVAL_RATIO))
        defend_warriors[0].specie.area.individuals_in_there += len([w for w in winner_survivals])
        # add the new individual in the area, if it is not already there
        winner_dead = [warr for warr in attack_warriors if warr not in winner_survivals]
        looser_survivals = random.sample([d for d in defend_warriors if d.strong == True], int(len([d for d in defend_warriors if d.strong == True]) * LOOSER_SURVIVAL_RATIO))
        looser_dead = [warr for warr in defend_warriors if warr not in looser_survivals]
        # end fight
        for survival in winner_survivals:
            survival.is_fighting = False # can come back to be considered for reproducing themselves
            survival.area = defend_warriors[0].area  # the attack survival change area
            survival.lifetime += ADDITIONAL_SURVIVAL_LF * survival.lifetime
        for survival2 in looser_survivals:
            survival2.is_fighting = False
            survival2.lifetime -= ADDITIONAL_SURVIVAL_LF * survival2.lifetime
        for dead in winner_dead:
            queue.append(dead)
            fes.put((time, 'death'))
        for dead2 in looser_dead:
            queue.append(dead2)
            fes.put((time, 'death'))
      else: # if attack_warriors[0].specie.serie < defend_warriors[0].specie.serie
        # defend warriors win
        # print(f'defend_warriors {defend_warriors[0].specie.name} wins')
        winner_survivals = random.sample([d for d in defend_warriors if d.strong == True], int(len([d for d in defend_warriors if d.strong == True]) * WINNER_SURVIVAL_RATIO))
        winner_dead = [warr for warr in defend_warriors if warr not in winner_survivals]
        looser_survivals = random.sample([a for a in attack_warriors if a.strong == True], int(len([a for a in attack_warriors if a.strong == True]) * LOOSER_SURVIVAL_RATIO))
        defend_warriors[0].specie.area.individuals_in_there += len([l for l in looser_survivals ])
        # this control is done because maybe an eagle survives in the wolf area, then wolfes are attacked by foxes
        # -> the survived eagle can be potentially taken as warrior so we must check what animals are in the area
        looser_dead = [warr for warr in attack_warriors if warr not in looser_survivals]
        # end fight
        for survival in winner_survivals:
            survival.is_fighting = False # can come back to be considered for reproducing themselves
            survival.lifetime += ADDITIONAL_LF_RATIO * survival.lifetime
        for survival2 in looser_survivals:
            survival2.is_fighting = False
            survival2.area = defend_warriors[0].area  # the attack survival change area
            survival2.lifetime -= ADDITIONAL_LF_RATIO * survival2.lifetime

        for dead in winner_dead:
            queue.append(dead)
            fes.put((time, 'death'))
        for dead2 in looser_dead:
            queue.append(dead2)
            fes.put((time, 'death'))
      attack_warriors.clear() # ready for the next battle
      defend_warriors.clear()
      # print(f'-- at {time + fighting_time} the fight ended --')
    fes.put((time + fighting_time, 'born'))



def simulate(alpha, probability_to_improve, wood_area):
    pop_size_per_specie = {}
    avg_children_per_specie = {}
    avg_LF_per_specie = {}
    avg_female_LF_per_specie = {}
    avg_male_LF_per_specie = {}
    times = []
    global_pop_per_time = []
    wolf_pop_per_time = []
    eagle_pop_per_time = []
    deer_pop_per_time = []
    fox_pop_per_time = []
    mouse_pop_per_time = []
    fights_time = []
    ind_in_area_1 = []
    ind_in_area_2 = []
    ind_in_area_3 = []
    ind_in_area_4 = []
    ind_in_area_5 = []

    seed = 142
    random.seed(seed)
    eco_wolf = initialize_ecosystem(area_idx=1, individuals_in_group=6, length=wood_area) # 1 branco di 6/km
    eco_eagle = initialize_ecosystem(area_idx=2, individuals_in_group=3, length=wood_area) # 1 coppia + 1 baby/km
    eco_deer = initialize_ecosystem(area_idx=3, individuals_in_group=7, length=wood_area) # 1 branco di 7/km
    eco_fox = initialize_ecosystem(area_idx=4, individuals_in_group=4, length=wood_area) # 2 coppie/km
    eco_mouse = initialize_ecosystem(area_idx=5, individuals_in_group=8, length=wood_area) # 8 individui/km

    wolf_specie, wolf_gen, wolf_pop = initialize_population(idx_specie=1, serie=3, name='Wolfes', ecosystem=eco_wolf, repr_rate=STANDARD_REPR_RATE, strong_ind_rate=W_STRONG_RATIO, initial_lf=W_INITIAL_LF, speed=W_SPEED, female_ratio=W_FEMALE_RATIO)
    eagle_specie, eagle_gen, eagle_pop = initialize_population(idx_specie=2, serie=3, name='Eagles', ecosystem=eco_eagle, repr_rate=STANDARD_REPR_RATE, strong_ind_rate=E_STRONG_RATIO, initial_lf=E_INITIAL_LF, speed=E_SPEED, female_ratio=E_FEMALE_RATIO)
    deer_specie, deer_gen, deer_pop = initialize_population(idx_specie=3, serie=2, name='Deers', ecosystem=eco_deer, repr_rate=STANDARD_REPR_RATE, strong_ind_rate=D_STRONG_RATIO, initial_lf=D_INITIAL_LF, speed=D_SPEED, female_ratio=D_FEMALE_RATIO)
    fox_specie, fox_gen, fox_pop = initialize_population(idx_specie=4, serie=2, name='Foxes', ecosystem=eco_fox, repr_rate=STANDARD_REPR_RATE, strong_ind_rate=F_STRONG_RATIO, initial_lf=F_INITIAL_LF, speed=F_SPEED, female_ratio=F_FEMALE_RATIO)
    mouses_specie, mouses_gen, mouse_pop = initialize_population(idx_specie=5, serie=1, name='Mice', ecosystem=eco_mouse, repr_rate=STANDARD_REPR_RATE, strong_ind_rate=M_STRONG_RATIO, initial_lf=M_INITIAL_LF, speed=M_SPEED, female_ratio=M_FEMALE_RATIO)

    areas = [eco_wolf, eco_eagle, eco_deer, eco_fox, eco_mouse]

    all_species = [wolf_specie, eagle_specie, deer_specie, fox_specie, mouses_specie]
    all_gens = {wolf_specie: wolf_gen, eagle_specie: eagle_gen, deer_specie: deer_gen, fox_specie: fox_gen, mouses_specie: mouses_gen}
    all_population = {**wolf_pop, **eagle_pop, **deer_pop, **fox_pop, **mouse_pop}
    pop_to_print = {**wolf_pop, **eagle_pop, **deer_pop, **fox_pop, **mouse_pop}

    time = 0
    queue = []
    attack_warriors = []
    defend_warriors = []
    fes = PriorityQueue()
    # the first event of birth is scheduled
    fes.put((0, 'born'))

    while not fes.empty() and all_population:
        time, event_type = fes.get()
        # print((time, event_type))
        if event_type == 'born':
            born(alpha, probability_to_improve, time, fes, queue, all_species, all_gens, all_population, attack_warriors, defend_warriors, pop_to_print, areas)
        elif event_type == 'death':
            death(time, fes, queue, all_population)
        elif event_type == 'fight':
            fight(fes, time, queue, attack_warriors, defend_warriors)
            fights_time.append(time)
        times.append(time)
        global_pop_per_time.append(len(all_population.keys()))
        wolf_pop_per_time.append(len([p for p in all_population.keys() if p.specie == all_species[0]]))
        eagle_pop_per_time.append(len([p for p in all_population.keys() if p.specie == all_species[1]]))
        deer_pop_per_time.append(len([p for p in all_population.keys() if p.specie == all_species[2]]))
        fox_pop_per_time.append(len([p for p in all_population.keys() if p.specie == all_species[3]]))
        mouse_pop_per_time.append(len([p for p in all_population.keys() if p.specie == all_species[4]]))
        if areas[0].individuals_in_there > 0:
          ind_in_area_1.append(areas[0].individuals_in_there)
        elif areas[0].individuals_in_there <= 0:
          ind_in_area_1.append(0)
        if areas[1].individuals_in_there > 0:
          ind_in_area_2.append(areas[1].individuals_in_there)
        elif areas[1].individuals_in_there <= 0:
          ind_in_area_2.append(0)
        if areas[2].individuals_in_there > 0:
          ind_in_area_3.append(areas[2].individuals_in_there)
        elif areas[2].individuals_in_there <= 0:
          ind_in_area_3.append(0)
        if areas[3].individuals_in_there > 0:
          ind_in_area_4.append(areas[3].individuals_in_there)
        elif areas[3].individuals_in_there <= 0:
          ind_in_area_4.append(0)
        if areas[4].individuals_in_there > 0:
          ind_in_area_5.append(areas[4].individuals_in_there)
        elif areas[4].individuals_in_there <= 0:
          ind_in_area_5.append(0)

    # print(f'simulation ended in time {time}\n')
    # print(f'TOT INDIVIDUALS: len(pop_to_print) = {len(pop_to_print)}')
    # print(f'\nqueue_size = {len(queue)}')

    flag = False
    for j in all_species:
      num = 0
      ch = []
      lf = []
      female_lf = []
      male_lf = []

      for k, v in pop_to_print.items():
          if k.specie == j:
            num += 1
            ch.append(len(v))
            lf.append(k.lifetime)
            if k.gender == 'f':
              female_lf.append(k.lifetime)
            else:
              male_lf.append(k.lifetime)
      pop_size_per_specie[j] = num
      avg_children_per_specie[j] = np.mean(ch)
      if female_lf:
        avg_female_LF_per_specie[j] = np.mean(female_lf)
      else:
        print(f'error female lf: not enough females for specie {j.name} with configuration: prob_to_improve={probability_to_improve}, wood_area={wood_area} -> discarded')
        flag = True
      avg_male_LF_per_specie[j] = np.mean(male_lf)
      avg_LF_per_specie[j] = np.mean(lf)
      ch.clear()
      lf.clear()

    return areas, times, global_pop_per_time, wolf_pop_per_time, eagle_pop_per_time, deer_pop_per_time, fox_pop_per_time, \
          fights_time, all_species, mouse_pop_per_time, pop_size_per_specie, avg_children_per_specie, avg_LF_per_specie, \
          areas, ind_in_area_1, ind_in_area_2, ind_in_area_3, ind_in_area_4, ind_in_area_5, \
          avg_female_LF_per_specie, avg_male_LF_per_specie, flag


def calculate_confidence_interval(data, conf):
    data = list(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    se = std / (len(data)**(1/2)) # this is the standard error

    interval = t.interval(confidence = conf, # confidence level
                          df = max(len(data) - 1, 1), # degree of freedom
                          loc = mean, # center of the distribution
                          scale = se # spread of the distribution
                          # (we use the standard error as we use the extimate of the mean)
                          )

    MOE = interval[1] - interval[0] # this is the margin of error
    re = (MOE / (2 * abs(mean))) # this is the relative error
    acc = 1 - re # this is the accuracy
    return interval, acc

while acc_sizes < ACC_ACCEPTANCE or acc_ch < ACC_ACCEPTANCE:
  # for computing the confidence intervals, I take different batches to get the higher accuracy with a fixed confidence level (0.97)
  for batch in range(batches):
    # print(f'Batch {batch}:')
    areas, times, global_pop_per_time, wolf_pop_per_time, eagle_pop_per_time, deer_pop_per_time, fox_pop_per_time, \
    fights_time, all_species, mouse_pop_per_time, pop_size_per_specie, avg_children_per_specie, avg_LF_per_specie, \
    areas, ind_in_area_1, ind_in_area_2, ind_in_area_3, ind_in_area_4, ind_in_area_5, \
    avg_female_LF_per_specie, avg_male_LF_per_specie, flag = simulate(ALPHA, PROBABILITY_TO_IMPROVE, WOOD_AREA)

    if flag == False:
      interval_1, acc_sizes = calculate_confidence_interval(pop_size_per_specie.values(), CONFIDENCE_LEVEL)
      interval_2, acc_ch = calculate_confidence_interval(avg_children_per_specie.values(), CONFIDENCE_LEVEL)
      interval_3, acc_lf = calculate_confidence_interval(avg_LF_per_specie.values(), CONFIDENCE_LEVEL)
    # print(f'\t\t\t\t\t FINE BATCH {batch}')

specie_data = {
    'Specie': ['Wolf', 'Eagle', 'Deer', 'Fox', 'Mouse'],
    'Initial LifeTime (days)': [W_INITIAL_LF, E_INITIAL_LF, D_INITIAL_LF, F_INITIAL_LF, M_INITIAL_LF],
    'Strong Ratio': [W_STRONG_RATIO, E_STRONG_RATIO, D_STRONG_RATIO, F_STRONG_RATIO, M_STRONG_RATIO],
    'Speed (km/h)': [W_SPEED, E_SPEED, D_SPEED, F_SPEED, M_SPEED],
    'Female Ratio': [W_FEMALE_RATIO, E_FEMALE_RATIO, D_FEMALE_RATIO, F_FEMALE_RATIO, M_FEMALE_RATIO]
}
species_df = pd.DataFrame(specie_data)
print(species_df)

for i in areas:
  print(i, i.specie_in_there)
print(f'\nCOMPUTATION OF CI FINISHED WITH ACC_SIZES = {acc_sizes} AND ACC_CH = {acc_ch} with configuration: prob_to_improve={PROBABILITY_TO_IMPROVE}, wood_area={WOOD_AREA}\n')

# with this configuration of the system: see how it change varying the parameters:
prob_to_improves = np.linspace(0.1, 0.9, 10)
woodareas = np.linspace(100, 1_000, 10)

df_comparable = pd.DataFrame(columns=['P(improve)', 'Wood km', 'time to extinction', 'Fights to survive', 'Mean LF (all species)', 'Mean # Children (all species)', \
                                      'Mean LF (Wolfes)',  'Mean # Children (Wolfes)', 'Mean LF (Eagles)',  'Mean # Children (Eagles)', \
                                      'Mean LF (Deers)',  'Mean # Children (Deers)', 'Mean LF (Foxes)',  'Mean # Children (Foxes)', 'Mean LF (Mice)',  'Mean # Children (Mice)'])

for proba_to_improve in prob_to_improves:
  for woodarea in woodareas:
    # print(f'configuration: prob_to_improve={proba_to_improve}, wood_area={woodarea}')
    areas, times, global_pop_per_time, wolf_pop_per_time, eagle_pop_per_time, deer_pop_per_time, fox_pop_per_time, \
    fights_time, all_species, mouse_pop_per_time, pop_size_per_specie, avg_children_per_specie, avg_LF_per_specie, \
    areas, ind_in_area_1, ind_in_area_2, ind_in_area_3, ind_in_area_4, ind_in_area_5, avg_female_LF_per_specie, \
    avg_male_LF_per_specie, flag = simulate(ALPHA, proba_to_improve, woodarea)
    if flag == False:
      new_record = {'P(improve)':proba_to_improve, 'Wood km':woodarea, 'time to extinction':times[len(times)-1], 'Fights to survive': len(fights_time), \
                    'Mean LF (all species)': np.mean(list(avg_LF_per_specie.values())), 'Mean # Children (all species)': np.mean(list(avg_children_per_specie.values())), \
                    'Mean LF (Wolfes)': avg_LF_per_specie[all_species[0]], 'Mean # Children (Wolfes)': avg_children_per_specie[all_species[0]], \
                    'Mean LF (Eagles)': avg_LF_per_specie[all_species[1]], 'Mean # Children (Eagles)': avg_children_per_specie[all_species[1]], \
                    'Mean LF (Deers)': avg_LF_per_specie[all_species[2]], 'Mean # Children (Deers)': avg_children_per_specie[all_species[2]], \
                    'Mean LF (Foxes)': avg_LF_per_specie[all_species[3]], 'Mean # Children (Foxes)': avg_children_per_specie[all_species[3]], \
                    'Mean LF (Mice)': avg_LF_per_specie[all_species[4]], 'Mean # Children (Mice)': avg_children_per_specie[all_species[4]]
                    }
      df_comparable.loc[len(df_comparable)] = new_record

best_config_df = df_comparable[df_comparable['time to extinction'] == max(df_comparable['time to extinction'])]
best_config = best_config_df[best_config_df['Fights to survive'] >= 1]
print(f'best configuration found! P(improve) = {best_config["P(improve)"].iloc[0]}, Wook km = {best_config["Wood km"].iloc[0]}\n')
areas, times, global_pop_per_time, wolf_pop_per_time, eagle_pop_per_time, deer_pop_per_time, fox_pop_per_time, \
fights_time, all_species, mouse_pop_per_time, pop_size_per_specie, avg_children_per_specie, avg_LF_per_specie, \
areas, ind_in_area_1, ind_in_area_2, ind_in_area_3, ind_in_area_4, ind_in_area_5, avg_female_LF_per_specie, \
avg_male_LF_per_specie, flag = simulate(ALPHA, best_config['P(improve)'].iloc[0], best_config['Wood km'].iloc[0])

#### Average Male/Female LifeTime vs. Specie\n(X = female, Y = male)
colors = ['red', 'goldenrod', 'green', 'blue', 'purple']
fig, ax2 = plt.subplots(figsize=(4, 6))
for i, specie in enumerate(avg_female_LF_per_specie.keys()):
    ax2.scatter(specie.name, avg_female_LF_per_specie[specie], c=colors[i], marker="x", s=80)
ax2.set_xlabel('Specie Name')
ax2.set_ylabel('Average Female LifeTime')
ax2.set_yticks(list(avg_female_LF_per_specie.values()))
ax2.grid(True)

ax2 = plt.twinx()
for i, specie in enumerate(avg_female_LF_per_specie.keys()):
    ax2.scatter(specie.name, avg_male_LF_per_specie[specie], c=colors[i], marker='1', s=80)
ax2.set_yticks(list(avg_male_LF_per_specie.values()))
ax2.set_ylabel('Average Male LifeTime')
plt.text(5.5, avg_male_LF_per_specie[all_species[0]], f'Initial Wolfes LF = {W_INITIAL_LF}', fontsize=11, color='red')
plt.text(5.5, avg_male_LF_per_specie[all_species[1]], f'Initial Eagles LF = {E_INITIAL_LF}', fontsize=11, color='goldenrod')
plt.text(5.5, avg_male_LF_per_specie[all_species[2]], f'Initial Deers LF = {D_INITIAL_LF}', fontsize=11, color='green')
plt.text(5.5, avg_male_LF_per_specie[all_species[3]], f'Initial Foxes LF = {F_INITIAL_LF}', fontsize=11, color='blue')
plt.text(5.5, abs(avg_male_LF_per_specie[all_species[4]] - avg_male_LF_per_specie[all_species[3]]) - 200, f'Initial Mice LF = {M_INITIAL_LF}', fontsize=11, color='purple')

plt.suptitle('Average Male/Female LifeTime vs. Specie\n(X = female, Y = male)')
plt.title(f"P(improve) = {best_config['P(improve)'].iloc[0]}, Wood Area = {best_config['Wood km'].iloc[0]}")
plt.grid(True)
plt.show()

#### subplot of 3 plots
#### Number of Individuals of the Population vs. Time
plt.figure(figsize=(18, 12))
plt.subplot(3, 1, 1)
plt.grid(True)
plt.title('Number of Individuals of the Population vs. Time')
plt.suptitle(f"best configuration found: P(improve) = {best_config['P(improve)'].iloc[0]}, wood area = {best_config['Wood km'].iloc[0]}")
plt.xlabel('Time')
plt.ylabel('Number of Individuals')
plt.plot(times, wolf_pop_per_time, label=all_species[0], c='red')
plt.plot(times, eagle_pop_per_time, label=all_species[1], c='goldenrod')
plt.plot(times, deer_pop_per_time, label=all_species[2], c='green')
plt.plot(times, fox_pop_per_time, label=all_species[3], c='blue')
plt.plot(times, mouse_pop_per_time, label=all_species[4], c='purple')
plt.xticks(np.linspace(min(times), max(times), 10))
plt.xlim(min(times) - 100, max(times))
plt.legend()

#### Number of Individuals in Area vs.Time
plt.subplot(3, 1, 2)
plt.grid(True)
plt.title('Number of Individuals in Area vs.Time')
plt.ylabel('Number of Individuals')
plt.xlabel('Time')
plt.plot(times, ind_in_area_1, label=areas[0], c='red')
plt.plot(times, ind_in_area_2, label=areas[1], c='goldenrod')
plt.plot(times, ind_in_area_3, label=areas[2], c='green')
plt.plot(times, ind_in_area_4, label=areas[3], c='blue')
plt.plot(times, ind_in_area_5, label=areas[4], c='purple')
plt.xlim(min(times) - 100, max(times))
plt.legend()

#### Number of Individuals in Area vs.Time (focus on fights)
if fights_time:
  plt.subplot(3, 1, 3)
  plt.title('Number of Individuals in Area vs.Time (focus on fights)')
  plt.ylabel('Number of Individuals')
  plt.xlabel('Time')
  plt.plot(times, ind_in_area_1, label=areas[0], c='red')
  plt.plot(times, ind_in_area_2, label=areas[1], c='goldenrod')
  plt.plot(times, ind_in_area_3, label=areas[2], c='green')
  plt.plot(times, ind_in_area_4, label=areas[3], c='blue')
  plt.plot(times, ind_in_area_5, label=areas[4], c='purple')
  plt.xticks(np.linspace(min(times), max(times), 10))
  for fight in fights_time:
      plt.axvline(fight, c='k', alpha=0.6, linestyle='--')
  plt.xticks(fights_time)
  plt.xlim(0, max(fights_time) + 1)
  colors = ['red', 'goldenrod', 'green', 'blue', 'purple']
  for i, area in enumerate(areas):
      plt.axhline(area.max_ind_per_area, c=colors[i], alpha=0.3)
  plt.legend()
plt.tight_layout()
plt.show()

#### heatmap
plt.figure(figsize=(6, 3))
df_comparablee = df_comparable[['P(improve)', 'Wood km', 'time to extinction',
       'Fights to survive','Mean LF (all species)', 'Mean # Children (all species)']]
ind_correlation_matrix = df_comparablee.corr()
sns.heatmap(ind_correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

#### subplot of 16 plots
#### Correlation: Extinction Time vs. Probability To Improve
plt.figure(figsize=(18, 26))
plt.subplot(10, 2, 1)
minw_sliced_df = df_comparable[df_comparable['Wood km'] == min(df_comparable['Wood km'])]
p_minw_sliceddf = minw_sliced_df.sort_values(by='P(improve)')
plt.plot(p_minw_sliceddf['P(improve)'], p_minw_sliceddf['time to extinction'],  label=f'Minimum wood area = {min(df_comparable["Wood km"])}')
sns.set(style="whitegrid")
sns.regplot(x='P(improve)', y='time to extinction', data=p_minw_sliceddf[['P(improve)', 'time to extinction']],  label='trend for Min wood area')
plt.xticks(np.linspace(min(df_comparable['P(improve)']), max(df_comparable['P(improve)']), 10))
plt.title('Correlation: Extinction Time vs. Probability To Improve (Small wood)')
plt.xlabel('P(improve)')
plt.ylabel('time to extinction')
plt.grid(True)
plt.legend()

plt.subplot(10, 2, 2)
maxw_sliced_df = df_comparable[df_comparable['Wood km'] == max(df_comparable['Wood km'])]
p_maxw_sliceddf = maxw_sliced_df.sort_values(by='P(improve)')
plt.plot(p_maxw_sliceddf['P(improve)'], p_maxw_sliceddf['time to extinction'], c='seagreen', label=f'Maximum wood area = {max(df_comparable["Wood km"])}')
sns.set(style="whitegrid")
sns.regplot(x='P(improve)', y='time to extinction', data=p_maxw_sliceddf[['P(improve)', 'time to extinction']], line_kws=dict(color='indianred'), scatter_kws=dict(color='indianred'), label='trend for Max wood area')
plt.title('Correlation: Extinction Time vs. Probability To Improve (Big wood)')
plt.xlabel('P(improve)')
plt.ylabel('time to extinction')
plt.xticks(np.linspace(min(df_comparable['P(improve)']), max(df_comparable['P(improve)']), 10))
plt.grid(True)
plt.legend()

#### Correlation: Extinction Time vs. Extension of Wood
plt.subplot(10, 2, 3)
plt.title('Correlation: Extinction Time vs. Extension of Wood (Low improvement)')
plt.xlabel('Wood km')
plt.ylabel('Extinction Time')
minp_sliced_df = df_comparable[df_comparable['P(improve)'] == min(df_comparable['P(improve)'])]
w_minp_sliceddf = minp_sliced_df.sort_values(by='Wood km')
plt.plot(w_minp_sliceddf['Wood km'], w_minp_sliceddf['time to extinction'],  label=f'Minimum P(improve) = {min(df_comparable["P(improve)"])}')
sns.set(style="whitegrid")
sns.regplot(x='Wood km', y='time to extinction', data=w_minp_sliceddf[['Wood km', 'time to extinction']],  label='trend for Min P(improve)')
plt.xticks(np.linspace(min(df_comparable['Wood km']), max(df_comparable['Wood km']), 10))
plt.grid(True)
plt.legend()

plt.subplot(10, 2, 4)
plt.title('Correlation: Extinction Time vs. Extension of Wood (High improvement)')
plt.xticks(np.linspace(min(df_comparable['Wood km']), max(df_comparable['Wood km']), 10))
maxp_sliced_df = df_comparable[df_comparable['P(improve)'] == max(df_comparable['P(improve)'])]
w_maxp_sliceddf = maxp_sliced_df.sort_values(by='Wood km')
plt.plot(w_maxp_sliceddf['Wood km'], w_maxp_sliceddf['time to extinction'],c='seagreen', label=f'Maximum P(improve) = {max(df_comparable["P(improve)"])}')
sns.set(style="whitegrid")
sns.regplot(x='Wood km', y='time to extinction', data=w_maxp_sliceddf[['Wood km', 'time to extinction']], line_kws=dict(color='indianred'), scatter_kws=dict(color='indianred'), label='trend for Max P(improve)')
plt.grid(True)
plt.legend()

#### Correlation: Extinction Time vs. Mean LifeTime considering all the species
plt.subplot(10, 2, 5)
plt.title('Correlation: Extinction Time vs. Mean LifeTime considering all the species (Small wood)')
plt.xlabel('Mean LF (all species)')
plt.ylabel('Extinction Time')
minmean_lf_sliced_df = df_comparable[df_comparable['Wood km'] == min(df_comparable['Wood km'])]
p_minmean_lf_sliceddf = minmean_lf_sliced_df.sort_values(by='Mean LF (all species)')
plt.plot(p_minmean_lf_sliceddf['Mean LF (all species)'], p_minmean_lf_sliceddf['time to extinction'], label=f'Minimum wood area = {min(df_comparable["Wood km"])}')
sns.set(style="whitegrid")
sns.regplot(x='Mean LF (all species)', y='time to extinction', data=p_minmean_lf_sliceddf[['Mean LF (all species)', 'time to extinction']], label='trend for Min wood area')
plt.yticks(np.linspace(min(p_minmean_lf_sliceddf['time to extinction'])-1000, max(p_minmean_lf_sliceddf['time to extinction']), 5))
plt.xticks(np.linspace(min(p_minmean_lf_sliceddf['Mean LF (all species)']), max(p_minmean_lf_sliceddf['Mean LF (all species)']), 10))
plt.legend()

plt.subplot(10, 2, 6)
plt.title('Correlation: Extinction Time vs. Mean LifeTime considering all the species (Big wood)')
maxmean_lf_sliced_df = df_comparable[df_comparable['Wood km'] == max(df_comparable['Wood km'])]
p__maxmean_lf_sliceddf = maxmean_lf_sliced_df.sort_values(by='Mean LF (all species)')
plt.plot(p__maxmean_lf_sliceddf['Mean LF (all species)'], p__maxmean_lf_sliceddf['time to extinction'], c='seagreen', label=f'Maximum wood area = {max(df_comparable["Wood km"])}')
sns.set(style="whitegrid")
sns.regplot(x='Mean LF (all species)', y='time to extinction', data=p__maxmean_lf_sliceddf[['Mean LF (all species)', 'time to extinction']], line_kws=dict(color='indianred'), scatter_kws=dict(color='indianred'), label='trend for Max wood area')
plt.yticks(np.linspace(min(p__maxmean_lf_sliceddf['time to extinction']), max(p__maxmean_lf_sliceddf['time to extinction']), 5))
plt.xticks(np.linspace(min(p__maxmean_lf_sliceddf['Mean LF (all species)']), max(p__maxmean_lf_sliceddf['Mean LF (all species)']), 10))
plt.grid(True)
plt.legend()
plt.tight_layout()

#### Correlation: Extinction Time vs. Mean # Children (all species) considering all the species
plt.subplot(10, 2, 7)
plt.title('Correlation: Extinction Time vs. Mean # Children considering all the species (Small wood)')
plt.xlabel('Mean # Children (all species)')
plt.ylabel('Extinction Time')
minmean_ch_sliced_df = df_comparable[df_comparable['Wood km'] == min(df_comparable['Wood km'])]
p_minmean_ch_sliceddf = minmean_ch_sliced_df.sort_values(by='Mean # Children (all species)')
plt.plot(p_minmean_ch_sliceddf['Mean # Children (all species)'], p_minmean_ch_sliceddf['time to extinction'], label=f'Minimum wood area = {min(df_comparable["Wood km"])}')
sns.set(style="whitegrid")
sns.regplot(x='Mean # Children (all species)', y='time to extinction', data=p_minmean_ch_sliceddf[['Mean # Children (all species)', 'time to extinction']], label='trend for Min wood area')
plt.yticks(np.linspace(min(p_minmean_ch_sliceddf['time to extinction'])-1000, max(p_minmean_ch_sliceddf['time to extinction']), 5))
plt.xticks(np.linspace(min(p_minmean_ch_sliceddf['Mean # Children (all species)']), max(p_minmean_ch_sliceddf['Mean # Children (all species)']), 10))
plt.legend()

plt.subplot(10, 2, 8)
plt.title('Correlation: Extinction Time vs. Mean # Children considering all the species (Big wood)')
maxmean_ch_sliced_df = df_comparable[df_comparable['Wood km'] == max(df_comparable['Wood km'])]
p_maxmean_ch_sliceddf = maxmean_ch_sliced_df.sort_values(by='Mean # Children (all species)')
plt.plot(p_maxmean_ch_sliceddf['Mean # Children (all species)'], p_maxmean_ch_sliceddf['time to extinction'], c='seagreen', label=f'Maximum wood area = {max(df_comparable["Wood km"])}')
plt.ylabel('Extinction Time')
sns.set(style="whitegrid")
sns.regplot(x='Mean # Children (all species)', y='time to extinction', data=p_maxmean_ch_sliceddf[['Mean # Children (all species)', 'time to extinction']], line_kws=dict(color='indianred'), scatter_kws=dict(color='indianred'), label='trend for Max wood area')
plt.xticks(np.linspace(min(p_maxmean_ch_sliceddf['Mean # Children (all species)']), max(p_maxmean_ch_sliceddf['Mean # Children (all species)']), 5))
plt.grid(True)
plt.legend()
plt.tight_layout()

#### Correlation: Mean # Children (all species) vs. P(improve)
plt.subplot(10, 2, 9)
plt.title('Correlation: Mean # Children (all species) vs. P(improve) (Small wood)')
plt.ylabel('Mean # Children (all species)')
meanch_pwmin_sliced_df = df_comparable[df_comparable['Wood km'] == min(df_comparable['Wood km'])]
meanch_p_wmin_sliceddf = meanch_pwmin_sliced_df.sort_values(by='P(improve)')
plt.plot(meanch_p_wmin_sliceddf['P(improve)'], meanch_p_wmin_sliceddf['Mean # Children (all species)'], label=f'Minimum Wood km = {min(df_comparable["Wood km"])}')
sns.set(style="whitegrid")
sns.regplot(x='P(improve)', y='Mean # Children (all species)', data=meanch_p_wmin_sliceddf[['P(improve)', 'Mean # Children (all species)']], label='trend for Min Wood km')
plt.yticks(np.linspace(min(meanch_p_wmin_sliceddf['Mean # Children (all species)']), max(meanch_p_wmin_sliceddf['Mean # Children (all species)']), 5))
plt.xticks(np.linspace(min(meanch_p_wmin_sliceddf['P(improve)']), max(meanch_p_wmin_sliceddf['P(improve)']), 10))
plt.grid(True)
plt.legend()

plt.subplot(10, 2, 10)
plt.title('Correlation: Mean # Children (all species) vs. P(improve) (Big wood)')
meanch_pwmax_sliced_df = df_comparable[df_comparable['Wood km'] == max(df_comparable['Wood km'])]
meanch_p_wmax_sliceddf = meanch_pwmax_sliced_df.sort_values(by='P(improve)')
plt.plot(meanch_p_wmax_sliceddf['P(improve)'], meanch_p_wmax_sliceddf['Mean # Children (all species)'],c='seagreen', label=f'Maximum Wood km = {max(df_comparable["Wood km"])}')
sns.set(style="whitegrid")
sns.regplot(x='P(improve)', y='Mean # Children (all species)', data=meanch_p_wmax_sliceddf[['P(improve)', 'Mean # Children (all species)']],line_kws=dict(color='indianred'), scatter_kws=dict(color='indianred'), label='trend for Max Wood km')
plt.yticks(np.linspace(min(meanch_p_wmax_sliceddf['Mean # Children (all species)']), max(meanch_p_wmax_sliceddf['Mean # Children (all species)']), 5))
plt.xticks(np.linspace(min(meanch_p_wmax_sliceddf['P(improve)']), max(meanch_p_wmax_sliceddf['P(improve)']), 10))
plt.legend()
plt.tight_layout()

########################################################
#### Correlation: Fights To Survive vs. Extension of Wood
plt.subplot(10, 2, 11)
plt.title('Correlation: Fights To Survive vs. Extension of Wood')
plt.xlabel('Fights to survive')
pmin_fight_sliced_df = df_comparable[df_comparable['P(improve)'] == min(df_comparable['P(improve)'])]
w_pmin_fight_sliceddf = pmin_fight_sliced_df.sort_values(by='Wood km')
plt.plot(w_pmin_fight_sliceddf['Wood km'], w_pmin_fight_sliceddf['Fights to survive'], label=f'Minimum P(improve) = {min(df_comparable["P(improve)"])}')
sns.set(style="whitegrid")
sns.regplot(x='Wood km', y='Fights to survive', data=w_pmin_fight_sliceddf[['Wood km', 'Fights to survive']], label='trend for Min P(improve)')
plt.xticks(np.linspace(min(df_comparable['Wood km']), max(df_comparable['Wood km']), 5))
plt.grid(True)

pmax_fight_sliced_df = df_comparable[df_comparable['P(improve)'] == max(df_comparable['P(improve)'])]
w_pmax_fight_sliceddf = pmax_fight_sliced_df.sort_values(by='Wood km')
plt.plot(w_pmax_fight_sliceddf['Wood km'], w_pmax_fight_sliceddf['Fights to survive'],c='seagreen', label=f'Maximum P(improve) = {max(df_comparable["P(improve)"])}')
sns.set(style="whitegrid")
sns.regplot(x='Wood km', y='Fights to survive', data=w_pmax_fight_sliceddf[['Wood km', 'Fights to survive']],line_kws=dict(color='indianred'), scatter_kws=dict(color='indianred'), label='trend for Max P(improve)')
plt.yticks(np.linspace(0, max(df_comparable['Fights to survive']), 5))
plt.legend()

#### Correlation: Mean LF (all species) vs. Probability To Improve
plt.subplot(10, 2, 12)
pminw_sliced_df = df_comparable[df_comparable['Wood km'] == min(df_comparable['Wood km'])]
meanlf_pminw_sliceddf = pminw_sliced_df.sort_values(by='P(improve)')
plt.plot(meanlf_pminw_sliceddf['P(improve)'], meanlf_pminw_sliceddf['Mean LF (all species)'], label=f'Minimum wood area = {min(df_comparable["Wood km"])}')
sns.set(style="whitegrid")
sns.regplot(x='P(improve)', y='Mean LF (all species)', data=meanlf_pminw_sliceddf[['P(improve)', 'Mean LF (all species)']], label='trend for Min wood area')

pmaxw_sliced_df = df_comparable[df_comparable['Wood km'] == max(df_comparable['Wood km'])]
meanlf_pmaxw_sliceddf = pmaxw_sliced_df.sort_values(by='P(improve)')
plt.plot(meanlf_pmaxw_sliceddf['P(improve)'], meanlf_pmaxw_sliceddf['Mean LF (all species)'], label=f'Maximum wood area = {max(df_comparable["Wood km"])}')
sns.set(style="whitegrid")
sns.regplot(x='P(improve)', y='Mean LF (all species)', data=meanlf_pmaxw_sliceddf[['P(improve)', 'Mean LF (all species)']], label='trend for Max wood area')

plt.title('Correlation: Mean LF considering all species vs. Probability To Improve')
plt.xlabel('P(improve)')
plt.ylabel('Mean LF (all species)')
plt.xticks(np.linspace(min(df_comparable['P(improve)']), max(df_comparable['P(improve)']), 10))
plt.grid(True)
plt.legend()

#### Correlation: Fights To Survive vs. Mean # Children considering all species
plt.subplot(10, 2, 13)
plt.title('Correlation: Fights To Survive vs. Mean # Children considering all species (Low improvement)')
plt.ylabel('Fights to survive')
meanch_pmin_sliced_df = df_comparable[df_comparable['P(improve)'] == min(df_comparable['P(improve)'])]
meanch_fights_pmin_sliceddf = meanch_pmin_sliced_df.sort_values(by='Mean # Children (all species)')
plt.plot(meanch_fights_pmin_sliceddf['Mean # Children (all species)'], meanch_fights_pmin_sliceddf['Fights to survive'], label=f'Minimum P(improve) = {min(df_comparable["P(improve)"])}')
sns.set(style="whitegrid")
sns.regplot(x='Mean # Children (all species)', y='Fights to survive', data=meanch_fights_pmin_sliceddf[['Mean # Children (all species)', 'Fights to survive']], label='trend for Min P(improve)')
plt.xticks(np.linspace(min(meanch_fights_pmin_sliceddf['Mean # Children (all species)']), max(meanch_fights_pmin_sliceddf['Mean # Children (all species)']), 10))
plt.yticks(list(range(max(meanch_fights_pmin_sliceddf['Fights to survive']) + 1)))
plt.grid(True)
plt.legend()

plt.subplot(10, 2, 14)
plt.title('Correlation: Fights To Survive vs. Mean # Children considering all species (High improvement)')
meanch_pmax_sliced_df = df_comparable[df_comparable['P(improve)'] == max(df_comparable['P(improve)'])]
meanch_fights_pmax_sliceddf = meanch_pmax_sliced_df.sort_values(by='Mean # Children (all species)')
plt.plot(meanch_fights_pmax_sliceddf['Mean # Children (all species)'], meanch_fights_pmax_sliceddf['Fights to survive'],c='seagreen', label=f'Maximum P(improve) = {max(df_comparable["P(improve)"])}')
sns.set(style="whitegrid")
sns.regplot(x='Mean # Children (all species)', y='Fights to survive', data=meanch_fights_pmax_sliceddf[['Mean # Children (all species)', 'Fights to survive']],line_kws=dict(color='indianred'), scatter_kws=dict(color='indianred'), label='trend for Max P(improve)')
plt.yticks(list(range(max(meanch_fights_pmax_sliceddf['Fights to survive']) + 1)))
plt.xticks(np.linspace(min(meanch_fights_pmax_sliceddf['Mean # Children (all species)']), max(meanch_fights_pmax_sliceddf['Mean # Children (all species)']), 10))
plt.legend()
plt.tight_layout()

#### Correlation: Fights To Survive vs. Mean LF considering all species
plt.subplot(10, 2, 15)
plt.title('Correlation: Fights To Survive vs. Mean LF considering all species (Low improvement)')
plt.ylabel('Fights to survive')
meanlf_pmin_sliced_df = df_comparable[df_comparable['P(improve)'] == min(df_comparable['P(improve)'])]
meanlf_fights_pmin_sliceddf = meanlf_pmin_sliced_df.sort_values(by='Mean LF (all species)')
plt.plot(meanlf_fights_pmin_sliceddf['Mean LF (all species)'], meanlf_fights_pmin_sliceddf['Fights to survive'], label=f'Minimum P(improve) = {min(df_comparable["P(improve)"])}')
sns.set(style="whitegrid")
sns.regplot(x='Mean LF (all species)', y='Fights to survive', data=meanlf_fights_pmin_sliceddf[['Mean LF (all species)', 'Fights to survive']], label='trend for Min P(improve)')
if max(meanlf_fights_pmin_sliceddf['Fights to survive']) > 1:
  plt.yticks(list(range(0, max(meanlf_fights_pmin_sliceddf['Fights to survive']) + 1, 2)))
else:
  plt.yticks(list(range(max(meanlf_fights_pmin_sliceddf['Fights to survive']) + 1)))
plt.xticks(np.linspace(min(meanlf_fights_pmin_sliceddf['Mean LF (all species)']), max(meanlf_fights_pmin_sliceddf['Mean LF (all species)']), 10))
plt.grid(True)
plt.legend()

plt.subplot(10, 2, 16)
plt.title('Correlation: Fights To Survive vs. Mean LF considering all species (High improvement)')
meanlf_pmax_sliced_df = df_comparable[df_comparable['P(improve)'] == max(df_comparable['P(improve)'])]
meanlf_fights_pmax_sliceddf = meanlf_pmax_sliced_df.sort_values(by='Mean LF (all species)')
plt.plot(meanlf_fights_pmax_sliceddf['Mean LF (all species)'], meanlf_fights_pmax_sliceddf['Fights to survive'],c='seagreen', label=f'Maximum P(improve) = {max(df_comparable["P(improve)"])}')
sns.set(style="whitegrid")
sns.regplot(x='Mean LF (all species)', y='Fights to survive', data=meanlf_fights_pmax_sliceddf[['Mean LF (all species)', 'Fights to survive']],line_kws=dict(color='indianred'), scatter_kws=dict(color='indianred'), label='trend for Max P(improve)')
if max(meanlf_fights_pmax_sliceddf['Fights to survive']) > 1:
  plt.yticks(list(range(0, max(meanlf_fights_pmax_sliceddf['Fights to survive']) + 1, 2)))
else:
  plt.yticks(list(range(max(meanlf_fights_pmax_sliceddf['Fights to survive']) + 1)))
plt.xticks(np.linspace(min(meanlf_fights_pmax_sliceddf['Mean LF (all species)']), max(meanlf_fights_pmax_sliceddf['Mean LF (all species)']), 10))
plt.legend()
plt.tight_layout()

plt.subplot(10, 2, 17)
plt.title('Correlation: Fights To Survive vs. P(improve) (Small wood)')
plt.ylabel('Fights to survive')
meanch_wmin_sliced_df = df_comparable[df_comparable['Wood km'] == min(df_comparable['Wood km'])]
meanch_fights_wmin_sliceddf = meanch_wmin_sliced_df.sort_values(by='P(improve)')
plt.plot(meanch_fights_wmin_sliceddf['P(improve)'], meanch_fights_wmin_sliceddf['Fights to survive'], label=f'Minimum Wood km = {min(df_comparable["Wood km"])}')
sns.set(style="whitegrid")
sns.regplot(x='P(improve)', y='Fights to survive', data=meanch_fights_wmin_sliceddf[['P(improve)', 'Fights to survive']], label='trend for Min Wood km')
plt.xticks(np.linspace(min(meanch_fights_wmin_sliceddf['P(improve)']), max(meanch_fights_wmin_sliceddf['P(improve)']), 10))
plt.yticks(list(range(max(meanch_fights_wmin_sliceddf['Fights to survive']) + 1)))
plt.grid(True)
plt.legend()

plt.subplot(10, 2, 18)
plt.title('Correlation: Fights To Survive vs. P(improve) (Big wood)')
meanch_wmax_sliced_df = df_comparable[df_comparable['Wood km'] == max(df_comparable['Wood km'])]
meanch_fights_wmax_sliceddf = meanch_wmax_sliced_df.sort_values(by='P(improve)')
plt.plot(meanch_fights_wmax_sliceddf['P(improve)'], meanch_fights_wmax_sliceddf['Fights to survive'],c='seagreen', label=f'Maximum Wood km = {max(df_comparable["Wood km"])}')
sns.set(style="whitegrid")
sns.regplot(x='P(improve)', y='Fights to survive', data=meanch_fights_wmax_sliceddf[['P(improve)', 'Fights to survive']],line_kws=dict(color='indianred'), scatter_kws=dict(color='indianred'), label='trend for Max Wood km')
plt.yticks(list(range(max(meanch_fights_wmax_sliceddf['Fights to survive']) + 1)))
plt.xticks(np.linspace(min(meanch_fights_wmax_sliceddf['P(improve)']), max(meanch_fights_wmax_sliceddf['P(improve)']), 10))
plt.legend()
plt.tight_layout()

#### Correlation: Mean LF considering all species vs. Wood km 
plt.subplot(10, 2, 19)
plt.title('Correlation: Mean LF considering all species vs. Wood km (Low improvement)')
plt.ylabel('Wood km')
meanlf_ppmin_sliced_df = df_comparable[df_comparable['P(improve)'] == min(df_comparable['P(improve)'])]
meanlf_wood_ppmin_sliceddf = meanlf_ppmin_sliced_df.sort_values(by='Wood km')
plt.plot( meanlf_wood_ppmin_sliceddf['Wood km'],meanlf_wood_ppmin_sliceddf['Mean LF (all species)'], label=f'Minimum P(improve) = {min(df_comparable["P(improve)"])}')
sns.set(style="whitegrid")
sns.regplot(y='Mean LF (all species)', x='Wood km', data=meanlf_wood_ppmin_sliceddf[['Wood km', 'Mean LF (all species)']], label='trend for Min P(improve)')
plt.xticks(meanlf_wood_ppmin_sliceddf['Wood km'])
plt.yticks(np.linspace(min(meanlf_wood_ppmin_sliceddf['Mean LF (all species)']), max(meanlf_wood_ppmin_sliceddf['Mean LF (all species)']), 5))
plt.grid(True)
plt.legend()

plt.subplot(10, 2, 20)
plt.title('Correlation: Mean LF considering all species vs. Wood km (High improvement)')
meanlf_ppmax_sliced_df = df_comparable[df_comparable['P(improve)'] == max(df_comparable['P(improve)'])]
meanlf_wood_ppmax_sliceddf = meanlf_ppmax_sliced_df.sort_values(by='Wood km')
plt.plot( meanlf_wood_ppmax_sliceddf['Wood km'],meanlf_wood_ppmax_sliceddf['Mean LF (all species)'],c='seagreen', label=f'Maximum P(improve) = {max(df_comparable["P(improve)"])}')
sns.set(style="whitegrid")
sns.regplot(y='Mean LF (all species)', x='Wood km', data=meanlf_wood_ppmax_sliceddf[['Wood km', 'Mean LF (all species)']],line_kws=dict(color='indianred'), scatter_kws=dict(color='indianred'), label='trend for Max P(improve)')
plt.xticks(meanlf_wood_ppmax_sliceddf['Wood km'])
plt.yticks(np.linspace(min(meanlf_wood_ppmax_sliceddf['Mean LF (all species)']), max(meanlf_wood_ppmax_sliceddf['Mean LF (all species)']), 5))
plt.legend()
plt.tight_layout()

plt.show()