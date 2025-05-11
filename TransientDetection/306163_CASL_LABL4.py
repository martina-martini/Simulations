######################################### TRANSIENT DETECTION AND BATCH MEANS METHOD###################################
# due to the big number of plots to compute, the runnning might takes at most 7 minutes
import random
import matplotlib.pyplot as plt
import numpy
from scipy.stats import t

# service rate is needed for the case of  exponentially distributed service times
SERVICE = 1.0

def generate_hyper_exp():
    lambda1, lambda2 = random.choices(arrival_lambdas, k=2)
    x1 = random.expovariate(1 / lambda1)
    x2 = random.expovariate(1 / lambda2)
    p = random.uniform(0, 1)
    y = p * x1 + (1 - p) * x2
    return y


arrival_lambdas = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 0.999]
service_distribution = [1, random.expovariate(SERVICE), generate_hyper_exp()]

SIM_TIME = 2_000

# the maximum capacity of the queue is set to a big number as if it is infinite
MAX_QUEUE_CAPACITY = 100000

k = 1  # number of servers/machines = 1

class Measure:
    def __init__(self, num_arr, num_dep, num_avg_users, time_last, avg_delay, num_drop):
        self.num_arrivals = num_arr
        self.num_departures = num_dep
        self.average_utilization = num_avg_users
        self.time_last_event = time_last
        self.average_delay_time = avg_delay
        self.num_dropped = num_drop

class Client:
    def __init__(self, arrival_time):
        self.arrival_time = arrival_time

class Queue:
    def __init__(self, capacity):
        self.queue = []
        self.capacity = capacity

    def is_full(self):
        return len(self.queue) >= self.capacity

class PriorityQueue:
    def __init__(self):
        self.events = []

    def put(self, el):
        index = None
        for i, event in enumerate(self.events):
            if el[0] < event[0]:
                index = i
                break
        if index is not None:
            self.events.insert(i, el)
        else:
            self.events.append(el)

def arrival(time, FES, queue, data, lambd, ser_time):
    global users
    global servers

    data.num_arrivals += 1 # complete data
    data.average_utilization += users * (time - data.time_last_event)
    data.time_last_event = time

    # compute the inter-arrival
    inter_arrival = random.expovariate(lambd)

    # schedule an arrival
    FES.put((time + inter_arrival, "arrival"))

    client = Client(time)

    users += 1

    # if the server is available -> make it busy
    if servers >= 1:
        service_time = ser_time
        servers -= 1  # the server is busy

        # schedule the end of service
        FES.put((time + service_time, 'departure'))
    else:  # add the client to the queue
        if not queue.is_full():
            queue.queue.append(client)
        else:
            data.num_dropped += 1
            users -= 1


def departure(time, FES, queue, data, ser_time):
    global users
    global servers

    data.num_departures += 1
    data.average_utilization += users * (time - data.time_last_event)
    data.time_last_event = time

    users -= 1

    # if the queue is not empty, some clients are waiting, so we take a client from the queue
    if len(queue.queue) > 0:
        client = queue.queue.pop(0)

        data.average_delay_time += (time - client.arrival_time)

        service_time = ser_time

        FES.put((time + service_time, 'departure'))
    else:
        servers += 1
        # the server is available again

# starting collecting data for the following statistics
departure_delay_per_time = {}
time_points = {}
for lambd in arrival_lambdas:
    time_points[lambd] = []
    departure_delay_per_time[lambd] = []

############################################## function to plot the average delays for each lambda #####################
def simulate_plot(lambd, queue_lenght, ser_time):
    data = Measure(0, 0, 0, 0, 0, 0)
    time = 0

    FES = PriorityQueue()
    queue = Queue(queue_lenght)

    FES.put((time, "arrival"))

    while time < SIM_TIME:
        if not FES.events:
            break

        (time, event_type) = FES.events.pop(0)

        if event_type == 'arrival':
            arrival(time, FES, queue, data, lambd, ser_time)
        elif event_type == 'departure':
            departure(time, FES, queue, data, ser_time)
            time_points[lambd].append(time)
            departure_delay_per_time[lambd].append(data.average_delay_time / data.num_departures)

    average_delay = data.average_delay_time / data.num_departures
    average_no_cust = data.average_utilization / time

    print(f'Average time spent waiting (queueing delay): {average_delay:.4f}s')

# for each service time (deterministic, exponential and hyperexponential) do the following
for ser_time in service_distribution:
    for lambd in arrival_lambdas:

        users = 0
        servers = k
        print(f'\nStarting simulation with arrival lambda = {lambd}')
        simulate_plot(queue_lenght=MAX_QUEUE_CAPACITY, lambd=lambd, ser_time=ser_time)

        plt.figure(figsize=(18, 4))
        # plot the delay for each time of departure
        plt.plot(time_points[lambd], departure_delay_per_time[lambd], marker='o', linestyle='-', color='b',
                 label='average delays')
        plt.axhline(y=numpy.mean(departure_delay_per_time[lambd]), color='green', label='Experimental Mean Delay')
        print(f'Experimental Mean Delay: {round(numpy.mean(departure_delay_per_time[lambd]), 3)}s')
        departure_delay_per_time[lambd].clear()
        time_points[lambd].clear()
        plt.xlabel('Departure Time')
        plt.ylabel('Average Delay')
        if ser_time == service_distribution[0]:
            plt.title(
                f'Average Delay vs Departure Time with lambda = {lambd} and Deterministic service time = {ser_time}')
        elif ser_time == service_distribution[1]:
            plt.title(
                f'Average Delay vs Departure Time with lambda = {lambd} and Exponential service time = {ser_time}')
        else:
            plt.title(
                f'Average Delay vs Departure Time with lambda = {lambd} and Hyper-exponential service time = {ser_time}')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()

######################################## collect the average delays for each lambda ####################################
average_delay_times = []


def departure(time, FES, queue, data, ser_time):
    global users
    global servers

    data.num_departures += 1
    data.average_utilization += users * (time - data.time_last_event)
    data.time_last_event = time

    users -= 1

    if len(queue.queue) > 0:
        client = queue.queue.pop(0)

        data.average_delay_time += (time - client.arrival_time)

        average_delay_times.append(time - client.arrival_time)

        service_time = ser_time

        FES.put((time + service_time, 'departure'))
    else:
        servers += 1


departure_delay_per_time = {}
time_points = {}
for lambd in arrival_lambdas:
    time_points[lambd] = []
    departure_delay_per_time[lambd] = []

def simulate(lambd, queue_lenght, ser_time):
    data = Measure(0, 0, 0, 0, 0, 0)
    time = 0

    FES = PriorityQueue()
    queue = Queue(queue_lenght)
    FES.put((time, "arrival"))

    while time < SIM_TIME:
        if not FES.events:
            break

        (time, event_type) = FES.events.pop(0)

        if event_type == 'arrival':
            arrival(time, FES, queue, data, lambd, ser_time)
        elif event_type == 'departure':
            departure(time, FES, queue, data, ser_time)
            time_points[lambd].append(time)
            departure_delay_per_time[lambd].append(data.average_delay_time / data.num_departures)

    average_delay = data.average_delay_time / data.num_departures
    average_no_cust = data.average_utilization / time

    return average_delay, average_no_cust


######################################### useful functions for later ###################################################
import numpy
import random
from scipy.stats import t


def Pollacek_Khinchine_mean_waiting_time(lambda_, det_service_time):
    if lambda_ != 1: # this condition is because if lambda = 1 the denominator is 0
        if det_service_time == 1: # deterministic case
            mu = 1
            var = 0
        elif det_service_time == service_distribution[1]: # exponential case
            mu = 1
            var = mu ** 2
        else:  # hyper-exponential case
            mu = 1
            var = 10

        ro = lambda_ / mu
        W_ = (ro + (lambda_ * mu * var)) / (2 * (mu - lambda_))

        return W_
    else:
        pass


def Little(lambda_, num_of_clients): # Little's Law
    W = num_of_clients / lambda_
    return W


def detect_transient(signal, window_size):
    variances = [np.var(signal[i:i + int(window_size)]) for i in range(len(signal) - int(window_size) + 1)]
    transient_indices = [i for i, var in enumerate(variances) if var > np.mean(variances)]

    return transient_indices
# the idea is to compute the variances of each interval -> if the variance of the single interval is higher than the
# mean variance, then the system is not stable yet, so we are in the transient phase

def batch_means(signal, batch_size, confidence):
# the batches are divided depending on the batch size and for each batch the mean of the points inside is computed
    means = [np.mean(signal[i:i + int(batch_size)]) for i in
             range(0, len(signal) - int(batch_size) + 1, int(batch_size))]

    batch_sizes = [len(signal[i:i + int(batch_size)]) for i in range(0, len(signal), int(batch_size))]
    standard_errors = [np.std(signal[i:i + int(batch_size)], ddof=1) / np.sqrt(size) for size in batch_sizes]

    # compute the t-value for a given confidence level
    t_value = t.ppf((1 + confidence) / 2, df=batch_size - 1)

    # compute the margin of error
    margins_of_error = [t_value * se for se in standard_errors]

    # compute the confidence intervals
    lower_bounds = [mean - margin for mean, margin in zip(means, margins_of_error)]
    upper_bounds = [mean + margin for mean, margin in zip(means, margins_of_error)]

    return means, lower_bounds, upper_bounds


##### computation of the cumulative average for each lambda, taking under consideration NUM_OF_SEEDS seeds #############
iteration = 0
k = 1
iterat = []
NUM_OF_SEEDS = 2_000
seeds = [random.randint(10, 1_000_000) for _ in range(NUM_OF_SEEDS)]

for ser_time in service_distribution:
    for lam in arrival_lambdas:
        average_delay_per_run = []
        # for each random seed, the average delay at the end of the system is stored -> taking all together the averages
        # we can plot the cumulative average and see how the system behaves changing the seeds (high number are used to
        # # make the samples as more independent as possible from each others)
        for seed in seeds:
            numpy.random.seed(seed)
            users = 0
            servers = k
            lambd = lam
            avg, no_cust = simulate(queue_lenght=MAX_QUEUE_CAPACITY, lambd=lambd, ser_time=ser_time)
            average_delay_per_run.append(avg)
            iteration += 1
            iterat.append(iteration)

        ss = []  # list of seeds
        aa = []  # list of averages
        sip = zip(seeds, average_delay_per_run)
        s_sid = sorted(sip, key=lambda x: x[0])
        for s, a in s_sid:
            ss.append(s)
            aa.append(a)

        # plot the average delays for each seed (ordered increasingly)
        plt.figure(figsize=(18, 4))
        plt.plot(ss, aa)
        if ser_time == service_distribution[0]:
            plt.title(f'Average Delay vs Seed with lambda = {lambd} and Deterministic service time = {ser_time}')
        elif ser_time == service_distribution[1]:
            plt.title(f'Average Delay vs Seed with lambda = {lambd} and Exponential service time = {ser_time}')
        else:
            plt.title(f'Average Delay vs Seed with lambda = {lambd} and Hyper-exponential service time = {ser_time}')
        plt.ylabel('Average Delay')
        plt.xlabel('Seed')
        plt.grid(True)
        plt.show()

########################################## Pollachezk and Little's law comparisons #####################################

        # plot again the cumulative average per seed (one different seed for each run)
        # then the Pollachezk formula and the Little's law are computed, plotted, evaluated and compared
        import numpy as np

        plt.figure(figsize=(18, 4))
        plt.title(f'Cumulative average delay per Seed with lambda = {lambd}')
        cum_avg = np.cumsum(aa) / np.arange(1, len(aa) + 1)
        plt.plot(cum_avg, label='Cumulative Average', color='blue')
        plt.axhline(y=Pollacek_Khinchine_mean_waiting_time(lambd, ser_time), color='brown',
                    label='Pollaczek–Khinchine Mean Waiting Time')
        # plt.axhline(y = Little(lambd, no_cust), color='brown', label="Average Waiting Time from Little's Law")
        print(
            f"Little's Law: mean number of customer in the system = {no_cust}, lambda = {lambd} ->  mean waiting time "
            f"W = {Little(lambd, no_cust)}")
        print(f"Pollacek Khinchine: mean waiting time W' = {Pollacek_Khinchine_mean_waiting_time(lambd, ser_time)}")
        print(f'Experimental mean waiting time = {numpy.mean(cum_avg)}')
        plt.legend()
        plt.ylabel('Cumulative Average Delay')
        plt.xlabel('Seed')
        plt.grid(True)
        plt.show()

############################################### transient detection ####################################################
        iteration = 0
        NUM_BATCHES = 20
        BATCH_SIZE = len(cum_avg) / NUM_BATCHES
        indexes = []

        # detect transient phase by checking which indexes are in correspondence of the points whose variance is higher
        # than the mean variance
        transient_indices = detect_transient(cum_avg, BATCH_SIZE)
        plt.figure(figsize=(18, 5))
        for i, avg in enumerate(cum_avg):
            indexes.append(i)
        plt.plot(indexes, cum_avg, label='Cumulative average delay', color='blue')
        plt.scatter(transient_indices, [cum_avg[i] for i in transient_indices], color='red', label='Transient Phase')
        plt.xlabel('Seed')
        plt.ylabel('Cumulative Average')
        plt.title(f'Detecting Transient phase with lambda = {lambd}')
        plt.legend()
        plt.grid(True)
        plt.show()

################################################### steady state selection and batch means #############################
        # select the steady state as the remaining points from the last index of the transient phase on
        last_transient_index = transient_indices[-1]
        selected_avg = cum_avg[last_transient_index:]
        selected_index = indexes[last_transient_index:]

        # compute the batches
        BATCH_SIZE = len(selected_index) // NUM_BATCHES
        CONFIDENCE_LEVEL = 0.9999

        plt.figure(figsize=(18, 4))

        # plot the steady state
        plt.plot(selected_index, selected_avg, color='blue', label='Steady State')
        lines = []
        s_batch_midpoints = []
        intervals = np.linspace(selected_index[0], selected_index[-1], NUM_BATCHES, dtype=int)
        last_line = 0
        for i in range(len(intervals) - 1):
            start_interval = intervals[i]
            end_interval = intervals[i + 1] - 1
            lines.append(start_interval)
            plt.axvline(x=start_interval, color='k', linestyle='--')
            s_batch_midpoints.append((start_interval + end_interval) // 2)
        last_line = end_interval
        lines.append(last_line)
        plt.axvline(x=last_line, color='k', linestyle='--')
        # plot also the batches and compute the means of all the batches, the display them in correspondence of the
        # middle point of each interval
        batch_means_values, lower_bounds, upper_bounds = batch_means(selected_avg, BATCH_SIZE, CONFIDENCE_LEVEL)

        plt.scatter(s_batch_midpoints, batch_means_values[:len(s_batch_midpoints)], marker='*', linestyle='--',
                    color='aqua', linewidth=4, label='Batch Mean')
        plt.axhline(numpy.mean(selected_avg), color='green', label='Experimental Mean')
        plt.xticks(lines)

        plt.xlabel('Run')
        plt.ylabel('Cumulative Average')
        plt.title(f'Batch Means with lambda = {lambd}')
        plt.ylim(min(cum_avg), max(cum_avg))

        plt.legend()
        plt.grid(True)
        plt.show()

################################################ batch means method and CI  ############################################

        plt.figure(figsize=(18, 5))
        plt.plot(selected_index, selected_avg, color='blue', label='Steady State')
        lines = []
        s_batch_midpoints = []
        intervals = np.linspace(selected_index[0], selected_index[-1], NUM_BATCHES, dtype=int)
        last_line = 0
        for i in range(len(intervals) - 1):
            start_interval = intervals[i]
            end_interval = intervals[i + 1] - 1
            lines.append(start_interval)
            plt.axvline(x=start_interval, color='k', linestyle='--')
            s_batch_midpoints.append((start_interval + end_interval) // 2)
        last_line = end_interval
        lines.append(last_line)
        plt.axvline(x=last_line, color='k', linestyle='--')
        plt.scatter(s_batch_midpoints, batch_means_values[:len(s_batch_midpoints)], marker='*', linestyle='--',
                    color='aqua', linewidth=4, label='Batch Mean')
        # plot confidence intervals in correspondence of the middle points of the batches
        for midpoint, lower, upper in zip(s_batch_midpoints, lower_bounds, upper_bounds):
            plt.plot([midpoint, midpoint], [lower, upper], color='orange', linestyle='-', linewidth=2)

        plt.axhline(numpy.mean(selected_avg), label='Experimental Mean', color='green')
        plt.xticks(lines)

        plt.xlabel('Run')
        plt.ylabel('Cumulative Average')
        plt.title(f'Confidence Intervals and Batch Means with lambda = {lambd}')
        # the plot is zommed in to better understand the confidence intervals (orange lines)
        plt.ylim(min(lower_bounds) - 0.0001, max(upper_bounds) + 0.0001)
        plt.legend()
        plt.grid(True)

        plt.show()

############################################## M/M/1 QUEUE #####################################################
# fixed parameters
LIMIT_QUEUE_CAPACITY = 1000 # max number of clients in the queue (i.e., finite waiting line)
k = 1 # number of servers
lambdas = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1, 1.1, 1.2]

mm1departure_delay_per_time = {}
mm1time_points = {}
mm1_dropping = {}
mm1_time = {}
for lambd in lambdas:
    mm1time_points[lambd] = []
    mm1departure_delay_per_time[lambd] = []
    mm1_dropping[lambd] = []
    mm1_time[lambd] = []

# conceptually is the same system as before but both the arrivals and the services are exponentially distributed with
# mean equal to lambda
def simulate_mm1plot(lambd, queue_lenght, ser_time):
    data = Measure(0, 0, 0, 0, 0, 0)
    time = 0

    FES = PriorityQueue()
    queue = Queue(queue_lenght)

    FES.put((time, "arrival"))

    while time < SIM_TIME:
        if not FES.events:
            break

        (time, event_type) = FES.events.pop(0)

        if event_type == 'arrival':
            arrival(time, FES, queue, data, lambd, ser_time)
            mm1_dropping[lambd].append(data.num_dropped / data.num_arrivals * 100)
            mm1_time[lambd].append(time)

        elif event_type == 'departure':
            departure(time, FES, queue, data, ser_time)
            mm1time_points[lambd].append(time)
            mm1departure_delay_per_time[lambd].append(data.average_delay_time / data.num_departures)

    mm1average_delay = data.average_delay_time / data.num_departures
    average_no_cust = data.average_utilization / time

    print(f'Average time spent waiting (queueing delay): {mm1average_delay:.4f}s')
    print(f'Dropped clients: {data.num_dropped} (Dropping probability: {data.num_dropped / data.num_arrivals * 100:.2f}%)')


for lambd in lambdas:
    users = 0
    servers = k
    print(f'\nStarting M/M/1 simulation with arrival lambda = {lambd}')
    simulate_mm1plot(queue_lenght=LIMIT_QUEUE_CAPACITY, lambd=lambd, ser_time=random.expovariate(lambd))

    plt.figure(figsize=(18, 4))
###################################### plot the average delays for each lambda #########################################
    plt.plot(mm1time_points[lambd], mm1departure_delay_per_time[lambd], marker='o', linestyle='-', color='b',
             label='average delays')
    plt.axhline(y=numpy.mean(mm1departure_delay_per_time[lambd]), color='green', label='Experimental Mean Delay')
    print(f'Experimental Mean Delay: {round(numpy.mean(mm1departure_delay_per_time[lambd]), 3)}s')

    plt.xlabel('Departure Time')
    plt.ylabel('Average Delay')
    plt.title(f'Average Delay vs Departure Time with lambda = {lambd} and Exponential service time = {ser_time}')

    plt.legend()
    plt.grid(True)

###################################### plot the dropped clients for each lambda ########################################
    plt.figure(figsize=(18, 4))

    plt.plot(mm1_time[lambd], mm1_dropping[lambd], marker='o', linestyle='-', color='b', label='dropping probabilities')
    plt.axhline(y=numpy.mean(mm1_dropping[lambd]), color='green', label='Experimental Mean Dropping Probability')
    print(f'Experimental Mean Dropping Probability: {round(numpy.mean(mm1_dropping[lambd]), 3)}s')

    plt.xlabel('Arrival Time')
    plt.ylabel('Dropping Probability')
    plt.title(f'Dropping Probability vs Arrival Time with lambda = {lambd} and Exponential service time = {ser_time}')

    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()