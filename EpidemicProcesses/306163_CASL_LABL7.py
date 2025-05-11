################################## IMPORT ######################################
import numpy as np
import pandas as pd
import random
from scipy.integrate import odeint
import matplotlib.pyplot as plt


############################## DEFINE S = N - I - R ############################
## the following function define the number of susceptible as the difference between the total number of
## individuals in the population (N) and the number of Infected (I) and Recovered (i.e., healed or dead) (R) people
# Note: the Recovered people are no longer infectable
def def_S(N, I, R):
    S = N - I - R
    return S

seed = 142
random.seed(seed)
# Note: since all the simulations done have different parameter, the inputs are put into the function in lines 307-378
# Generally it follows the general structure of the simulate() function which is conducted given the following situations:
# 1. N = 50M (number of individuals of the population), h = 0.1 (fraction of hospitalized Infected), H_max = 100K (available hospital beds)
# 2. N = 50M (number of individuals of the population), h = 0.06 (fraction of hospitalized Infected), H_max = 100K (available hospital beds)
# 3. N = 50M (number of individuals of the population), h = 0.06 (fraction of hospitalized Infected), H_max = 50K (available hospital beds)
# 4. N = 10K (number of individuals of the population), h = 0.1 (fraction of hospitalized Infected), H_max = 200 (available hospital beds)

# the other parameters are taken fixed as follows:
# gamma = 1/14 (recovery rate), f = 0.03 (fatality rate), mu = 0.00001 (general death rate), D_max = 100K (max number of deaths allowed)
# R_t, that remains equal to R_0 = 4 (avg number of susceptible people infected by an Infected individual)

# finally, the following parameters are initialized but they assume a certain value during the simulation:
# lambda_ = 0.0 (infection rate), rho = 1.0 (strength of the restrictions)

# Note: lambda_, gamma and rho change their value if a certain condition is verified

############################## START OF THE SIMULATION ########################
def simulate(N, R_0, gamma, h, f, mu, I, R, S, D, H, lambda_, rho, H_max, D_max):
    year = list(range(366))
    # the simulation time lasts 1 year, since the discussion must be done during this period of time
    t0 = 0  # starting timestamp

    times = []
    S_list = []
    I_list = []
    R_list = []
    D_list = []
    H_list = []
    l_list = []

    times.append(t0)
    S_list.append(S)
    I_list.append(I)
    R_list.append(R)
    D_list.append(D)
    H_list.append(H)
    l_list.append(lambda_)
    # print(f'Day {t0+1}: S = {S}, I = {I}, R = {R}, D = {D}, H = {H}, rho = {rho} -> tot = {S+R+I}')

    for t in year:  # 366 days
        S = def_S(N, I, R)  # compute the number of susceptible people
        if t == 0:  # if it is the first day of the simulation
            R_t = R_0  # set the R_t (avg number of people infectable by an Infected individual) to R_0 = 4
            lambda_ = R_t * (gamma + mu)  # compute the contact rate following the Wikipedia (et al.) approach
            # availble at https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology
            new_I = round(lambda_ * S * I)  # compute the number of new infected people
            I = I + new_I  # add the new infected people to the initial number of Infected

            new_H = round(h * I)  # compute the number of Hospitalized people from the Infected ones
            H = H + new_H  # add the number of Hospitalized people to the previous Hospitalized ones
            I = I - new_H

            new_R = round(gamma * I)  # compute the number of Recovered people from the Infected ones
            R = R + new_R  # add the number of new Recovered people to the previous Recovered ones
            I = I - new_R  # remove the number of new Recovered people from the Infected ones

            new_D = round((mu + f) * I)  # compute the number of new Dead people from the Infected ones
            # new_D = round(f * I)
            R = R + new_D  # add the number of new Dead people to the Recovered ones
            D = D + new_D  # add the number of new Dead people to the previous Dead ones
            I = I - new_D  # remove the number of new Dead people to the Infected ones

            S = def_S(N, I, R)  # compute the number of remaining Susceptible people

            # store the results in the lists
            times.append(t + 1)
            S_list.append(S)
            I_list.append(I)
            R_list.append(R)
            D_list.append(D)
            H_list.append(H)
            l_list.append(lambda_)

        # Note: the strength of restriction rho is multiplied to the number of Infected I -> lower is rho, higher is the restriction
        elif t != 0:  # if is not the first day of simulation, check the fraction of Infected people over the entire population
            # if I/N <= 0.1: # if there are a few people infected, do not add any restriction -> basic behaviour
            #   rho = 1.0
            if (I / N) > 0.2 and (I / N) <= 0.3:  # if there are not too much infected compared with the population
                rho = 0.8  # add some restrictions but not too high
            elif (I / N) > 0.3 and (
                    I / N) <= 0.5:  # if there are an almost-high number of infected compared with the population
                rho = 0.1  # increase the restriction
            elif (I / N) > 0.5 or (H + new_H) >= (H_max - 1):  # if more than half of the population is infected or the hospidalized beds are almost fullfilled or both
                rho = 0.002  # higher the restriction

            if (D + new_D) >= (D_max - 1):  # if the number of deaths is almost reached the limit
                rho = 0.00001  # higher as much as possible restriction

            # lambda contact rate, rho strength of the restriction measures
            new_I = round(lambda_ * I * rho)  # compute the number of new Infected people
            I = I + new_I

            new_H = round(h * I)  # compute the number of Hospitalized people
            if (H + new_H) >= H_max:  # if the number of new H + old H are more then the available hospital beds, the new_H people are 'severe' cases and remain in the population
                # -> this means that they have a probability to infect someone else higher than the others, so the contact rate is higher
                lambda_ = lambda_ + (h * lambda_)  # lambda is incremented by a factor given by the ratio between the new_H and the I
                # (i.e., new_H/I) which is = h because they are not isolated, so the transmissivity rate is assumed to be higher
            else:  # if there are enough hospital beds
                H = H + new_H
                I = I - new_H
                lambda_ = lambda_ - (h * lambda_)  # lambda is decremented by a factor given by h because the 'severe' case are isolated
                gamma = gamma - mu  # the recovery rate is decreased by the death rate (chosen because very low)

            if I == 0:  # if there are no Infected people
                gamma = 1 / 14  # gamma is restored to the initial value

            new_R = round(gamma * (I + H))  # compute the number of Recovered people -> Note: both H and I are taken because when I store new_H, I remove them from I
            R = R + new_R
            frac = round(random.uniform(0, 1), 1)  # compute a random fraction of people in order to simulate that some people in hospital is healed/dead (i.e., Recovered)
            # and some other 'non-Hospitalized' infected people is healed/dead (i.e., Recovered) people
            if (H - round(frac * new_R)) > 0 and (I - round((1 - frac) * new_R)) > 0:  # if removing a fraction of people from H and from I does not produce negative result
                I = I - round((1 - frac) * new_R)  # a fraction of recovered is taken from the infected people
                H = H - round(frac * new_R)  # the other fraction is removed from the Hospitalized people
            else:  # if one of the 2 differences produces negative result
                if (H - round(frac * new_R)) < 0:  # in the case in which removing a fraction of Recovered people form H gives negative result,
                    H = H - H  # remove as much recovered people from H and ...
                    I = I + (H - round(frac * new_R))  # ... remove the remaining number of recovered people from I
                elif (I - round((1 - frac) * new_R)) < 0:  # in the case in which removing a fraction of Recovered people form I gives negative result,
                    I = I - I  # remove as much recovered people from I and ...
                    H = H + (I - round( (1 - frac) * new_R))  # ... remove the remaining number of recovered people from H
                    # Note: there is a '+' because (I - round((1 - frac) * new_R)) is negative, so H (>0) + (I - round((1 - frac) * new_R)) (<0) means doing 'H minus a quantity'

            new_D = round((mu + f) * (I + H))  # compute the number of dead people
            # new_D = round(f * I)
            R = R + new_D
            D = D + new_D
            frac_d = round(random.uniform(0, 1), 1)  # compute a fraction of people and DO THE SAME AS BEFORE (not reported to avoid redundances)
            if (H - round(frac_d * new_D)) > 0 and (I - round((1 - frac_d) * new_D)) > 0:
                I = I - round((1 - frac_d) * new_D)  # a fraction of deaths is taken from the infected people
                H = H - round(frac_d * new_D)  # the other fraction from the Hospitalized
            else:
                if (H - round(frac_d * new_D)) < 0:
                    H = H - H
                    I = I + (H - round(frac_d * new_D))
                elif (I - round((1 - frac_d) * new_D)) < 0:
                    I = I - I
                    H = H + (I - round((1 - frac_d) * new_D))

            S = def_S(N, I, R)

            if S <= 0:  # if all the susceptible people are all infected
                # Note: if, otherwise, not all the susceptible are infected, the you do not enter in this 'if' block and the 'for' loop keeps running with the previous basic behaviour
                # S = 0 # remove as much susceptible people as possible
                I = I + S  # the remaining infected people are not taken from the susceptible ones
                ## Note: to clarify, for example: if you have N = 5_000, I = 1705 and R = 3418, then S = N - I - R = 5000 - 1705 - 3418 = - 123
                ## so, if I remove as much susceptible as possible, S goes to 0 and the remaining 123 people that 'should be theoretically infected' are not taken
                ## so, you compute I + S = 1705 + (-123) = 1582 so that S = 5000 - 1582 - 3418 = 0
                if (I / N) > 0.2 and (I / N) <= 0.3:  # if there are not too much infected compared with the population -> DO THE SAME AS BEFORE (not reported to avoid redundances)
                    rho = 0.8  # not too high restriction
                elif (I / N) > 0.3 and (I / N) <= 0.5:
                    rho = 0.1
                elif (I / N) > 0.5 or (H + new_H) >= (H_max - 1):  # if more than half of the population is infected or the hospidalized beds are almost fullfilled or both
                    rho = 0.002  # higher the restriction

                if (D + new_D) >= (D_max - 1):
                    rho = 0.00001  # as high as possible restriction

                new_H = round(h * I)
                if (H + new_H) >= H_max:
                    lambda_ = lambda_ + (h * lambda_)  # lambda is incremented by a factor given by the ratio between the new_H and the I
                    # (i.e., new_H/I) which is = h because they are not isolated
                else:  # if there are enough hospital beds
                    H = H + new_H
                    I = I - new_H
                    lambda_ = lambda_ - (h * lambda_)  # lambda is decremented by a factor given by h because the 'severe' case are isolated
                    gamma = gamma - mu  # the recovery rate is decreased by the death rate

                if I == 0:
                    gamma = 1 / 14

                new_R = round(gamma * I)
                R = R + new_R
                frac = round(random.uniform(0, 1), 1)  # compute a fraction of people
                if (H - round(frac * new_R)) > 0 and (I - round((1 - frac) * new_R)) > 0:
                    I = I - round((1 - frac) * new_R)  # a fraction of deaths is taken from the infected people
                    H = H - round(frac * new_R)  # the other fraction from the Hospitalized
                else:
                    if (H - round(frac * new_R)) < 0:
                        H = H - H
                        I = I + (H - round(frac * new_R))
                    elif (I - round((1 - frac) * new_R)) < 0:
                        I = I - I
                        H = H + (I - round((1 - frac) * new_R))

                        # new_D = round((mu + f) * I)
                new_D = round(f * I)
                R = R + new_D
                D = D + new_D
                frac_d = round(random.uniform(0, 1), 1)  # compute a fraction of people
                if (H - round(frac_d * new_D)) > 0 and (I - round((1 - frac_d) * new_D)) > 0:
                    I = I - round((1 - frac_d) * new_D)  # a fraction of deaths is taken from the infected people
                    H = H - round(frac_d * new_D)  # the other fraction from the Hospitalized
                else:
                    if (H - round(frac_d * new_D)) < 0:
                        H = H - H
                        I = I + (H - round(frac_d * new_D))
                    elif (I - round((1 - frac_d) * new_D)) < 0:
                        I = I - I
                        H = H + (I - round((1 - frac_d) * new_D))
                        # at the end of the computations, S is put equal to 0, as said before
                S = 0

            # the computations are appended
            times.append(t + 1)
            S_list.append(S)
            I_list.append(I)
            R_list.append(R)
            D_list.append(D)
            H_list.append(H)
            l_list.append(lambda_)

            # print(f'Day {t+1}: S = {S}, I = {I}, R = {R}, D = {D}, H = {H}, rho = {rho}, lambda = {lambda_} -> tot = {S+R+I}') # to print day by day computations, use this line

    print(f'The number of deaths over a population N = {N} (at the end of the year) is {D_list[-1]}\n')

    def SIR_mean_field(y, t, lamb, gamma, rho):
        S, I, R = y
        # differential equation of the mean field (deterministic quantities)
        dSdt = -lamb * S * I * rho
        dIdt = lamb * S * I * rho - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]

    S0 = N - 1  # initial number of susceptible
    I0 = 1  # initial number of infected
    R0 = 0  # initial number of recovered
    y0 = [S0, I0, R0]

    # parameters (fixed)
    lambda_ = R_0 * (gamma + mu)  # motivation given above
    gamma = 1 / 14
    rho = 1.0  # initial value of rho
    t_det = np.linspace(0, 366, 366)  # Simulate for 200 days
    # solve the differential equations using odeint() predefined function
    solution = odeint(SIR_mean_field, y0, t_det, args=(lambda_, gamma, rho))
    S_det, I_det, R_det = solution.T

    return N, h, D_max, H_max, S_list, I_list, R_list, H_list, D_list, t_det, S_det, I_det, R_det, l_list


############################### OUTPUT METRICS #################################
def plot_res(N, h, D_max, H_max, S_list, I_list, R_list, H_list, D_list, t_det, S_det, I_det, R_det, l_list):
    plt.figure(figsize=(22, 16))
    plt.subplot(4, 1, 1)
    plt.plot(t_det, S_det, label='Susceptible deterministic', linestyle='--', c='blue', alpha=0.6)
    plt.plot(t_det, I_det, label='Infected deterministic', linestyle='--', c='gold', alpha=0.8)
    plt.plot(t_det, R_det, label='Recovered deterministic', linestyle='--', c='yellowgreen', alpha=0.7)
    plt.plot(S_list, label='Susceptible')
    plt.plot(I_list, label='Infected')
    plt.plot(R_list, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of Individuals of the Population')
    plt.title('SIR: Stochastic Model Simulation vs. Deterministic Mean Field (focus on the wave)')
    plt.suptitle(f'Parameters of the simulation: N = {N}, h = {h}, H_max = {H_max}, D_max = {D_max}')
    plt.legend(loc='upper right')
    plt.grid(True)
    max_x = 77
    plt.xlim(0, max_x)
    plt.xticks(list(range(0, max_x + 1, 7)))  # one tick each week

    plt.subplot(4, 1, 2)
    plt.plot(H_list, label='Hospitalized', c='purple')
    plt.ylabel('Number of Individuals of the Population')
    plt.xlabel('Time (days)')
    plt.grid(True)
    plt.xticks(list(range(0, len(H_list) + 1, 7)))  # one tick each week
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(D_list, label='Dead', c='Brown')
    plt.ylabel('Number of Individuals of the Population')
    plt.xlabel('Time (days)')
    plt.grid(True)
    plt.xticks(list(range(0, len(D_list) + 1, 7)))  # one tick each week
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(l_list, label='lambda', c='red')
    plt.ylabel('Infection Rate')
    plt.xlabel('Time (days)')
    plt.xticks(list(range(0, len(D_list) + 1, 7)))  # one tick each week
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# CASE 1. N = 50M, h = 0.1, H_max = 10K, D_max = 100k
N, h, D_max, H_max, S_list, I_list, R_list, H_list, D_list, t_det, S_det, I_det, R_det, l_list = simulate(N=50_000_000,
                                                                                                          R_0=4,
                                                                                                          gamma=1 / 14,
                                                                                                          h=0.1, f=0.03,
                                                                                                          mu=0.00001,
                                                                                                          I=1, R=0,
                                                                                                          S=def_S(
                                                                                                              N=50_000_000,
                                                                                                              I=1,
                                                                                                              R=0),
                                                                                                          D=0, H=0,
                                                                                                          lambda_=0.0,
                                                                                                          rho=1.0,
                                                                                                          H_max=10_000,
                                                                                                          D_max=100_000)
plot_res(N, h, D_max, H_max, S_list, I_list, R_list, H_list, D_list, t_det, S_det, I_det, R_det, l_list)
print('\n')

# CASE 2. N = 50M, h = 0.06, H_max = 10K, D_max = 100K
N, h, D_max, H_max, S_list, I_list, R_list, H_list, D_list, t_det, S_det, I_det, R_det, l_list = simulate(N=50_000_000,
                                                                                                          R_0=4,
                                                                                                          gamma=1 / 14,
                                                                                                          h=0.06,
                                                                                                          f=0.03,
                                                                                                          mu=0.00001,
                                                                                                          I=1, R=0,
                                                                                                          S=def_S(
                                                                                                              N=50_000_000,
                                                                                                              I=1,
                                                                                                              R=0),
                                                                                                          D=0, H=0,
                                                                                                          lambda_=0.0,
                                                                                                          rho=1.0,
                                                                                                          H_max=10_000,
                                                                                                          D_max=100_000)
plot_res(N, h, D_max, H_max, S_list, I_list, R_list, H_list, D_list, t_det, S_det, I_det, R_det, l_list)
print('\n')

# CASE 3. N = 50M, h = 0.1, H_max = 50K, D_max = 50K
N, h, D_max, H_max, S_list, I_list, R_list, H_list, D_list, t_det, S_det, I_det, R_det, l_list = simulate(N=50_000_000,
                                                                                                          R_0=4,
                                                                                                          gamma=1 / 14,
                                                                                                          h=0.1, f=0.03,
                                                                                                          mu=0.00001,
                                                                                                          I=1, R=0,
                                                                                                          S=def_S(
                                                                                                              N=50_000_000,
                                                                                                              I=1,
                                                                                                              R=0),
                                                                                                          D=0, H=0,
                                                                                                          lambda_=0.0,
                                                                                                          rho=1.0,
                                                                                                          H_max=50_000,
                                                                                                          D_max=100_000)
plot_res(N, h, D_max, H_max, S_list, I_list, R_list, H_list, D_list, t_det, S_det, I_det, R_det, l_list)
print('\n')

# CASE 4. n = 10K, h = 0.1, H_max = 200, D_max = 200
N, h, D_max, H_max, S_list, I_list, R_list, H_list, D_list, t_det, S_det, I_det, R_det, l_list = simulate(N=10_000,
                                                                                                          R_0=4,
                                                                                                          gamma=1 / 14,
                                                                                                          h=0.1, f=0.03,
                                                                                                          mu=0.00001,
                                                                                                          I=1, R=0,
                                                                                                          S=def_S(
                                                                                                              N=10_000,
                                                                                                              I=1,
                                                                                                              R=0),
                                                                                                          D=0, H=0,
                                                                                                          lambda_=0.0,
                                                                                                          rho=1.0,
                                                                                                          H_max=200,
                                                                                                          D_max=200)
plot_res(N, h, D_max, H_max, S_list, I_list, R_list, H_list, D_list, t_det, S_det, I_det, R_det, l_list)
