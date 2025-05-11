############################## RAYLEIGH DISTRIBUTION ###################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
import math
import random

sizes = [100, 10_000, 100_000]
# define the lower bound and the upper bounds
lb = 0
ub = 1

slb = 0
sub = 10
sigma = random.randint(slb, sub)


def rayleigh_pdf(x, sigma):
    pdf = (x / (sigma ** 2)) * np.exp(-x ** 2 / (2 * sigma ** 2))
    return pdf


def plot_Rayleigh_histograms(size, ax):
    # Generate uniform random variables
    y = uniform(loc=lb, scale=ub - lb).rvs(size)

    # Calculate the values from the inverse of the cumulative distribution function
    transformed_values = [math.sqrt(2 * ((sigma) ** 2) * (math.log(1 / (1 - x)))) for x in y]
    # Plot the histogram
    ax.hist(transformed_values, bins=75, density=True, alpha=0.7, label=f'Empirical Histogram (size = {size})')
    ax.set_xlabel('Value')
    ax.set_ylabel('Probability density')

    # Plot the analytical PDF
    x_values = np.linspace(0, max(transformed_values), size)
    pdf_values = rayleigh_pdf(x_values, sigma)
    ax.plot(x_values, pdf_values, 'r-', label='Analytical PDF (Rayleigh)')

    # Compute and plot the first and the second moments to compare them with the mean and variance
    mean = np.mean(transformed_values)
    var = np.var(transformed_values)
    expectation = np.sqrt((np.pi / 2) * (sigma ** 2))
    variance = ((4 - np.pi) / 2) * (sigma ** 2)

    ax.scatter(x=mean, y=0.0, marker='*', s=80, c='k', label=f'mean = {round(mean, 3)}')
    ax.scatter(x=expectation, y=0.00, marker='o', s=80, c='orange', label=f'first moment = {round(expectation, 3)}')
    ax.scatter(x=variance, y=0.0, marker='+', s=80, c='k', label=f'variance = {round(variance, 3)}')
    ax.scatter(x=var, y=0.00, marker='x', s=80, c='orange', label=f'second moment = {round(var, 3)}')

    print(
        f'For size = {size}: mean = {mean} and variance = {var} vs expectation = {expectation} and variance = {variance}')

    ax.text(0.4, 0.5, f'sigma = {sigma} in a range of [{slb} - {sub}]', transform=plt.gca().transAxes, fontsize=10,
            color='black')
    ax.set_title('Empirical and analytical comparison for Rayleigh distribution')
    ax.legend()


fig, axs = plt.subplots(1, len(sizes), figsize=(20, 5))

for i, size in enumerate(sizes):
    plot_Rayleigh_histograms(size, axs[i])

plt.tight_layout()
plt.show()

############################## LOGNORMAL DISTRIBUTION ###################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
import math
import random

sizes = [100, 10_000, 100_000]
# define the lower bound and the upper bounds
lb = 0
ub = 1

slb = 0
sub = 1
sigma = random.uniform(slb, sub)
mlb = 0
mub = 10
mu = random.uniform(mlb, mub)


def lognormal_pdf(x, mu, sigma):
    pdf = (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma ** 2))
    return pdf


def plot_lognormal(size, ax):
    # Generate uniform random variables
    us = uniform(loc=lb, scale=ub - lb).rvs(size)
    vs = uniform(loc=lb, scale=ub - lb).rvs(size)
    # Compute the Bs and the thetas
    Bs = [np.sqrt(- (2 * np.log(u))) for u in us]
    thetas = [2 * np.pi * v for v in vs]
    half = int(len(Bs) / 2)
    # Compute the Zs distributed as Standard Normal
    z1s = [B * np.cos(thetas[i]) for i, B in enumerate(Bs[:half])]
    z2s = [B * np.sin(thetas[i]) for i, B in enumerate(Bs[half:])]

    zs = z1s + z2s
    # Generate the normal values and take the exponential from them to obtain the lognormal values
    normal_values = [mu + (sigma * z) for z in zs]
    lognormal_values = [np.exp(value) for value in normal_values]
    ax.hist(lognormal_values, bins=75, density=True, alpha=0.7, label=f'Histogram for size = {size}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Probability density')

    # Plot the analytical PDF
    x_values = np.linspace(0, max(lognormal_values), size)
    pdf_values = lognormal_pdf(x_values, mu, sigma)
    ax.text(0.6, 0.5, f'sigma = {round(sigma, 3)} \nin a range of [{slb} - {sub}]', transform=plt.gca().transAxes,
            fontsize=10, color='black')
    ax.text(0.6, 0.6, f'mu = {round(mu, 3)} \nin a range of [{mlb} - {mub}]', transform=plt.gca().transAxes,
            fontsize=10, color='black')
    ax.plot(x_values, pdf_values, 'r-', label='Analytical PDF (Lognormal)')
    # Compute the first and the second moments to compare them with the mean and variance
    mean = np.mean(lognormal_values)
    var = np.var(lognormal_values)
    expectation = np.exp(mu + (0.5 * (sigma) ** 2))
    variance = np.exp(2 * mu + (sigma) ** 2) * ((np.exp(sigma ** 2)) - 1)

    print(
        f'For size = {size}: mean = {mean} and variance = {var} vs expectation = {expectation} and variance = {variance}')

    ax.scatter(x=mean, y=0.0, marker='*', s=80, c='k', label=f'mean = {round(mean, 3)}')
    ax.scatter(x=expectation, y=0.00, marker='o', s=80, c='orange', label=f'first moment = {round(expectation, 3)}')
    # ax.scatter(x=variance, y=0.0, marker='+', s=80, c='k', label=f'variance = {round(variance, 3)}')
    # ax.scatter(x=var, y=0.00, marker='x', s = 80, c='orange', label=f'second moment = {round(var, 3)}')
    # Note: variance is not displayed to avoid confusion but it is computed and printed

    ax.set_title('Empirical and analytical comparison for Lognormal distribution')
    ax.legend(loc='upper right')


fig, axs = plt.subplots(1, len(sizes), figsize=(20, 5))

for i, size in enumerate(sizes):
    plot_lognormal(size, axs[i])

plt.tight_layout()
plt.show()

############################## BETA DISTRIBUTION ###################################
import random
import math
import matplotlib.pyplot as plt
from scipy.stats import uniform
import numpy as np
from scipy.special import gamma
# define the lower bound and the upper bounds
sizes = [100, 10_000, 100_000]
lb = 0
ub = 1

alb = 2
aub = 10

a = random.randint(alb, aub)
b = random.randint(alb, aub)

percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def gamma(alpha):
    return math.factorial(alpha - 1)


# Define the Beta PDF function
def beta_pdf(x, a, b):
    pdf = (gamma(a + b) / (gamma(a) * gamma(b))) * (x ** (a - 1)) * ((1 - x) ** (b - 1))
    return pdf


def plot_beta(percentages, size, ax):
    perc = random.choice(percentages)
    k = 1 / perc  # Probability to acceptance is P = 1/k

    # Generate uniform random variables
    xs = uniform(loc=lb, scale=ub - lb).rvs(size)
    ys = uniform(loc=lb, scale=(k * ub) - lb).rvs(size)

    return_x = []
    for i, y in enumerate(ys):
        if y < beta_pdf(xs[i], a, b):
            return_x.append(xs[i])

    ax.hist(return_x, bins=75, density=True, alpha=0.7,
            label=f'Histogram for size = {size} \nand probability to accept= {perc}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Probability density')

    # Plot the analytical PDF
    x_values = np.linspace(min(return_x), max(return_x), size)  # Use a fixed number of samples (e.g., 1000)
    pdf_values = beta_pdf(x_values, a, b)
    ax.plot(x_values, pdf_values, 'r-', label='Analytical PDF (Beta)')

    ax.text(0.5, 0.25, f'b = {b} in a range of [{alb} - {aub}]', transform=plt.gca().transAxes, fontsize=10,
            color='black')
    ax.text(0.5, 0.3, f'a = {a} in a range of [{alb} - {aub}]', transform=plt.gca().transAxes, fontsize=10,
            color='black')
    ax.text(0.5, 0.2, f'k = {round(k, 3)}', transform=plt.gca().transAxes, fontsize=10, color='black')
    # Compute the first and the second moments to compare them with the mean and variance
    mean = np.mean(return_x)
    var = np.var(return_x)
    expectation = a / (a + b)
    variance = (a * b) / (((a + b) ** 2) * (a + b + 1))

    ax.scatter(x=mean, y=0.0, marker='*', s=80, c='k', label=f'mean = {round(mean, 3)}')
    ax.scatter(x=expectation, y=0.00, marker='o', s=80, c='orange', label=f'first moment = {round(expectation, 3)}')
    # ax.scatter(x=variance, y=0.0, marker='+', s=80, c='k', label=f'variance = {round(variance, 3)}')
    # ax.scatter(x=var, y=0.00, marker='x', s = 80, c='orange', label=f'second moment = {round(var, 3)}')
    print(
        f'For size = {size}: mean = {mean} and variance = {var} vs expectation = {expectation} and variance = {variance}')
    # Note: the variance is not display to avoid confusion but it is computed and printed

    ax.set_title('Empirical and analytical comparison for Beta distribution')
    ax.legend(loc='upper right')


# Create a single figure with subplots
fig, axs = plt.subplots(1, len(sizes), figsize=(20, 5))

for i, size in enumerate(sizes):
    plot_beta(percentages, size, axs[i])

plt.tight_layout()  # Ensure subplots are properly spaced
plt.show()

############################## CHI SQUARED DISTRIBUTION ###################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
import math
import random

sizes = [100, 10_000, 100_000]
# define the lower bound and the upper bounds
lb = 0
ub = 1

dlb = 2
dub = 10
dof = random.randint(dlb, dub)


def chi_pdf(x, k):
    pdf = (1 / ((2 ** (k / 2)) * math.gamma(k / 2)) * (x ** (k / 2 - 1)) * np.exp(-x / 2))
    return pdf


def plot_chi(size, ax):
    us = []
    vs = []
    # Generate sequences of uniform random variables
    for _ in range(size):
        u = uniform(loc=0, scale=1).rvs(dof)
        v = uniform(loc=0, scale=1).rvs(dof)
        us.append(u)
        vs.append(v)
    # Compute the Bs and the thetas
    Bs = [np.sqrt(- (2 * np.log(u))) for u in us]
    thetas = [2 * np.pi * v for v in vs]
    half = int(len(Bs) / 2)
    # Compute the Zs
    z1s = [B * np.cos(thetas[i]) for i, B in enumerate(Bs[:half])]
    z2s = [B * np.sin(thetas[i]) for i, B in enumerate(Bs[half:])]

    zs = z1s + z2s
    # Generate the chi values by elevating the normal values to the square
    # Note that Zs are standard normally distributed so their mu = 0 and sigma = 1
    square_normal_values = [(z) ** 2 for z in zs]
    chi_values = [sum(square_normal_values[i]) for i in range(len(square_normal_values))]

    ax.hist(chi_values, bins=75, density=True, alpha=0.7, label=f'Histogram for size = {size}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Probability density')

    # Plot the analytical PDF
    x_values = np.linspace(0, max(chi_values), size)
    pdf_values = chi_pdf(x_values, dof)
    ax.text(0.5, 0.7, f'degrees of freedom = {dof}', transform=plt.gca().transAxes, fontsize=10, color='black')

    ax.plot(x_values, pdf_values, 'r-', label='Analytical PDF (Chi squared)')
    # Compute the first and the second moment to compare them with the mean and the variance
    mean = np.mean(chi_values)
    var = np.var(chi_values)
    expectation = dof
    variance = 2 * dof

    ax.scatter(x=mean, y=0.0, marker='*', s=80, c='k', label=f'mean = {round(mean, 3)}')
    ax.scatter(x=expectation, y=0.00, marker='o', s=80, c='orange', label=f'first moment = {round(expectation, 3)}')
    ax.scatter(x=variance, y=0.0, marker='+', s=80, c='k', label=f'variance = {round(variance, 3)}')
    ax.scatter(x=var, y=0.00, marker='x', s=80, c='orange', label=f'second moment = {round(var, 3)}')

    print(
        f'For size = {size}: mean = {mean} and variance = {var} vs expectation = {expectation} and variance = {variance}')

    ax.set_title('Empirical and analytical comparison for Chi squared distribution')
    ax.legend(loc='upper right')


fig, axs = plt.subplots(1, len(sizes), figsize=(20, 5))

for i, size in enumerate(sizes):
    plot_chi(size, axs[i])

plt.tight_layout()  # Ensure subplots are properly spaced
plt.show()

############################## RICE DISTRIBUTION ###################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, rice
import math
import random

sizes = [100, 10_000, 100_000]
# define the lower bound and the upper bounds
lb = 0
ub = 1

nlb = 1
nub = 4
nu = random.randint(nlb, nub)

slb = 1
sub = 4
sigma = random.randint(slb, sub)


def plot_rice(size, ax):
    # Generate uniform random variables
    us = uniform(loc=lb, scale=(ub - lb)).rvs(size)
    vs = uniform(loc=lb, scale=(ub - lb)).rvs(size)
    # Compute the Bs and the thetas
    Bs = [np.sqrt(- (2 * np.log(u))) for u in us]
    thetas = [2 * np.pi * v for v in vs]
    half = int(len(Bs) / 2)
    # Compute the Zs
    z1s = [B * np.cos(thetas[i]) for i, B in enumerate(Bs[:half])]
    z2s = [B * np.sin(thetas[i]) for i, B in enumerate(Bs[half:])]

    zs = z1s + z2s
    # Generate X distributed as Normal with mean equal to nu * cos(theta), variance equal to sigma^2
    # Generate also Y distributed as Normal with mean equal to nu * sin(theta), variance equal to sigma^2
    # square_xs = [((nu* np.cos(thetas[i])) + (sigma*z))**2 for i, z in enumerate(z1s)]
    # square_ys = [((nu* np.sin(thetas[i])) + (sigma*z))**2 for i, z in enumerate(z2s)]

    # but considering cos(theta) = 1 and sin(theta) = 0
    # so that mu = nu * cos(theta) = nu
    # and mu = nu * sen(theta) = 0

    square_xs = [((0) + (sigma * z)) ** 2 for z in z1s]
    square_ys = [((nu) + (sigma * z)) ** 2 for z in z2s]

    # Generate Rician random variables by computing the square root of the sum of X^2 and Y^2
    r_values = [np.sqrt(square_xs[i] + sq_y) for i, sq_y in enumerate(square_ys)]

    ax.hist(r_values, bins=75, density=True, alpha=0.7, label=f'Histogram for size = {size}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Probability density')

    # Plot the analytical PDF
    x_values = np.linspace(0, np.max(r_values), size)

    rv = rice(b=nu / sigma)
    ax.plot(x_values, rv.pdf(x_values), 'r-', label='Analytical pdf (Rice)')
    ax.text(0.6, 0.5, f'sigma = {round(sigma, 3)} \nin a range of [{slb} - {sub}]', transform=plt.gca().transAxes,
            fontsize=10, color='black')
    ax.text(0.6, 0.6, f'mu = {round(nu, 3)} \nin a range of [{nlb} - {nub}]', transform=plt.gca().transAxes,
            fontsize=10, color='black')
    # Note that the comparison between the first two moments and the mean and the variance is too complicated so it's skipped
    ax.set_title('Empirical and analytical comparison for Rice distribution')
    ax.legend(loc='upper right')


fig, axs = plt.subplots(1, len(sizes), figsize=(20, 5))

for i, size in enumerate(sizes):
    plot_rice(size, axs[i])

plt.tight_layout()  # Ensure subplots are properly spaced
plt.show()