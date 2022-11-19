from scipy import stats
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt

x = []
y = []
dist = []

PIXEL_CONVERT = 0.1155 * (10 ** -6)
T = 0.5
TEMP = 296.5
ETA = 0.094
SPHERE_RADIUS = (1.9 * (10 ** -6)) / 2
GAMMA = 6 * np.pi * ETA * SPHERE_RADIUS


# Rayleigh Function
def func(r, D):
    return (r / (2 * D * T)) * np.exp(-(r**2) / (4 * D * T))


# File Reading
for j in range(13):
        filename = 'tracker {}.txt'.format(str(j+1))
        f = open('tracker 1.txt', 'r')

        for _ in range(2):
            discard = f.readline()

        contents = f.readlines()

        for i in range(len(contents)):
            line = contents[i]
            line = line.replace('\t', ' ').replace('\n', '')
            line = line.split()
            x.append(PIXEL_CONVERT * float(line[0]))
            y.append(PIXEL_CONVERT * float(line[1]))
            if i != 0:
                delta_x = x[-1] - x[-2]
                delta_y = y[-1] - y[-2]
                dist.append(np.sqrt((delta_x ** 2) + (delta_y ** 2)))

'''
# Removing Outliers
dist.sort()
pointer = 0
for k in range(len(dist)):
    if dist[k] < 1.6 * (10 ** -6):
        pointer += 1
dist = dist[:pointer-1]
'''
'''
# Creating Histogram
hist = np.histogram(dist, bins=15)
hist_dist = stats.rv_histogram(hist)


# Fitting Curve and Calculations
X = np.linspace(0, 1.8 * (10 ** -6), 200)
popt, pcov = optimize.curve_fit(func, X, hist_dist.pdf(X), bounds=(0, 1 * (10**-12)))
D1 = popt[0]
D1_err = np.sqrt(np.diag(pcov))
k1 = (D1 * GAMMA) / TEMP

GAMMA_ERR = 1.25957 * (10 ** -7)
k1_err = k1 * np.sqrt(((D1_err / D1) ** 2) + ((0.5 / 296.5) ** 2) + ((GAMMA_ERR / GAMMA) ** 2))


# Max Likelihood Calculations
sum = 0
for q in dist:
    sum += q ** 2

max_likelihood_est = (1 / (2 * len(dist))) * sum
D2 = max_likelihood_est / (2 * T)
k2 = (D2 * GAMMA) / TEMP


# Max Likelihood Error
r_sum = 0
for r in dist:
    r_sum += (2 * r * (10 ** -7)) ** 2
r_sum_err = np.sqrt(r_sum)
time_term = sum * (-1 / (4 * len(dist) * (T**2)))
D_term = 1 / (4 * len(dist) * T)
D2_err = np.sqrt(((time_term * 0.03) ** 2) + ((D_term * r_sum_err) ** 2))

k2_err = k2 * np.sqrt(((D2_err / D2) ** 2) + ((0.5 / 296.5) ** 2) + ((GAMMA_ERR / GAMMA) ** 2))


# R-Sqaured
residuals = 0
for_mean = 0


for x in X:
    residuals += (hist_dist.pdf(x) - func(x, D2)) ** 2
    for_mean += hist_dist.pdf(x)

mean = for_mean / len(X)

with_mean = 0
for x in X:
    with_mean += (hist_dist.pdf(x) - mean) ** 2

r_sqaured = 1 - (residuals / with_mean)

# Display
print(r_sqaured)
print("D1: ", D1)
print("D1 error: ", D1_err)
print("k1: ", k1)
print("k1 error: ", k1_err)
print('')
print("D2: ", D2)
print("D2 error: ", D2_err)
print("k2: ", k2)
print("k2 error: ", k2_err)
'''

Y = np.linspace(0, 0, 200)
#plt.title("Absolute Distance Travelled Per 0.5s Time Interval")
#plt.title("Residual Plot of Probability Distribution vs Fitted Rayleigh Distribution")
#plt.xlabel("Absolute Distance Travelled (m)")
#plt.ylabel("Residual Distribution")
#plt.hist(dist, density=True, bins=15)
#plt.plot(X, hist_dist.pdf(X), label="RV Probability Distribution")
#plt.plot(X, func(X, D1), label="Rayleigh Distribution (Curve Fitting Method)")
#plt.plot(X, func(X, D2), label="Rayleigh Distribution (Max-Likelihood Method)", color='#00BB00')
#plt.plot(X, hist_dist.pdf(X) - func(X, D2))
#plt.plot(X, Y, color='#00BB00')

X_new = np.linspace(0, 60, 120)

data_used_2 = [0]
data_used_3 = [0]
for p in range(0, 119):
    data_used_2.append(data_used_2[p] + dist[p+119])
    data_used_3.append(data_used_3[p] + dist[p+238])

plt.title("Total Distance Traveled of Brownian Particle")
plt.xlabel("Time (s)")
plt.ylabel("Total Distance (m)")
plt.plot(X_new, data_used_2)
plt.plot(X_new, data_used_3

#plt.legend()
plt.show()