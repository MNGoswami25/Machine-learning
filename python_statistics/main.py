import numpy as np
import statistics as stats

height = [110, 120, 140, 150, 125, 160, 178, 178, 141, 152, 156, 145, 165]

mean = np.mean(height)

n = len(height)
np.sort(height)
if n % 2 == 0:
    median = (height[n // 2 - 1] + height[n // 2]) / 2
else:
    median = height[(n - 1) // 2]

mode=stats.mode(height)

var=np.var(height)

std=np.std(height)

print("Mean:", mean)
print("Median:", median)
print("Mode:",mode)
print("Variance: ",var)
print("Standard Deviation: ",std)
