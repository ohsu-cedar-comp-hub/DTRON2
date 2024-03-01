import numpy as np
import ot  # Python Optimal Transport library
import matplotlib.pyplot as plt


# Example 1D distributions X and Y
np.random.seed(0)  # for reproducibility
X = np.random.normal(0, 1, 500)  # Normal distribution with mean 0 and std 1
Y = np.random.normal(2, 0.5, 500)  # Normal distribution with mean 1 and std 1

# Reshape distributions for the OT problem
X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)

# Compute the cost matrix (Euclidean distance in 1D is absolute difference)
cost_matrix = ot.dist(X, Y, metric='euclidean')

# Compute uniform weights for each distribution
n = len(X)
m = len(Y)
a = ot.unif(n)
b = ot.unif(m)

# Solve the Optimal Transport problem
transport_plan = ot.emd(a, b, cost_matrix)

# Displaying the first few elements of the transport plan
# print(transport_plan[:10])



# Plot the original distributions
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.hist(X, bins=20, alpha=0.5, label='X')
plt.hist(Y, bins=20, alpha=0.5, label='Y')
plt.title("Original Distributions")
plt.legend()

# Plot the transformed X (assuming each point in X moves to the corresponding point in Y)
# This is a simplification for visualization purposes.

X_transformed = np.zeros_like(X)
for i in range(len(X)):
    X_transformed[i] = np.sum(transport_plan[i, :] * Y.T) / np.sum(transport_plan[i, :])

plt.subplot(1, 2, 2)
plt.hist(X_transformed, bins=20, alpha=0.5, label='Transformed X')
plt.hist(Y, bins=20, alpha=0.5, label='Y')
plt.title("Transformed Distribution")
plt.legend()

plt.tight_layout()
plt.savefig("test_opt.pdf",format='pdf')