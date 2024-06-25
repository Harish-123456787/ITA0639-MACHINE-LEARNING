import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Generate synthetic data from two Gaussian distributions
np.random.seed(0)

# Parameters for Gaussian distributions
mean1 = [0, 0]
cov1 = [[1, 0.5], [0.5, 1]]
mean2 = [3, 3]
cov2 = [[1, -0.5], [-0.5, 1]]

# Generate data points
data1 = np.random.multivariate_normal(mean1, cov1, 100)
data2 = np.random.multivariate_normal(mean2, cov2, 100)
X = np.vstack((data1, data2))

# Plot the generated data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
plt.title('Generated Data Points')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()

# Implementation of EM algorithm for GMM
def expectation_maximization(X, n_clusters, max_iters=100, tol=1e-4):
    n_samples, n_features = X.shape
    
    # Initialize parameters: means, covariances, and mixing coefficients
    np.random.seed(0)
    pi = np.ones(n_clusters) / n_clusters  # Mixing coefficients
    means = np.random.randn(n_clusters, n_features)
    covs = np.array([np.eye(n_features)] * n_clusters)
    
    # Log likelihood to track convergence
    log_likelihoods = []
    
    # EM algorithm
    for _ in range(max_iters):
        # E-step: Compute responsibilities (posterior probabilities)
        responsibilities = np.zeros((n_samples, n_clusters))
        for k in range(n_clusters):
            responsibilities[:, k] = pi[k] * multivariate_normal.pdf(X, mean=means[k], cov=covs[k])
        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
        
        # M-step: Update parameters
        Nk = np.sum(responsibilities, axis=0)
        means_new = np.dot(responsibilities.T, X) / Nk[:, np.newaxis]
        covs_new = []
        for k in range(n_clusters):
            diff = X - means_new[k]
            cov_k = np.dot(responsibilities[:, k] * diff.T, diff) / Nk[k]
            covs_new.append(cov_k)
        covs_new = np.array(covs_new)
        pi_new = Nk / n_samples
        
        # Compute log likelihood
        log_likelihood = np.sum(np.log(np.sum(pi[k] * multivariate_normal.pdf(X, means[k], covs[k]) for k in range(n_clusters))))
        log_likelihoods.append(log_likelihood)
        
        # Check for convergence
        if len(log_likelihoods) > 1 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            break
        
        # Update parameters
        means = means_new
        covs = covs_new
        pi = pi_new
    
    return means, covs, pi, responsibilities, log_likelihoods

# Run EM algorithm for GMM
n_clusters = 2
means, covs, pi, responsibilities, log_likelihoods = expectation_maximization(X, n_clusters)

# Print parameters
print("Means:")
print(means)
print("Covariances:")
print(covs)
print("Mixing coefficients:")
print(pi)

# Plot the convergence of log likelihood
plt.figure(figsize=(8, 6))
plt.plot(log_likelihoods)
plt.title('Log Likelihood Convergence')
plt.xlabel('Iterations')
plt.ylabel('Log Likelihood')
plt.grid(True)
plt.show()

# Plot the data points with cluster assignments
plt.figure(figsize=(8, 6))
colors = responsibilities.argmax(axis=1)
plt.scatter(X[:, 0], X[:, 1], c=colors, cmap='viridis', alpha=0.6)
plt.scatter(means[:, 0], means[:, 1], marker='x', color='r', s=100, label='Cluster Centers')
plt.title('EM Algorithm for GMM')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()
