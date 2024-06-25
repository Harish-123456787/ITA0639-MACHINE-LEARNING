import numpy as np
import pandas as pd
from collections import Counter

# Define a function to calculate entropy
def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

# Define a function to calculate information gain
def information_gain(X, y, threshold):
    # Parent entropy
    parent_entropy = entropy(y)
    
    # Generate split
    left_mask = X <= threshold
    right_mask = X > threshold
    
    if sum(left_mask) == 0 or sum(right_mask) == 0:
        return 0
    
    # Compute the weighted average entropy of the children
    n = len(y)
    n_left, n_right = sum(left_mask), sum(right_mask)
    
    e_left, e_right = entropy(y[left_mask]), entropy(y[right_mask])
    
    child_entropy = (n_left / n) * e_left + (n_right / n) * e_right
    
    # Information gain is difference in entropy
    ig = parent_entropy - child_entropy
    return ig

# Define the Decision Tree Classifier
class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
    
    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)
    
    def predict(self, X):
        return [self._predict(inputs) for inputs in X]
    
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = {
            'predicted_class': predicted_class
        }
        
        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] <= thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node['feature_index'] = idx
                node['threshold'] = thr
                node['left'] = self._grow_tree(X_left, y_left, depth + 1)
                node['right'] = self._grow_tree(X_right, y_right, depth + 1)
        return node
    
    def _best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None
        
        # Calculate the number of classes
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
        best_ig = 0
        best_idx, best_thr = None, None
        
        # Iterate over all features
        for idx in range(n):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                e_left = entropy(np.array(num_left))
                e_right = entropy(np.array(num_right))
                ig = entropy(y) - (i / m) * e_left - ((m - i) / m) * e_right
                
                if thresholds[i] == thresholds[i - 1]:
                    continue
                
                if ig > best_ig:
                    best_ig = ig
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr
    
    def _predict(self, inputs):
        node = self.tree_
        while 'threshold' in node:
            if inputs[node['feature_index']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node['predicted_class']

# Load the Iris dataset
from sklearn.datasets import load_iris
data = load_iris()
X, y = data.data, data.target

# Train the Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)

# Predict the class of a new sample
sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example: Iris-setosa
prediction = clf.predict(sample)
print(f"Predicted class: {data.target_names[prediction][0]}")

# Test the model on the test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
