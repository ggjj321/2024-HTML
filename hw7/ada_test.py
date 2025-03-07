import numpy as np

def decision_stump_multi_feature(X, y, weights):
    """
    Multi-dimensional Decision Stump implementation.
    
    Parameters:
    X : ndarray of shape (N, d), feature values for all samples
    y : ndarray of shape (N,), labels for all samples (+1 or -1)
    weights : ndarray of shape (N,), weights for all samples
    
    Returns:
    best_stump : dict, containing the best (feature, s, theta, E_in)
    """
    N, d = X.shape
    best_stump = {'feature': None, 's': None, 'theta': None, 'E_in': float('inf')}
    
    # Iterate over all features
    for feature in range(d):
        feature_values = X[:, feature]
        
        # Find the best stump for this feature
        sorted_indices = np.argsort(feature_values)
        X_sorted = feature_values[sorted_indices]
        y_sorted = y[sorted_indices]
        weights_sorted = weights[sorted_indices]
        
        thresholds = []
        for i in range(len(X_sorted) - 1):
            if X_sorted[i] != X_sorted[i + 1]:
                thresholds.append((X_sorted[i] + X_sorted[i + 1]) / 2)
        thresholds = [-float('inf')] + thresholds + [float('inf')]

        for theta in thresholds:
            for s in [-1, 1]:
                # Compute predictions
                predictions = s * np.sign(X_sorted - theta)
                predictions[predictions == 0] = s
                
                # Compute weighted error
                errors = (predictions != y_sorted).astype(float)
                weighted_error = np.dot(weights_sorted, errors)
                
                if weighted_error < best_stump['E_in']:
                    best_stump = {
                        'feature': feature,
                        's': s,
                        'theta': theta,
                        'E_in': weighted_error
                    }
    
    return best_stump

def adaboost(X, y, T):
    """
    AdaBoost implementation with multi-dimensional Decision Stump as weak classifier.
    
    Parameters:
    X : ndarray of shape (N, d), feature values for all samples
    y : ndarray of shape (N,), labels for all samples (+1 or -1)
    T : int, number of boosting rounds
    
    Returns:
    classifiers : list of (feature, s, theta, alpha) for each weak classifier
    """
    N, d = X.shape
    weights = np.ones(N) / N  # Initialize uniform weights
    classifiers = []

    for t in range(T):
        # Find the best decision stump across all features
        stump = decision_stump_multi_feature(X, y, weights)
        epsilon = stump['E_in']
        
        # Compute alpha for the weak classifier
        if epsilon == 0:  # Avoid division by zero
            alpha = float('inf')
        else:
            alpha = np.log((1 - epsilon) / epsilon)
        
        # Update sample weights
        feature_values = X[:, stump['feature']]
        predictions = stump['s'] * np.sign(feature_values - stump['theta'])
        predictions[predictions == 0] = stump['s']
        weights *= np.exp(-alpha * y * predictions)
        weights /= np.sum(weights)  # Normalize weights
        
        # Store the weak classifier
        classifiers.append({
            'feature': stump['feature'],
            's': stump['s'],
            'theta': stump['theta'],
            'alpha': alpha
        })
    
    return classifiers



def strong_classifier(X, classifiers):
    """
    Combine weak classifiers into a strong classifier.
    
    Parameters:
    X : ndarray of shape (N, d), feature values for all samples
    classifiers : list of (feature, s, theta, alpha) for each weak classifier
    
    Returns:
    predictions : ndarray of shape (N,), final predictions (+1 or -1)
    """
    print(classifiers)
    N = X.shape[0]
    final_pred = np.zeros(N)
    
    for clf in classifiers:
        feature_values = X[:, clf['feature']]
        weak_pred = clf['s'] * np.sign(feature_values - clf['theta'])
        weak_pred[weak_pred == 0] = clf['s']
        final_pred += clf['alpha'] * weak_pred
    
    return np.sign(final_pred)

# Example dataset
X = np.array([
    [1, 2],
    [2, 3],
    [3, 1],
    [4, 5],
    [5, 2]
])  # Features (N=5, d=2)
y = np.array([1, 1, -1, -1, 1])  # Labels
T = 5  # Number of boosting rounds

# Train AdaBoost
classifiers = adaboost(X, y, T)

# Display weak classifiers
print("Weak Classifiers:")
for t, clf in enumerate(classifiers):
    print(f"Round {t+1}: {clf}")

# Test strong classifier
predictions = strong_classifier(X, classifiers)
print("\nFinal Predictions:", predictions)
