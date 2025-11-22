
import math
from collections import Counter

# ----- KNN Implementation -----
def euclidean(p1, p2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(p1, p2)))

def normalize(data):
    transposed = list(zip(*[row[0] for row in data]))
    mins = [min(col) for col in transposed]
    maxs = [max(col) for col in transposed]
    normalized = [
        ([(x - mn) / (mx - mn) if mx != mn else 0 for x, mn, mx in zip(row[0], mins, maxs)], row[1])
        for row in data
    ]
    return normalized

def predict_knn(new_point, data, k):
    data = normalize(data)
    new_point = [(x - min(c)) / (max(c) - min(c)) if max(c) != min(c) else 0 for x, c in zip(new_point, zip(*[r[0] for r in data]))]
    distances = [(euclidean(new_point, point), label) for point, label in data]
    neighbors = sorted(distances, key=lambda x: x[0])[:k]
    votes = [label for _, label in neighbors]
    return Counter(votes).most_common(1)[0][0]

# Example usage:
# dataset = [([feature1, feature2], 'label'), ...]
# result = predict_knn([new_feature1, new_feature2], dataset, k)

# ----- Logistic Regression Prediction -----
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

z = 1.72
prob = sigmoid(z)
print("Sigmoid Probability:", prob)
print("Class:", "Positive" if prob >= 0.5 else "Negative")
