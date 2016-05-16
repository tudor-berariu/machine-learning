from iris_dataset import IrisDataset
import numpy as np

def perceptron(X, t):
    (N, D) = X.shape
    w, b = np.zeros(D), 0
    num_iter, is_done = 0, False
    while not is_done and num_iter < 1000:
        is_done = True
        p = np.random.permutation(N)
        for i in p:
            y = np.dot(w, X[i]) + b
            if y * t[i] <= 0:
                w += X[i] * t[i]
                b += t[i]
                is_done = False
        num_iter += 1
    return (True, w, b) if is_done else (False, None, None)

def kernel_perceptron(X, t):
    (N, D) = X.shape
    a, b = np.zeros(N), 0
    num_iter, is_done = 0, False
    while not is_done and num_iter < 1000:
        is_done = True
        p = np.random.permutation(N)
        for i in p:
            y = b + np.dot(a, np.dot(X, X[i]))
            if y * t[i] <= 0:
                a[i] += t[i]
                b += t[i]
                is_done = False
        num_iter += 1
    return (True, a, b) if is_done else (False, None, None)

def separate_classes(dataset, algorithm):
    for (class_id, class_name) in dataset.class_names.items():
        print("Separating %s..." % class_name)
        p = np.random.permutation(dataset.labels.size)
        vectors = dataset.vectors[p]
        labels = (dataset.labels[p] == class_id) * 2 - 1
        success, weights, bias = algorithm(vectors, labels)
        if success:
            print("Weights: %s; Bias: %s" % (str(weights), str(bias)))
        else:
            print("Fail")

if __name__ == "__main__":
    iris = IrisDataset()
    # Using standard perceptron
    print("------PERCEPTRON------")
    separate_classes(iris, perceptron)
    print("----------------------")
    # Using kernelized perceptron
    print("--KERNEL--PERCEPTRON--")
    separate_classes(iris, kernel_perceptron)
    print("----------------------")
