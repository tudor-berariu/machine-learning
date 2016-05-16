import numpy as np
from os.path import exists
from os import system

class IrisDataset:
    def __init__(self):
        if not exists("iris.data"):
            base_url = \
                    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            dataset_name = "iris/iris.data"
            system("wget %s%s" % (base_url, dataset_name))
        assert(exists("iris.data"))

        # Read examples
        with open("iris.data") as f:
            examples = map(lambda s: s.strip().split(","),
                           filter(lambda s: len(s) > 2,
                                  f.readlines())
                       )
        classes = map(lambda e: e[-1], examples)

        i = 0
        self.ids = {}
        self.class_names = {}
        for class_name in set(classes):
            self.ids[class_name] = i
            self.class_names[i] = class_name
            i += 1
        self.labels = np.array(map(lambda name: self.ids[name], classes))
        self.vectors = np.array(map(lambda e: map(float, e[:-1]), examples))


if __name__ == "__main__":
    iris = IrisDataset()
    print("%d examples from %d classes" % (iris.labels.size, len(iris.ids)))
    for class_name in iris.class_names.values():
        print(class_name)
