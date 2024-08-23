
class Benchmark:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def multimodality_score(self, s_l):
        ...

    def frechet_inception_distance_score(self):
        ...

    def diversity_score(self, s_d):
        ...
