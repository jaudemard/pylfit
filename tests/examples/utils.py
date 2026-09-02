from pylfit.datasets import DiscreteStateTransitionsDataset

import random


def add_noise(dataset, noise=0.10):
    """Add noise to a dataset

    Add noise in the transitions of a dataset by randomly flipping its value.

    """

    dataset = dataset.copy()

    noise_f_ind = random.sample(range(0,len(dataset.data)),round(noise*len(dataset.data)))
    noise_t_ind = random.sample(range(0,len(dataset.data)),round(noise*len(dataset.data)))

    _data = []

    for tr_ind, transition in enumerate(dataset.data):

        if tr_ind in noise_f_ind:
            q = int(abs(random.normalvariate(0,1)//1+1))
            if q > len(transition[0]):
                q = len(transition[0])
            index_f = random.sample(range(0,len(transition[0])),q)

            for ind_f in index_f:
                domain = dataset.features[ind_f][1].copy()
                domain.remove(str(transition[0][ind_f]))
                transition[0][ind_f] = random.sample(domain,1)[0]
      
        if tr_ind in noise_t_ind:
            q = int(abs(random.normalvariate(0,1)//1+1))
            if q > len(transition[1]):
                q = len(transition[1])
            index_t = random.sample(range(0,len(transition[1])),q)

            for ind_t in index_t:
                domain = dataset.features[ind_t][1].copy()
                domain.remove(str(transition[1][ind_t]))
                transition[1][ind_t] = random.sample(domain,1)[0]
        _data.append(transition)

    return DiscreteStateTransitionsDataset(_data, dataset.features, dataset.targets)

def subset_dataset(
        dataset: DiscreteStateTransitionsDataset,
        split:float = 0.5,
        seed:int = None
        ) -> DiscreteStateTransitionsDataset:
    """Select a subset of a pylfit dataset observations"""
    if seed:
        random.seed(seed)

    ind = random.sample(range(0,len(dataset.data)), int(split*len(dataset.data)))

    subset_data = [dataset.data[i] for i in ind]

    return DiscreteStateTransitionsDataset(subset_data, dataset.features, dataset.targets)