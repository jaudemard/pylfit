import os 
import pylfit 

from pylfit.datasets.knowledge_network import PKN

if __name__ == "__main__":
    data_path = "./tests/datasets/sandbox.csv"
    targets = ["a","b","c"]
    features = ["a_1","b_1","c_1"]

    dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_csv(path = data_path, feature_names=features, target_names=targets)

    background_knowledge = [
        'a(1) :- b_1(1), c_1(0)',
        'a(0) :- b_1(0), c_1(0)',
        'b(1) :- c_1(1), b_1(0), d_1(0)'
        ]

    pkn = PKN.from_string(rules_list=background_knowledge)

    model = pylfit.models.DMVLP(features=dataset.features, targets=dataset.targets)
    model.compile(algorithm="gula")

    model.fit(dataset=dataset, options = {"background_rules" : pkn.rules, "verbose" : 1})
    # model.compile(algorithm='gula')

    # print(model.summary())




