import os 
import pylfit 
import argparse
import random
from pylfit.objects.legacyAtom import LegacyAtom

from pylfit.datasets.knowledge_network import PKN
from pylfit.datasets.discreteStateTransitionsDataset import DiscreteStateTransitionsDataset


def generate_dataset(data_path):
        with open(data_path,"r") as data:
            header = data.readline().strip().split(',')
        features = [feature for feature in header if 't_1' in feature]
        targets = [target for target in header if 't_1' not in target]

        dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_csv(path = data_path, feature_names=features, target_names=targets)

        return dataset


def test_observation_partiality(dataset,background_rules):
    """Remove observations in the dataset that are covered by the prior rules.
    
    Create a testing dataset to check the model ability to retrieve relations
    from unobserved states by using prior knowledge rules.
    """
    kept_data = []
    for rule in background_rules:
        for transition in dataset.data:
            match = rule.partial_matches(transition[0], unknown_values = '?')
            if match == 0:
                 kept_data.append(transition)

    _dataset = DiscreteStateTransitionsDataset(kept_data, dataset.features, dataset.targets) 
    
    return _dataset


def random_select_obs(dataset,percentage):
    random_index = random.sample(range(0,len(dataset.data)),int(percentage*len(dataset.data)))
    _data = [dataset.data[i] for i in random_index]

    return DiscreteStateTransitionsDataset(_data, dataset.features, dataset.targets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("pkn integration development")
    parser.add_argument('type')
    parser.add_argument('-t', '--test', default='0')
    parser.add_argument('-c','--control', action='store_true')

    args = parser.parse_args()

    if args.type == "mammalian":
        dataset = generate_dataset('./tests/datasets/mammalian.csv')

        # compile
        model = pylfit.models.DMVLP(features=dataset.features, targets=dataset.targets)
        model.compile(algorithm="gula")

        # fit
        if args.test:
            # Test the first scenario
            if args.test == '1':
                print("\nScenario 1: expanding generability of the final model by covering unobserved states with prior knowledge.")
                if args.control:
                    print('Control mode: will not integrate prior knowledge')
                background_knowledge = ['E2F_t(0) :- CycA_t_1(1), p27_t_1(0).']

                # Fit prior rule to dataset
                pkn = PKN.from_string(string_format_rules=background_knowledge)
                pkn.rules, dataset = PKN.fit_dataset(pkn.rules, dataset)

                new_dataset = random_select_obs(dataset,0.5)

                print(f"\nLength of the original dataset: {len(dataset.data)}")
                print(f"Length of the testing dataset: {len(new_dataset.data)}") 

            if args.control:
                model.fit(dataset=new_dataset, options = {"background_rules" : [], "verbose" : 0})
                rules_file = f"./tests/tmp/mammalian_rules_test_{args.test}_control.txt"
            else:
                # Fit the model
                model.fit(dataset=new_dataset, options = {"background_rules" : pkn.rules, "verbose" : 0})
                rules_file = f"./tests/tmp/mammalian_rules_test_{args.test}.txt"


            # Save the rules
            with open(rules_file, "w+") as learned_rules:
                for rule in model.rules:
                    learned_rules.write(f"{rule}\n")
        
        # Run without testing
        else:
            model.fit(dataset=dataset, options = {"background_rules" : [],"verbose" : 1})
            with open("./tests/tmpt/mammalian_rules_no_test.txt","w+") as learned_rules:
                for rule in model.rules:
                    learned_rules.write(f"{rule}\n")
        
    
    elif args.type == "sandbox":
        data_path = "./tests/datasets/sandbox.csv"
        targets = ["a","b","c"]
        features = ["a_1","b_1","c_1"]

        dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_csv(path = data_path, feature_names=features, target_names=targets)

        background_knowledge = [
            'a(1) :- b_1(1), c_1(0)',
            'a(0) :- b_1(0), c_1(0)',
            'b(1) :- c_1(1), b_1(0), d_1(0)',
            'b(1) :- b_1(1), e_1(0)'
            ]

        pkn = PKN.from_string(string_format_rules=background_knowledge)

        model = pylfit.models.DMVLP(features=dataset.features, targets=dataset.targets)
        model.compile(algorithm="gula")

        model.fit(dataset=dataset, options = {"background_rules" : pkn.rules, "verbose" : 1})
        # model.fit(dataset=dataset, options = {"verbose" : 1})
        # model.compile(algorithm='gula')

        print(model.summary())




