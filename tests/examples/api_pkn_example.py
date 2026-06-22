import os
import pylfit
import random
import datetime

from pylfit.objects.legacyAtom import LegacyAtom
from pylfit.datasets.knowledge_network import PKN
from pylfit.datasets.discreteStateTransitionsDataset import DiscreteStateTransitionsDataset
from pylfit.preprocessing import discrete_state_transitions_dataset_from_csv

import logging
logger = logging.getLogger(__name__)

def compute_accuracy(targets_true, targets_pred):
    """Compute accuracy"""
    accurate_pred = 0

    total_pred = len(targets_pred[0]) * len(targets_pred)

    for feature_state in targets_true.keys():
        for j, s in enumerate(targets_true[feature_state]):
            if targets_pred[feature_state][j] == s:
                accurate_pred += 1

    return accurate_pred / total_pred

if __name__ == "__main__":
    logging.basicConfig(filename='tests/tmp/pkn_examples.log',
    level=logging.INFO,
    format='%(message)s'
    )
    separator ="\n"+"-"*32+"\n"

    random.seed(64)
    dataset_path = './tests/datasets/mammalian.csv'

    logging.info(f"Start: {datetime.datetime.now()}")
    logging.info(f"Reference dataset: {dataset_path}")
    logging.info(f"seed: 64")

    with open(dataset_path,"r") as data_file:
        header = data_file.readline().strip().split(',')
    features = [feature for feature in header if "t_1" in feature]
    targets = [target for target in header if "t_1" not in target]

    full_dataset = discrete_state_transitions_dataset_from_csv(dataset_path, feature_names=features, target_names=targets)
    all_observed_states = [transition[0] for transition in full_dataset.data]


    dataset_sample_size = [0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.9]
    partial_datasets = []

    logging.info("\nGenerating partial datasets...")
    for sample_size in dataset_sample_size:
        random_index = random.sample(range(0,len(full_dataset.data)),int(sample_size*len(full_dataset.data)))
        _data = [full_dataset.data[i] for i in random_index]

        partial_datasets.append(DiscreteStateTransitionsDataset(_data, full_dataset.features, full_dataset.targets))

    # ----------------------------------------
    # Normal Run
    # ----------------------------------------
    logging.info(f"{separator}Normal run with GULA on reference dataset{separator}")
    
    model = pylfit.models.DMVLP(features=full_dataset.features, targets=full_dataset.targets)
    model.compile(algorithm="gula")
    
    logging.info("\tlaunching learning on all observations from the reference...")
    model.fit(dataset=full_dataset)

    reference_rules = model.rules


    logging.info("\tlaunching prediction on the reference program...")
    targets_true = model.predict([all_observed_states[0]])


    # ----------------------------------------
    # Test 1
    # Expanging generality of the final model by covering unobserved states with prior knowledge
    # ----------------------------------------
    logging.info(f"{separator}Test 1")
    logging.info(f"Expanding generalisability of the program by covering unobserved states with prior knowledge{separator}")    

    rules_percentages = [0.1,0.2,0.3,0.5,1]

    partial_prior_knowledge = []
    for percentage in rules_percentages:
        index = random.sample(range(0,len(reference_rules)),int(percentage*len(reference_rules)))
        partial_prior_knowledge.append([reference_rules[i] for i in index])

    logging.info("\tlaunching learning on partial datasets, as control values...")

    test_1_control_accuracy = []
    for i, partial_dataset in enumerate(partial_datasets):
        logging.info(f"\tcontrol {i+1}, {dataset_sample_size[i]}% of the observations")
        model = pylfit.models.DMVLP(features=partial_dataset.features, targets=partial_dataset.targets)
        model.compile(algorithm="gula")
        model.fit(partial_dataset)
        
        pred = model.predict(all_observed_states)

        accuracy = compute_accuracy(targets_true, pred)

        print(accuracy)

        break