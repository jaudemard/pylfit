import os
import pylfit
import random
import datetime
import numpy as np

from pylfit.objects.legacyAtom import LegacyAtom
from pylfit.datasets.knowledge_network import PKN
from pylfit.datasets.discreteStateTransitionsDataset import DiscreteStateTransitionsDataset
from pylfit.preprocessing import discrete_state_transitions_dataset_from_csv
from pylfit.models.dmvlp import DMVLP
from pylfit.models.wdmvlp import WDMVLP


import logging
logger = logging.getLogger(__name__)

def compute_accuracy(true_target_states, pred_target_states):
    """Compute accuracy"""
    accurate_pred = 0

    total_pred = len(pred_target_states[list(pred_target_states.keys())[0]]) * len(pred_target_states)

    for feature_state in true_target_states.keys():
        
        if len(true_target_states[feature_state].keys()) == 1:
            true_target_state = list(true_target_states[feature_state].keys())[0]
        else:
            raise KeyError("Non-deterministic real target state.")
        
        for j, s in enumerate(true_target_state):
            if str(pred_target_states[feature_state][j]) == str(s):
                accurate_pred += 1

    return accurate_pred / total_pred

# def generate_noise(data):



if __name__ == "__main__":
    logging.basicConfig(filename='tests/tmp/pkn_examples.log',
    level=logging.INFO,
    format='%(message)s',
    filemode='w'
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
    # targets = [target for target in header if "t_1" not in target]
    targets = ['CycA_t']

    full_dataset = discrete_state_transitions_dataset_from_csv(dataset_path, feature_names=features, target_names=targets)
    all_observed_states = [transition[0] for transition in full_dataset.data]


    logging.info("\nSplit dataset into train, validation and tests (64%, 16%, and 20%)...")

    ind = random.sample(range(0,len(full_dataset.data)),int(0.8*len(full_dataset.data)))
    _data = [full_dataset.data[i] for i in ind]
    
    train_ind = random.sample(ind,len(_data))

    train_set = DiscreteStateTransitionsDataset([full_dataset.data[i] for i in train_ind], full_dataset.features, full_dataset.targets)
    # validation_set = DiscreteStateTransitionsDataset([full_dataset.data[i] for i in ind if i not in train_ind], full_dataset.features, full_dataset.targets)

    _data = [full_dataset.data[i] for i in range(0, len(full_dataset.data)) if i not in ind]
    test_set = [transition[0] for transition in _data]

    logging.info("\nGenerating partial datasets...")

    dataset_sample_size = [0.01,0.02,0.05,0.1,0.20]
    partial_datasets = []

    for sample_size in dataset_sample_size:
        ind = random.sample(range(0,len(train_set.data)),int(sample_size*len(train_set.data)))
        _data = [train_set.data[i] for i in ind]

        partial_datasets.append(DiscreteStateTransitionsDataset(_data, full_dataset.features, full_dataset.targets))

    # ----------------------------------------
    # Normal Run
    # ----------------------------------------
    logging.info(f"{separator}Normal run with GULA on reference dataset{separator}")
    
    model = DMVLP(features=full_dataset.features, targets=full_dataset.targets)
    model.compile(algorithm="gula")
    
    logging.info("\tlaunching learning on all observations from the reference...")
    model.fit(dataset=full_dataset)

    reference_rules = model.rules


    logging.info("\tlaunching prediction on the reference program...")
    targets_true = model.predict(test_set)


    # ----------------------------------------
    # Test 1
    # Expanding generality of the final model by covering unobserved states with prior knowledge
    # ----------------------------------------
    logging.info(f"{separator}Test 1")
    logging.info(f"Expanding generalisability of the program by covering unobserved states with prior knowledge{separator}")


    # rules_percentages = [0.1,0.2,0.3,0.5,1]

    # partial_prior_knowledge = []
    # for percentage in rules_percentages:
    #     index = random.sample(range(0,len(reference_rules)),int(percentage*len(reference_rules)))
    #     partial_prior_knowledge.append([reference_rules[i] for i in index])

    # ----------------------------------------
    # Control Run

    logging.info("\tlaunching control...")

    test_1_control_accuracy = []
    for i, partial_dataset in enumerate(partial_datasets):

        logging.info(f"\tcontrol {i+1}, {dataset_sample_size[i]*100}% ({len(partial_dataset.data)})")
        model = WDMVLP(features=partial_dataset.features, targets=partial_dataset.targets)
        model.compile(algorithm="gula")
        model.fit(partial_dataset)
        pred = model.predict(test_set)

        predicted_targets = {}
        for feature_state in pred.keys():
            predictions = pred[feature_state]
            max_score_targets = []
            for target in targets:
                values = list(predictions[target].keys())
                score = [predictions[target][value][0] for value in values]
                max_score_targets.append(values[score.index(max(score))])
            predicted_targets[feature_state] = max_score_targets

        # print("TRUE: ", targets_true)
        # print("PRED: ", predicted_targets)

        accuracy = compute_accuracy(targets_true, predicted_targets)

        logging.info(f"\tAccuracy: {accuracy}")


    # ----------------------------------------
    # PKN Run

    logging.info("\tlaunching PKN integration...")



    test_1_control_accuracy = []
    for i, partial_dataset in enumerate(partial_datasets):

        logging.info(f"\tcontrol {i+1}, {dataset_sample_size[i]*100}% ({len(partial_dataset.data)})")
        model = WDMVLP(features=partial_dataset.features, targets=partial_dataset.targets)
        model.compile(algorithm="gula")
        model.fit(partial_dataset)
        pred = model.predict(test_set)

        predicted_targets = {}
        for feature_state in pred.keys():
            predictions = pred[feature_state]
            max_score_targets = []
            for target in targets:
                values = list(predictions[target].keys())
                score = [predictions[target][value][0] for value in values]
                max_score_targets.append(values[score.index(max(score))])
            predicted_targets[feature_state] = max_score_targets

        # print("TRUE: ", targets_true)
        # print("PRED: ", predicted_targets)

        accuracy = compute_accuracy(targets_true, predicted_targets)

        logging.info(f"\tAccuracy: {accuracy}")
