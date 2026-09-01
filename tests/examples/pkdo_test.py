"""Testing file for PKDO algorithm."""
from pylfit.preprocessing.tabular_dataset import discrete_state_transitions_dataset_from_csv
from pylfit.algorithms import PKDO
from pylfit.objects import Rule

from utils import add_noise

import pandas as pd
import random
import sys

# Hyper Parameters 
# ---------------------------------------------------------

NOISE = 0

ACCURACY_THRESHOLD = 0.90

OBSERVATION_COVERAGE = 1.00

BACKGROUND_RULES = []

# Dataset Load 
# ---------------------------------------------------------

with open("tests/datasets/mammalian.csv","r") as data_file:
    header = data_file.readline().strip().split(',')

features = [f for f in header if "t_1" in f]
targets = [t for t in header if t not in features]

dataset = discrete_state_transitions_dataset_from_csv(path="tests/datasets/mammalina.csv", feature_names=features, target_names=targets)

# Add noise
noisy_dataset = add_noise(dataset, NOISE)

# Background knowledge rules
# ---------------------------------------------------------

background_rules = [Rule.from_string(rule,noisy_dataset.features, noisy_dataset.targets) for rule in BACKGROUND_RULES]

# PKDO Learning phase
# ---------------------------------------------------------

learned_rules = PKDO.fit(noisy_dataset, background_rules, ACCURACY_THRESHOLD, True)

# Summary
# ---------------------------------------------------------
print("\n\n---------Summary----------\n")
print(f"{len(learned_rules)} rules found.")

for rule in learned_rules:
    print(rule)

real_rules = []
with open("tests/tmp/mammalian_rules.txt","r") as real_rule_file:
    for r_rule in real_rule_file.readlines():
        r_rule = Rule.from_string(r_rule, noisy_dataset.features, noisy_dataset.targets)
        found = False
        for l_rule in background_rules:
            if l_rule.head == r_rule.head:
                found = True
        if found:
            real_rules.append(r_rule)


real_rules_accuracy = []
for rule in real_rules_accuracy:
    positives, negatives = PKDO.interprete(noisy_dataset, rule.head)
    real_rules_accuracy.append(PKDO._accuracy(rule, positives, negatives))

summary = pd.DataFrame({"rule": real_rules, "score": real_rules_accuracy})