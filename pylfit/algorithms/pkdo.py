"""A LFIT algorithm to learn prior knowledge and data optimal model.

Identify a set of rules that fit the prior knowledge and are adapted to the data.
"""

import pylfit
from ..utils import eprint
from ..algorithms.algorithm import Algorithm
from ..objects.legacyAtom import LegacyAtom
from ..datasets import DiscreteStateTransitionsDataset

import itertools

class PKDO (Algorithm):
    """
    Define the Prior Knowledge Data Optimal algorithm.

    INPUT: a list of rules and a set of pairs of discrete states
    OUTPUT: a list of discrete rules
    """

    @staticmethod
    def fit(dataset, rules, verbose=0):
        """
        
        Args:
            dataset: pylfit.datasets.DiscreteStateTransitionsDataset
                state transitions of the system
            rules: list of pylfit.objects.Rule
        """
        if not isinstance(dataset, DiscreteStateTransitionsDataset):
            raise ValueError(f"Dataset type not supported, PKDO expect {str(DiscreteStateTransitionsDataset.__name__)}")

        sorted_rules = {}
        for rule in rules:
            if rule.head.var not in sorted_rules:
                sorted_rules[rule.head.var] = []
            sorted_rules[rule.head.var].append(rule)

        for target in dataset.targets:
            if target[0] not in sorted_rules:
                eprint(f"No prior knowledge rules identified for {target}. No rules learned for the target")

        raise NotImplementedError("fit not implemented")


    @staticmethod
    def fit_thread():
        raise NotImplementedError("fit_thread not implemented")


    @staticmethod
    def interprete(transitions, head):
        """
        Split transitions into positve/negative states for the given head atom
        """
        positives = []
        negatives = []

        for s1, S2 in transitions:
            negative = True
            for s2 in S2:
                if head.matches(s2) or s2[head.state_position] == LegacyAtom._UNKNOWN_VALUE:
                    positives.append(s1)
                    negative = False
                    break
            if negative:
                negatives.append(s1)
                
        return positives, negatives 
    
    @staticmethod
    def fit_var_val():
        raise NotImplementedError("fit_var_val not implemented")

    
    @staticmethod
    def extract_influences(rules):
        """Extract influences from rules as single-atom rules
        
        Args:
            rules: list of pylfit.objects.Rule
                A list of prior knowledge logical rules
        Returns:
            list of pylfit.objects.Rule
                single atoms rules
        """
        raise NotImplementedError("extract_influences not implemented")
        

    

