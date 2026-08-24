"""A LFIT algorithm to learn prior knowledge and data optimal model.

Identify a set of rules that fit the prior knowledge and are adapted to the data.
"""

import pylfit

import itertools
import numpy as np
from typing import Collection, Dict, List, Mapping, Optional, Set, Tuple, Union
import sys

from ..utils import eprint
from .algorithm import Algorithm
from ..objects import LegacyAtom, Rule, Atom
from ..datasets import DiscreteStateTransitionsDataset



class PKDO (Algorithm):
    """Rules learner based on prior knowledge specialization and data."""

    @staticmethod
    def fit(
        dataset: DiscreteStateTransitionsDataset,
        rules: List[Rule], 
        accuracy_threshold: float = None,
        verbose: bool = False
        ) -> Set[Rule]:
        """
        Args:
            dataset: pylfit.datasets.DiscreteStateTransitionsDataset
                state transitions of the system
            rules: list of pylfit.objects.Rule
        """
        if not isinstance(dataset, DiscreteStateTransitionsDataset):
            raise ValueError(f"Dataset type not supported, PKDO expect {str(DiscreteStateTransitionsDataset.__name__)}")

        if not accuracy_threshold or (accuracy_threshold == 0):
            eprint("No maximum discrepancy provided, running GULA with conditions would be faster (and complete).")

        sorted_rules = {}
        for rule in rules:
            if rule.head not in sorted_rules:
                sorted_rules[rule.head] = []
            sorted_rules[rule.head].append(rule)

        pkdo_rules = []

        for head, influences in sorted_rules.items():

            # Check if there is duplicates / domination in the prior -> quality check

            if verbose:
                eprint(f"Start learning for {head} from prior knowledge and dataset.")

            pkdo_rules += PKDO._fit_target_value(dataset, head, sorted_rules[head], accuracy_threshold, verbose)

        return pkdo_rules
                
    @staticmethod
    def interprete(dataset, head):
        """
        Split transitions into positve/negative states for the given head atom
        """
        positives = set(tuple(s1) for s1,s2 in dataset.data if head.matches(s2) or s2[head.state_position] == LegacyAtom._UNKNOWN_VALUE)
        negatives = set(tuple(s1) for s1,s2 in dataset.data if tuple(s1) not in positives)

        return list(positives), list(negatives)

    @staticmethod
    def _rule_key(rule: Rule) -> frozenset:
        """Canonical, hashable key for a rule's body."""
        return frozenset((rule.body[atom].state_position, rule.body[atom].value) for atom in rule.body)
    
    @staticmethod
    def _fit_target_value(
        dataset: DiscreteStateTransitionsDataset,
        head: Tuple[LegacyAtom, str],
        influences: List[Rule],
        accuracy_threshold: float = None,
        verbose: bool = False
        ) -> Set[Rule]:
        """Learns the target minimal rules from prior knowledge specialization
        """
        positives, negatives = PKDO.interprete(dataset, head)

        candidates = influences

        if accuracy_threshold is None:
            raise NotImplementedError

        else:
            final_rules = set()
            seen = set()
            frontier = influences

            while frontier:
                # dedupe this round's frontier against everything seen so far + check for domination [TODO]

                unique_frontier = []
                for rule in frontier:
                    key = PKDO._rule_key(rule)
                    if key in seen:
                        continue
                    seen.add(key)
                    unique_frontier.append(rule)

                if not unique_frontier:
                    break

                candidates = []  # (rule, (optimistic_accuracy, complexity))
                for rule in unique_frontier:
                    accuracy = PKDO._compute_accuracy(rule, positives, negatives)

                    print(rule, accuracy)

                    if accuracy >= accuracy_threshold:
                        if verbose:
                            eprint(f"Accepted (accuracy={accuracy:.3f}): {rule}")
                        final_rules.add(rule)
                        continue

                    # Is there a maximum accuracy [TODO]
                    # opt_acc = PKDO._optimistic_accuracy(rule, positives, negatives)
                    # print(opt_acc)
                    # sys.exit()
                    # if opt_acc < accuracy_threshold:
                    #     if verbose:
                    #         eprint(f"Pruned, unreachable (optimistic={opt_acc:.3f} "
                    #             f"< {accuracy_threshold}): {rule}")
                    #     continue

                    candidates.append((rule,(accuracy,len(rule.body))))

                # test to see if there is duplicates in the candidates [TODO]


                if not candidates:
                    break

                new_candidates = PKDO._pareto_front(candidates)

                print("surviving_rules", new_candidates)

                if verbose and len(new_candidates) < len(candidates):
                    eprint(f"Frontier reduced {len(candidates)} -> "
                        f"{len(new_candidates)} by dominance")

                next_frontier = []
                for rule in new_candidates:
                    next_frontier.extend(PKDO._least_specialization(rule, dataset))
                frontier = next_frontier

        return final_rules

    @staticmethod
    def _dominates(score_a: Tuple[float, int], score_b: Tuple[float, int]) -> bool:
        """True if score_a Pareto-dominates score_b"""
        acc_a, cx_a = score_a
        acc_b, cx_b = score_b
        a_dominate = acc_a >= acc_b and cx_a <= cx_b
        return a_dominate

    @staticmethod
    def _pareto_front(
        scored_rules: List[Tuple[Rule, Tuple[float,int]]]
        ) -> List[Rule]:
        """Keeps only the non-dominated rules from a list of (rule, score)."""
        front = []
        for rule, score in scored_rules:
            dominated = any(
                PKDO._dominates(other_score, score)
                for other_rule, other_score in scored_rules
                if other_rule is not rule
            )
            if not dominated:
                front.append(rule)
        return front


    @staticmethod
    def _compute_ecd_score(rule, negatives, positives, prior_knowledge):
        # e : accuracy ?
        # c : rule complexity
        # d : distance to prior ?
        e = PKDO._compute_accuracy(rule,negatives,positives)
        c = len(rule.body)
        # d = PKDO._compute_distance_to_prior(rule,prior_knowledge)
        return (e,c)

    @staticmethod
    def _compute_distance_to_prior(rule,prior_knowledge):
        raise NotImplementedError

    @staticmethod
    def _least_specialization(
        rule: Rule,
        dataset: DiscreteStateTransitionsDataset
        ) -> List[Rule]:
        """Generates all specialization of a rule."""
        base = [var for var in rule.body]

        specializations = []
        for pos, feature in enumerate(dataset.features):
            var = feature[0]
            domain = set(feature[1])

            if var in base:
                continue
            else:
                for val in domain:
                    new_rule = rule.copy()
                    atom = LegacyAtom(var, domain, val, pos)
                    new_rule.add_condition(atom)
                    specializations.append(new_rule)

        return specializations

    # @staticmethod
    # def _rule_pruning(rule):
    #     pruned = []


    @staticmethod
    def _compute_accuracy(
        rule: Rule,
        positives: List[Tuple[str]],
        negatives: List[Tuple[str]]
        ) -> float:
        """Computes a rule accuracy score.

        Count true positives and true negatives of a rule prediction
        on the observed initial states, to return the ratio of correct
        prediction over the total amount of unique initial states.

        Args:
            rule: a logical rule
            negatives: a list of negative initial states for the rule head
            positives: a list of positive initial states for the rule head
        
        Returns:
            accuracy score
        """
        tn = 0
        tp = 0

        for neg in negatives:
            if not rule.matches(neg):
                tn += 1

        for pos in positives:
            if rule.matches(pos):
                tp += 1
                
        return tn+tp / (len(negatives)+len(positives))