"""Module for handling prior knowledge networks.

Still in development and testing.
"""

import libsbml
import os
import collections
import pylfit
import itertools
from typing import Dict, List, Any

class PKN():
    def __init__(self, sbml_path, dataset=None):
        """
        Constructor of prior knowledge's rules based on a SBML-Qual model.
        
        :param sbml_path: Path of the SBML-Qual model.
        :param dataset: a pylift.datasets.Dataset object to constraint 
        rules extraction and check domain definition coherence
        """
        reader = libsbml.SBMLReader()
        self.model = reader.readSBML(sbml_path).getModel()
        self.qual_model = self.model.getPlugin("qual")

        # Check Qual extension
        if self.qual_model is None:
            raise ValueError("Model has no SBML-Qual content.")
        
        if dataset:
            self.dataset = dataset
            self._dataset_features_names = [feature[0] for feature in self.dataset.features]
        
        self.features = self._extract_features()
        self.rules = self._extract_rules()

        if dataset:
            raise NotImplementedError("Match to dataset features has yet to be implemented.")


    def _extract_features(self, dataset=None):
        """
        Extract features and their domain. Return a dict of Id and the corresponding Void Atom

        :param dataset: A pylfit.dataset.Dataset 
        """
        features = {}
        
        for species in self.qual_model.getListOfQualitativeSpecies():
            if dataset:
                if species.getId() not in self._dataset_features_names:
                    continue
                else:
                    # state_position = 
                    raise NotImplementedError("Retrieving features is not implemented yet.")
            else:
                state_position = None


            domain = set([str(i) for i in range(0,species.getMaxLevel()+1)])
            # Create void atoms for each feature
            features[species.getId()] = pylfit.objects.LegacyAtom(species.getId(),
                                        domain,
                                        pylfit.objects.LegacyAtom._VOID_VALUE,
                                        state_position)
        
        return features


    def _parse_condition(self, node):
        """
        Parse a single condition node (comparison operation).
        Returns dict with variable, operator, and value.
        """
        node_type = node.getType()
        
        # Map libSBML node types to operator strings
        op_map = {
            libsbml.AST_RELATIONAL_EQ: 'eq',
            libsbml.AST_RELATIONAL_NEQ: 'neq',
            libsbml.AST_RELATIONAL_GEQ: 'geq',
            libsbml.AST_RELATIONAL_GT: 'gt',
            libsbml.AST_RELATIONAL_LEQ: 'leq',
            libsbml.AST_RELATIONAL_LT: 'lt',
        }
        
        if node_type in op_map:
            left = node.getChild(0)
            right = node.getChild(1)
            
            # Assume left is variable, right is constant
            if left.isName() and right.isInteger():
                return {
                    'variable': left.getName(),
                    'operator': op_map[node_type],
                    'value': right.getInteger()
                }
        
            # If it's different
            raise NotImplementedError(f"Not implemented for,\n",
                                        f"left: {type(left)}\n",
                                        f"right: {type(right)}\n")
    

    def _extract_and_conditions(self, node):
        """
        Extract all conditions from an AND node.
        Returns list of condition dicts.
        """
        branches = None
        if node.getType() == libsbml.AST_LOGICAL_AND:
            conditions = []
            for i in range(node.getNumChildren()):
                child_node = node.getChild(i)
                # Get nested OR
                if child_node.getType() == libsbml.AST_LOGICAL_OR:
                    branches = self._extract_or_branches(child_node)
                    print("OR BRANCHES:", branches)
                # else get the condition
                else:
                    cond = self._parse_condition(node.getChild(i))
                    if cond:
                        conditions.append(cond)

            # Flatten OR branches as multiple 'AND-only' conditions
            if branches is not None:
                all_conditions = []
                for or_branch in branches:
                    all_conditions += [or_branch + conditions]
                return all_conditions
            else:
                return conditions
        
        # Single condition (no AND)
        else:
            cond = self._parse_condition(node)
            return [cond] if cond else []
    

    def _extract_or_branches(self, node):
        """
        Extract all OR branches from a math expression.
        Returns a list of conditions, with only AND relationship.
        """
        if node.getType() == libsbml.AST_LOGICAL_OR:
            branches = []
            # Extract the different conditions in OR
            for i in range(node.getNumChildren()):
                child = node.getChild(i)
                if child.getType() == libsbml.AST_LOGICAL_OR:
                    return NotImplementedError("Found a OR nested in a OR.",
                    "It can't be handled here, and is considered a bad practice.")
                conditions = self._extract_and_conditions(child)
                if conditions:
                    branches.append(conditions)
            return branches
        
        # No OR
        else:
            conditions = self._extract_and_conditions(node)
            if conditions:
                return [conditions]
            else:
                return []
    

    def _condition_to_lfit_body(self, condition):
        """
        Convert a single condition to list of possible atoms.
        For non-equality operators, returns multiple atoms (one per valid value).

        :param condition: A dict with structure, {"variable": .. , "operator": .. , "value": .. }
        """
        var_name = condition['variable']
        operator = condition['operator']
        value = condition['value']
        
        if var_name not in self.features:
            return []
        
        feature = self.features[var_name]
        domain_values = sorted([int(v) for v in feature.domain])
        
        # Determine valid values based on operator
        if operator == 'eq':
            valid_values = [value]
        elif operator == 'neq':
            valid_values = [v for v in domain_values if v != value]
        elif operator == 'geq':
            valid_values = [v for v in domain_values if v >= value]
        elif operator == 'gt':
            valid_values = [v for v in domain_values if v > value]
        elif operator == 'leq':
            valid_values = [v for v in domain_values if v <= value]
        elif operator == 'lt':
            valid_values = [v for v in domain_values if v < value]
        else:
            return []
        
        # Create an atom for each valid value
        atoms = []
        for val in valid_values:
            atom = pylfit.objects.LegacyAtom(var_name, feature.domain, str(val), None)
            atoms.append(atom)
        
        return atoms



    def _extract_rules(self, dataset_feature=None):
        """
        Extract the logical rules of the model.

        If dataset_feature is not none, restrain rules extraction 
        to rules where only the features of 
        the dataset are in the head or body of the rules.
        
        :param dataset_feature: Array of strings, of the features in 
        the dataset, to restrain rules extraction.

        """
        rules = []
        for transition in self.qual_model.getListOfTransitions():

            # Get the target variable, or head, of the rules
            list_of_targets = [output for output in transition.getListOfOutputs()]
            target_id = list_of_targets[0].getQualitativeSpecies()
            target = self.features[target_id] # import the void atom for this feature

            # Get the default level of the head
            list_of_function_terms = transition.getListOfFunctionTerms()
            default_level = list_of_function_terms.getDefaultTerm().getResultLevel()
            
            # Extract rules
            for function_term in transition.getListOfFunctionTerms():
                # Get the head of the rules
                head = pylfit.objects.LegacyAtom(target_id,
                        target.domain,
                        function_term.getResultLevel(),
                        None)
                print("Target: ", target_id, ", level: ", function_term.getResultLevel())

                math_expression = function_term.getMath()

                if math_expression is None:
                    continue
                
                # Extract OR branches where each branch is a list of AND conditions
                branches = self._extract_or_branches(math_expression)

                # Flatten branches from nested OR extraction
                rules_bodies = []
                for item in branches:
                    if item and isinstance(item[0], list):
                        # Nested OR in nested OR is not implemented, and untested
                        if not isinstance(item[0][0], dict):
                            raise NotImplementedError(f"This type of nested OR isn't handled, in transition {transition.getId()}.")
                        rules_bodies.extend(item)
                    else:
                        rules_bodies.append(item)

                
                print(transition.getId())
                print(rules_bodies)
                
                # For each conditions, parse to ensure lfit compliance
                for conditions_set in rules_bodies:

                    print("###New condition set")

                    variants = []
                    simple_condition = []

                    for cond in conditions_set:
                        print("Conditions", cond)
                        atoms_in_cond = self._condition_to_lfit_body(cond)
                        # There's multiple values allowed for the rule
                        if len(atoms_in_cond) > 1:
                            variants.append(atoms_in_cond)
                        # There's one single value of the atom for this rule
                        else:
                            simple_condition += atoms_in_cond
                    
                    # Create a simple rule if it already respect lfit 
                    if len(variants) == 0:
                        body = {atom.variable: atom for atom in simple_condition}
                        rules.append(pylfit.objects.Rule(head, body))
                    
                    # Create a cartesian product of the variant if it doesn't
                    else:
                        body_conditions = [[simple_condition]] + variants
                        for rule_variant in itertools.product(*body_conditions):
                            body = {}
                            for el in rule_variant:
                                if isinstance(el, list):
                                    body = body | {atom.variable:atom for atom in el}
                                else:
                                    body[el.variable] = el
                            rules.append(pylfit.objects.Rule(head,body))
        for rule in rules:
            print(rule)
                        

    def _map_to_dataset(self, dataset):
        """
        Try to map features names to the dataset features.
        """
        return NotImplementedError('yet to be done')

    
if __name__ == "__main__":
    sbml_file = "/home/user/lfit/pkn/toy_data/aghamiri_2020-Executable_file_for_CaSQ_derived_MAPK_model.sbml"
    #sbml_file = "/home/user/lfit/pkn/toy_data/Selvaggio_2020-Microenvironment_control_of_hybrid_Epithelial_Mesenchymal_phenotypes.sbml"
    

    model = PKN(sbml_file)

    # print(model.rules)