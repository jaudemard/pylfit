"""Module for handling prior knowledge networks.

Still in development and testing.
"""
from ..objects.rule import Rule
from ..objects.legacyAtom import LegacyAtom
from ..datasets.discreteStateTransitionsDataset import DiscreteStateTransitionsDataset


# import libsbml
import itertools
import numpy
from typing import Dict, List, Any

class PKN:
    def __init__(self, rules = [], sbml_path=None, dataset=None):
        """
        Constructor of prior knowledge's rules based on a SBML-Qual model.
        
        :param sbml_path: Path of the SBML-Qual model.
        :param dataset: a pylift.datasets.Dataset object to constraint 
        rules extraction and check domain definition coherence
        """

        self.rules = rules

        if sbml_path:
            reader = libsbml.SBMLReader()
            self.model = reader.readSBML(sbml_path).getModel()
            self.qual_model = self.model.getPlugin("qual")

            # Check Qual extension
            if self.qual_model is None:
                raise ValueError("Model has no SBML-Qual content.")
            
            self.all_features = self._extract_features()
            self.rules = self._extract_rules()
        
        if dataset:
            self.dataset = dataset
            self._dataset_features = dataset
            self._dataset_targets = {var_name: {'domain': var_domain, 'state_position': var_id } for var_id, (var_name, var_domain) in enumerate(self.dataset.targets)}
        
    

        if dataset:
            raise NotImplementedError("Match to dataset features has yet to be implemented.")
        

#--------------
# Constructors
#--------------


    @staticmethod
    def from_tabular(path, source_col='source', target_col='target', relation_col='relation', relation_sign={'+':'+','-':'-'}):
        """
        Create the prior knowledge rules from tabular data.

        Args:
            path: String
            source_col: String or Int
                The source nodes column index or name
            target_col: String or Int
                The target nodes column index or name
            relation_col: String or Int
                The relation (edges) column index or name
            relation_sign: Dict of {String: String}
        Returns:
            PKN        
        """
        

        # return PKN()


    @staticmethod
    def from_string(string_format_rules):
        """
        Create the PKN rules from list of string.        
        """
        features = {}
        targets = {}

        for string_format in string_format_rules:
            tokens = string_format.strip().split(":-")

            if len(tokens[0]) > 0:
                head = tokens[0].split('(')
                if head[0] not in targets:
                    targets[head[0]] = {head[1].strip(')')}
                else:
                    targets[head[0]].update(head[1].strip(')'))
            
            body_string = tokens[1].strip(".").split(',')
            if len(body_string) > 0:
                for var in body_string:
                    condition = var.split("(")
                    if condition[0] not in features:
                        features[condition[0]] = {condition[1].strip(')')}
                    else:
                        features[condition[0]].update(condition[1].strip(')'))

        features = [(var, list(values)) for var, values in features.items()]
        targets = [(var, list(values)) for var, values in targets.items()]

        rules_list = []
        for string_format in string_format_rules:
            rules_list += [Rule.from_string(string_format, features, targets)]
        
        # Create the rules
        return PKN(rules = rules_list)


    @staticmethod
    def fit_dataset(prior_rules_list, dataset):
        """
        Fit the background rules to the dataset, and vice versa.
        """
        targets = {target[0]: target[1] for target in dataset.targets}
        features = {feature[0]: feature[1] for feature in dataset.features}
        new_atoms = {}
        # Get the rules of interest in the PKN to not add useless rule
        prior_rules_list_fitted = []
        for rule in prior_rules_list:
            if rule.head.variable in targets:
                prior_rules_list_fitted.append(rule)
                if rule.head.value not in targets[rule.head.variable]:
                    raise ValueError(f"For prior rule {rule.to_string()}\n{rule.head} value isn't found in the dataset variable domain.")
                
                # Identify unseen in data atoms
                for var in rule.body:
                    if (var not in features) and (var not in new_atoms):
                        new_atoms[var] = rule.body[var].domain
                    elif var in new_atoms:
                        new_atoms[var].update(rule.body[var].value)
                    
        prior_rules_list = prior_rules_list_fitted

        # Assure a minimum of two values in every atoms domains
        _new_atoms = {}
        for atom, domain in new_atoms.items():
            _new_atoms[atom] = domain
            if len(domain) == 1:
                _new_atoms[atom].add(f"not_{list(domain)[0]}")
        new_atoms = _new_atoms

        new_atoms = [(var, list(values)) for var, values in new_atoms.items()]

        if len(new_atoms) > 0:
            _data = []
            _features = dataset.features + (new_atoms)
            # Add unknown value in the transitions for the new atoms
            appendice = numpy.full(len(new_atoms), LegacyAtom._UNKNOWN_VALUE)
            for transition in dataset.data:
                _data.append((numpy.append(transition[0], appendice), transition[1]))
            # Create the adapted dataset
            dataset = DiscreteStateTransitionsDataset(_data, _features, dataset.targets)

        # Re-create rules so they match the dataset features and targets vectors
        prior_rules_list_fitted = []
        for rule in prior_rules_list:
            str_rule = rule.to_string()
            new_rule = Rule.from_string(str_rule, dataset.features, dataset.targets)
            prior_rules_list_fitted.append(new_rule)

        return prior_rules_list_fitted, dataset


    def _extract_features(self, dataset=None):
        """
        Extract features and their domain. Return a dict of Id and the corresponding Void Atom

        :param dataset: A pylfit.dataset.Dataset 
        """
        features = {}
        
        for species in self.qual_model.getListOfQualitativeSpecies():
            # if dataset:
            #     if (species.getId() not in self._dataset_features) or (species.getId() not in self._dataset_targets):
            #         continue
            #     else:
            #         # state_position = 
            #         raise NotImplementedError("Retrieving features is not implemented yet.")
            # else:
            #     state_position = None

            domain = set([str(i) for i in range(0,species.getMaxLevel()+1)])
            # Create void atoms for each feature
            features[species.getId()] = LegacyAtom(species.getId(),
                                        domain,
                                        LegacyAtom._VOID_VALUE,
                                        None)
        
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
        
        if var_name not in self.all_features:
            return []
        
        feature = self.all_features[var_name]
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
            atom = LegacyAtom(var_name, feature.domain, str(val), None)
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
                 
            if self.dataset is not None:
                if target_id not in self._dataset_targets:
                    continue
                else:
                    target = self._dataset_targets[target_id]
            else:  
                target = self.all_features[target_id] # import the void atom for this feature

            # Get the default level of the head
            list_of_function_terms = transition.getListOfFunctionTerms()
            default_level = list_of_function_terms.getDefaultTerm().getResultLevel()
            
            # Extract rules
            for function_term in transition.getListOfFunctionTerms():
                # Get the head of the rules
                head = LegacyAtom(target_id,
                        target.domain,
                        function_term.getResultLevel(),
                        target.state_position)

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
                
                # For each conditions, parse to ensure lfit compliance
                for conditions_set in rules_bodies:

                    variants = []
                    simple_condition = []

                    not_in_background_knowledge = False

                    for cond in conditions_set:
                        atoms_in_cond = self._condition_to_lfit_body(cond)
                        for atom in atoms_in_cond:
                            if self.dataset is not None:
                                if atom.variable not in self._dataset_features:
                                    not_in_background_knowledge = True
                                    break
                        
                        # If one of the atom is not in background knowledge, skip the rule
                        if not_in_background_knowledge:
                            break

                        # There's multiple values allowed for the rule
                        if len(atoms_in_cond) > 1:
                            variants.append(atoms_in_cond)
                        # There's one single value of the atom for this rule
                        else:
                            simple_condition += atoms_in_cond
                    
                    # Create a simple rule if it already respect lfit 
                    if len(variants) == 0:
                        body = {atom.variable: atom for atom in simple_condition}
                        rules.append(Rule(head, body))
                    
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
                            rules.append(Rule(head,body))
        
        return rules
                        

    def _map_to_dataset(self, dataset):
        """
        Try to map features names to the dataset features.
        """
        return NotImplementedError('yet to be done')


    
if __name__ == "__main__":
    sbml_file = "/home/user/lfit/pkn/toy_data/aghamiri_2020-Executable_file_for_CaSQ_derived_MAPK_model.sbml"
    #sbml_file = "/home/user/lfit/pkn/toy_data/Selvaggio_2020-Microenvironment_control_of_hybrid_Epithelial_Mesenchymal_phenotypes.sbml"
    

    model = PKN(sbml_path=sbml_file)

    print(model.rules)

    # print(model.rules)