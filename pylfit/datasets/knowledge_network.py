"""Module for handling prior knowledge networks."""

import libsbml
import os
import collections
import pylfit
from typing import Dict, List, Any
#from ..objects import LegacyAtom

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
        
        self.features = self._extract_features()
        self.rules = self._extract_rules()

        if dataset:
            raise NotImplementedError("Match to dataset features has yet to be implemented.")


    def _extract_features(self, dataset=None):
        """
        Extract features and their domain. Return a dict of Id and the corresponding Void Atom

        :param dataset: 
        """
        features = {}
        
        for species in self.qual_model.getListOfQualitativeSpecies():
            domain = set([str(i) for i in range(0,species.getMaxLevel()+1)])
            # Create void atoms for each feature
            features[species.getId()] = pylfit.objects.LegacyAtom(species.getId(), domain, pylfit.objects.LegacyAtom._VOID_VALUE, None)
        
        return features


    def _handle_or(self, Dict):
        """
        Create two rule if a OR is in the logical expression.
        """
        return NotImplementedError("yet to be done")
    

    def _ast_dict_to_legacy(self, Dict):
        """
        Transform the dict from parsed AST into lefacy rules.
        """
        return NotImplementedError("yet to be done")

    
    def _parse_ast_node(self, node):
        """
        Recursively parse libSBML ASTNode and convert to dictionary structure.
        
        :param node: libSBML ASTNode object
        :return: Dictionary representing the logical expression
        """
        if node is None:
            return None
        
        node_type = node.getType()
        
        # Handle variable names (ci elements)
        if node_type == libsbml.AST_NAME:
            return {
                'type': 'variable',
                'value': node.getName()
            }
        
        # Handle integer constants (cn elements)
        elif node_type == libsbml.AST_INTEGER:
            return {
                'type': 'constant',
                'value': node.getInteger()
            }
        
        # Handle real constants (shouldn't be seen in sbml-qual but who knows)
        elif node_type == libsbml.AST_REAL:
            return {
                'type': 'constant',
                'value': int(node.getReal())
            }
        
        # Handle relational operators
        elif node_type == libsbml.AST_LOGICAL_AND:
            operands = [self._parse_ast_node(node.getChild(i)) for i in range(node.getNumChildren())]
            return {'op': 'and', 'operands': operands}
        
        elif node_type == libsbml.AST_LOGICAL_OR:
            operands = [self._parse_ast_node(node.getChild(i)) for i in range(node.getNumChildren())]
            return {'op': 'or', 'operands': operands}
        
        elif node_type == libsbml.AST_LOGICAL_NOT:
            return {'op': 'not', 'operands': [self._parse_ast_node(node.getChild(0))]}
        
        elif node_type == libsbml.AST_RELATIONAL_EQ:
            return {
                'op': 'eq',
                'operands': [
                    self._parse_ast_node(node.getChild(0)),
                    self._parse_ast_node(node.getChild(1))
                ]
            }
        
        elif node_type == libsbml.AST_RELATIONAL_NEQ:
            return {
                'op': 'neq',
                'operands': [
                    self._parse_ast_node(node.getChild(0)),
                    self._parse_ast_node(node.getChild(1))
                ]
            }
        
        elif node_type == libsbml.AST_RELATIONAL_GEQ:
            return {
                'op': 'geq',
                'operands': [
                    self._parse_ast_node(node.getChild(0)),
                    self._parse_ast_node(node.getChild(1))
                ]
            }
        
        elif node_type == libsbml.AST_RELATIONAL_GT:
            return {
                'op': 'gt',
                'operands': [
                    self._parse_ast_node(node.getChild(0)),
                    self._parse_ast_node(node.getChild(1))
                ]
            }
        
        elif node_type == libsbml.AST_RELATIONAL_LEQ:
            return {
                'op': 'leq',
                'operands': [
                    self._parse_ast_node(node.getChild(0)),
                    self._parse_ast_node(node.getChild(1))
                ]
            }
        
        elif node_type == libsbml.AST_RELATIONAL_LT:
            return {
                'op': 'lt',
                'operands': [
                    self._parse_ast_node(node.getChild(0)),
                    self._parse_ast_node(node.getChild(1))
                ]
            }
        
        else:
            # Fallback for unsupported node types
            return {'type': 'unknown', 'value': str(node)}
    

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
                target.domain, function_term.getResultLevel(), None)

                math_expression = function_term.getMath()

                if math_expression is None:
                    continue
                
                # Parse the AST node into a manageable dict
                parsed_expr = self._parse_ast_node(math_expression)

        return parsed_expr


    def _map_to_dataset(self, dataset):
        """
        Try to map features names to the dataset features.
        """
        return NotImplementedError('yet to be done')

    
if __name__ == "__main__":
    sbml_file = "/home/user/lfit/pkn/aghamiri_et_al_2020_data/F2 - Executable_file_for_CaSQ_derived_MAPK_model.sbml"

    model = PKN(sbml_file)

    print(model.rules)