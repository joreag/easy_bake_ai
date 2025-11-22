import pickle
import os
import sys

# Ensure we can unpickle CognitiveNodes
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cognitive_node import CognitiveNode

class KnowledgeGraphQueryEngine:
    """
    The interface between the raw Knowledge Graph pickle and the
    Diagnostic/Inference tools.
    """
    def __init__(self, graph_path):
        self.nodes = self._load_graph(graph_path)

    def _load_graph(self, path):
        if not os.path.exists(path):
            print(f"[QueryEngine] Error: Graph not found at {path}")
            return {}
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[QueryEngine] Error loading graph: {e}")
            return {}

    def get_all_canonical_names(self):
        """Returns a list of all human-readable labels in the graph."""
        names = []
        for node in self.nodes.values():
            if node.label:
                names.append(node.label)
        return list(set(names))

    def get_solution_rules_by_canonical_name(self, name):
        """
        Searches for a node by label and returns its solution rules 
        or definition to serve as Ground Truth.
        """
        results = []
        name_lower = name.lower()
        for node in self.nodes.values():
            if node.label and node.label.lower() == name_lower:
                props = node.properties
                # Prioritize specific rules, then definitions, then raw answers
                if 'solution_rules' in props:
                    results.extend(props['solution_rules'])
                elif 'definition' in props:
                    results.append(props['definition'])
                elif 'answer' in props:
                    results.append(props['answer'])
                elif 'correct_solution' in props:
                    results.append(props['correct_solution'])
        return results