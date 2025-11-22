import json
import os
import re
import pickle
import math
import sys
import argparse

# Boilerplate for finding 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.cognitive_node import CognitiveNode

class KnowledgeGraphBuilder:
    def __init__(self, pre_graph_path):
        self.raw_lessons = self._load_pre_graph(pre_graph_path)
        self.nodes = {}

    def _load_pre_graph(self, filepath: str):
        try:
            with open(filepath, 'r') as f: return json.load(f)
        except Exception as e: sys.exit(f"FATAL: Could not load pre_graph file: {e}")

    def _generate_unique_id(self, label: str) -> str:
        s = str(label).lower()
        s = re.sub(r'[^a-z0-9\s-]', '', s)
        s = re.sub(r'[\s-]+', '_', s).strip('_')
        return s or "unnamed_concept"

    def build_graph(self):
        if not self.raw_lessons:
            print("No raw lessons to build graph."); return
        
        print(f"\n===== Building Cognitive Nodes (Input size: {len(self.raw_lessons)}) =====")
        
        count_new_schema = 0
        count_old_schema = 0

        for i, item in enumerate(self.raw_lessons):
            try:
                if not isinstance(item, dict): continue

                # --- SCHEMA A: The New "CognitiveNode" Format (SOTA Data) ---
                if 'node_id' in item and 'label' in item and 'properties' in item:
                    # Pass-through directly, it's already formatted perfectly
                    node = CognitiveNode(
                        node_id=item['node_id'],
                        label=item['label'],
                        node_type=item.get('node_type', 'concept'),
                        properties=item['properties'],
                        source_lessons=item.get('source_lessons', [])
                    )
                    self.nodes[item['node_id']] = node
                    count_new_schema += 1

                # --- SCHEMA B: The Original "Lesson" Format (Old Curriculum) ---
                elif 'problem_description' in item:
                    problem_desc = item['problem_description']
                    node_id = self._generate_unique_id(problem_desc)
                    
                    properties = {
                        "level": item.get('level', -1),
                        "canonical_name": problem_desc
                    }
                    properties.update(item.get('properties', {}))

                    # Logic extraction (kept from your original file)
                    hypothesis_templates = item.get("hypothesis_templates", [])
                    training_data = item.get("training_data", [])
                    if isinstance(hypothesis_templates, list) and hypothesis_templates:
                        properties["solution_rules"] = hypothesis_templates

                    node = CognitiveNode(
                        node_id=node_id,
                        label=problem_desc,
                        node_type=item.get('node_type', 'CognitiveConcept'),
                        properties=properties,
                        source_lessons=[f"lesson_{i}"]
                    )
                    self.nodes[node_id] = node
                    count_old_schema += 1

                else:
                    # If it's neither, it might be an unstructured chunk. Skip silently to avoid log spam.
                    continue

            except Exception as e:
                print(f"[WARN] Failed to process item {i}: {e}")

        print(f"Graph Build Summary: {count_new_schema} SOTA Nodes, {count_old_schema} Lesson Nodes.")

        # --- Edge Weaving (Generic) ---
        print(f"\n===== Weaving Edges =====")
        for item in self.raw_lessons:
            # Handle dependencies regardless of schema
            dependencies = item.get('dependencies', [])
            source_id = item.get('node_id') or self._generate_unique_id(item.get('problem_description', ''))
            
            if source_id in self.nodes and isinstance(dependencies, list):
                for dep in dependencies:
                    if not isinstance(dep, str): continue
                    # Try to find the target ID (this is heuristic, as IDs are hashes now)
                    # In a real graph, dependencies should probably be explicit IDs.
                    # For now, we skip complex dependency linking for SOTA data as they are atomic.
                    pass

    def save_graph(self, output_filename='knowledge_graph.pkl'):
        print(f"\nSerializing Knowledge Graph ({len(self.nodes)} nodes) to '{output_filename}'...")
        with open(output_filename, 'wb') as f:
            pickle.dump(self.nodes, f)
        print("Knowledge Graph build complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pre_graph_path')
    parser.add_argument('output_filename')
    args = parser.parse_args()
    builder = KnowledgeGraphBuilder(args.pre_graph_path)
    builder.build_graph()
    builder.save_graph(output_filename=args.output_filename)