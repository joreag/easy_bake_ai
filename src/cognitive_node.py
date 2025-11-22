class CognitiveNode:
    """
    Represents a node in the Knowledge Graph as a pure data container.
    This object is simple and easily serializable (pickle-able).
    """
    def __init__(self, node_id: str, label: str, node_type: str, properties: dict, source_lessons: list):
        self.node_id = node_id
        self.label = label
        self.node_type = node_type
        self.properties = properties
        self.source_lessons = source_lessons
        self.edges = [] # List of {'target': node_id, 'label': relationship}

    def add_edge(self, target_node_id: str, relationship_label: str):
        """Adds a directed edge from this node to another."""
        self.edges.append({"target": target_node_id, "label": relationship_label})

    def __repr__(self):
        return f"CognitiveNode(id={self.node_id}, label='{self.label}')"
