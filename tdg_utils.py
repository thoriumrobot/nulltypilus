import json
import networkx as nx
import numpy as np
import logging
import re
from collections import defaultdict
import javalang
import tensorflow as tf
import traceback
import os
import pdb

class JavaTDG:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.classnames = set()

    def add_node(self, node_id, node_type, name, line_number=None, nullable=False, actual_type=None):
        self.graph.add_node(node_id, attr={'type': node_type, 'name': name, 'line_number': line_number, 'nullable': nullable, 'actual_type': actual_type})

    def add_edge(self, from_node, to_node, edge_type):
        self.graph.add_edge(from_node, to_node, type=edge_type)
        self.graph.add_edge(to_node, from_node, type=f"reverse_{edge_type}")

    def add_classname(self, classname):
        self.classnames.add(classname)

class NodeIDMapper:
    def __init__(self):
        self.id_to_int = {}
        self.int_to_id = {}
        self.counter = 0

    def get_int(self, node_id):
        if node_id not in self.id_to_int:
            self.id_to_int[node_id] = self.counter
            self.int_to_id[self.counter] = node_id
            self.counter += 1
        return self.id_to_int[node_id]

    def get_id(self, node_int):
        return self.int_to_id.get(node_int, None)

node_id_mapper = NodeIDMapper()

def split_identifier(name):
    return re.findall(r'[a-z]+|[A-Z][a-z]*|[0-9]+', name)

def extract_features(attr):
    if attr is None:
        logging.warning("Encountered NoneType for attr. Using default values.")
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    type_mapping = {'class': 0, 'method': 1, 'field': 2, 'parameter': 3, 'variable': 4, 'literal': 5, 'file': 6, 'return': 7}
    name_mapping = defaultdict(lambda: len(name_mapping))
    type_name_mapping = defaultdict(lambda: len(type_name_mapping))

    node_type = attr.get('type', '')
    node_name = attr.get('name', '')
    actual_type = attr.get('actual_type', '')
    nullable = float(attr.get('nullable', 0))

    type_id = type_mapping.get(node_type, len(type_mapping))

    subtokens = split_identifier(node_name)
    subtoken_ids = [name_mapping[sub] for sub in subtokens]

    actual_type_id = type_name_mapping[actual_type] if actual_type else -1

    max_subtokens = 3
    padded_subtoken_ids = subtoken_ids[:max_subtokens] + [-1] * (max_subtokens - len(subtoken_ids))

    feature_vector = [
        float(type_id),
        float(padded_subtoken_ids[0]),
        float(padded_subtoken_ids[1]),
        float(padded_subtoken_ids[2]),
        float(actual_type_id),
        nullable
    ]

    return feature_vector

def balance_dataset(features, labels, node_ids, adjacency_matrix, prediction_node_ids):
    # Separate nullable and non-nullable nodes
    nullable_indices = [i for i, label in enumerate(labels) if label == 1.0]
    non_nullable_indices = [i for i, label in enumerate(labels) if label == 0.0]

    # Shuffle to ensure randomness
    np.random.shuffle(non_nullable_indices)

    # Select an equal number of non-nullable nodes
    selected_non_nullable = non_nullable_indices[:len(nullable_indices)]
    balanced_indices = np.concatenate([nullable_indices, selected_non_nullable])

    # Shuffle the selected indices to mix nullable and non-nullable nodes
    np.random.shuffle(balanced_indices)

    # Return the balanced dataset
    balanced_features = features[balanced_indices]
    balanced_labels = labels[balanced_indices]
    balanced_node_ids = node_ids[balanced_indices]
    balanced_adj_matrix = adjacency_matrix[np.ix_(balanced_indices, balanced_indices)]
    balanced_prediction_node_ids = [i for i in prediction_node_ids if i in balanced_indices]

    return balanced_features, balanced_labels, balanced_node_ids, balanced_adj_matrix, balanced_prediction_node_ids

def data_generator(file_list, balance=False, max_nodes=8000):
    graphs = []
    
    pdb.set_trace()

    if balance:
        # Process pre-extracted graphs
        for file_path in file_list:
            try:
                result = load_tdg_data(file_path)
                if len(result) != 5:
                    logging.error(f"Graph from {file_path} returned {len(result)} values. Expected 5. Skipping this graph.")
                    continue

                features, labels, node_ids, adjacency_matrix, prediction_node_ids = result

                features = np.array(features, dtype=np.float32)
                adjacency_matrix = np.array(adjacency_matrix, dtype=np.float32)
                
                # Skip if the graph is empty or invalid
                if features.size == 0 or adjacency_matrix.size == 0:
                    logging.warning(f"Skipping empty or invalid graph in file: {file_path}")
                    continue

                features, labels, node_ids, adjacency_matrix, prediction_node_ids = balance_dataset(features, labels, node_ids, adjacency_matrix, prediction_node_ids)

                graphs.append((features, labels, node_ids, adjacency_matrix, prediction_node_ids))
            except Exception as e:
                logging.error(f"Error processing graph in file {file_path}: {e}")
                continue
    else:
        # Prediction: Combine all Java source code into a single graph
        tdg = JavaTDG()
        for file_path in file_list:
            process_java_file(file_path, tdg)

        # Preprocess the combined graph
        try:
            result = preprocess_tdg(tdg)
            if len(result) != 5:
                logging.error(f"Combined graph returned {len(result)} values. Expected 5. Skipping this graph.")
                return
            
            features, labels, node_ids, adjacency_matrix, prediction_node_ids = map(np.array, result)
            
            if features.size == 0 or adjacency_matrix.size == 0:
                logging.warning(f"Skipping empty or invalid graph in file: {file_path}")
                return
            graphs.append((features, labels, node_ids, adjacency_matrix, prediction_node_ids))
        except Exception as e:
            logging.error(f"Error processing combined graph: {e}")
            return

    # Accumulate and split graphs into batches of max_nodes
    for padded_features, padded_labels, padded_node_ids, padded_adj_matrix, prediction_node_ids in accumulate_and_split_graphs(graphs, max_nodes):
        yield (padded_features, padded_adj_matrix), (padded_labels, prediction_node_ids)

def accumulate_and_split_graphs(graphs, max_nodes=8000):
    accumulated_features = []
    accumulated_labels = []
    accumulated_node_ids = []
    accumulated_adj_matrix = []
    accumulated_prediction_node_ids = []

    current_node_count = 0

    for features, labels, node_ids, adjacency_matrix, prediction_node_ids in graphs:
        G = nx.Graph()  # Convert the directed graph to an undirected graph to identify connected components
        for i in range(adjacency_matrix.shape[0]):
            for j in range(adjacency_matrix.shape[1]):
                if adjacency_matrix[i, j] != 0:
                    G.add_edge(i, j)

        # Get connected components (subgraphs)
        connected_components = list(nx.connected_components(G))

        for component in connected_components:
            component_size = len(component)

            # If the component is too large, partition it into smaller subgraphs
            if component_size > max_nodes:
                subgraphs = partition_large_component(G.subgraph(component), max_nodes)
                for subgraph in subgraphs:
                    handle_component(subgraph, features, labels, node_ids, adjacency_matrix, prediction_node_ids, max_nodes, accumulated_features, accumulated_labels, accumulated_node_ids, accumulated_adj_matrix, accumulated_prediction_node_ids, current_node_count)
            else:
                handle_component(component, features, labels, node_ids, adjacency_matrix, prediction_node_ids, max_nodes, accumulated_features, accumulated_labels, accumulated_node_ids, accumulated_adj_matrix, accumulated_prediction_node_ids, current_node_count)

    if accumulated_features:
        yield pad_batch(accumulated_features, accumulated_labels, accumulated_node_ids, accumulated_adj_matrix, accumulated_prediction_node_ids, max_nodes)

def handle_component(component, features, labels, node_ids, adjacency_matrix, prediction_node_ids, max_nodes, accumulated_features, accumulated_labels, accumulated_node_ids, accumulated_adj_matrix, accumulated_prediction_node_ids, current_node_count):
    component_size = len(component)
    if current_node_count + component_size > max_nodes:
        # If adding this component exceeds max_nodes, yield the current batch
        yield pad_batch(accumulated_features, accumulated_labels, accumulated_node_ids, accumulated_adj_matrix, accumulated_prediction_node_ids, max_nodes)

        # Reset accumulation for the next batch
        accumulated_features.clear()
        accumulated_labels.clear()
        accumulated_node_ids.clear()
        accumulated_adj_matrix.clear()
        accumulated_prediction_node_ids.clear()
        current_node_count = 0

    # Add nodes from the current component
    component_features = features[list(component), :]
    component_labels = labels[list(component)]
    component_node_ids = node_ids[list(component)]
    component_adj_matrix = adjacency_matrix[np.ix_(list(component), list(component))]

    accumulated_features.append(component_features)
    accumulated_labels.append(component_labels)
    accumulated_node_ids.append(component_node_ids)
    accumulated_adj_matrix.append(component_adj_matrix)
    accumulated_prediction_node_ids.append([i for i in prediction_node_ids if i in component])

    current_node_count += component_size

def partition_large_component(G, max_nodes):
    """
    Partition a large connected component into smaller subgraphs using graph partitioning.
    The Metis algorithm from networkx can be used here to minimize edge cuts.
    """
    num_parts = (G.number_of_nodes() // max_nodes) + 1
    partition = nxmetis.partition(G, num_parts)[1]  # This returns a list of sets, each set is a partition
    
    # Convert the partitioned sets into individual subgraphs
    subgraphs = [G.subgraph(part).copy() for part in partition]
    return subgraphs

def pad_batch(features, labels, node_ids, adjacency_matrix, prediction_node_ids, max_nodes):
    feature_dim = 6  # Typilus uses 6-dimensional feature vectors

    # Initialize padded arrays
    padded_features = np.zeros((max_nodes, feature_dim), dtype=np.float32)
    padded_labels = np.zeros((max_nodes,), dtype=np.float32)
    padded_node_ids = np.zeros((max_nodes,), dtype=np.int32)
    padded_adj_matrix = np.zeros((max_nodes, max_nodes), dtype=np.float32)

    # Combine features, labels, node_ids, and adjacency matrices correctly
    combined_features = np.concatenate(features, axis=0)
    combined_labels = np.concatenate(labels, axis=0)
    combined_node_ids = np.concatenate(node_ids, axis=0)

    # Ensure combined features are within max_nodes
    num_nodes = min(combined_features.shape[0], max_nodes)

    # Adjust adjacency matrix
    combined_adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    offset = 0
    for adj in adjacency_matrix:
        size = adj.shape[0]
        if offset + size > max_nodes:
            break  # Stop if adding this adjacency matrix would exceed max_nodes
        combined_adj_matrix[offset:offset+size, offset:offset+size] = adj
        offset += size

    # Flatten prediction_node_ids if it's a list of lists
    flat_prediction_node_ids = [item for sublist in prediction_node_ids for item in sublist]

    # Map the prediction node IDs to integers
    mapped_prediction_node_ids = [node_id_mapper.get_int(node_id) for node_id in flat_prediction_node_ids if node_id in node_id_mapper.id_to_int]

    # Apply the padding to the final batch
    padded_features[:num_nodes, :] = combined_features[:num_nodes]
    padded_labels[:num_nodes] = combined_labels[:num_nodes]
    padded_node_ids[:num_nodes] = combined_node_ids[:num_nodes]
    padded_adj_matrix[:num_nodes, :num_nodes] = combined_adj_matrix

    return padded_features, padded_labels, padded_node_ids, padded_adj_matrix, mapped_prediction_node_ids

def create_tf_dataset(file_list, batch_size, balance=False, is_tdg=True):
    def generator():
        for (features, adjacency_matrix), (labels, prediction_node_ids) in data_generator(file_list, balance, max_nodes=8000):
            features = np.array(features, dtype=np.float32)
            adjacency_matrix = np.array(adjacency_matrix, dtype=np.float32)
            labels = np.array(labels, dtype=np.float32)

            if len(labels.shape) == 1:
                labels = np.expand_dims(labels, -1)

            max_prediction_nodes = 100
            padded_prediction_node_ids = []

            # Ensure prediction_node_ids is always a list or array
            for node_ids in prediction_node_ids:
                if not isinstance(node_ids, (list, np.ndarray)):
                    node_ids = [node_ids]  # If node_ids is a single integer, wrap it in a list

                if len(node_ids) > max_prediction_nodes:
                    node_ids = node_ids[:max_prediction_nodes]  # Truncate if too long
                else:
                    padding_length = max_prediction_nodes - len(node_ids)
                    node_ids = np.pad(node_ids, (0, padding_length), 'constant', constant_values=-1)  # Pad with -1
                
                padded_prediction_node_ids.append(node_ids)

            prediction_node_ids = np.array(padded_prediction_node_ids, dtype=np.int32)

            prediction_mask = np.zeros(features.shape[0], dtype=bool)
            prediction_mask[prediction_node_ids] = True

            yield (features, adjacency_matrix, prediction_mask), labels

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            (tf.TensorSpec(shape=(None, 6), dtype=tf.float32),  # Fixed 6-dimensional feature vectors
             tf.TensorSpec(shape=(None, None), dtype=tf.float32),  # Dynamic adjacency matrix
             tf.TensorSpec(shape=(None,), dtype=tf.bool)),  # Prediction mask
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32)  # Labels
        )
    )

    dataset = dataset.shuffle(buffer_size=10000).padded_batch(
        batch_size,
        padded_shapes=(
            (tf.TensorShape([None, 6]),  # Fixed feature vectors with shape (None, 6)
             tf.TensorShape([None, None]),  # Dynamic adjacency matrix
             tf.TensorShape([None])),  # Prediction mask
            tf.TensorShape([None, 1])  # Labels
        ),
        padding_values=(
            (tf.constant(0.0),  # Padding value for features
             tf.constant(0.0),  # Padding value for adjacency matrix
             tf.constant(False)),  # Padding value for prediction mask
            tf.constant(0.0)  # Padding value for labels
        )
    )
    return dataset

def preprocess_tdg(tdg):
    features = []
    labels = []
    node_ids = []
    prediction_node_ids = []
    all_node_ids = set(tdg.graph.nodes)

    # Convert node_id to integer using node_id_mapper
    node_id_map = {node: node_id_mapper.get_int(node) for node in all_node_ids}

    num_valid_nodes = len(all_node_ids)
    if num_valid_nodes == 0:
        return np.zeros((1, 6)), np.zeros((1,)), np.zeros((1,)), np.zeros((1, 1)), []

    adjacency_matrix = np.zeros((num_valid_nodes, num_valid_nodes), dtype=np.float32)

    for node_id, attr in tdg.graph.nodes(data='attr'):
        if attr is None:
            logging.warning(f"Node {node_id} has no attributes. Skipping.")
            continue

        feature_vector = extract_features(attr)
        node_index = node_id_map[node_id]
        features.append(feature_vector)
        node_ids.append(node_index)

        if attr.get('type') in ['method', 'field', 'parameter']:
            prediction_node_ids.append(node_index)
            labels.append(float(attr.get('nullable', 0)))

    for from_node, to_node in tdg.graph.edges():
        if from_node in all_node_ids and to_node in all_node_ids:
            from_id = node_id_map[from_node]
            to_id = node_id_map[to_node]
            adjacency_matrix[from_id, to_id] = 1.0
            adjacency_matrix[to_id, from_id] = 1.0
    
    features = np.array(features, dtype=np.float32)
    adjacency_matrix = np.array(adjacency_matrix, dtype=np.float32)

    if features.size == 0 or adjacency_matrix.size == 0:
        logging.warning("Skipping empty or invalid graph.")
        return np.zeros((1, 6)), np.zeros((1,)), np.zeros((1,)), np.zeros((1, 1)), []

    return features, np.array(labels, dtype=np.float32), np.array(node_ids, dtype=np.int32), adjacency_matrix, prediction_node_ids

def create_combined_tdg(file_list):
    combined_tdg = JavaTDG()
    file_mappings = {}  # Map node IDs to file paths
    line_number_mapping = {}  # Track line numbers for nodes across files

    for file_path in file_list:
        class_tdg = JavaTDG()
        process_java_file(file_path, class_tdg)

        for node_id, node_data in class_tdg.graph.nodes(data=True):
            # Convert node_id to integer using node_id_mapper
            combined_node_id = node_id_mapper.get_int(node_id)
            
            if 'attr' not in node_data:
                logging.warning(f"Node {node_id} is missing 'attr'. Skipping.")
                continue
            
            combined_tdg.add_node(
                combined_node_id, 
                node_data['attr']['type'], 
                node_data['attr']['name'], 
                line_number=node_data['attr'].get('line_number'), 
                nullable=node_data['attr'].get('nullable'), 
                actual_type=node_data['attr'].get('actual_type')
            )
            
            # Maintain the original file mapping and line numbers
            file_mappings[combined_node_id] = file_path
            line_number_mapping[combined_node_id] = node_data['attr'].get('line_number')
        
        for from_node, to_node, edge_data in class_tdg.graph.edges(data=True):
            combined_tdg.add_edge(
                node_id_mapper.get_int(from_node), 
                node_id_mapper.get_int(to_node), 
                edge_data['type']
            )
    
    return combined_tdg, file_mappings, line_number_mapping

def load_tdg_data(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        tdg = JavaTDG()
        tdg.graph = nx.node_link_graph(data)
        return preprocess_tdg(tdg)
    except Exception as e:
        logging.error(f"Error processing {json_path}: {e}")
        return ([], [], [], [], [])

def has_nullable_annotation(annotations):
    return any(annotation.name == 'Nullable' for annotation in annotations)

def get_actual_type(node):
    if hasattr(node, 'type') and hasattr(node.type, 'name'):
        return node.type.name
    return None

def get_superclass_name(node):
    """
    Extracts the superclass name from a class declaration node, if present.
    """
    if node.extends:
        return node.extends.name
    return None

def process_field_declaration(field, class_id, tdg):
    """
    Processes field declarations and connects them to the TDG.
    """
    for decl in field.declarators:
        field_id = f"{class_id}.{decl.name}"
        line_number = field.position.line if field.position else None
        actual_type = get_actual_type(decl)
        nullable = has_nullable_annotation(field.annotations)
        
        # Add the field to the TDG
        tdg.add_node(field_id, "field", decl.name, line_number=line_number, actual_type=actual_type, nullable=nullable)
        tdg.add_edge(class_id, field_id, "has_field")

        # Handle assignment to field via method call (e.g., field initialization)
        if isinstance(decl.initializer, javalang.tree.MethodInvocation):
            method_call_id = f"{class_id}.{decl.initializer.member}()"
            tdg.add_edge(field_id, method_call_id, "assigned_from_method")

def process_method_invocation(method_id, class_id, method_invocation, tdg):
    """
    Handles method invocations, ensuring they are correctly linked to the TDG.
    """
    called_method_id = f"{class_id}.{method_invocation.member}()"
    tdg.add_edge(method_id, called_method_id, "calls")
    return called_method_id

def process_expression(expression, method_id, class_id, tdg):
    """
    Recursively processes expressions to extract method invocations and variable references.
    """
    # If the expression is a method invocation
    if isinstance(expression, javalang.tree.MethodInvocation):
        method_call_id = process_method_invocation(method_id, class_id, expression, tdg)
        return method_call_id

    # If the expression is a member reference (i.e., a variable)
    if isinstance(expression, javalang.tree.MemberReference):
        referenced_var_id = f"{method_id}.{expression.member}"
        return referenced_var_id

    # Recursively process binary operations (e.g., x + y)
    if isinstance(expression, javalang.tree.BinaryOperation):
        left_result = process_expression(expression.operandl, method_id, class_id, tdg)
        right_result = process_expression(expression.operandr, method_id, class_id, tdg)
        return left_result, right_result

    return None

def process_assignment(statement, method_id, class_id, tdg):
    """
    Processes assignments to variables, handling complex expressions involving method calls or other variables.
    """
    if isinstance(statement, javalang.tree.Assignment):
        assigned_var_id = f"{method_id}.{statement.left.name}"

        # Process the right-hand expression of the assignment (may contain method calls or variables)
        results = process_expression(statement.expression, method_id, class_id, tdg)
        
        # Handle results from expressions (could be method calls or variable references)
        if isinstance(results, tuple):  # If both left and right are processed (binary operations)
            for result in results:
                if result:
                    tdg.add_edge(assigned_var_id, result, "assigned_from_expression")
        elif results:
            tdg.add_edge(assigned_var_id, results, "assigned_from_expression")

        return assigned_var_id
    return None

def process_java_file(file_path, tdg):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        if not content.strip():
            logging.warning(f"File {file_path} is empty, skipping.")
            return

        tree = javalang.parse.parse(content)
        file_name = os.path.basename(file_path)
        logging.info(f"Processing file {file_path}")

        file_id = file_name
        tdg.add_node(file_id, "file", file_name)

        for path, node in tree:
            if isinstance(node, javalang.tree.ClassDeclaration):
                class_id = f"{file_name}.{node.name}"
                line_number = node.position.line if node.position else None
                tdg.add_node(class_id, "class", node.name, line_number=line_number)
                tdg.add_classname(node.name)
                tdg.add_edge(file_id, class_id, "contains")

                # Process each method in the class
                for method in node.methods:
                    method_id = f"{class_id}.{method.name}()"
                    line_number = method.position.line if method.position else None
                    nullable = has_nullable_annotation(method.annotations)
                    tdg.add_node(method_id, "method", method.name, line_number=line_number, nullable=nullable)
                    tdg.add_edge(class_id, method_id, "contains")

                    # Check for overridden methods (inheritance)
                    if any(annotation.name == "Override" for annotation in method.annotations):
                        superclass_name = get_superclass_name(node)
                        if superclass_name:
                            superclass_method_id = f"{superclass_name}.{method.name}()"
                            tdg.add_edge(method_id, superclass_method_id, "overrides")

                    # Add method return value as a node
                    return_id = f"{method_id}.return"
                    tdg.add_node(return_id, "return", "return_value", line_number=line_number)
                    tdg.add_edge(method_id, return_id, "has_return")

                    # Add method parameters and variables
                    for param in method.parameters:
                        param_id = f"{method_id}.{param.name}"
                        line_number = param.position.line if param.position else None
                        actual_type = get_actual_type(param)
                        nullable = has_nullable_annotation(param.annotations)
                        tdg.add_node(param_id, "parameter", param.name, line_number=line_number, actual_type=actual_type, nullable=nullable)
                        tdg.add_edge(method_id, param_id, "has_parameter")

                    # Process method body statements (assignments and standalone method calls)
                    for statement in method.body:
                        # Handle standalone method calls
                        if isinstance(statement, javalang.tree.MethodInvocation):
                            process_method_invocation(method_id, class_id, statement, tdg)

                        # Handle variable assignments (with method calls or variables)
                        process_assignment(statement, method_id, class_id, tdg)

                    # Add variables used in the method as nodes and connect them
                    for local_var in method.body:
                        if isinstance(local_var, javalang.tree.VariableDeclarator):
                            var_id = f"{method_id}.{local_var.name}"
                            line_number = local_var.position.line if local_var.position else None
                            actual_type = get_actual_type(local_var)
                            tdg.add_node(var_id, "variable", local_var.name, line_number=line_number, actual_type=actual_type)
                            tdg.add_edge(method_id, var_id, "has_variable")

                # Process field declarations
                for field in node.fields:
                    process_field_declaration(field, class_id, tdg)

            # Handle top-level method declarations
            elif isinstance(node, javalang.tree.MethodDeclaration):
                method_id = f"{file_name}.{node.name}()"
                line_number = node.position.line if node.position else None
                tdg.add_node(method_id, "method", node.name, line_number=line_number)
                for param in node.parameters:
                    param_id = f"{method_id}.{param.name}"
                    line_number = param.position.line if param.position else None
                    actual_type = get_actual_type(param)
                    nullable = has_nullable_annotation(param.annotations)
                    tdg.add_node(param_id, "parameter", param.name, line_number=line_number, actual_type=actual_type, nullable=nullable)
                    tdg.add_edge(method_id, param_id, "has_parameter")

            # Handle field declarations at the top level
            elif isinstance(node, javalang.tree.FieldDeclaration):
                process_field_declaration(node, file_name, tdg)

            # Handle variables and null literals
            elif isinstance(node, javalang.tree.VariableDeclarator):
                var_id = f"{file_name}.{node.name}"
                line_number = node.position.line if node.position else None
                actual_type = get_actual_type(node)
                tdg.add_node(var_id, "variable", node.name, line_number=line_number, actual_type=actual_type)

            elif isinstance(node, javalang.tree.Literal) and node.value == "null":
                if node.position:
                    null_id = f"{file_name}.null_{node.position.line}_{node.position.column}"
                    tdg.add_node(null_id, "literal", "null", line_number=node.position.line)
                    parent = path[-2] if len(path) > 1 else None
                    parent_id = get_parent_id(file_name, parent)
                    if parent_id:
                        tdg.add_edge(parent_id, null_id, "contains")

    except javalang.parser.JavaSyntaxError as e:
        logging.error(f"Syntax error in file {file_path}: {e}")
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        logging.error(traceback.format_exc())

def get_parent_id(file_name, parent):
    if parent is None:
        return None
    if hasattr(parent, 'name'):
        return f"{file_name}.{parent.name}"
    if isinstance(parent, javalang.tree.MethodInvocation):
        return f"{file_name}.{parent.member}"
    if isinstance(parent, javalang.tree.Assignment):
        if parent.position:
            return f"{file_name}.assignment_{parent.position.line}_{parent.position.column}"
    return None
