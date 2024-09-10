import os
import sys
import numpy as np
from tensorflow.keras.models import load_model
import logging
import tensorflow as tf
from tdg_utils import preprocess_tdg, process_java_file, create_combined_tdg, NodeIDMapper, node_id_mapper, name_mapping, type_name_mapping
import pickle

class BooleanMaskLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        output, mask = inputs
        return tf.boolean_mask(output, mask)

    def compute_output_shape(self, input_shape):
        output_shape, mask_shape = input_shape
        return (None, output_shape[-1])

def annotate_file(file_path, annotations, output_file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for annotation in annotations:
        file_name, node_name, line_num = annotation

        if line_num is None:
            logging.warning(f"Line number is None for node {node_name} in file {file_path}")
            continue

        line = lines[line_num - 1]

        if 0 <= line_num - 1 < len(lines) and "@Nullable {node_name}" not in line:
            lines[line_num - 1] = line.replace(node_name, f"@Nullable {node_name}")
        else:
            logging.warning(f"Line number {line_num} is out of range in file {file_path}")

    with open(output_file_path, 'w') as file:
        file.writelines(lines)

def process_project(project_dir, model, output_dir):
    # Create a list of Java files to process
    file_list = [os.path.join(root, file)
                 for root, _, files in os.walk(project_dir)
                 for file in files if file.endswith('.java')]

    # Create a combined TDG for all Java files in the project
    combined_tdg, file_mappings, line_number_mapping = create_combined_tdg(file_list)
    
    # Preprocess the combined TDG
    features, _, node_ids, adjacency_matrix, prediction_node_ids = preprocess_tdg(combined_tdg)

    if features.size == 0 or adjacency_matrix.size == 0:
        logging.warning(f"No valid TDG created for project {project_dir}. Skipping.")
        return

    # Create the prediction mask
    num_nodes = features.shape[0]
    max_nodes = 8000
    feature_dim = features.shape[1] if features.ndim > 1 else 6  # Adjust feature dimension for Typilus

    if num_nodes > max_nodes:
        logging.warning(f"Number of nodes ({num_nodes}) exceeds max_nodes ({max_nodes}). Truncating the graph.")
        num_nodes = max_nodes
        adjacency_matrix = adjacency_matrix[:num_nodes, :num_nodes]
        features = features[:num_nodes, :]

    if num_nodes < max_nodes:
        padded_features = np.zeros((max_nodes, feature_dim), dtype=np.float32)
        padded_adjacency_matrix = np.zeros((max_nodes, max_nodes), dtype=np.float32)
        padded_features[:num_nodes, :] = features[:num_nodes, :]
        padded_adjacency_matrix[:num_nodes, :num_nodes] = adjacency_matrix[:num_nodes, :num_nodes]
        features = padded_features
        adjacency_matrix = padded_adjacency_matrix

    # Prepare the prediction mask
    prediction_mask = np.zeros((max_nodes,), dtype=bool)
    valid_prediction_node_ids = [node_id_mapper.get_int(node_id) for node_id in prediction_node_ids if node_id_mapper.get_int(node_id) < max_nodes]
    prediction_mask[valid_prediction_node_ids] = True

    # Prepare the inputs for the model
    features = np.expand_dims(features, axis=0)
    adjacency_matrix = np.expand_dims(adjacency_matrix, axis=0)
    prediction_mask = np.expand_dims(prediction_mask, axis=0)

    # Run the prediction
    predictions = model.predict([features, adjacency_matrix, prediction_mask])

    if predictions.shape[0] != len(valid_prediction_node_ids):
        logging.error(f"Model output shape {predictions.shape} does not match the expected number of prediction indices.")
        return

    annotations = []
    counter = 0
    for node_index in valid_prediction_node_ids:
        prediction = predictions[counter, 0]
        counter += 1
        if prediction > 0.2:  # Adjust the threshold if needed
            mapped_node_id = node_id_mapper.get_id(node_index)
            if mapped_node_id is None:
                logging.warning(f"Node index {node_index} not found in NodeIDMapper. Skipping.")
                continue

            file_name = file_mappings.get(mapped_node_id)
            if not file_name:
                logging.warning(f"No file mapping found for node_id {mapped_node_id}. Skipping annotation.")
                continue

            line_num = line_number_mapping.get(mapped_node_id, 0)
            if line_num == 0:
                logging.warning(f"Line number for node_id {mapped_node_id} not found. Skipping annotation.")
                continue

            node_name = combined_tdg.graph.nodes[mapped_node_id]['attr']['name']
            annotations.append((file_name, node_name, line_num))

    # Annotate the files based on predictions
    file_names = set([ann[0] for ann in annotations])
    for file_name in file_names:
        base_file_name = os.path.basename(file_name)
        relative_subdir = os.path.relpath(file_name, project_dir)
        output_file_path = os.path.join(output_dir, relative_subdir)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        file_annotations = [ann for ann in annotations if ann[0] == file_name]
        annotate_file(file_name, file_annotations, output_file_path)

    logging.info(f"Annotation complete for project {project_dir}")

def main(project_dir, model_path, output_dir):
    model = load_model(model_path, custom_objects={'BooleanMaskLayer': BooleanMaskLayer})
    load_mappings(os.path.join(os.path.dirname(model_path), 'mappings.pkl'))
    
    node_id_mapper = NodeIDMapper()  # Initialize a new NodeIDMapper

    # Process the entire project
    process_project(project_dir, model, output_dir)

# Load the mappings at the start of the prediction process
def load_mappings(mapping_file):
    global name_mapping, type_name_mapping
    with open(mapping_file, 'rb') as f:
        mappings = pickle.load(f)
        name_mapping = defaultdict(lambda: len(mappings['name_mapping']), mappings['name_mapping'])
        type_name_mapping = defaultdict(lambda: len(mappings['type_name_mapping']), mappings['type_name_mapping'])
    logging.info(f"Loaded mappings from {mapping_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python predict.py <ProjectDir> <ModelPath> <OutputDir>")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)

    project_dir = sys.argv[1]
    model_path = sys.argv[2]
    output_dir = sys.argv[3]

    main(project_dir, model_path, output_dir)
