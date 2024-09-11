from tensorflow.keras.layers import Input, Dense, Dropout, GRU, Lambda
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import Model
import numpy as np
import tensorflow as tf
import logging
from tdg_utils import load_tdg_data, preprocess_tdg, create_tf_dataset, name_mapping, type_name_mapping
from sklearn.model_selection import train_test_split  # Missing import added
import sys
import os
import pickle

class BooleanMaskLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        output, mask = inputs
        return tf.boolean_mask(output, mask)

    def compute_output_shape(self, input_shape):
        output_shape, mask_shape = input_shape
        return (None, output_shape[-1])

def typilus_loss(y_true, y_pred):
    classification_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return classification_loss  

def build_typilus_model(input_dim, max_nodes):
    # Input layers for node features, adjacency matrix, and prediction mask
    node_features_input = Input(shape=(max_nodes, input_dim), name="node_features")
    adj_input = Input(shape=(max_nodes, max_nodes), name="adjacency_matrix")
    prediction_mask = Input(shape=(max_nodes,), dtype=tf.bool, name="prediction_mask")

    def message_passing(x, adj):
        # Perform message passing by multiplying adjacency matrix with node features
        message = tf.matmul(adj, x)
        return message

    # Initial message passing
    x = Lambda(lambda inputs: message_passing(inputs[0], inputs[1]),
               output_shape=(max_nodes, input_dim))([node_features_input, adj_input])
    
    # First GRU layer for message passing update
    x = GRU(256, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    
    # Second message passing and GRU layer
    x = Lambda(lambda inputs: message_passing(inputs[0], inputs[1]),
               output_shape=(max_nodes, 256))([x, adj_input])
    x = GRU(256, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    
    # Third message passing and GRU layer
    x = Lambda(lambda inputs: message_passing(inputs[0], inputs[1]),
               output_shape=(max_nodes, 256))([x, adj_input])
    x = GRU(128, return_sequences=True)(x)
    x = Dropout(0.2)(x)

    # Final dense layer
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Output layer with sigmoid activation
    output = Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

    # Mask the output based on the prediction mask
    masked_output = BooleanMaskLayer()([output, prediction_mask])

    # Create the model
    model = Model(inputs=[node_features_input, adj_input, prediction_mask], outputs=masked_output)

    # Compile the model with Adam optimizer and the custom loss function
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=typilus_loss, metrics=['accuracy'])

    return model

def main(json_output_dir, model_output_path, batch_size):
    file_list = [os.path.join(json_output_dir, file) for file in os.listdir(json_output_dir) if file.endswith('.json')]
    train_files, val_files = train_test_split(file_list, test_size=0.2, random_state=42)

    train_dataset = create_tf_dataset(train_files, batch_size, balance=True, is_tdg=True)
    val_dataset = create_tf_dataset(val_files, batch_size, balance=True, is_tdg=True)

    (sample_feature, sample_adj, prediction_mask), sample_labels = next(iter(train_dataset))
    input_dim = sample_feature.shape[-1] if sample_feature.shape[0] > 0 else 6  
    max_nodes = sample_feature.shape[1] if sample_feature.shape[0] > 0 else 1

    model = build_typilus_model(input_dim, max_nodes)
    
    checkpoint = ModelCheckpoint(
        filepath=model_output_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=25, mode='min')

    history = model.fit(train_dataset, epochs=50, validation_data=val_dataset, callbacks=[checkpoint, early_stopping])
    
    # Call this function at the end of training
    save_mappings(os.path.join(os.path.dirname(model_output_path), 'mappings.pkl'))

def save_mappings(output_path):
    mappings = {
        'name_mapping': dict(name_mapping),
        'type_name_mapping': dict(type_name_mapping)
    }
    with open(output_path, 'wb') as f:
        pickle.dump(mappings, f)
    logging.info(f"Saved mappings to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train_typilus.py <JsonOutputDir> <ModelOutputPath>")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)
    json_output_dir = sys.argv[1]
    model_output_path = sys.argv[2]
    batch_size = 1

    main(json_output_dir, model_output_path, batch_size)
