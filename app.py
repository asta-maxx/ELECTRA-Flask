import base64
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS,cross_origin
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from transformers import Trainer, TrainingArguments, TrainerCallback
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim import AdamW, SGD, Adam, RMSprop
import shutil
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import os
import matplotlib.pyplot as plt
import tempfile
import io

app = Flask(__name__)
CORS(app)

class ModelTrainer:
    tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator")
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_dir = os.path.join(self.temp_dir, 'model')
        os.makedirs(self.model_dir, exist_ok=True)

    def handle_imbalance(self, train_texts, train_labels, technique, tokenizer=None):
        # Tokenize if tokenizer is passed
        if tokenizer:
            encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='np')
            train_texts = encodings["input_ids"]

        # Handle imbalance using different techniques
        unique_classes, class_counts = np.unique(train_labels, return_counts=True)
        min_samples = class_counts.min()
        n_neighbors = min(5, max(1, min_samples - 1))

        if technique == 'SMOTE':
            smote = SMOTE(k_neighbors=n_neighbors)
            # Ensure all features are float32 during SMOTE resampling
            train_texts = train_texts.astype(np.float32)  # Convert to float32 for compatibility with tensor operations
            train_labels = train_labels.astype(np.float32)  # Convert to float32 if needed
        elif technique == 'ADASYN':
            adasyn = ADASYN(n_neighbors=n_neighbors)
            train_texts, train_labels = adasyn.fit_resample(train_texts, train_labels)
        elif technique == 'Random Oversampling':
            ros = RandomOverSampler()
            train_texts, train_labels = ros.fit_resample(train_texts, train_labels)
        elif technique == 'Random Undersampling':
            rus = RandomUnderSampler()
            train_texts, train_labels = rus.fit_resample(train_texts, train_labels)

        return train_texts, train_labels

    def train_model(self, train_texts, train_labels, training_params):
        label_encoder = LabelEncoder()
        train_labels = label_encoder.fit_transform(train_labels)

        tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator")
        model = ElectraForSequenceClassification.from_pretrained("google/electra-small-discriminator",
                                                                num_labels=len(label_encoder.classes_))

        encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')
        train_encodings, val_encodings, train_labels_split, val_labels_split = train_test_split(
            encodings["input_ids"], train_labels, test_size=0.2, random_state=training_params['seed']
        )

        class TextDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        train_dataset = TextDataset({'input_ids': train_encodings}, train_labels_split)
        val_dataset = TextDataset({'input_ids': val_encodings}, val_labels_split)

        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")

        def compute_metrics(pred):
            logits, labels = pred
            preds = np.argmax(logits, axis=1)
            metrics_results = {'accuracy': accuracy_score(labels, preds)}
            if 'f1' in training_params['metrics']:
                metrics_results['f1'] = f1_score(labels, preds, average='weighted')
            if 'precision' in training_params['metrics']:
                metrics_results['precision'] = precision_score(labels, preds, average='weighted')
            if 'recall' in training_params['metrics']:
                metrics_results['recall'] = recall_score(labels, preds, average='weighted')
            return metrics_results

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=training_params['num_epochs'],
            per_device_train_batch_size=training_params['batch_size'],
            learning_rate=training_params['learning_rate'],
            warmup_steps=training_params['warmup_steps'],
            weight_decay=training_params['weight_decay'],
            adam_epsilon=training_params['adam_epsilon'],
            max_grad_norm=training_params['max_grad_norm'],
            logging_dir='./logs',
            logging_steps=training_params['logging_steps'],
            save_steps=training_params['save_steps'],
            seed=training_params['seed'],
            fp16=training_params['fp16'],
            evaluation_strategy=training_params['evaluation_strategy'],
        )

        class ProgressCallback(TrainerCallback):
            def on_epoch_end(self, args, state, control, **kwargs):
                progress = state.epoch / training_params['num_epochs']
                print(progress)

        if training_params['optimizer_choice'] == 'AdamW':
            optimizer = AdamW(model.parameters(), lr=training_params['learning_rate'])
        elif training_params['optimizer_choice'] == 'SGD':
            optimizer = SGD(model.parameters(), lr=training_params['learning_rate'])
        elif training_params['optimizer_choice'] == 'Adam':
            optimizer = Adam(model.parameters(), lr=training_params['learning_rate'])
        elif training_params['optimizer_choice'] == 'RMSprop':
            optimizer = RMSprop(model.parameters(), lr=training_params['learning_rate'])

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[ProgressCallback()],
            optimizers=(optimizer, None)
        )

        trainer.train()

        model.save_pretrained('./model')
        tokenizer.save_pretrained('./model')

        with open('./model/label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)

        eval_results = trainer.evaluate()

        trainer.model.eval()
        with torch.no_grad():
            val_inputs = val_encodings.to(trainer.model.device)
            logits = trainer.model(val_inputs)["logits"]
            preds = np.argmax(logits.cpu().numpy(), axis=1)

        return eval_results.get("eval_accuracy", None), preds, val_labels_split

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Invalid file format. Only CSV files are allowed'}), 400

    try:
        # Load CSV into a pandas DataFrame
        df = pd.read_csv(file)
        # Convert the DataFrame to JSON
        data_full = df.to_dict(orient='records')  # Full dataset as a list of records (rows)
        columns = list(df.columns)
        preview = df.head(5).to_dict(orient='records')  # First 5 rows as a preview
        shape = df.shape

        # Return JSON response with the full data and metadata
        return jsonify({
            'columns': columns,
            'preview': preview,
            'shape': shape,
            'fullData': data_full  # Full dataset
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/train', methods=['POST'])
def train():
    try:
        # Ensure the temp directory exists
        if not os.path.exists('temp'):
            os.makedirs('temp')

        # Retrieve the JSON data from the request
        payload = request.get_json()
        if not payload:
            return jsonify({'error': 'No data provided'}), 400

        # Extract training parameters and the dataset from the request payload
        training_params = payload.get('training_params')
        imbalance_technique = payload.get('imbalance_technique')
        data = payload.get('data')

        if not training_params or not data:
            return jsonify({'error': 'Missing training parameters or data'}), 400

        # Load the dataset into a DataFrame
        df = pd.DataFrame(data)

        # Separate features and target column
        target_column = training_params['target_column']
        train_texts = (
            df.drop(columns=[target_column])
              .fillna("")  # Replace NaN values with empty strings
              .astype(str)
              .agg(' '.join, axis=1)
              .tolist()
        )
        train_labels = df[target_column].tolist()
        label_encoder = LabelEncoder()
        train_labels = label_encoder.fit_transform(train_labels)

        # Initialize the model trainer
        trainer = ModelTrainer()

        # Handle class imbalance if necessary
        if imbalance_technique != 'None':
            tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator")
            encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='np')
            input_ids = encodings["input_ids"]
            train_texts, train_labels = trainer.handle_imbalance(input_ids, train_labels, imbalance_technique)

            # Convert resampled numeric IDs back to strings for downstream use
            train_texts = tokenizer.batch_decode(train_texts, skip_special_tokens=True)

        # Train the model
        final_accuracy, preds, actuals = trainer.train_model(train_texts, train_labels, training_params)

        # Create a zip file of the model
        model_archive_path = os.path.join(trainer.temp_dir, 'model_archive')
        shutil.make_archive(model_archive_path, 'zip', trainer.model_dir)
        print({
         "status": "success",
    "final_accuracy": f"{final_accuracy:.2f}" if final_accuracy is not None else "Accuracy not available.",
    "predictions": preds.tolist() if isinstance(preds, np.ndarray) else preds,
    "actuals": actuals.tolist() if isinstance(actuals, np.ndarray) else actuals,
    "model_archive": f"{model_archive_path}.zip"
    })
        # Return response with results
        return jsonify({
    "status": "success",
    "final_accuracy": f"{final_accuracy:.2f}" if final_accuracy is not None else "Accuracy not available.",
    "predictions": preds.tolist() if isinstance(preds, np.ndarray) else preds,
    "actuals": actuals.tolist() if isinstance(actuals, np.ndarray) else actuals,
    "model_archive": f"{model_archive_path}.zip"
})
    

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/visualize', methods=['POST'])
def visualize():
    # Get the request data
    request_data = request.json

    # Validate required parameters
    if not all(key in request_data for key in ["data", "plotType", "xColumn", "yColumn"]):
        return jsonify({"error": "Missing required parameters"}), 400

    data = request_data["data"]
    plot_type = request_data["plotType"]
    x_column = request_data["xColumn"]
    y_column = request_data["yColumn"]

    # Prepare the data for Chart.js
    x_values = [row[x_column] for row in data]
    y_values = [row[y_column] for row in data]
    x_values.sort()
    y_values.sort()
    # Structure the data for the Chart.js
    chart_data = {
        "labels": x_values,
        "datasets": [{
            "label": f"{x_column} vs {y_column}",
            "data": y_values,
            "borderColor": "rgba(75, 192, 192, 1)",
            "backgroundColor": "rgba(75, 192, 192, 0.2)",
            "fill": False,
        }]
    }

    # Return the data to the frontend
    return jsonify({
        "plotType": plot_type,
        "chartData": chart_data
    })

@app.route('/api/download-model', methods=['GET'])
def download_model():
    try:
        model_path = os.path.join('temp', 'model_archive.zip')
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model file not found.'}), 404
        return send_file(
            model_path,
            as_attachment=True,
            download_name='model.zip'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
     app.run(debug=False ,port=8080,use_reloader=False)