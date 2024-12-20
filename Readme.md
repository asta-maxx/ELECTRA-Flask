Electra Flask Training Application

This project is a web-based platform designed to train machine learning models using a React frontend and a Flask backend. It supports user-provided datasets and allows for configuration of training parameters. The application provides a visual interface for monitoring results and downloading trained models.

Prerequisites

Make sure you have the following installed on your system:
	1.	Node.js and npm
	•	Download and install Node.js from Node.js official site.
	2.	Python (version 3.7 or later)
	•	Install Python from Python official site.
	3.	Pip (Python package manager)
	•	Pip is included by default in Python installations.
	4.	Required Python Libraries
	•	Flask, NumPy, and any other dependencies listed in requirements.txt.

Installation Guide

Follow the steps below to set up and run the application.

1. Clone the Repository

git clone https://github.com/your-repository/Electra_flask.git
cd Electra_flask

2. Install Dependencies

Frontend Dependencies

Navigate to the project directory and install the required npm packages:

cd Electra_flask
npm install

Backend Dependencies

Navigate back to the root directory and install the Python dependencies:

pip install -r requirements.txt

Usage Guide

1. Running the Frontend

To start the React development server:

cd Electra_flask
npm run dev

The frontend will be available at http://localhost:3000.

2. Running the Backend

To start the Flask server:
python app.py

The backend will be available at http://localhost:8080.

3. Access the Application

Open your browser and navigate to http://localhost:3000 to use the application.

Features

	1.	Dataset Upload
  • Users can upload their datasets (CSV format), which are validated and previewed in the app.
	2.	Model Training Configuration
	•	Configure parameters such as epochs, learning rate, batch size, optimizer, and imbalance handling techniques.
	•	Select the target column for training.
	3.	Training Results
	•	View training results such as accuracy, predictions, and actual labels.
	•	Download the trained model archive.
	4.	Visualization
	•	Display results in an intuitive and user-friendly interface.

Directory Structure

Electra_flask/
│
├── frontend/               # React frontend
│   ├── src/                # Source code for React
│   ├── public/             # Static files
│   ├── package.json        # Frontend dependencies
│   └── ...                 # Other React-related files
│
├── app.py                  # Flask app main file
├── requirements.txt
└── README.md               # Project documentation

API Endpoints

/api/train (POST)

Description: Accepts training parameters and dataset, initiates model training, and returns the results.

Request Body:

{
  "training_params": {
    "num_epochs": 3,
    "optimizer_choice": "AdamW",
    "learning_rate": 0.00005,
    "batch_size": 8,
    "target_column": "label",
    ...
  },
  "data": [
    {"feature1": 1, "feature2": 0, "label": 1},
    ...
  ]
}

Response:

{
  "status": "success",
  "final_accuracy": "95.67",
  "predictions": [0, 1, 1, 0],
  "actuals": [0, 1, 1, 1],
  "model_archive": "models/trained_model.zip"
}

Notes

	1.	Frontend and Backend Communication
Ensure that the Flask backend runs on http://localhost:8080 for the frontend to communicate with it properly.
	2.	Dataset Format
Upload datasets in CSV format with headers matching the required input fields.
	3.	Model Download
After training, a downloadable link for the model archive (.zip) will be provided in the results page.

Troubleshooting

Issue: Frontend Not Loading

	•	Ensure you have run npm install in the frontend directory.
	•	Make sure the frontend is running at http://localhost:3000 by executing npm run dev.

Issue: Backend Not Responding

	•	Check that you have installed all dependencies using pip install -r requirements.txt.
	•	Ensure the backend server is running at http://localhost:8080 by executing python app.py.

Issue: Results Not Displaying

	•	Confirm that the Flask backend successfully processes the request and returns the expected JSON.
	•	Verify that the onTrainingComplete function correctly passes results to the results visualization page.

Contributions

Feel free to fork the repository, raise issues, or submit pull requests to improve the project.

License

This project is licensed under the MIT License. See the LICENSE file for details.
