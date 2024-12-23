
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
# Electra Flask Training Application

This project is a web-based platform designed to train Deep learning models using a **React** frontend and a **Flask** backend. It supports user-provided datasets and allows for the configuration of training parameters. The application provides a visual interface for monitoring results and downloading trained models.

## Prerequisites 📦

Make sure you have the following installed on your system:

1. **Node.js** and **npm** (for the frontend)
   - Download and install from [Node.js official site](https://nodejs.org/)
2. **Python** (version 3.7 or later, for the backend)
   - Install Python from [Python official site](https://www.python.org/)
3. **Pip** (Python package manager)
   - Pip is included by default in Python installations.
4. **Required Python Libraries**
   - Install Flask, NumPy, and other dependencies listed in `requirements.txt`.

## Installation Guide 🛠️

Follow the steps below to set up and run the application.

### 1. Clone the Repository

```bash
git clone https://github.com/your-repository/Electra_flask.git
cd Electra_flask
```

### 2. Install Dependencies

#### Frontend Dependencies

Navigate to the frontend directory and install the required npm packages:

```bash
cd Electra_flask/frontend
npm install
```

#### Backend Dependencies

Navigate back to the root directory and install the Python dependencies:

```bash
cd ../backend
pip install -r requirements.txt
```

## Usage Guide 🚀

### 1. Running the Frontend

To start the React development server:

```bash
cd Electra_flask/frontend
npm run dev
```

The frontend will be available at [http://localhost:5173](http://localhost:5173).

### 2. Running the Backend

To start the Flask server:

```bash
python app.py
```

The backend will be available at [http://localhost:8080](http://localhost:8080).

### 3. Access the Application

Open your browser and navigate to [http://localhost:5173](http://localhost:5173) to use the application.

## Features 🌟

1. **Dataset Upload** 🗂️
   - Users can upload their datasets (CSV format), which are validated and previewed in the app.
2. **Model Training Configuration** ⚙️
   - Configure parameters such as epochs, learning rate, batch size, optimizer, and imbalance handling techniques.
   - Select the target column for training.
3. **Training Results** 📊
   - View training results such as accuracy, predictions, and actual labels.
   - Download the trained model archive.
4. **Visualization** 📈
   - Display results in an intuitive and user-friendly interface.

## Directory Structure 📁

```plaintext
Electra_flask/
│
├── frontend/      # React frontend
│   ├── src/       # Source code for React
│   ├── public/    # Static files
│   ├── package.json  # Frontend dependencies
│   └── ...        # Other React-related files
│
├── app.py         # Flask app main file
├── requirements.txt  # Backend dependencies│
└── README.md      # Project documentation
```

## API Endpoints 🖥️

### `/api/train` (POST)

**Description**: Accepts training parameters and dataset, initiates model training, and returns the results.

#### Request Body:
```json
{
  "training_params": {
    "num_epochs": 3,
    "optimizer_choice": "AdamW",
    "learning_rate": 0.00005,
    "batch_size": 8,
    "target_column": "label"
  },
  "data": [
    {"feature1": 1, "feature2": 0, "label": 1},
    {"feature1": 0, "feature2": 1, "label": 0}
  ]
}
```

#### Response:
```json
{
  "status": "success",
  "final_accuracy": "95.67",
  "predictions": [0, 1, 1, 0],
  "actuals": [0, 1, 1, 1],
  "model_archive": "models/trained_model.zip"
}
```

## Notes 📝

1. **Frontend and Backend Communication** 🔗
   - Ensure that the Flask backend runs on [http://localhost:8080](http://localhost:8080) for the frontend to communicate with it properly.
   
2. **Dataset Format** 🗃️
   - Upload datasets in CSV format with headers matching the required input fields.
   
3. **Model Download** 📥
   - After training, a downloadable link for the model archive (.zip) will be provided in the results page.

## Troubleshooting ⚠️

### Issue: Frontend Not Loading

- Ensure you have run `npm install` in the frontend directory.
- Make sure the frontend is running at [http://localhost:5173](http://localhost:5173) by executing `npm run dev`.

### Issue: Backend Not Responding

- Check that you have installed all dependencies using `pip install -r requirements.txt`.
- Ensure the backend server is running at [http://localhost:8080](http://localhost:8080) by executing `python app.py`.

### Issue: Results Not Displaying

- Confirm that the Flask backend successfully processes the request and returns the expected JSON.
- Verify that the `onTrainingComplete` function correctly passes results to the results visualization page.

### Issue: Module error (accelarate)

- Use
```
pip install -U accelerate
```
## License 📝

This project is licensed under the MIT License.


## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/stanlee47"><img src="https://avatars.githubusercontent.com/u/116173029?v=4?s=100" width="100px;" alt="stanlee47"/><br /><sub><b>stanlee47</b></sub></a><br /><a href="#infra-stanlee47" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/asta-maxx/ELECTRA-Flask/commits?author=stanlee47" title="Tests">⚠️</a> <a href="https://github.com/asta-maxx/ELECTRA-Flask/commits?author=stanlee47" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
