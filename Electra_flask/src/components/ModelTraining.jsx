import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';


function ModelTraining({ data, onTrainingComplete }) {
  const navigate = useNavigate();
  const [trainingParams, setTrainingParams] = useState({
    num_epochs: 3,
    optimizer_choice: 'AdamW',
    learning_rate: 5e-5,
    batch_size: 8,
    warmup_steps: 0,
    weight_decay: 0.01,
    adam_epsilon: 1e-8,
    max_grad_norm: 1.0,
    save_steps: 500,
    logging_steps: 10,
    seed: 42,
    fp16: false,
    evaluation_strategy: 'epoch',
    metrics: ['accuracy'],
    imbalance_technique: 'None',
    target_column: '', // to store selected target column
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [columns, setColumns] = useState([]); // to store column names of the dataset

  useEffect(() => {
    if (data && data.columns) {
      setColumns(data.columns); // Set columns when data is available
    }
  }, [data]);

  const handleParamChange = (param, value) => {
    setTrainingParams(prev => ({
      ...prev,
      [param]: value
    }));
  };

 
    // const formData = new FormData();
    // formData.append('file', fileInput.current.files[0]);  // Assuming you have a file input for CSV file
    // formData.append('training_params', JSON.stringify(trainingParams)); // Append other params as JSON
    // formData.append('imbalance_technique', trainingParams.imbalance_technique);
    // formData.append('handle_imbalance', true); // or false based on the selection

  const handleTraining = async () => {
    if (!trainingParams.target_column) {
      setError('Please select a target column for training.');
      return;
    }
  
    // Ensure the data object contains the necessary CSV data
    if (!data || !data.columns || !data.preview) {
      setError('No valid data available.');
      return;
    }
  
    try {
      setLoading(true);
      setError(null);
  
      // Prepare the request payload
      const payload = {
        training_params: trainingParams,
        imbalance_technique: trainingParams.imbalance_technique,
        handle_imbalance: true, // or false based on the selection
        data: data.fullData, // Pass the entire data object
      };
  
      // Send the request to the Flask backend
      const response = await axios.post('http://localhost:8080/api/train', payload, {
        headers: {
          'Content-Type': 'application/json',
        },
      });
  
      // After getting the response, pass the results to the onTrainingComplete callback
      onTrainingComplete(response.data);
      // Navigate to the results page
      navigate('/results', { state: { results: response.data.results } });
    } catch (error) {
      setError(error.response?.data?.error || 'Error during training');
    } finally {
      setLoading(false);
    }
};

  if (!data) {
    return (
      <div className="text-center">
        <p className="text-xl">Please upload data first</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="bg-white p-6 rounded-lg shadow">
        <h2 className="text-2xl font-bold mb-4">Model Training Configuration</h2>
        
        {/* Data Preview and Target Column Selection */}
        <div className="mb-6">
          <h3 className="text-xl font-semibold">Dataset Preview</h3>
          <div className="overflow-x-auto mt-4">
            <table className="min-w-full divide-y divide-gray-200">
              <thead>
                <tr>
                  {columns.map((column) => (
                    <th
                      key={column}
                      className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider bg-gray-50"
                    >
                      {column}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {data.preview.map((row, idx) => (
                  <tr key={idx} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    {columns.map((column) => (
                      <td
                        key={column}
                        className="px-6 py-4 whitespace-nowrap text-sm text-gray-500"
                      >
                        {row[column]}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Target Column Selection */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700">
            Select Target Column
          </label>
          <select
            value={trainingParams.target_column}
            onChange={(e) => handleParamChange('target_column', e.target.value)}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
          >
            <option value="">-- Select Target Column --</option>
            {columns.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
        </div>

        {/* Training Parameters Configuration */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Number of Epochs
              </label>
              <input
                type="number"
                min="1"
                max="10"
                value={trainingParams.num_epochs}
                onChange={(e) => handleParamChange('num_epochs', parseInt(e.target.value))}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700">
                Optimizer
              </label>
              <select
                value={trainingParams.optimizer_choice}
                onChange={(e) => handleParamChange('optimizer_choice', e.target.value)}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              >
                <option value="AdamW">AdamW</option>
                <option value="SGD">SGD</option>
                <option value="Adam">Adam</option>
                <option value="RMSprop">RMSprop</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700">
                Learning Rate
              </label>
              <input
                type="number"
                step="0.00001"
                min="0.00001"
                max="0.01"
                value={trainingParams.learning_rate}
                onChange={(e) => handleParamChange('learning_rate', parseFloat(e.target.value))}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700">
                Batch Size
              </label>
              <input
                type="number"
                min="1"
                max="64"
                value={trainingParams.batch_size}
                onChange={(e) => handleParamChange('batch_size', parseInt(e.target.value))}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              />
            </div>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Imbalance Handling Technique
              </label>
              <select
                value={trainingParams.imbalance_technique}
                onChange={(e) => handleParamChange('imbalance_technique', e.target.value)}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              >
                <option value="None">None</option>
                <option value="SMOTE">SMOTE</option>
                <option value="ADASYN">ADASYN</option>
                <option value="Random Oversampling">Random Oversampling</option>
                <option value="Random Undersampling">Random Undersampling</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700">
                Evaluation Strategy
              </label>
              <select
                value={trainingParams.evaluation_strategy}
                onChange={(e) => handleParamChange('evaluation_strategy', e.target.value)}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              >
                <option value="no">No</option>
                <option value="steps">Steps</option>
                <option value="epoch">Epoch</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700">
                Evaluation Metrics
              </label>
              <div className="mt-2 space-y-2">
                {['accuracy', 'f1', 'precision', 'recall'].map((metric) => (
                  <label key={metric} className="inline-flex items-center mr-4">
                    <input
                      type="checkbox"
                      checked={trainingParams.metrics.includes(metric)}
                      onChange={(e) => {
                        const newMetrics = e.target.checked
                          ? [...trainingParams.metrics, metric]
                          : trainingParams.metrics.filter(m => m !== metric);
                        handleParamChange('metrics', newMetrics);
                      }}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="ml-2 text-sm text-gray-700">{metric}</span>
                  </label>
                ))}
              </div>
            </div>

            <div>
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={trainingParams.fp16}
                  onChange={(e) => handleParamChange('fp16', e.target.checked)}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span className="ml-2 text-sm text-gray-700">Enable FP16 (Mixed Precision)</span>
              </label>
            </div>
          </div>
        </div>

        <div className="mt-6 flex justify-end space-x-4">
          <button
            onClick={handleTraining}
            disabled={loading}
            className="px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
          >
            {loading ? 'Training...' : 'Train Model'}
          </button>
          
        </div>
        {error && (
          <div className="mt-4 p-4 bg-red-50 text-red-700 rounded-md">
            {error}
          </div>
        )}
      </div>
    </div>
  );
}

export default ModelTraining;