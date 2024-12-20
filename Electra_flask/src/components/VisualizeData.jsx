import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Line, Scatter, Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend } from 'chart.js';

// Register the necessary Chart.js components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend);

function VisualizeData({ data }) {
  const [columns, setColumns] = useState([]); // Stores column names
  const [plotChoice, setPlotChoice] = useState(''); // Tracks selected plot type
  const [xColumn, setXColumn] = useState(''); // Tracks selected X column
  const [yColumn, setYColumn] = useState(''); // Tracks selected Y column
  const [chartData, setChartData] = useState(null); // Stores chart data
  const [loading, setLoading] = useState(false); // Loading state for the request
  const [error, setError] = useState(null); // Error state

  // Populate columns from the dataset
  useEffect(() => {
    if (data && data.columns) {
      setColumns(data.columns);
    }
  }, [data]);

  // Handle submission to Flask
  const handleSubmit = async () => {
    if (!plotChoice || !xColumn || !yColumn) {
      setError('Please select all required options');
      return;
    }

    setError(null);
    setLoading(true);

    try {
      const response = await axios.post('http://localhost:8080/api/visualize', {
        data: data.fullData, // Pass the dataset's preview rows
        plotType: plotChoice,
        xColumn,
        yColumn,
      });

      if (response.data && response.data.chartData) {
        setChartData(response.data.chartData); // Update chart data
      } else {
        setError('Failed to generate chart. Please try again.');
      }
    } catch (err) {
      setError('Error communicating with the server. Please check your connection.');
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
        <h2 className="text-2xl font-bold mb-4">Visualize Data</h2>

        {/* Data Preview */}
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

        {/* Plot Parameters */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            {/* Plot Type Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700">Choose the Plot</label>
              <select
                onChange={(e) => setPlotChoice(e.target.value)}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              >
                <option value="">Select Plot</option>
                <option value="scatter">Scatter Plot</option>
                <option value="line">Line Plot</option>
                <option value="bar">Bar Chart</option>
              </select>
            </div>

            {/* X Column Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700">Choose the X Column</label>
              <select
                onChange={(e) => setXColumn(e.target.value)}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              >
                <option value="">Select X Column</option>
                {columns.map((column) => (
                  <option key={column} value={column}>
                    {column}
                  </option>
                ))}
              </select>
            </div>

            {/* Y Column Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700">Choose the Y Column</label>
              <select
                onChange={(e) => setYColumn(e.target.value)}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              >
                <option value="">Select Y Column</option>
                {columns.map((column) => (
                  <option key={column} value={column}>
                    {column}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* Submit Button */}
        <div className="mt-6">
          <button
            onClick={handleSubmit}
            className="px-4 py-2 bg-blue-500 text-white rounded-md shadow hover:bg-blue-600"
          >
            Generate Plot
          </button>
        </div>

        {/* Loading and Error States */}
        {loading && <p className="mt-4 text-lg text-blue-500">Generating plot...</p>}
        {error && <p className="mt-4 text-lg text-red-500">{error}</p>}
      </div>

      {/* Display Chart */}
      {chartData && (
        <div className="bg-white p-6 rounded-lg shadow mt-6">
          <h3 className="text-xl font-semibold">Generated Plot</h3>
          <div className="mt-4">
            {plotChoice === 'line' && <Line data={chartData} />}
            {plotChoice === 'scatter' && <Scatter data={chartData} />}
            {plotChoice === 'bar' && <Bar data={chartData} />}
          </div>
        </div>
      )}
    </div>
  );
}

export default VisualizeData;