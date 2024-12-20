// App.js
import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import DataUpload from './components/DataUpload';
import ModelTraining from './components/ModelTraining';
import ResultsVisualization from './components/ResultsVisualization';
import Navigation from './components/Navigation';
import VisualizeData from './components/VisualizeData';

function App() {
  const [uploadedData, setUploadedData] = useState(null);
  const [trainingResults, setTrainingResults] = useState(null);

  return (
    <Router>
      <div className="min-h-screen bg-gray-100">
        <Navigation />
        <div className="container mx-auto px-4 py-8">
          <Routes>
            <Route 
              path="/" 
              element={
                <DataUpload 
                  onDataUploaded={setUploadedData} 
                />
              } 
            />
            <Route 
              path="/visualize" 
              element={
                <VisualizeData 
                  data={uploadedData}
                />
              } 
            />
            <Route 
  path="/train" 
  element={
    <ModelTraining 
      data={uploadedData} 
      onTrainingComplete={setTrainingResults} 
    />
  } 
/>
<Route 
  path="/results" 
  element={<ResultsVisualization />} 
/>
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;