import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    setPrediction(null);
    setError(null);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!file) {
      setError('Please select an image');
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await axios.post('http://localhost:5000/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setPrediction(response.data);
      setError(null);
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred');
      setPrediction(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Bird Species Classifier</h1>
      <form onSubmit={handleSubmit}>
        <input type="file" accept="image/*" onChange={handleFileChange} />
        <button type="submit" disabled={loading}>
          {loading ? 'Processing...' : 'Predict'}
        </button>
      </form>
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {prediction && (
        <div>
          <h2>Prediction: {prediction.prediction}</h2>
          <p>Confidence: {prediction.confidence}</p>
          {file && (
            <img
              src={URL.createObjectURL(file)}
              alt="Uploaded"
              style={{ maxWidth: '300px' }}
            />
          )}
        </div>
      )}
    </div>
  );
}

export default App;
