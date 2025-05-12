import { useState } from 'react';
import { Card, CardContent, Typography, Box, Button, CircularProgress } from '@mui/material';
import Papa from 'papaparse';
import CSVTable from '../components/CSVTable';
import LoadingBox from '../components/LoadingBox'; // Import the new LoadingBox component
import axios from 'axios'; // Add axios for easier multipart/form upload

export default function MLPipelinePage() {
  const [fileName, setFileName] = useState(null);
  const [tableData, setTableData] = useState([]);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false); // <-- Add this
  const [s3Key, setS3Key] = useState(null);
  const rowsToShow = 10;

  const handleFileUpload = async (file) => {
    if (file) {
      setFileName(file.name);

      // Parse for preview
      Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        complete: (result) => {
          setTableData(result.data);
        },
        error: () => {
          alert('Error parsing the CSV file.');
        },
      });

      setUploading(true); // Start uploading
      try {
        const key = await uploadToS3(file);
        setS3Key(key);
        if (key) {
          alert('File uploaded to S3 successfully!');
        } else {
          alert('Failed to upload file to S3.');
        }
      } catch (err) {
        alert('Error uploading file to S3.');
      }
      setUploading(false); // End uploading
    } else {
      setFileName(null);
      setTableData([]);
      setS3Key(null);
    }
  };

  const uploadToS3 = async (file) => {
    // 1. Request presigned POST URL from backend
    const res = await fetch('http://44.210.146.47:5001/generate_upload_url', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename: file.name }),
    });
    const presigned = await res.json();
    if (!presigned.url) {
      alert('Failed to get upload URL');
      return null;
    }
    
    // 2. Prepare form data for S3 upload
    const formData = new FormData();
    Object.entries(presigned.fields).forEach(([k, v]) => formData.append(k, v));
    formData.append('file', file);

    // 3. Upload file to S3
    await axios.post(presigned.url, formData, { headers: { 'Content-Type': 'multipart/form-data' } });

    // 4. Return the S3 key for backend use
    return presigned.fields.key;
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type === 'text/csv') {
      handleFileUpload(file);
    } else {
      alert('Please upload a valid CSV file.');
    }
  };

  const handleInputChange = (event) => {
    const file = event.target.files[0];
    if (file && file.type === 'text/csv') {
      handleFileUpload(file);
    } else {
      alert('Please upload a valid CSV file.');
    }
  };

  const handleSubmit = async ({ featureCols, labelCol, dataTypes }) => {
    if (!fileName || !labelCol || featureCols.length === 0 || !s3Key) {
      alert('Please ensure you have selected a label, features, uploaded a CSV file, and the file is uploaded to S3.');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('http://44.210.146.47:5001/start_pipeline', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          key: s3Key,
          feature_cols: featureCols,
          label_col: labelCol,
          data_types: dataTypes,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        alert(`Error: ${error.error}`);
        setLoading(false);
        return;
      }

      const result = await response.json();
      setResults(result);
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex justify-center items-center bg-gray-100 p-4">
      <Card sx={{ maxWidth: 800, width: '100%', p: 2, boxShadow: 3 }}>
        <CardContent>
          {/* Show Loading Spinner */}
          {loading ? (
            <LoadingBox message="Training the model, please wait..." /> // Use the new LoadingBox component
          ) : uploading ? (
            <Box sx={{ textAlign: 'center', mt: 4 }}>
              <CircularProgress />
              <Typography variant="body1" sx={{ mt: 2 }}>
                Uploading file to S3...
              </Typography>
            </Box>
          ) : results ? (
            // Show Results
            <Box>
              <Typography variant="h5" align="center" gutterBottom>
                Model Training Results
              </Typography>
              <Typography variant="body1" align="center">
                <strong>Best Model:</strong> {results.best_model_name}
              </Typography>
              <Typography variant="body1" align="center">
                <strong>Best RMSE:</strong> {results.best_rmse}
              </Typography>
              <Typography variant="body1" align="center" sx={{ mt: 2 }}>
                <strong>Detailed Results:</strong>
              </Typography>
              <Box sx={{ mt: 2 }}>
                {results.results.map((modelResult, index) => (
                  <Typography key={index} variant="body2" align="center">
                    Model: {modelResult.model}, RMSE: {modelResult.rmse}, R²: {modelResult.r2}
                  </Typography>
                ))}
              </Box>
              {results && results.model_path && (
                <Button
                  variant="contained"
                  color="secondary"
                  sx={{ mt: 2, display: 'block', margin: '20px auto' }}
                  onClick={() => {
                    // Direct download of zip file
                    window.location.href = `http://44.210.146.47:5001/download_model_zip?model_path=${encodeURIComponent(results.model_path)}`;
                  }}
                >
                  Download Trained Model (ZIP)
                </Button>
              )}
            </Box>
          ) : (
            // Show File Upload and Table
            <>
              <Typography variant="h4" component="h1" align="center" gutterBottom>
                ML Pipeline
              </Typography>

              <Typography variant="body1" align="center" color="textSecondary" gutterBottom>
                Please upload your dataset in CSV format to begin processing. You can drag and drop the file into the box below or click to select a file.
              </Typography>

              <Box
                sx={{
                  border: '2px dashed #ccc',
                  borderRadius: '8px',
                  p: 4,
                  textAlign: 'center',
                  mt: 2,
                  cursor: 'pointer',
                  '&:hover': { borderColor: '#1976d2' },
                }}
                onDragOver={handleDragOver}
                onDrop={handleDrop}
              >
                <Typography variant="body2" color="textSecondary">
                  Drag and drop your CSV file here
                </Typography>
                <Typography variant="caption" color="textSecondary">
                  (Only .csv files are supported)
                </Typography>
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleInputChange}
                  style={{ display: 'none' }}
                  id="fileInput"
                />
                <label htmlFor="fileInput">
                  <Button
                    variant="contained"
                    color="primary"
                    sx={{ mt: 2 }}
                    component="span"
                  >
                    Choose File
                  </Button>
                </label>
              </Box>

              {fileName && (
                <Typography variant="body2" align="center" color="textPrimary" sx={{ mt: 2 }}>
                  <strong>Uploaded File:</strong> {fileName}
                </Typography>
              )}

              {tableData.length > 0 && (
                <CSVTable
                  data={tableData}
                  rowsToShow={rowsToShow}
                  onSubmit={handleSubmit} // Pass the handleSubmit function to CSVTable
                />
              )}
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}