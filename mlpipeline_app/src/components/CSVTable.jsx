import { useState } from 'react';
import { 
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow, 
  Paper, Typography, Select, MenuItem, Button, Box, Slider 
} from '@mui/material';

export default function CSVTable({ data, rowsToShow, onSubmit }) {
  const [columnSelections, setColumnSelections] = useState(
    data.length > 0 ? Object.keys(data[0]).reduce((acc, key) => ({ ...acc, [key]: 'Ignore' }), {}) : {}
  );

  const [columnDataTypes, setColumnDataTypes] = useState(
    data.length > 0 ? Object.keys(data[0]).reduce((acc, key) => ({ ...acc, [key]: 'numerical' }), {}) : {}
  );
  
  // Change to use decimal format internally (0.8 instead of 80)
  const [trainSplit, setTrainSplit] = useState(0.8);

  const handleSelectionChange = (column, value) => {
    setColumnSelections((prev) => {
      const updatedSelections = { ...prev };

      // Ensure only one column can be "Label"
      if (value === 'Label') {
        Object.keys(updatedSelections).forEach((key) => {
          if (updatedSelections[key] === 'Label') {
            updatedSelections[key] = 'Ignore'; // Reset the previous "Label" column
          }
        });
      }

      updatedSelections[column] = value;
      return updatedSelections;
    });
  };

  const handleDataTypeChange = (column, value) => {
    setColumnDataTypes((prev) => ({
      ...prev,
      [column]: value,
    }));
  };
  
  // Update slider handler to work with decimals
  const handleTrainSplitChange = (event, newValue) => {
    // Convert percentage to decimal (slider shows percentages, but we store decimals)
    setTrainSplit(newValue / 100);
  };

  const handleSubmit = () => {
    const featureCols = Object.keys(columnSelections).filter((col) => columnSelections[col] === 'Feature');
    const labelCol = Object.keys(columnSelections).find((col) => columnSelections[col] === 'Label');

    if (!labelCol || featureCols.length === 0) {
      alert('Please ensure you have selected a label and at least one feature.');
      return;
    }

    // Pass the selected columns, data types, and train/test split to the parent component
    onSubmit({
      featureCols,
      labelCol,
      dataTypes: columnDataTypes,
      trainTestSplit: trainSplit // Now this will be a decimal value (0.8, etc.)
    });
  };

  // Check if a "Label" is already selected
  const isLabelSelected = Object.values(columnSelections).includes('Label');

  if (data.length === 0) {
    return null; // If no data, don't render the table
  }

  return (
    <>
      <Typography variant="body2" align="center" color="textSecondary" sx={{ mt: 2 }}>
        Showing the first {rowsToShow} rows of the dataset:
      </Typography>
      
      {/* Updated Train/Test Split Slider Box */}
      <Box 
        sx={{ 
          mt: 3, 
          mb: 3, 
          p: 2, 
          borderRadius: 2,
          bgcolor: 'rgba(25, 118, 210, 0.05)', 
          border: '1px solid rgba(25, 118, 210, 0.2)'
        }}
      >
        <Typography variant="subtitle1" gutterBottom fontWeight="medium">
          Data Split Configuration
        </Typography>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          Adjust the ratio of training data to testing data:
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
          <Typography variant="body2" sx={{ minWidth: '45px', fontWeight: 'bold' }}>
            {/* Display as percentage to user */}
            {Math.round(trainSplit * 100)}%
          </Typography>
          <Slider
            // Convert decimal to percentage for the slider
            value={trainSplit * 100}
            onChange={handleTrainSplitChange}
            aria-labelledby="train-test-split-slider"
            valueLabelDisplay="auto"
            step={5}
            marks={[
              { value: 50, label: '50%' },
              { value: 70, label: '70%' },
              { value: 80, label: '80%' },
              { value: 90, label: '90%' }
            ]}
            min={50}
            max={90}
            sx={{ mx: 2 }}
          />
          <Typography variant="body2" sx={{ color: 'text.secondary', minWidth: '130px' }}>
            {/* Display both formats for clarity */}
            Train: {Math.round(trainSplit * 100)}% ({trainSplit.toFixed(2)})
          </Typography>
        </Box>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
          API will receive value: {trainSplit.toFixed(2)}
        </Typography>
      </Box>
      
      <TableContainer component={Paper} sx={{ mt: 2, maxHeight: 400 }}>
        <Table stickyHeader>
          <TableHead>
            {/* Row for Feature/Label/Ignore Selection */}
            <TableRow>
              {Object.keys(data[0]).map((key) => (
                <TableCell key={key} align="center" sx={{ py: 1 }}>
                  <Select
                    value={columnSelections[key]}
                    onChange={(e) => handleSelectionChange(key, e.target.value)}
                    size="small"
                    sx={{ fontSize: '0.875rem', minWidth: 80 }}
                  >
                    <MenuItem value="Feature">Feature</MenuItem>
                    <MenuItem value="Label" disabled={isLabelSelected && columnSelections[key] !== 'Label'}>
                      Label
                    </MenuItem>
                    <MenuItem value="Ignore">Ignore</MenuItem>
                  </Select>
                </TableCell>
              ))}
            </TableRow>
            {/* Row for Data Type Selection */}
            <TableRow>
              {Object.keys(data[0]).map((key) => (
                <TableCell key={key} align="center" sx={{ py: 1 }}>
                  <Select
                    value={columnDataTypes[key]}
                    onChange={(e) => handleDataTypeChange(key, e.target.value)}
                    size="small"
                    sx={{ fontSize: '0.875rem', minWidth: 80 }}
                  >
                    <MenuItem value="numerical">numerical</MenuItem>
                    <MenuItem value="categorical">categorical</MenuItem>
                  </Select>
                </TableCell>
              ))}
            </TableRow>
            {/* Row for Column Headers */}
            <TableRow>
              {Object.keys(data[0]).map((key) => (
                <TableCell key={key} align="center" sx={{ py: 1 }}>
                  {key}
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {data.slice(0, rowsToShow).map((row, index) => (
              <TableRow key={index}>
                {Object.values(row).map((value, i) => (
                  <TableCell key={i} align="center" sx={{ py: 1 }}>
                    {value}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      <Button 
        variant="contained" 
        color="primary" 
        onClick={handleSubmit} 
        sx={{ mt: 2 }}
      >
        Submit
      </Button>
    </>
  );
}