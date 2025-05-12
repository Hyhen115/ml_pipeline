import { useState } from 'react';
import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, Typography, Select, MenuItem, Button } from '@mui/material';

export default function CSVTable({ data, rowsToShow, onSubmit }) {
  const [columnSelections, setColumnSelections] = useState(
    data.length > 0 ? Object.keys(data[0]).reduce((acc, key) => ({ ...acc, [key]: 'Ignore' }), {}) : {}
  );

  const [columnDataTypes, setColumnDataTypes] = useState(
    data.length > 0 ? Object.keys(data[0]).reduce((acc, key) => ({ ...acc, [key]: 'numerical' }), {}) : {}
  );

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

  const handleSubmit = () => {
    const featureCols = Object.keys(columnSelections).filter((col) => columnSelections[col] === 'Feature');
    const labelCol = Object.keys(columnSelections).find((col) => columnSelections[col] === 'Label');

    if (!labelCol || featureCols.length === 0) {
      alert('Please ensure you have selected a label and at least one feature.');
      return;
    }

    // Pass the selected columns and data types to the parent component
    onSubmit({
      featureCols,
      labelCol,
      dataTypes: columnDataTypes,
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
      <Button variant="contained" color="primary" onClick={handleSubmit} sx={{ mt: 2 }}>
        Submit
      </Button>
    </>
  );
}