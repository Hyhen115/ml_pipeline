import React from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';

export default function LoadingBox({ message = "Loading, please wait..." }) {
  return (
    <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '200px' }}>
      <CircularProgress />
      <Typography variant="body1" sx={{ ml: 2 }}>
        {message}
      </Typography>
    </Box>
  );
}