import { useNavigate } from 'react-router-dom';
import { Card, CardContent, Typography, Button } from '@mui/material';
import { AddCircleOutline } from '@mui/icons-material';

function StartingCard() {
  const navigate = useNavigate();

  return (
    <Card
      sx={{
        backgroundColor: 'rgba(255, 255, 255, 0.1)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(255, 255, 255, 0.2)',
        borderRadius: '12px',
        boxShadow: '0 4px 20px rgba(0, 0, 0, 0.2)',
        width: '100%',
        maxWidth: 500,
        textAlign: 'center',
        padding: 4,
      }}
    >
      <CardContent>
        <Typography
          variant="h4"
          component="h1"
          gutterBottom
          sx={{ color: 'white', textShadow: '1px 1px 2px rgba(0,0,0,0.5)' }}
        >
          Welcome to the MLPipeline App
        </Typography>
        <Typography
          variant="body1"
          color="white"
          paragraph
          sx={{ textShadow: '1px 1px 2px rgba(0,0,0,0.5)' }}
        >
          Optimize and accelerate your machine learning pipeline with modern tools and seamless integration.
        </Typography>
        <Button
          variant="contained"
          color="primary"
          startIcon={<AddCircleOutline />}
          onClick={() => navigate('/ml-pipeline')}
        >
          Get Started
        </Button>
      </CardContent>
    </Card>
  );
}

export default StartingCard;
