import { Container } from '@mui/material';
import StartingCard from '../components/StartingCard';

function StartingPage() {
  return (
    <Container
      maxWidth="sm"
      className="flex justify-center items-center min-h-screen p-4"
    >
      <StartingCard />
    </Container>
  );
}

export default StartingPage;