import { Routes, Route } from 'react-router-dom';
import StartingPage from './pages/StartingPage';
import MLPipelinePage from './pages/MLPipelinePage';

function App() {
  return (
    <Routes>
      <Route path="/" element={<StartingPage />} />
      <Route path="/ml-pipeline" element={<MLPipelinePage />} />
    </Routes>
  );
}

export default App;