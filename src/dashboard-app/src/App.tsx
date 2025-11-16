import { Route, Routes } from 'react-router-dom';
import NavBar from './components/NavBar';
import EvaluationDashboard from './pages/EvaluationDashboard';
import SummarisationDemo from './pages/SummarisationDemo';

const App = () => {
  return (
    <>
      <NavBar />
      <main className="container">
        <Routes>
          <Route path="/" element={<EvaluationDashboard />} />
          <Route path="/demo" element={<SummarisationDemo />} />
        </Routes>
      </main>
    </>
  );
};

export default App;
