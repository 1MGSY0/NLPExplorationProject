import { useEffect, useState } from 'react';
import DatasetOverview from '../components/DatasetOverview';
import ModelInsights from '../components/ModelInsights';
import MetricChart from '../components/MetricChart';
import MetricTable from '../components/MetricTable';
import { METRIC_LABELS } from '../constants/metrics';
import { MetricKey, Reports } from '../types';

const METRIC_OPTIONS: { value: MetricKey; label: string }[] = Object.entries(METRIC_LABELS).map(([value, label]) => ({
  value: value as MetricKey,
  label
}));

const EvaluationDashboard = () => {
  const [reports, setReports] = useState<Reports | null>(null);
  const [selectedMetric, setSelectedMetric] = useState<MetricKey>('f1');
  const [status, setStatus] = useState<'idle' | 'loading' | 'error'>('idle');
  const [errorMessage, setErrorMessage] = useState('');

  useEffect(() => {
    let isMounted = true;
    const loadReports = async () => {
      setStatus('loading');
      try {
        const response = await fetch('/reports.json');
        if (!response.ok) {
          throw new Error('Unable to load reports.json');
        }
        const data = (await response.json()) as Reports;
        if (isMounted) {
          setReports(data);
          setStatus('idle');
        }
      } catch (error) {
        if (isMounted) {
          setStatus('error');
          setErrorMessage(error instanceof Error ? error.message : 'Unknown error');
        }
      }
    };

    loadReports();

    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <section className="card">
      <h1>Evaluation Dashboard</h1>
      <p>Compare across extractive summarisation on the CNN/DailyMail dataset</p>

      {status === 'loading' && <p className="notice">Loading reports…</p>}
      {status === 'error' && <p className="error">{errorMessage}</p>}

      {reports && (
        <>
          <DatasetOverview reports={reports} />

          <ModelInsights reports={reports} />

          <MetricTable reports={reports} />

          <div className="metric-controls">
            <label htmlFor="metric-select">Bar chart metric:</label>
            <select
              id="metric-select"
              value={selectedMetric}
              onChange={(event) => setSelectedMetric(event.target.value as MetricKey)}
            >
              {METRIC_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          <MetricChart reports={reports} metric={selectedMetric} />
        </>
      )}
    </section>
  );
};

export default EvaluationDashboard;
