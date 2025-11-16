import { Reports } from '../types';

interface ModelInsightsProps {
  reports: Reports;
}

const ModelInsights = ({ reports }: ModelInsightsProps) => {
  const entries = Object.entries(reports);

  return (
    <div className="model-specs-grid">
      {entries.map(([model, report]) => (
        <article className="spec-card" key={model}>
          <header>
            <h2>{report.label}</h2>
            <p className="insight-summary">{report.insight}</p>
          </header>

          <div className="spec-body">
            <div className="insight-section">
              <h3>Key Parameters</h3>
              <ul>
                {report.params.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
            <div className="insight-section">
              <h3>Data Processing</h3>
              <ul>
                {report.processing.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
          </div>
        </article>
      ))}
    </div>
  );
};

export default ModelInsights;
