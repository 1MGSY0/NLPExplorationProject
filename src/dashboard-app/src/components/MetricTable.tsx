import { METRIC_LABELS } from '../constants/metrics';
import { Reports, MetricKey } from '../types';

interface MetricTableProps {
  reports: Reports;
}

const formatValue = (value: number) => value.toFixed(3);

const MetricTable = ({ reports }: MetricTableProps) => {
  const rows = Object.entries(reports);

  return (
    <div className="table-wrapper">
      <table>
        <thead>
          <tr>
            <th>Model</th>
            {Object.entries(METRIC_LABELS).map(([key, label]) => (
              <th key={key}>{label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(([model, report]) => (
            <tr key={model}>
              <td>{report.label ?? model}</td>
              {(Object.keys(METRIC_LABELS) as MetricKey[]).map((metric) => (
                <td key={metric}>{formatValue(report.metrics[metric])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default MetricTable;
