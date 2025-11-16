import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { METRIC_DESCRIPTIONS, METRIC_LABELS } from '../constants/metrics';
import { MetricKey, Reports } from '../types';

interface MetricChartProps {
  reports: Reports;
  metric: MetricKey;
}

const MetricChart = ({ reports, metric }: MetricChartProps) => {
  const data = Object.entries(reports).map(([model, report]) => ({
    model,
    label: report.label ?? model.toUpperCase(),
    value: Number(report.metrics[metric].toFixed(3))
  }));

  return (
    <ResponsiveContainer width="100%" height={320}>
      <BarChart data={data} margin={{ top: 20, left: 8, right: 8, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="label" interval={0} angle={-12} textAnchor="end" height={80} />
        <YAxis domain={[0, 'auto']} />
        <Tooltip content={<MetricTooltip metric={metric} />} />
        <Bar dataKey="value" name={METRIC_LABELS[metric]} fill="#2563eb" radius={[6, 6, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
};

type TooltipPayload = {
  value: number;
  payload: {
    model: string;
    label: string;
  };
};

const MetricTooltip = ({
  active,
  payload,
  metric
}: {
  active?: boolean;
  payload?: TooltipPayload[];
  metric: MetricKey;
}) => {
  if (!active || !payload || payload.length === 0) {
    return null;
  }
  const { label } = payload[0].payload;
  const metricValue = payload[0].value;
  return (
    <div className="metric-tooltip">
      <strong>{label}</strong>
      <div>Value: {metricValue?.toFixed(3)}</div>
      <p>{METRIC_DESCRIPTIONS[metric]}</p>
    </div>
  );
};

export default MetricChart;
