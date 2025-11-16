import { useMemo } from 'react';
import { Cell, Legend, Pie, PieChart, ResponsiveContainer } from 'recharts';
import { Reports } from '../types';

const DATASET_HEADERS = ['id', 'article', 'highlights'];
const DATASET_DESCRIPTION =
  'Articles are split into sentences, cleaned, tokenised, and automatically labelled by checking how similar they are to the highlights.';
const SPLIT_COLORS = ['#2563eb', '#f97316', '#10b981'];

type SplitGroup = {
  id: string;
  train: number;
  val: number;
  test: number;
  models: string[];
};

interface DatasetOverviewProps {
  reports: Reports;
}

const DatasetOverview = ({ reports }: DatasetOverviewProps) => {
  const splitGroups = useMemo<SplitGroup[]>(() => {
    const map = new Map<string, SplitGroup>();

    Object.entries(reports).forEach(([modelKey, report]) => {
      const { train, val, test } = report.dataSplit;
      const key = `${train}-${val}-${test}`;
      const label = report.label ?? modelKey.toUpperCase();

      if (!map.has(key)) {
        map.set(key, { id: key, train, val, test, models: [label] });
      } else {
        map.get(key)!.models.push(label);
      }
    });

    return Array.from(map.values());
  }, [reports]);

  return (
    <section className="dataset-overview">
      <div className="dataset-intro">
        <div>
          <h2>Dataset Overview</h2>
          <p>{DATASET_DESCRIPTION}</p>
        </div>
        <div className="dataset-headers">
          <h3>CSV Headers</h3>
          <ul>
            {DATASET_HEADERS.map((header) => (
              <li key={header}>{header}</li>
            ))}
          </ul>
        </div>
      </div>

      <div className="dataset-splits-grid">
        {splitGroups.map((group) => {
          const total = group.train + group.val + group.test;
          const pieData = [
            { name: 'Train', value: group.train },
            { name: 'Validation', value: group.val },
            { name: 'Test', value: group.test }
          ];

          return (
            <article className="dataset-card" key={group.id}>
              <header>
                <div>
                  <h3>
                    {group.train.toLocaleString()} / {group.val.toLocaleString()} / {group.test.toLocaleString()}
                  </h3>
                  <p>Total documents: {total.toLocaleString()}</p>
                </div>
                <div className="dataset-model-tags">
                  {group.models.map((model) => (
                    <span key={model} className="dataset-tag">
                      {model}
                    </span>
                  ))}
                </div>
              </header>

              <div className="dataset-card-body">
                <ResponsiveContainer width="100%" height={200}>
                  <PieChart>
                    <Pie dataKey="value" data={pieData} innerRadius={50} outerRadius={75} paddingAngle={2}>
                      {pieData.map((_, index) => (
                        <Cell key={`${group.id}-slice-${index}`} fill={SPLIT_COLORS[index]} />
                      ))}
                    </Pie>
                    <Legend verticalAlign="bottom" height={36} />
                  </PieChart>
                </ResponsiveContainer>

                <div className="split-values">
                  {pieData.map((entry, index) => (
                    <div key={entry.name} className="split-pill" style={{ borderColor: SPLIT_COLORS[index] }}>
                      <strong>{entry.name}</strong>
                      <span>
                        {entry.value.toLocaleString()} ({total ? ((entry.value / total) * 100).toFixed(1) : 0}%)
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
};

export default DatasetOverview;
