import { FormEvent, useMemo, useState } from 'react';
import { ModelKey, SummariesResponse } from '../types';

const MODELS: ModelKey[] = ['tfidf', 'word2vec', 'gru', 'bert'];
const DEFAULT_TOP_K = 3;

type TokenMetrics = {
  precision: number;
  recall: number;
  f1: number;
  overlap: number;
  predictionTokens: number;
  referenceTokens: number;
};

const tokenize = (text: string) => (text.toLowerCase().match(/[a-z0-9']+/g) ?? []);

const countTokens = (tokens: string[]) => {
  const counts = new Map<string, number>();
  tokens.forEach((token) => counts.set(token, (counts.get(token) ?? 0) + 1));
  return counts;
};

const computeTokenMetrics = (prediction: string, reference: string): TokenMetrics | null => {
  const predictionTokens = tokenize(prediction);
  const referenceTokens = tokenize(reference);
  if (!predictionTokens.length || !referenceTokens.length) {
    return null;
  }
  const predCounts = countTokens(predictionTokens);
  const refCounts = countTokens(referenceTokens);
  let overlap = 0;
  predCounts.forEach((count, token) => {
    if (refCounts.has(token)) {
      overlap += Math.min(count, refCounts.get(token) ?? 0);
    }
  });
  const precision = overlap / predictionTokens.length;
  const recall = overlap / referenceTokens.length;
  const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
  return {
    precision,
    recall,
    f1,
    overlap,
    predictionTokens: predictionTokens.length,
    referenceTokens: referenceTokens.length,
  };
};

const SummarisationDemo = () => {
  const [article, setArticle] = useState('');
  const [reference, setReference] = useState('');
  const [results, setResults] = useState<SummariesResponse | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'error'>('idle');
  const [errorMessage, setErrorMessage] = useState('');

  const referenceProvided = reference.trim().length > 0;

  const metrics = useMemo(() => {
    if (!results || !referenceProvided) {
      return null;
    }
    const referenceText = reference.trim();
    const aggregated: Partial<Record<ModelKey, TokenMetrics | null>> = {};
    MODELS.forEach((model) => {
      const prediction = results[model]?.join(' ') ?? '';
      aggregated[model] = computeTokenMetrics(prediction, referenceText);
    });
    return aggregated as Record<ModelKey, TokenMetrics | null>;
  }, [referenceProvided, reference, results]);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!article.trim()) {
      setErrorMessage('Please paste an article first.');
      setStatus('error');
      return;
    }

    setStatus('loading');
    setErrorMessage('');
    setResults(null);

    try {
      const response = await fetch('http://localhost:8000/summarise', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ article, top_k: DEFAULT_TOP_K })
      });

      if (!response.ok) {
        throw new Error(`Backend responded with ${response.status}`);
      }

      const data = (await response.json()) as SummariesResponse;
      setResults(data);
      setStatus('idle');
    } catch (error) {
      setStatus('error');
      setErrorMessage(error instanceof Error ? error.message : 'Unknown error');
    }
  };

  return (
    <section className="card">
      <h1>Summarisation Demo</h1>
      <form onSubmit={handleSubmit} className="demo-form">
        <div className="form-field">
          <label htmlFor="article-input">Article</label>
          <textarea
            id="article-input"
            className="article-input"
            placeholder="Paste or type an article..."
            value={article}
            onChange={(event) => setArticle(event.target.value)}
          />
        </div>

        <div className="form-field">
          <label htmlFor="reference-input">Reference highlight (optional)</label>
          <textarea
            id="reference-input"
            className="reference-input"
            placeholder="Provide a highlight to unlock quick overlap metrics..."
            value={reference}
            onChange={(event) => setReference(event.target.value)}
          />
        </div>

        <button type="submit" disabled={status === 'loading'}>
          {status === 'loading' ? 'Generating…' : 'Generate summaries'}
        </button>
      </form>

      {status === 'error' && <p className="error">{errorMessage}</p>}
      {status === 'loading' && <p className="notice">Contacting backend…</p>}

      {results && (
        <>
          {referenceProvided && metrics && (
            <div className="metrics-table-wrapper">
              <table className="metrics-table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1</th>
                    <th>Overlap</th>
                    <th>Pred tokens</th>
                    <th>Ref tokens</th>
                  </tr>
                </thead>
                <tbody>
                  {MODELS.map((model) => {
                    const metric = metrics[model];
                    return (
                      <tr key={model}>
                        <td>{model.toUpperCase()}</td>
                        <td>{metric ? metric.precision.toFixed(3) : '—'}</td>
                        <td>{metric ? metric.recall.toFixed(3) : '—'}</td>
                        <td>{metric ? metric.f1.toFixed(3) : '—'}</td>
                        <td>{metric ? metric.overlap : '—'}</td>
                        <td>{metric ? metric.predictionTokens : '—'}</td>
                        <td>{metric ? metric.referenceTokens : '—'}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          <div className="results-grid">
            {MODELS.map((model) => (
              <article key={model} className="result-card">
                <h3>{model.toUpperCase()}</h3>
                {results[model]?.length ? (
                  <ol className="summary-list">
                    {results[model]!.map((sentence, index) => (
                      <li key={index}>{sentence}</li>
                    ))}
                  </ol>
                ) : (
                  <p className="notice">No sentences returned.</p>
                )}
              </article>
            ))}
          </div>
        </>
      )}
    </section>
  );
};

export default SummarisationDemo;
