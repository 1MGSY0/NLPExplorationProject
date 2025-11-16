import { MetricKey } from '../types';

export const METRIC_LABELS: Record<MetricKey, string> = {
  precision: 'Precision',
  recall: 'Recall',
  f1: 'F1',
  rouge1: 'ROUGE-1',
  rouge2: 'ROUGE-2',
  rougeL: 'ROUGE-L',
  bleu: 'BLEU',
  meteor: 'METEOR'
};

export const METRIC_DESCRIPTIONS: Record<MetricKey, string> = {
  precision: 'How many selected sentences were actually relevant.',
  recall: 'How many reference sentences were recovered.',
  f1: 'probability that each sentence is summary worthy.',
  rouge1: 'Unigram overlap with reference highlights.',
  rouge2: 'Bigram overlap that rewards short phrase alignment.',
  rougeL: 'Longest common subsequence which measures summarisation quality',
  bleu: 'Not too useful as it is designed for full sentence translations.',
  meteor: 'meaningful when highlights paraphrase the article'
};
