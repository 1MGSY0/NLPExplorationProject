export type MetricKey =
	| 'precision'
	| 'recall'
	| 'f1'
	| 'rouge1'
	| 'rouge2'
	| 'rougeL'
	| 'bleu'
	| 'meteor';

export type ModelKey = 'tfidf' | 'word2vec' | 'gru' | 'bert';

export interface DetailedMetrics {
	precision: number;
	recall: number;
	f1: number;
	rouge1: number;
	rouge2: number;
	rougeL: number;
	bleu: number;
	meteor: number;
}

export interface DataSplit {
	train: number;
	val: number;
	test: number;
}

export interface ModelReport {
	label: string;
	metrics: DetailedMetrics;
	dataSplit: DataSplit;
	params: string[];
	processing: string[];
	insight: string;
}

export type Reports = Record<ModelKey, ModelReport>;

export type SummariesResponse = Partial<Record<ModelKey, string[]>>;
