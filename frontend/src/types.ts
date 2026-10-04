// Tipos de los JSON estáticos generados por GitHub Actions (src/pipeline/export.py).

export type Num = number | null;

export interface Freshness {
  generated_at: string;
  data_as_of: string | null;
  data_source: string | null;
}

export interface IndexSummary {
  symbol: string;
  price: Num;
  change_1d: Num;
  ret21: Num;
  ret63: Num;
  rsi14: Num;
  trend: string;
}

export interface OpportunityRow {
  symbol: string;
  name: string | null;
  sector: string | null;
  price: Num;
  change_1d: Num;
  statistical_score: Num;
  news_score: Num;
  fundamental_score: Num;
  neural_score: Num;
  final_score: Num;
  opportunity_probability: Num;
  confidence: Num;
  risk_score: Num;
  risk_level: string | null;
  rank: Num;
  is_opportunity: boolean;
  signal_label: string;
  in_tracking: boolean;
  summary: string | null;
  last_update: string | null;
}

export type TradeStatus = "PENDING" | "OPEN" | "CLOSED" | "INVALIDATED";

export interface Trade {
  trade_id: string;
  symbol: string;
  name?: string | null;
  status: TradeStatus;
  strategy?: string;
  mode?: string;
  signal_date: string;
  signal_price: Num;
  entry_date: string | null;
  entry_price: Num;
  exit_date: string | null;
  exit_price: Num;
  exit_reason?: string | null;
  invalidation_reason?: string | null;
  actual_return: Num;
  pnl?: Num;
  max_drawdown: Num;
  holding_days: Num;
  final_score: Num;
  statistical_score: Num;
  news_score: Num;
  fundamental_score: Num;
  neural_score: Num;
  risk_score?: Num;
  opportunity_probability: Num;
  confidence: Num;
  expected_return: Num;
  expected_drawdown: Num;
  reason: string | null;
  model_version: string | null;
  current_price?: Num;
}

export interface NewsItem {
  symbol?: string;
  title: string;
  source: string;
  url: string;
  published_at: string | null;
  sentiment: Num;
  relevance?: Num;
  themes?: string[];
  provider?: string;
}

export interface YearRow {
  year: number;
  return: number;
  partial?: boolean;
}

export interface Metrics {
  [key: string]: unknown;
  total_return?: Num;
  cagr?: Num;
  volatility?: Num;
  sharpe?: Num;
  sortino?: Num;
  max_drawdown?: Num;
  win_rate?: Num;
  avg_return?: Num;
  median_return?: Num;
  avg_gain?: Num;
  avg_loss?: Num;
  profit_factor?: Num;
  trades?: Num;
  avg_holding_days?: Num;
  exposure?: Num;
  yearly?: YearRow[];
  start?: string;
  end?: string;
}

export interface PaperMetrics extends Metrics {
  total_trades: number;
  closed_trades: number;
  open_trades: number;
  pending_trades: number;
  invalidated_trades: number;
  by_year?: YearRow[];
  notes: string[];
}

export interface MarketJson extends Freshness {
  disclaimer: string;
  mode: string;
  status: { state: "ok" | "no_data" | "synthetic" | string; message: string; last_update: string | null;
    last_run: { task?: string; finished_at?: string; n_errors?: number; n_warnings?: number } | null };
  market: { name: string; timezone: string; indices: IndexSummary[] };
  universe: { size: Num; analyzed: Num; eligible: Num; skipped: Num; deep_analyzed: Num; quality_rejected: Num };
  counts: { opportunities: number; open_trades: number; pending_trades: number; closed_trades: number };
  top_opportunities: OpportunityRow[];
  system_performance: PaperMetrics;
  best_trades: Trade[];
  worst_trades: Trade[];
  model: { production_version: string | null; available: boolean; reason: string | null; training_date: string | null;
    training_period: { start: string; end: string } | null; features_version: string | null; oos_auc: Num;
    last_training: string | null; last_candidate: string | null; last_candidate_promoted: boolean | null };
  backtest: { strategy: string; metrics: Metrics; benchmark: Metrics; period: { start: string; end: string };
    generated_at: string; model_version: string | null } | null;
  live_evaluation: { available: boolean; evaluated_days: number; samples?: number; base_rate?: Num;
    top10_final_hit_rate?: Num; opportunities_hit_rate?: Num; note?: string };
  history_days: number;
}

export interface OpportunitiesJson extends Freshness {
  model_available: boolean;
  weights: Record<string, number>;
  opportunity_threshold: number;
  rows: OpportunityRow[];
}

export interface Explanation {
  summary: string;
  reasons: { code: string; category: string; text: string; why_it_matters: string }[];
  risks: { code: string; category: string; text: string; why_it_matters: string }[];
  news_explanation?: string | null;
  fundamental_facts?: string[];
  model_explanation?: ModelExplanation | null;
  disclaimers: { facts: string; model: string };
}

export interface ModelExplanation {
  method: string;
  disclaimer: string;
  base_probability: number;
  factors: { feature: string; label: string; effect: number; direction: string; value: number }[];
}

export interface TrackedTrade extends Trade {
  reasons: string[];
  why_it_matters: string[];
  risks_at_entry: string[];
  news_explanation?: string | null;
  model_explanation?: ModelExplanation | null;
  current_scores?: Record<string, Num | string> | null;
  sessions_tracked: Num;
  news_since_entry: NewsItem[];
}

export interface ManualPosition {
  symbol: string;
  name: string;
  entry_date: string | null;
  entry_price: number;
  shares: number;
  current_price: Num;
  return: Num;
  days: Num;
  reasons: string[];
  why_it_matters: string[];
  reasons_auto: boolean;
  why_it_matters_auto: boolean;
  risks: string[];
  current_scores: Record<string, Num> | null;
  news_since_entry: NewsItem[];
}

export interface TrackingJson extends Freshness {
  mode: string;
  trades: TrackedTrade[];
  manual_positions: ManualPosition[];
}

export interface HistoryJson extends Freshness {
  trades: Trade[];
  equity: { date: string; equity: number; benchmark: Num; invested?: Num; open_positions?: Num }[];
  metrics: PaperMetrics;
  snapshots: { date: string; model_version: string | null; benchmark_return_since: Num;
    opportunities: { symbol: string; final_score: Num; price: Num; return_since: Num; return_20d: Num }[] }[];
  strategies: string[];
}

export interface StrategyBlock {
  description: string;
  available?: boolean;
  reason?: string;
  metrics?: Metrics;
  invalidated_entries?: number;
}

export interface BacktestJson {
  available: boolean;
  message?: string;
  generated_at?: string;
  exported_at?: string;
  data_as_of?: string;
  period?: { start: string; end: string; sessions: number };
  neural_available?: boolean;
  model_version?: string | null;
  symbols?: number;
  assumptions?: Record<string, unknown>;
  strategies?: Record<string, StrategyBlock>;
  benchmarks?: Record<string, StrategyBlock>;
  curves?: { dates: string[]; series: Record<string, (number | null)[]> };
  drawdowns?: { dates: string[]; series: Record<string, (number | null)[]> };
  notes?: string[];
  trades_sample?: Record<string, unknown>[];
}

export interface FoldInfo {
  name: string;
  train_start: string;
  train_end: string;
  val_start: string;
  val_end: string;
  test_start: string;
  test_end: string;
  train_samples: number;
  test_samples: number;
  best_epoch: number;
  train_base_rate: number;
  metrics: Record<string, Num>;
  baselines_auc: Record<string, Num>;
}

export interface ModelVersion {
  model_version: string;
  status: string;
  training_date?: string;
  training_period?: { start: string; end: string };
  validation_period?: { start: string; end: string };
  features_version?: string;
  data_as_of?: string;
  lookback?: number;
  n_parameters?: number;
  base_rate?: number;
  threshold?: number;
  hyperparameters?: Record<string, unknown>;
  label_definition?: Record<string, unknown>;
  training?: { best_epoch: number; epochs_run: number; train_samples: number; val_samples: number; seconds: number;
    history: { epoch: number; train_loss?: number; val_loss: number; val_auc: Num }[] };
  walk_forward?: { folds: FoldInfo[]; aggregate: Record<string, Num | string>; baselines: Record<string, Num> };
  feature_importance?: { feature: string; label: string; auc_drop: number }[];
  backtest?: Record<string, Record<string, Num>>;
  dataset?: Record<string, unknown>;
  promotion_checks?: { name: string; passed: boolean; detail: string }[];
  compared_with?: string | null;
  promoted_at?: string;
  registered_at?: string;
}

export interface ModelJson extends Freshness {
  production_version: string | null;
  registry: { model_version: string; status: string; training_date?: string; training_period?: { start: string; end: string };
    features_version?: string; oos_auc: Num; top_decile_lift: Num; combined_sharpe: Num; promoted_at?: string;
    checks: { name: string; passed: boolean; detail: string }[] }[];
  production: ModelVersion | null;
  latest_candidate: ModelVersion | null;
  live_evaluation: { available: boolean; evaluated_days: number; samples?: number; base_rate?: Num; final_score_auc?: Num;
    statistical_score_auc?: Num; neural_auc?: Num; neural_samples?: number; top10_final_hit_rate?: Num;
    opportunities_hit_rate?: Num; note?: string;
    by_date?: { date: string; base_rate: number; n: number; top10_final_hit_rate: Num; top10_neural_hit_rate: Num;
      opportunities_hit_rate: Num }[] };
  features: { name: string; label: string; group: string }[];
  architecture: Record<string, unknown>;
  walk_forward_config: Record<string, unknown>;
  promotion_config: Record<string, unknown>;
  status_reason: string | null;
}

export interface NewsJson extends Freshness {
  providers: string[];
  items: NewsItem[];
}

export interface StockIndexRow {
  symbol: string;
  name: string;
  sector: string | null;
  price: Num;
  change_1d: Num;
  final_score: Num;
  statistical_score: Num;
  is_opportunity: boolean;
  in_tracking: boolean;
  has_chart: boolean;
}

export interface StocksIndexJson extends Freshness {
  stocks: StockIndexRow[];
}

export interface StockJson extends Freshness {
  symbol: string;
  name: string;
  sector: string | null;
  industry: string | null;
  indices: string | null;
  price?: { last: Num; change_1d: Num; change_1m: Num; change_3m: Num; change_1y: Num; high52: Num; low52: Num; date: string };
  series: { t0: string; dt: number[]; decimals: number; open: Num[]; high: Num[]; low: Num[]; close: Num[];
    ma7: Num[]; ma25: Num[]; ma99: Num[]; volume: Num[] } | null;
  oscillators?: { start_index: number; rsi14: Num[]; macd: Num[]; macd_signal: Num[]; macd_hist: Num[] };
  latest_indicators?: Record<string, Num>;
  signals?: { date: string; type: string; price: Num }[];
  explanation: Explanation | null;
  statistical_breakdown?: { component: string; score: Num; weight: number }[];
  scores: (Record<string, Num> & { rank: Num; eligible: boolean; is_opportunity: boolean; signal_label: string;
    risk_level: string | null }) | null;
  breakdown: { final_score: Num; weights_used: Record<string, number>; configured_weights?: Record<string, number>;
    contributions: Record<string, number>; missing: string[] } | null;
  news: { news_score: Num; label: string; explanation: string; count: number; positive: number; negative: number;
    neutral: number; themes: string[]; sentiment: Num; items: NewsItem[] } | null;
  news_archive: NewsItem[];
  fundamentals: { fundamental_score: Num; metric_scores: Record<string, number>; values: Record<string, unknown>;
    available: string[]; unavailable: string[]; note: string } | null;
  fundamental_labels: Record<string, string>;
  neural: { available: boolean; reason?: string; model_version?: string | null; base_rate?: Num; threshold?: Num;
    opportunity_probability?: Num; confidence?: Num; expected_return?: Num; expected_drawdown?: Num; neural_score?: Num;
    explanation?: ModelExplanation | null };
  trades: { paper: Trade[]; backtest: Record<string, unknown>[] };
  backtest_stats?: { trades: number; win_rate: Num; avg_return: Num; period?: { start: string; end: string } } | null;
  score_history: { date: string; final_score: Num; statistical_score: Num; probability: Num }[];
}
