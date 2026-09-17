/**
 * What each model's inputs are worth, measured by
 * `data_jobs/build_importance.py` and read here at build time like the
 * rest of the static data.
 *
 * Two methods share one shape because both report the same quantity — an
 * increase in log loss when an input is taken away — but they are not the
 * same measurement, so `method` travels with every model and the chart
 * says which one it is drawing. See the job's own docstring.
 */
import importance from '@/public/data/importance.json';

export interface ImportanceFeature {
  name: string;
  /** Increase in log loss when this input is removed or shuffled. A
   * negative value means the model scored *better* without it. */
  value: number;
  /** Spread across permutation repeats; null for a single ablation. */
  sd: number | null;
  detail: string;
  /** Permutation only: the column had no variation over the window, so a
   * zero here is missing data rather than a useless feature. */
  constant?: boolean;
}

export interface ImportanceModel {
  name: string;
  method: 'permutation importance' | 'component ablation' | string;
  metric: string;
  /** Log loss with everything in place. */
  baseline: number;
  n: number;
  window: string;
  features: ImportanceFeature[];
  caveat: string;
}

export interface ImportanceFile {
  generated_at: string;
  models: Record<string, ImportanceModel>;
}

const data = importance as unknown as ImportanceFile;

export function importanceFor(key: string): ImportanceModel | null {
  return data.models[key] ?? null;
}

export function importanceGeneratedAt(): string {
  return data.generated_at;
}
