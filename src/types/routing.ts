export type AgenticMode = 'off' | 'auto' | 'always';
export type Route = 'direct' | 'agentic';
export type DecidedBy = 'rule' | 'model' | 'fallback' | 'config' | 'mode' | 'override';

export interface RouteInfo {
  route: Route;
  decidedBy: DecidedBy;
  reason: string;
  pAgentic?: number | null;
}

export interface TriageResponse {
  route: Route;
  p_agentic: number | null;
  decided_by: 'rule' | 'model' | 'fallback' | 'config';
  reason: string;
  classifier_version: string;
}

export const AGENTIC_MODES: readonly AgenticMode[] = ['off', 'auto', 'always'];
export const isAgenticMode = (v: unknown): v is AgenticMode =>
  typeof v === 'string' && (AGENTIC_MODES as readonly string[]).includes(v);
