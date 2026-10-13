export type GuardOutcome = 'allow' | 'verify' | 'consent' | 'decline' | 'escalate';

export interface GuardRefusal {
  code: 'consent_required' | 'declined' | 'escalated';
  reason: string;
  policyId?: string;
  consentToken?: string;
}

export interface ApprovalOffer {
  action: string;
  reason: string;
  approvalToken: string;
}

export interface SecuritySummary {
  tainted: boolean;
  shadow: boolean;
  dropped: { source: string; ref?: string; reason: string }[];
  skipped: { action: string; reason: string }[];
  approvals: ApprovalOffer[];
}

const GUARD_REFUSAL_CODES = new Set<GuardRefusal['code']>([
  'consent_required',
  'declined',
  'escalated',
]);

export function isGuardRefusal(data: unknown): data is GuardRefusal {
  if (!data || typeof data !== 'object') {
    return false;
  }
  const code = (data as { code?: unknown }).code;
  return typeof code === 'string' && GUARD_REFUSAL_CODES.has(code as GuardRefusal['code']);
}

export function normalizeGuardRefusal(
  body: Record<string, unknown>,
  status: number
): GuardRefusal | null {
  const codeRaw = body.code;
  if (typeof codeRaw === 'string' && GUARD_REFUSAL_CODES.has(codeRaw as GuardRefusal['code'])) {
    return {
      code: codeRaw as GuardRefusal['code'],
      reason: typeof body.reason === 'string' ? body.reason : 'Request blocked',
      policyId:
        typeof body.policy_id === 'string'
          ? body.policy_id
          : typeof body.policyId === 'string'
            ? body.policyId
            : undefined,
      consentToken:
        typeof body.consent_token === 'string'
          ? body.consent_token
          : typeof body.consentToken === 'string'
            ? body.consentToken
            : undefined,
    };
  }

  if (status === 409) {
    return {
      code: 'consent_required',
      reason: typeof body.reason === 'string' ? body.reason : 'Consent required',
      policyId: typeof body.policy_id === 'string' ? body.policy_id : undefined,
      consentToken:
        typeof body.consent_token === 'string' ? body.consent_token : undefined,
    };
  }

  if (status === 403) {
    const outcome = body.outcome;
    const code: GuardRefusal['code'] =
      outcome === 'escalate' || codeRaw === 'escalated' ? 'escalated' : 'declined';
    return {
      code,
      reason: typeof body.reason === 'string' ? body.reason : 'Request declined',
      policyId: typeof body.policy_id === 'string' ? body.policy_id : undefined,
    };
  }

  return null;
}

export function normalizeSecuritySummary(raw: unknown): SecuritySummary | undefined {
  if (!raw || typeof raw !== 'object') {
    return undefined;
  }
  const sec = raw as Record<string, unknown>;
  const dropped = Array.isArray(sec.dropped)
    ? sec.dropped.map((item) => {
        const row = item as Record<string, unknown>;
        return {
          source: String(row.source ?? 'unknown'),
          ref: typeof row.ref === 'string' ? row.ref : undefined,
          reason: typeof row.reason === 'string' ? row.reason : '',
        };
      })
    : [];
  const skipped = Array.isArray(sec.skipped)
    ? sec.skipped.map((item) => {
        const row = item as Record<string, unknown>;
        return {
          action: String(row.action ?? 'unknown'),
          reason: typeof row.reason === 'string' ? row.reason : '',
        };
      })
    : [];
  const approvals = Array.isArray(sec.approvals)
    ? sec.approvals.map((item) => {
        const row = item as Record<string, unknown>;
        return {
          action: String(row.action ?? 'unknown'),
          reason: typeof row.reason === 'string' ? row.reason : '',
          approvalToken:
            typeof row.approval_token === 'string'
              ? row.approval_token
              : typeof row.approvalToken === 'string'
                ? row.approvalToken
                : '',
        };
      })
    : [];

  return {
    tainted: Boolean(sec.tainted),
    shadow: Boolean(sec.shadow),
    dropped,
    skipped,
    approvals,
  };
}
