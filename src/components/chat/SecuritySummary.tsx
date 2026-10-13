'use client';

import React from 'react';
import { SecuritySummary as SecuritySummaryType } from '@/types/security';
import { Tooltip } from './Tooltip';

interface SecuritySummaryProps {
  security: SecuritySummaryType;
  onApprove: () => void;
  busy: boolean;
}

function isEmptySummary(security: SecuritySummaryType): boolean {
  return (
    security.dropped.length === 0 &&
    security.skipped.length === 0 &&
    security.approvals.length === 0
  );
}

function summarySegments(security: SecuritySummaryType): string[] {
  const segments: string[] = [];
  const droppedCount = security.dropped.length;
  if (droppedCount > 0) {
    segments.push(
      `${droppedCount} web result${droppedCount === 1 ? '' : 's'} left out`
    );
  }

  const searchSkipped = security.skipped.filter((s) =>
    s.action.includes('web_search')
  ).length;
  if (searchSkipped > 0) {
    segments.push(`${searchSkipped} search${searchSkipped === 1 ? '' : 'es'} skipped`);
  }

  const otherSkipped = security.skipped.length - searchSkipped;
  if (otherSkipped > 0) {
    segments.push(
      `${otherSkipped} action${otherSkipped === 1 ? '' : 's'} skipped`
    );
  }

  if (security.approvals.length > 0) {
    segments.push('code not run: needs your approval');
  }

  return segments;
}

function tooltipText(security: SecuritySummaryType): string {
  const lines: string[] = [];
  for (const item of security.dropped) {
    const label = item.ref || item.source;
    lines.push(`${label}: ${item.reason}`);
  }
  for (const item of security.skipped) {
    lines.push(`${item.action}: ${item.reason}`);
  }
  for (const item of security.approvals) {
    lines.push(`${item.action}: ${item.reason}`);
  }
  return lines.join('\n');
}

export function SecuritySummary({ security, onApprove, busy }: SecuritySummaryProps) {
  if (security.shadow || isEmptySummary(security)) {
    return null;
  }

  const line = summarySegments(security).join(' · ');
  const reasons = tooltipText(security);

  return (
    <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-gray-500">
      <Tooltip text={reasons} align="start">
        <span className="cursor-default underline decoration-dotted underline-offset-2" tabIndex={0}>
          {line}
        </span>
      </Tooltip>
      {security.approvals.length > 0 && (
        <button
          type="button"
          onClick={onApprove}
          disabled={busy}
          className="rounded border border-gray-600 px-2 py-0.5 text-gray-300 hover:bg-gray-800 disabled:opacity-50"
        >
          Approve and re-run
        </button>
      )}
    </div>
  );
}
