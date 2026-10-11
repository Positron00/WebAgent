'use client';

import React from 'react';
import { AgenticMode, RouteInfo } from '@/types/routing';

interface RouteLabelProps {
  info: RouteInfo;
  mode: AgenticMode;
  canOverride: boolean;
  onOverride: () => void;
}

export function RouteLabel({ info, mode, canOverride, onOverride }: RouteLabelProps) {
  const label = info.route === 'agentic' ? 'Deep research' : 'Answered directly';

  const showOverride =
    mode === 'auto' && info.route === 'direct' && canOverride;

  return (
    <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-gray-500">
      <span title={info.reason}>{label}</span>
      {showOverride && (
        <button
          type="button"
          onClick={onOverride}
          className="text-indigo-400 hover:text-indigo-300 underline-offset-2 hover:underline"
        >
          Research this in depth
        </button>
      )}
    </div>
  );
}
