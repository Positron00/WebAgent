'use client';

import React from 'react';
import { AgenticMode, RouteInfo } from '@/types/routing';
import { Tooltip } from './Tooltip';

interface RouteLabelProps {
  info: RouteInfo;
  mode: AgenticMode;
  canOverride: boolean;
  onOverride: () => void;
}

export function RouteLabel({ info, mode, canOverride, onOverride }: RouteLabelProps) {
  const label = info.route === 'agentic' ? 'Deep research' : 'Answered directly';

  const showOverride = mode === 'auto' && info.route === 'direct' && canOverride;

  return (
    <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-gray-500">
      <Tooltip text={info.reason} align="start">
        {/* Focusable so keyboard users can read why this route was chosen. */}
        <span
          tabIndex={0}
          className="cursor-help underline decoration-dotted underline-offset-2 focus:outline-none focus-visible:ring-1 focus-visible:ring-indigo-400 rounded-sm"
        >
          {label}
        </span>
      </Tooltip>
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
