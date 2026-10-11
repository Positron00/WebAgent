'use client';

import React, { useCallback, useRef } from 'react';
import { BeakerIcon } from '@heroicons/react/24/outline';
import { AgenticMode, AGENTIC_MODES } from '@/types/routing';

const LABELS: Record<AgenticMode, string> = {
  off: 'Off',
  auto: 'Auto',
  always: 'Always',
};

const TITLES: Record<AgenticMode, string> = {
  off: 'Answer directly, no research agents',
  auto: 'Use research agents when the question needs them',
  always: 'Always use research agents (slower)',
};

interface AgenticModeToggleProps {
  value: AgenticMode;
  onChange: (mode: AgenticMode) => void;
  disabled?: boolean;
}

export function AgenticModeToggle({ value, onChange, disabled = false }: AgenticModeToggleProps) {
  const buttonRefs = useRef<(HTMLButtonElement | null)[]>([]);

  const selectMode = useCallback(
    (mode: AgenticMode) => {
      if (!disabled) onChange(mode);
    },
    [disabled, onChange]
  );

  const handleKeyDown = (event: React.KeyboardEvent, index: number) => {
    if (disabled) return;
    let nextIndex = index;
    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
      event.preventDefault();
      nextIndex = (index + 1) % AGENTIC_MODES.length;
    } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
      event.preventDefault();
      nextIndex = (index - 1 + AGENTIC_MODES.length) % AGENTIC_MODES.length;
    } else {
      return;
    }
    const nextMode = AGENTIC_MODES[nextIndex];
    selectMode(nextMode);
    buttonRefs.current[nextIndex]?.focus();
  };

  const displayValue = disabled ? 'off' : value;

  return (
    <div className="flex items-center gap-1" role="radiogroup" aria-label="Agentic mode">
      <BeakerIcon className="w-3 h-3 text-gray-300 shrink-0" aria-hidden="true" />
      <span className="text-[10px] text-gray-300" aria-hidden="true">
        Agentic:
      </span>
      {AGENTIC_MODES.map((mode, index) => {
        const selected = displayValue === mode;
        return (
          <button
            key={mode}
            ref={(el) => {
              buttonRefs.current[index] = el;
            }}
            type="button"
            role="radio"
            aria-checked={selected}
            tabIndex={selected ? 0 : -1}
            title={TITLES[mode]}
            disabled={disabled}
            onClick={() => selectMode(mode)}
            onKeyDown={(e) => handleKeyDown(e, index)}
            className={`px-1.5 py-0.5 rounded text-[10px] transition-colors ${
              selected
                ? 'bg-indigo-600 text-white'
                : 'bg-indigo-600/10 text-white/40 hover:bg-indigo-600/40 hover:text-white/80'
            } disabled:opacity-50`}
          >
            {LABELS[mode]}
          </button>
        );
      })}
    </div>
  );
}
