'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { BeakerIcon } from '@heroicons/react/24/outline';
import { AgenticMode, AGENTIC_MODES } from '@/types/routing';
import { useDismiss } from '@/hooks/useDismiss';

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

// Text colour of the current mode on the toolbar; Always keeps the indigo of the old selected pill
const TEXT_COLORS: Record<AgenticMode, string> = {
  off: 'text-gray-400',
  auto: 'text-cyan-400',
  always: 'text-indigo-400',
};

interface AgenticModeToggleProps {
  value: AgenticMode;
  onChange: (mode: AgenticMode) => void;
  disabled?: boolean;
}

/** Toolbar button showing the Agentic mode; opens a menu to change it, like Style and Focus. */
export function AgenticModeToggle({ value, onChange, disabled = false }: AgenticModeToggleProps) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);

  const displayValue = disabled ? 'off' : value;

  const close = useCallback((returnFocus: boolean) => {
    setOpen(false);
    if (returnFocus) triggerRef.current?.focus();
  }, []);

  useDismiss(containerRef, open, close);

  useEffect(() => {
    if (disabled) setOpen(false);
  }, [disabled]);

  const selectMode = (mode: AgenticMode) => {
    if (disabled) return;
    onChange(mode);
    close(true);
  };

  return (
    <div className="relative" ref={containerRef}>
      <button
        ref={triggerRef}
        type="button"
        onClick={() => setOpen(!open)}
        disabled={disabled}
        className="p-1 hover:bg-white/5 rounded transition-colors flex items-center gap-0.5 text-[10px] disabled:opacity-50"
        aria-expanded={open}
        aria-haspopup="true"
        aria-label={`Agentic mode: ${LABELS[displayValue]}`}
      >
        <BeakerIcon className="w-3 h-3 text-gray-300" aria-hidden="true" />
        <span className="text-xs text-gray-300">Agentic</span>
        <span className="text-xs text-gray-500 px-0.5" aria-hidden="true">·</span>
        <span className={`text-xs ${TEXT_COLORS[displayValue]}`}>{LABELS[displayValue]}</span>
      </button>

      {open && (
        <div className="absolute bottom-8 right-0 bg-gray-800 border border-gray-700 rounded-lg shadow-lg p-2 min-w-[240px] z-10">
          <h4 id="agentic-mode-heading" className="text-white text-sm font-medium mb-2">
            Agentic Mode
          </h4>
          <div className="space-y-2" role="radiogroup" aria-labelledby="agentic-mode-heading">
            {AGENTIC_MODES.map((mode) => (
              <div key={mode} className="flex items-start">
                <input
                  type="radio"
                  id={`agentic-mode-${mode}`}
                  name="agenticMode"
                  className="mt-0.5 w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 focus:ring-blue-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600"
                  checked={displayValue === mode}
                  aria-describedby={`agentic-mode-${mode}-hint`}
                  onChange={() => selectMode(mode)}
                />
                <div className="ml-2 text-xs">
                  <label htmlFor={`agentic-mode-${mode}`} className="block font-medium text-gray-300">
                    {LABELS[mode]}
                  </label>
                  <span id={`agentic-mode-${mode}-hint`} className="block text-gray-400">
                    {TITLES[mode]}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
