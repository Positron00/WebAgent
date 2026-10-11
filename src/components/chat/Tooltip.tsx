'use client';

import React, { useId } from 'react';

interface TooltipProps {
  text: string;
  children: React.ReactElement;
  /** `center` sits over the middle of the trigger; `start` lines up with its left edge. */
  align?: 'center' | 'start';
}

/**
 * Shows `text` above its child at once on hover and on keyboard focus.
 * The native `title` tooltip appears only after a delay, never on focus or touch,
 * and not at all in some embedded browsers.
 */
export function Tooltip({ text, children, align = 'center' }: TooltipProps) {
  const id = useId();
  const position = align === 'center' ? 'left-1/2 -translate-x-1/2' : 'left-0';

  return (
    <span className="relative inline-flex group/tooltip">
      {React.cloneElement(children, { 'aria-describedby': id })}
      <span
        id={id}
        role="tooltip"
        className={`pointer-events-none absolute bottom-full ${position} z-50 mb-1.5 w-max max-w-[16rem] rounded bg-gray-900 px-2 py-1 text-[11px] leading-snug text-gray-100 shadow-lg ring-1 ring-white/10 opacity-0 transition-opacity duration-100 group-hover/tooltip:opacity-100 group-focus-within/tooltip:opacity-100`}
      >
        {text}
      </span>
    </span>
  );
}
