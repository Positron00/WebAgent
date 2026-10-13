'use client';

import React, { useEffect, useRef } from 'react';

interface ConsentDialogProps {
  reason: string;
  onSendAnyway: () => void;
  onEditMessage: () => void;
}

export function ConsentDialog({ reason, onSendAnyway, onEditMessage }: ConsentDialogProps) {
  const dialogRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    dialogRef.current?.focus();
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        onEditMessage();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onEditMessage]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="consent-dialog-title"
        tabIndex={-1}
        className="w-full max-w-md rounded-lg border border-gray-700 bg-gray-950 p-5 shadow-xl outline-none"
      >
        <h2 id="consent-dialog-title" className="text-lg font-medium text-white mb-3">
          Send this message?
        </h2>
        <p className="text-sm text-gray-300 mb-6">{reason}</p>
        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={onEditMessage}
            className="px-4 py-2 text-sm text-gray-300 hover:text-white rounded hover:bg-gray-800"
          >
            Edit message
          </button>
          <button
            type="button"
            onClick={onSendAnyway}
            className="px-4 py-2 text-sm bg-blue-600 hover:bg-blue-700 text-white rounded"
          >
            Send anyway
          </button>
        </div>
      </div>
    </div>
  );
}
