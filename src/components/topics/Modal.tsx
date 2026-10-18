"use client";

import React, { useEffect, useRef } from "react";

const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

interface ModalProps {
  titleId: string;
  title: string;
  onClose: () => void;
  children: React.ReactNode;
  /** Optional id of an element describing the dialog. */
  describedBy?: string;
}

/**
 * A modal dialog that traps Tab and Shift+Tab inside itself, closes on Escape and gives focus
 * back to whatever had it when it opened.
 */
export function Modal({
  titleId,
  title,
  onClose,
  children,
  describedBy,
}: ModalProps) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef(onClose);
  useEffect(() => {
    closeRef.current = onClose;
  }, [onClose]);

  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    const dialog = dialogRef.current;
    const first = dialog?.querySelector<HTMLElement>(FOCUSABLE);
    (first ?? dialog)?.focus();

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closeRef.current();
        return;
      }
      if (event.key !== "Tab" || !dialogRef.current) return;
      const items = Array.from(
        dialogRef.current.querySelectorAll<HTMLElement>(FOCUSABLE),
      );
      if (items.length === 0) {
        event.preventDefault();
        return;
      }
      const firstItem = items[0];
      const lastItem = items[items.length - 1];
      const active = document.activeElement;
      if (
        event.shiftKey &&
        (active === firstItem || !dialogRef.current.contains(active))
      ) {
        event.preventDefault();
        lastItem.focus();
      } else if (
        !event.shiftKey &&
        (active === lastItem || !dialogRef.current.contains(active))
      ) {
        event.preventDefault();
        firstItem.focus();
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      previous?.focus?.();
    };
  }, []);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={describedBy}
        tabIndex={-1}
        className="w-full max-w-md rounded-lg border border-gray-700 bg-gray-950 p-5 shadow-xl outline-none text-left"
      >
        <h2 id={titleId} className="text-lg font-medium text-white mb-3">
          {title}
        </h2>
        {children}
      </div>
    </div>
  );
}
