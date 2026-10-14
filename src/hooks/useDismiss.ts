import { RefObject, useEffect } from 'react';

/**
 * Closes an open menu on a click outside `containerRef` or on Escape.
 * `close(true)` means the user pressed Escape, so focus should go back to the menu's button.
 */
export function useDismiss(
  containerRef: RefObject<HTMLElement>,
  open: boolean,
  close: (returnFocus: boolean) => void,
) {
  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: MouseEvent) => {
      if (!containerRef.current?.contains(event.target as Node)) close(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') close(true);
    };
    document.addEventListener('mousedown', onPointerDown);
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('mousedown', onPointerDown);
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [containerRef, open, close]);
}
