import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { ConsentDialog } from '../ConsentDialog';

describe('ConsentDialog', () => {
  it('exposes dialog semantics, focuses the dialog, and wires the buttons', () => {
    const onSendAnyway = jest.fn();
    const onEditMessage = jest.fn();

    render(
      <ConsentDialog reason="Looks sensitive" onSendAnyway={onSendAnyway} onEditMessage={onEditMessage} />
    );

    const dialog = screen.getByRole('dialog', { name: 'Send this message?' });
    expect(dialog).toHaveAttribute('aria-modal', 'true');
    expect(dialog).toHaveFocus();
    expect(screen.getByText('Looks sensitive')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Send anyway' }));
    expect(onSendAnyway).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole('button', { name: 'Edit message' }));
    expect(onEditMessage).toHaveBeenCalledTimes(1);
  });

  it('treats Escape as Edit message', () => {
    const onEditMessage = jest.fn();
    render(
      <ConsentDialog reason="Looks sensitive" onSendAnyway={jest.fn()} onEditMessage={onEditMessage} />
    );

    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onEditMessage).toHaveBeenCalledTimes(1);
  });
});
