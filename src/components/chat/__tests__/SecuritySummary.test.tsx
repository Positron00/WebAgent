import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { SecuritySummary } from '../SecuritySummary';

const base = {
  tainted: false,
  shadow: false,
  dropped: [],
  skipped: [],
  approvals: [],
};

describe('SecuritySummary', () => {
  it('renders counts and tooltip reasons', () => {
    render(
      <SecuritySummary
        busy={false}
        onApprove={jest.fn()}
        security={{
          ...base,
          dropped: [
            { source: 'web_result', ref: 'https://a.test', reason: 'injection' },
            { source: 'web_result', ref: 'https://b.test', reason: 'injection' },
          ],
          skipped: [{ action: 'tool::web_search', reason: 'off topic' }],
        }}
      />
    );

    expect(screen.getByText('2 web results left out · 1 search skipped')).toBeInTheDocument();
    expect(screen.getByRole('tooltip')).toHaveTextContent('https://a.test: injection');
    expect(screen.getByRole('tooltip')).toHaveTextContent('tool::web_search: off topic');
  });

  it('renders nothing in shadow mode', () => {
    const { container } = render(
      <SecuritySummary
        busy={false}
        onApprove={jest.fn()}
        security={{
          ...base,
          shadow: true,
          dropped: [{ source: 'web_result', reason: 'x' }],
        }}
      />
    );
    expect(container).toBeEmptyDOMElement();
  });

  it('shows Approve and re-run when approvals exist', () => {
    const onApprove = jest.fn();
    render(
      <SecuritySummary
        busy={false}
        onApprove={onApprove}
        security={{
          ...base,
          approvals: [
            {
              action: 'tool::execute_code',
              reason: 'Needs you',
              approvalToken: 'tok',
            },
          ],
        }}
      />
    );

    const button = screen.getByRole('button', { name: 'Approve and re-run' });
    fireEvent.click(button);
    expect(onApprove).toHaveBeenCalledTimes(1);
    expect(screen.getByText(/code not run: needs your approval/)).toBeInTheDocument();
  });

  it('disables Approve and re-run while busy', () => {
    render(
      <SecuritySummary
        busy={true}
        onApprove={jest.fn()}
        security={{
          ...base,
          approvals: [
            { action: 'tool::execute_code', reason: 'Needs you', approvalToken: 'tok' },
          ],
        }}
      />
    );

    expect(screen.getByRole('button', { name: 'Approve and re-run' })).toBeDisabled();
  });
});
