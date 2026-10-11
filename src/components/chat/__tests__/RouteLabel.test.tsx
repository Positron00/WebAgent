import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { RouteLabel } from '../RouteLabel';
import { AgenticMode, RouteInfo } from '@/types/routing';

const direct: RouteInfo = { route: 'direct', decidedBy: 'rule', reason: 'Auto · simple question' };
const agentic: RouteInfo = { route: 'agentic', decidedBy: 'model', reason: 'Auto · needs research' };

function renderLabel(info: RouteInfo, mode: AgenticMode, canOverride = true, onOverride = jest.fn()) {
  render(<RouteLabel info={info} mode={mode} canOverride={canOverride} onOverride={onOverride} />);
  return onOverride;
}

const overrideButton = () => screen.queryByRole('button', { name: 'Research this in depth' });

describe('RouteLabel', () => {
  it('shows "Answered directly" with the reason in a focusable tooltip', () => {
    renderLabel(direct, 'auto');
    const label = screen.getByText('Answered directly');
    expect(label).toHaveAccessibleDescription(direct.reason);
    expect(label).toHaveAttribute('tabindex', '0');
    expect(screen.getByRole('tooltip')).toHaveTextContent(direct.reason);
  });

  it('shows "Deep research" with the reason in a tooltip', () => {
    renderLabel(agentic, 'auto');
    expect(screen.getByText('Deep research')).toHaveAccessibleDescription(agentic.reason);
  });

  it('shows the override button in auto mode for a direct answer, and clicking calls onOverride', () => {
    const onOverride = renderLabel(direct, 'auto');
    const button = overrideButton();
    expect(button).toBeInTheDocument();

    fireEvent.click(button!);
    expect(onOverride).toHaveBeenCalledTimes(1);
  });

  it('hides the override button for an agentic answer in auto mode', () => {
    renderLabel(agentic, 'auto');
    expect(overrideButton()).not.toBeInTheDocument();
  });

  it.each<AgenticMode>(['off', 'always'])('hides the override button in %s mode', (mode) => {
    renderLabel(direct, mode);
    expect(overrideButton()).not.toBeInTheDocument();
  });

  it('hides the override button when canOverride is false', () => {
    renderLabel(direct, 'auto', false);
    expect(overrideButton()).not.toBeInTheDocument();
  });
});
