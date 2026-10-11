import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { AgenticModeToggle } from '../AgenticModeToggle';

const TITLES = {
  off: 'Answer directly, no research agents',
  auto: 'Use research agents when the question needs them',
  always: 'Always use research agents (slower)',
};

describe('AgenticModeToggle', () => {
  it('renders three radios with the expected titles', () => {
    render(<AgenticModeToggle value="off" onChange={jest.fn()} />);

    expect(screen.getByRole('radiogroup', { name: 'Agentic mode' })).toBeInTheDocument();
    const radios = screen.getAllByRole('radio');
    expect(radios).toHaveLength(3);
    expect(radios.map((r) => r.textContent)).toEqual(['Off', 'Auto', 'Always']);
    expect(radios.map((r) => r.getAttribute('title'))).toEqual([
      TITLES.off,
      TITLES.auto,
      TITLES.always,
    ]);
  });

  it('marks only the selected mode with aria-checked', () => {
    render(<AgenticModeToggle value="auto" onChange={jest.fn()} />);

    expect(screen.getByTitle(TITLES.off)).toHaveAttribute('aria-checked', 'false');
    expect(screen.getByTitle(TITLES.auto)).toHaveAttribute('aria-checked', 'true');
    expect(screen.getByTitle(TITLES.always)).toHaveAttribute('aria-checked', 'false');
  });

  it('ArrowRight selects the next mode and wraps from Always to Off', () => {
    const onChange = jest.fn();
    const { rerender } = render(<AgenticModeToggle value="off" onChange={onChange} />);

    fireEvent.keyDown(screen.getByTitle(TITLES.off), { key: 'ArrowRight' });
    expect(onChange).toHaveBeenLastCalledWith('auto');

    rerender(<AgenticModeToggle value="always" onChange={onChange} />);
    fireEvent.keyDown(screen.getByTitle(TITLES.always), { key: 'ArrowRight' });
    expect(onChange).toHaveBeenLastCalledWith('off');
  });

  it('ArrowLeft selects the previous mode and wraps from Off to Always', () => {
    const onChange = jest.fn();
    const { rerender } = render(<AgenticModeToggle value="always" onChange={onChange} />);

    fireEvent.keyDown(screen.getByTitle(TITLES.always), { key: 'ArrowLeft' });
    expect(onChange).toHaveBeenLastCalledWith('auto');

    rerender(<AgenticModeToggle value="off" onChange={onChange} />);
    fireEvent.keyDown(screen.getByTitle(TITLES.off), { key: 'ArrowLeft' });
    expect(onChange).toHaveBeenLastCalledWith('always');
  });

  it('clicking a radio calls onChange with that mode', () => {
    const onChange = jest.fn();
    render(<AgenticModeToggle value="off" onChange={onChange} />);

    fireEvent.click(screen.getByTitle(TITLES.always));
    expect(onChange).toHaveBeenCalledWith('always');
  });

  it('when disabled, shows Off checked and ignores clicks and keys', () => {
    const onChange = jest.fn();
    render(<AgenticModeToggle value="always" onChange={onChange} disabled />);

    expect(screen.getByTitle(TITLES.off)).toHaveAttribute('aria-checked', 'true');
    expect(screen.getByTitle(TITLES.always)).toHaveAttribute('aria-checked', 'false');

    fireEvent.click(screen.getByTitle(TITLES.auto));
    fireEvent.keyDown(screen.getByTitle(TITLES.off), { key: 'ArrowRight' });
    expect(onChange).not.toHaveBeenCalled();
  });
});
