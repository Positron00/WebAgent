import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { AgenticModeToggle } from '../AgenticModeToggle';

const radio = (name: string) => screen.getByRole('radio', { name });
const trigger = () => screen.getByRole('button', { name: /^Agentic mode:/ });
const openMenu = () => fireEvent.click(trigger());

const TITLES = {
  off: 'Answer directly, no research agents',
  auto: 'Use research agents when the question needs them',
  always: 'Always use research agents (slower)',
};

describe('AgenticModeToggle', () => {
  it('shows the current mode on a button and keeps the menu closed until clicked', () => {
    render(<AgenticModeToggle value="auto" onChange={jest.fn()} />);

    expect(trigger()).toHaveAccessibleName('Agentic mode: Auto');
    expect(trigger()).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByRole('radiogroup')).not.toBeInTheDocument();
  });

  it('opens a menu of three radios, each described by its hint', () => {
    render(<AgenticModeToggle value="off" onChange={jest.fn()} />);
    openMenu();

    expect(trigger()).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByRole('radiogroup', { name: 'Agentic Mode' })).toBeInTheDocument();
    expect(screen.getAllByRole('radio')).toHaveLength(3);
    expect(radio('Off')).toHaveAccessibleDescription(TITLES.off);
    expect(radio('Auto')).toHaveAccessibleDescription(TITLES.auto);
    expect(radio('Always')).toHaveAccessibleDescription(TITLES.always);
  });

  it('checks only the selected mode', () => {
    render(<AgenticModeToggle value="auto" onChange={jest.fn()} />);
    openMenu();

    expect(radio('Off')).not.toBeChecked();
    expect(radio('Auto')).toBeChecked();
    expect(radio('Always')).not.toBeChecked();
  });

  it('choosing a mode calls onChange and closes the menu', () => {
    const onChange = jest.fn();
    render(<AgenticModeToggle value="off" onChange={onChange} />);
    openMenu();

    fireEvent.click(radio('Always'));
    expect(onChange).toHaveBeenCalledWith('always');
    expect(screen.queryByRole('radiogroup')).not.toBeInTheDocument();
  });

  it('Escape and a click outside close the menu without changing the mode', () => {
    const onChange = jest.fn();
    render(
      <div>
        <AgenticModeToggle value="off" onChange={onChange} />
        <p>outside</p>
      </div>,
    );

    openMenu();
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(screen.queryByRole('radiogroup')).not.toBeInTheDocument();
    expect(trigger()).toHaveFocus();

    openMenu();
    fireEvent.mouseDown(screen.getByText('outside'));
    expect(screen.queryByRole('radiogroup')).not.toBeInTheDocument();
    expect(onChange).not.toHaveBeenCalled();
  });

  it('when disabled, shows Off and does not open', () => {
    const onChange = jest.fn();
    render(<AgenticModeToggle value="always" onChange={onChange} disabled />);

    expect(trigger()).toHaveAccessibleName('Agentic mode: Off');
    expect(trigger()).toBeDisabled();
    fireEvent.click(trigger());
    expect(screen.queryByRole('radiogroup')).not.toBeInTheDocument();
    expect(onChange).not.toHaveBeenCalled();
  });
});
