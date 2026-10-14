import { fireEvent, render, screen } from '@testing-library/react';
import { MessageInput } from '../MessageInput';
import { AgenticMode } from '@/types/routing';

const appState: { agenticMode: AgenticMode; modeReady: boolean } = {
  agenticMode: 'off',
  modeReady: true,
};

jest.mock('@/contexts/AppContext', () => ({
  useApp: () => ({
    isOffline: false,
    accessibility: {
      reducedMotion: false,
      highContrast: false,
      fontSize: 'medium',
      promptStyle: 'balanced',
      knowledgeFocus: 'general',
      citeSources: true,
    },
    setAccessibility: jest.fn(),
    agenticMode: appState.agenticMode,
    setAgenticMode: jest.fn(),
    modeReady: appState.modeReady,
  }),
}));

const baseProps = {
  value: 'What changed in EU AI rules?',
  onChange: jest.fn(),
  onSubmit: jest.fn(),
  onFileSelect: jest.fn(),
  isLoading: false,
};

describe('MessageInput with the Agentic mode', () => {
  beforeEach(() => {
    appState.agenticMode = 'off';
    appState.modeReady = true;
  });

  it('disables sending and shows Off until the mode is known (REQ-054)', () => {
    appState.agenticMode = 'auto';
    appState.modeReady = false;
    render(<MessageInput {...baseProps} />);

    expect(screen.getByLabelText('Submit message')).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Agentic mode: Off' })).toBeDisabled();
  });

  it('enables sending once the mode is known', () => {
    render(<MessageInput {...baseProps} />);
    expect(screen.getByLabelText('Submit message')).not.toBeDisabled();
  });

  it('shows the image note only in Always with an image attached (REQ-003)', () => {
    appState.agenticMode = 'always';
    const { rerender } = render(<MessageInput {...baseProps} imagePreviewUrl="blob:x" />);
    expect(screen.getByText('Images are only used by direct answers')).toBeInTheDocument();

    rerender(<MessageInput {...baseProps} />);
    expect(screen.queryByText('Images are only used by direct answers')).not.toBeInTheDocument();
  });

  it('shows no image note in Auto, where image questions go direct', () => {
    appState.agenticMode = 'auto';
    render(<MessageInput {...baseProps} imagePreviewUrl="blob:x" />);
    expect(screen.queryByText('Images are only used by direct answers')).not.toBeInTheDocument();
  });
});

describe('MessageInput toolbar menus', () => {
  const menus = [
    { button: /^AI style:/, heading: 'AI Response Style' },
    { button: /^Knowledge focus:/, heading: 'Knowledge Focus' },
    { button: /^Settings$/, heading: 'Appearance' },
  ];

  beforeEach(() => {
    appState.agenticMode = 'off';
    appState.modeReady = true;
  });

  it.each(menus)('$heading closes on Escape and returns focus to its button', ({ button, heading }) => {
    render(<MessageInput {...baseProps} />);
    const trigger = screen.getByRole('button', { name: button });

    fireEvent.click(trigger);
    expect(trigger).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByRole('heading', { name: heading })).toBeInTheDocument();

    fireEvent.keyDown(document, { key: 'Escape' });
    expect(screen.queryByRole('heading', { name: heading })).not.toBeInTheDocument();
    expect(trigger).toHaveAttribute('aria-expanded', 'false');
    expect(trigger).toHaveFocus();
  });

  it.each(menus)('$heading closes on a click outside but not inside', ({ button, heading }) => {
    render(<MessageInput {...baseProps} />);
    fireEvent.click(screen.getByRole('button', { name: button }));

    fireEvent.mouseDown(screen.getByRole('heading', { name: heading }));
    expect(screen.getByRole('heading', { name: heading })).toBeInTheDocument();

    fireEvent.mouseDown(screen.getByLabelText('Submit message'));
    expect(screen.queryByRole('heading', { name: heading })).not.toBeInTheDocument();
  });

  it('opening one menu closes the one already open', () => {
    render(<MessageInput {...baseProps} />);
    fireEvent.click(screen.getByRole('button', { name: /^AI style:/ }));

    const focus = screen.getByRole('button', { name: /^Knowledge focus:/ });
    fireEvent.mouseDown(focus);
    fireEvent.click(focus);
    expect(screen.queryByRole('heading', { name: 'AI Response Style' })).not.toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Knowledge Focus' })).toBeInTheDocument();
  });
});

describe('MessageInput toolbar values', () => {
  it('shows the current style and focus on their buttons, like Agentic', () => {
    appState.agenticMode = 'auto';
    appState.modeReady = true;
    render(<MessageInput {...baseProps} />);

    expect(screen.getByRole('button', { name: 'AI style: Balanced' })).toHaveTextContent('Style·Balanced');
    expect(screen.getByRole('button', { name: 'Knowledge focus: General' })).toHaveTextContent('Focus·General');
    expect(screen.getByRole('button', { name: 'Agentic mode: Auto' })).toHaveTextContent('Agentic·Auto');
    // Each value is coloured by what it is
    expect(screen.getByText('Balanced')).toHaveClass('text-indigo-400');
    expect(screen.getByText('General')).toHaveClass('text-gray-400');
    expect(screen.getByText('Auto')).toHaveClass('text-cyan-400');
  });
});
