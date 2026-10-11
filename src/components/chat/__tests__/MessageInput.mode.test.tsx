import { render, screen } from '@testing-library/react';
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
    expect(screen.getByRole('radio', { name: 'Off' })).toHaveAttribute('aria-checked', 'true');
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
