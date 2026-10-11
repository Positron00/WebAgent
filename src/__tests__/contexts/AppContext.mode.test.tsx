import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { AppProvider, useApp } from '@/contexts/AppContext';

type SettingsHandler = () => Promise<unknown>;

/**
 * Mocks global.fetch for any URL. `/api/settings` uses the given handler;
 * everything else (e.g. the `/api/ping` connection check) gets a plain OK.
 */
function mockFetch(settings: SettingsHandler) {
  const fetchMock = jest.fn((url: RequestInfo | URL) => {
    if (String(url).includes('/api/settings')) return settings();
    return Promise.resolve({ ok: true, status: 200, json: async () => ({}) });
  });
  global.fetch = fetchMock as unknown as typeof fetch;
  return fetchMock;
}

const settingsCalls = (fetchMock: jest.Mock) =>
  fetchMock.mock.calls.filter(([url]) => String(url).includes('/api/settings'));

function Consumer() {
  const { agenticMode, modeReady, setAgenticMode } = useApp();
  return (
    <div>
      <span data-testid="mode">{agenticMode}</span>
      <span data-testid="ready">{String(modeReady)}</span>
      <button onClick={() => setAgenticMode('auto')}>set auto</button>
    </div>
  );
}

function renderApp() {
  return render(
    <AppProvider>
      <Consumer />
    </AppProvider>
  );
}

describe('AppContext agentic mode', () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it('shows off and not ready before /api/settings answers', async () => {
    let resolveSettings!: (value: unknown) => void;
    mockFetch(() => new Promise((r) => (resolveSettings = r)));
    renderApp();

    expect(screen.getByTestId('mode')).toHaveTextContent('off');
    expect(screen.getByTestId('ready')).toHaveTextContent('false');

    await act(async () => {
      resolveSettings({ ok: true, status: 200, json: async () => ({ defaultAgenticMode: 'off' }) });
    });
    await waitFor(() => expect(screen.getByTestId('ready')).toHaveTextContent('true'));
  });

  it('uses the server default and does not save it', async () => {
    mockFetch(() =>
      Promise.resolve({ ok: true, status: 200, json: async () => ({ defaultAgenticMode: 'auto' }) })
    );
    renderApp();

    await waitFor(() => expect(screen.getByTestId('ready')).toHaveTextContent('true'));
    expect(screen.getByTestId('mode')).toHaveTextContent('auto');
    expect(window.localStorage.getItem('agentic_mode')).toBeNull();
  });

  it('falls back to off when /api/settings fails', async () => {
    mockFetch(() => Promise.reject(new TypeError('Failed to fetch')));
    renderApp();

    await waitFor(() => expect(screen.getByTestId('ready')).toHaveTextContent('true'));
    expect(screen.getByTestId('mode')).toHaveTextContent('off');
  });

  it('uses a saved mode without calling /api/settings', async () => {
    window.localStorage.setItem('agentic_mode', 'always');
    const fetchMock = mockFetch(() =>
      Promise.resolve({ ok: true, status: 200, json: async () => ({ defaultAgenticMode: 'off' }) })
    );
    renderApp();

    await waitFor(() => expect(screen.getByTestId('ready')).toHaveTextContent('true'));
    expect(screen.getByTestId('mode')).toHaveTextContent('always');
    expect(settingsCalls(fetchMock)).toHaveLength(0);
  });

  it('setAgenticMode updates the mode and saves it', async () => {
    mockFetch(() =>
      Promise.resolve({ ok: true, status: 200, json: async () => ({ defaultAgenticMode: 'off' }) })
    );
    renderApp();
    await waitFor(() => expect(screen.getByTestId('ready')).toHaveTextContent('true'));

    fireEvent.click(screen.getByText('set auto'));

    expect(screen.getByTestId('mode')).toHaveTextContent('auto');
    expect(window.localStorage.getItem('agentic_mode')).toBe('auto');
  });
});
