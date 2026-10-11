import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { AppProvider } from '@/contexts/AppContext';
import { ChatProvider, useChat } from '@/contexts/ChatContext';
import { apiClient } from '@/utils/apiClient';

jest.mock('@/utils/apiClient', () => ({
  apiClient: { sendChatMessage: jest.fn() },
}));

jest.mock('@/utils/logger', () => ({
  logger: { debug: jest.fn(), info: jest.fn(), warn: jest.fn(), error: jest.fn() },
}));

const sendChatMessage = apiClient.sendChatMessage as jest.Mock;

function reply(content: string) {
  return { choices: [{ message: { role: 'assistant', content }, finish_reason: 'stop' }] };
}

function Probe() {
  const { state, sendMessage } = useChat();
  return (
    <div>
      <button onClick={() => sendMessage('What is a tuple?')}>send</button>
      <ul>
        {state.messages.map((m) => (
          <li key={m.id}>{`${m.role}: ${m.content}`}</li>
        ))}
      </ul>
      {state.isLoading && <span>loading</span>}
      {state.error && <span>{`error: ${state.error}`}</span>}
    </div>
  );
}

function renderChat() {
  return render(
    <AppProvider>
      <ChatProvider>
        <Probe />
      </ChatProvider>
    </AppProvider>
  );
}

describe('ChatContext', () => {
  beforeEach(() => {
    sendChatMessage.mockReset();
    window.localStorage.clear();
  });

  it('keeps the question visible while waiting and after the answer arrives', async () => {
    let resolve!: (value: unknown) => void;
    sendChatMessage.mockReturnValue(new Promise((r) => (resolve = r)));
    renderChat();

    fireEvent.click(screen.getByText('send'));
    expect(await screen.findByText('user: What is a tuple?')).toBeInTheDocument();
    expect(screen.getByText('loading')).toBeInTheDocument();

    await act(async () => resolve(reply('An immutable sequence.')));

    const items = screen.getAllByRole('listitem').map((li) => li.textContent);
    expect(items).toEqual(['user: What is a tuple?', 'assistant: An immutable sequence.']);
    expect(screen.queryByText('loading')).not.toBeInTheDocument();
  });

  it('sends the conversation including the new question', async () => {
    sendChatMessage.mockResolvedValue(reply('ok'));
    renderChat();

    fireEvent.click(screen.getByText('send'));
    await screen.findByText('assistant: ok');

    const [messages] = sendChatMessage.mock.calls[0];
    expect(messages[messages.length - 1]).toMatchObject({ role: 'user', content: 'What is a tuple?' });
  });

  it('keeps the question and shows the error when the request fails', async () => {
    sendChatMessage.mockRejectedValue(new Error('Backend unavailable'));
    renderChat();

    fireEvent.click(screen.getByText('send'));

    expect(await screen.findByText('error: Backend unavailable')).toBeInTheDocument();
    expect(screen.getByText('user: What is a tuple?')).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByText('loading')).not.toBeInTheDocument());
  });
});
