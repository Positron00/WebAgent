import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { AppProvider } from '@/contexts/AppContext';
import { ChatProvider, useChat } from '@/contexts/ChatContext';
import { apiClient } from '@/utils/apiClient';
import { ChatMessage } from '@/types/chat';
import { RouteInfo } from '@/types/routing';

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

const DIRECT: RouteInfo = { route: 'direct', decidedBy: 'rule', reason: 'Auto · simple' };
const AGENTIC: RouteInfo = { route: 'agentic', decidedBy: 'model', reason: 'Auto · research' };

function routedReply(content: string, route: RouteInfo, logs?: string[]) {
  return {
    choices: [{ message: { role: 'assistant', content, logs }, finish_reason: 'stop' }],
    route,
  };
}

function Probe() {
  const { state, sendMessage, researchInDepth } = useChat();
  const firstAssistant = state.messages.find((m) => m.role === 'assistant');
  return (
    <div>
      <button onClick={() => sendMessage('What is a tuple?')}>send</button>
      <button onClick={() => sendMessage('And a list?')}>send second</button>
      {firstAssistant?.id && (
        <button onClick={() => researchInDepth(firstAssistant.id!)}>research</button>
      )}
      <ul>
        {state.messages.map((m, i) => (
          <li key={`${m.id}-${i}`}>{`${m.role}: ${m.content}`}</li>
        ))}
      </ul>
      <pre data-testid="messages">{JSON.stringify(state.messages)}</pre>
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

function readMessages(): ChatMessage[] {
  return JSON.parse(screen.getByTestId('messages').textContent || '[]');
}

function assistantMessages(): ChatMessage[] {
  return readMessages().filter((m) => m.role === 'assistant');
}

describe('ChatContext', () => {
  beforeEach(() => {
    sendChatMessage.mockReset();
    window.localStorage.clear();
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ defaultAgenticMode: 'off' }),
    }) as unknown as typeof fetch;
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

    const [messages, , , , , mode] = sendChatMessage.mock.calls[0];
    expect(messages[messages.length - 1]).toMatchObject({ role: 'user', content: 'What is a tuple?' });
    expect(mode).toBe('off');
  });

  it('keeps the question and shows the error when the request fails', async () => {
    sendChatMessage.mockRejectedValue(new Error('Backend unavailable'));
    renderChat();

    fireEvent.click(screen.getByText('send'));

    expect(await screen.findByText('error: Backend unavailable')).toBeInTheDocument();
    expect(screen.getByText('user: What is a tuple?')).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByText('loading')).not.toBeInTheDocument());
  });

  it('does not add mock sources to a direct reply when citeSources is on', async () => {
    sendChatMessage.mockResolvedValue(routedReply('direct answer', DIRECT));
    renderChat();

    fireEvent.click(screen.getByText('send'));
    await screen.findByText('assistant: direct answer');

    const [, , , , citeSources] = sendChatMessage.mock.calls[0];
    expect(citeSources).toBe(true);
    expect(assistantMessages()[0].sources).toBeUndefined();
  });

  it('keeps sources only when the route is agentic', async () => {
    const sources = [{ title: 'Example', url: 'https://example.com' }];
    const direct = routedReply('direct answer', DIRECT);
    sendChatMessage.mockResolvedValue({
      ...direct,
      choices: [{ ...direct.choices[0], message: { ...direct.choices[0].message, sources } }],
    });
    renderChat();

    fireEvent.click(screen.getByText('send'));
    await screen.findByText('assistant: direct answer');
    expect(assistantMessages()[0].sources).toBeUndefined();
  });

  it('stores the route from the response on the assistant message', async () => {
    sendChatMessage.mockResolvedValue(routedReply('researched', AGENTIC));
    renderChat();

    fireEvent.click(screen.getByText('send'));
    await screen.findByText('assistant: researched');

    expect(assistantMessages()[0].route).toEqual(AGENTIC);
  });

  it('keeps logs when the route is agentic', async () => {
    sendChatMessage.mockResolvedValue(routedReply('researched', AGENTIC, ['step 1', 'step 2']));
    renderChat();

    fireEvent.click(screen.getByText('send'));
    await screen.findByText('assistant: researched');

    expect(assistantMessages()[0].logs).toEqual(['step 1', 'step 2']);
  });

  it('drops logs when the route is direct', async () => {
    sendChatMessage.mockResolvedValue(routedReply('direct answer', DIRECT, ['should not keep']));
    renderChat();

    fireEvent.click(screen.getByText('send'));
    await screen.findByText('assistant: direct answer');

    expect(assistantMessages()[0].logs).toBeUndefined();
  });

  it('researchInDepth resends up to the user message with forceAgentic and appends a reply', async () => {
    // Message ids come from Date.now(); make them unique so the two replies differ.
    let now = 1_700_000_000_000;
    const nowSpy = jest.spyOn(Date, 'now').mockImplementation(() => now++);

    try {
      sendChatMessage
        .mockResolvedValueOnce(routedReply('direct answer', DIRECT))
        .mockResolvedValueOnce(routedReply('deep answer', { ...AGENTIC, decidedBy: 'override' }));
      renderChat();

      fireEvent.click(screen.getByText('send'));
      await screen.findByText('assistant: direct answer');

      fireEvent.click(screen.getByText('research'));
      await screen.findByText('assistant: deep answer');

      expect(sendChatMessage).toHaveBeenCalledTimes(2);
      const [sent, image, , , , , options] = sendChatMessage.mock.calls[1];
      expect(options).toEqual({ forceAgentic: true });
      expect(image).toBeNull();
      expect(sent).toHaveLength(1);
      expect(sent[0]).toMatchObject({ role: 'user', content: 'What is a tuple?' });

      const items = screen.getAllByRole('listitem').map((li) => li.textContent);
      expect(items).toEqual([
        'user: What is a tuple?',
        'assistant: direct answer',
        'assistant: deep answer',
      ]);
      const [first, second] = assistantMessages();
      expect(first.route).toEqual(DIRECT);
      expect(second.route?.route).toBe('agentic');
      expect(second.id).not.toBe(first.id);
    } finally {
      nowSpy.mockRestore();
    }
  });

  it('researchInDepth puts the reply under its own question when the chat has moved on', async () => {
    sendChatMessage
      .mockResolvedValueOnce(routedReply('direct answer', DIRECT))
      .mockResolvedValueOnce(routedReply('list answer', DIRECT))
      .mockResolvedValueOnce(routedReply('deep answer', { ...AGENTIC, decidedBy: 'override' }));
    renderChat();

    fireEvent.click(screen.getByText('send'));
    await screen.findByText('assistant: direct answer');
    fireEvent.click(screen.getByText('send second'));
    await screen.findByText('assistant: list answer');

    fireEvent.click(screen.getByText('research'));
    await screen.findByText('assistant: deep answer');

    expect(screen.getAllByRole('listitem').map((li) => li.textContent)).toEqual([
      'user: What is a tuple?',
      'assistant: direct answer',
      'assistant: deep answer',
      'user: And a list?',
      'assistant: list answer',
    ]);
  });
});
