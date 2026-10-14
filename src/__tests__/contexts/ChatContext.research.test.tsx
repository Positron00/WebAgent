import React from "react";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { AppProvider, useApp } from "@/contexts/AppContext";
import { ChatProvider, useChat } from "@/contexts/ChatContext";
import { apiClient } from "@/utils/apiClient";
import { ResearchFailed, ResearchRunOptions } from "@/types/research";

jest.mock("@/utils/apiClient", () => {
  const actual =
    jest.requireActual<typeof import("@/utils/apiClient")>("@/utils/apiClient");
  return {
    ...actual,
    apiClient: { sendChatMessage: jest.fn(), cancelTask: jest.fn() },
  };
});

jest.mock("@/utils/logger", () => ({
  logger: {
    debug: jest.fn(),
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
  },
}));

const sendChatMessage = apiClient.sendChatMessage as jest.Mock;
const cancelTask = apiClient.cancelTask as jest.Mock;

const reply = (content: string) => ({
  choices: [{ message: { role: "assistant", content }, finish_reason: "stop" }],
  route: { route: "agentic", decidedBy: "rule", reason: "Always" },
});

/** Behaves like a research run that only ends when it is stopped. */
function runUntilStopped(_messages: unknown, ...rest: unknown[]) {
  const { research } = rest[5] as { research: ResearchRunOptions };
  research.onTaskId?.("t1");
  research.onProgress?.({
    step: "Planning",
    latestLog: "Supervisor: planning",
    elapsedMs: 1000,
  });
  return new Promise((_resolve, reject) => {
    research.signal.addEventListener("abort", () =>
      reject(new ResearchFailed("cancelled", "Research stopped.")),
    );
  });
}

function Probe() {
  const { agenticMode } = useApp();
  const chat = useChat();
  const { state, researchRun, researchFailure } = chat;
  return (
    <div>
      <span>{`mode: ${agenticMode}`}</span>
      <button onClick={() => chat.sendMessage("Compare two databases")}>
        send
      </button>
      <button onClick={() => chat.stopResearch()}>stop</button>
      <button onClick={() => chat.retryResearch()}>retry</button>
      <button onClick={() => chat.answerDirectlyInstead()}>direct</button>
      <ul>
        {state.messages.map((m) => (
          <li key={m.id}>{`${m.role}: ${m.content}`}</li>
        ))}
      </ul>
      {state.isLoading && <span>loading</span>}
      {state.error && <span>{`error: ${state.error}`}</span>}
      {researchRun && (
        <span>{`run: ${researchRun.taskId} ${researchRun.progress?.step}`}</span>
      )}
      {researchFailure && (
        <span>{`failure: ${researchFailure.kind} ${researchFailure.message}`}</span>
      )}
    </div>
  );
}

async function renderChat() {
  render(
    <AppProvider>
      <ChatProvider>
        <Probe />
      </ChatProvider>
    </AppProvider>,
  );
  await screen.findByText("mode: always");
}

describe("ChatContext research runs", () => {
  beforeEach(() => {
    sendChatMessage.mockReset();
    cancelTask.mockReset().mockResolvedValue(undefined);
    window.localStorage.clear();
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ defaultAgenticMode: "always" }),
    }) as unknown as typeof fetch;
  });

  it('shows the run, and Stop cancels it, ends loading and adds "Research stopped."', async () => {
    sendChatMessage.mockImplementation(runUntilStopped);
    await renderChat();

    fireEvent.click(screen.getByText("send"));
    expect(await screen.findByText("run: t1 Planning")).toBeInTheDocument();
    expect(screen.getByText("loading")).toBeInTheDocument();

    fireEvent.click(screen.getByText("stop"));

    expect(screen.queryByText("loading")).not.toBeInTheDocument();
    expect(
      await screen.findByText("assistant: Research stopped."),
    ).toBeInTheDocument();
    expect(cancelTask).toHaveBeenCalledWith("t1");
    expect(screen.queryByText(/^run:/)).not.toBeInTheDocument();
    expect(screen.queryByText(/error:|failure:/)).not.toBeInTheDocument();
  });

  it('adds "Research stopped." when the backend reports the run cancelled', async () => {
    sendChatMessage.mockRejectedValue(
      new ResearchFailed("cancelled", "Research stopped."),
    );
    await renderChat();

    fireEvent.click(screen.getByText("send"));

    expect(
      await screen.findByText("assistant: Research stopped."),
    ).toBeInTheDocument();
    expect(screen.queryByText("loading")).not.toBeInTheDocument();
  });

  it.each(["timeout", "budget", "unreachable", "failed"] as const)(
    "a %s failure is shown with retry info, not as an error banner",
    async (kind) => {
      sendChatMessage.mockRejectedValue(
        new ResearchFailed(kind, `why ${kind}`),
      );
      await renderChat();

      fireEvent.click(screen.getByText("send"));

      expect(
        await screen.findByText(`failure: ${kind} why ${kind}`),
      ).toBeInTheDocument();
      expect(screen.queryByText("loading")).not.toBeInTheDocument();
      expect(screen.queryByText(/error:/)).not.toBeInTheDocument();
      expect(screen.getAllByRole("listitem")).toHaveLength(1);
    },
  );

  it("Try again resends the same messages in the same mode", async () => {
    sendChatMessage
      .mockRejectedValueOnce(new ResearchFailed("timeout", "too long"))
      .mockResolvedValueOnce(reply("the report"));
    await renderChat();

    fireEvent.click(screen.getByText("send"));
    await screen.findByText(/^failure: timeout/);
    fireEvent.click(screen.getByText("retry"));

    expect(
      await screen.findByText("assistant: the report"),
    ).toBeInTheDocument();
    expect(screen.queryByText(/^failure:/)).not.toBeInTheDocument();
    const [first, second] = sendChatMessage.mock.calls;
    expect(second[0]).toEqual(first[0]);
    expect(second[5]).toBe("always");
  });

  it("Answer directly instead resends the same messages with mode off", async () => {
    sendChatMessage
      .mockRejectedValueOnce(new ResearchFailed("unreachable", "down"))
      .mockResolvedValueOnce(reply("direct answer"));
    await renderChat();

    fireEvent.click(screen.getByText("send"));
    await screen.findByText(/^failure: unreachable/);
    await act(async () => {
      fireEvent.click(screen.getByText("direct"));
    });

    await waitFor(() => expect(sendChatMessage).toHaveBeenCalledTimes(2));
    const [first, second] = sendChatMessage.mock.calls;
    expect(second[0]).toEqual(first[0]);
    expect(second[5]).toBe("off");
    expect(second[6].forceAgentic).toBeUndefined();
    expect(
      await screen.findByText("assistant: direct answer"),
    ).toBeInTheDocument();
  });
});
