import { fireEvent, render, screen, useChat as mockedChat } from "@/test-utils";
import { MessageList } from "../MessageList";
import { ChatMessage } from "@/types/chat";

const messages: ChatMessage[] = [
  { id: "u1", role: "user", content: "Compare two databases" },
];
const chat = mockedChat();

describe("MessageList research states and diagnostics", () => {
  afterEach(() => {
    chat.researchRun = null;
    chat.researchFailure = null;
    jest.clearAllMocks();
  });

  it("shows ResearchStatus with Stop while a research run is loading", () => {
    chat.researchRun = {
      taskId: "t1",
      startedAt: Date.now(),
      progress: {
        step: "Planning",
        latestLog: "Supervisor: planning",
        elapsedMs: 0,
      },
    };
    render(<MessageList messages={messages} isLoading={true} />);

    expect(screen.getByText("Thinking… · Planning")).toBeInTheDocument();
    expect(screen.queryByText("Thinking...")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Stop research" }));
    expect(chat.stopResearch).toHaveBeenCalledTimes(1);
  });

  it("keeps the plain spinner for direct answers", () => {
    render(<MessageList messages={messages} isLoading={true} />);
    expect(screen.getByText("Thinking...")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Stop research" }),
    ).not.toBeInTheDocument();
  });

  it("shows ResearchFailed instead of the error box", () => {
    chat.researchFailure = {
      kind: "budget",
      message: "budget",
      retry: { messages, imageBase64: null },
    };
    render(
      <MessageList messages={messages} isLoading={false} error="ignored" />,
    );

    expect(
      screen.getByText("The research used its token budget and was stopped."),
    ).toBeInTheDocument();
    expect(screen.queryByText("ignored")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    fireEvent.click(
      screen.getByRole("button", { name: "Answer directly instead" }),
    );
    expect(chat.retryResearch).toHaveBeenCalledTimes(1);
    expect(chat.answerDirectlyInstead).toHaveBeenCalledTimes(1);
  });

  it.each([
    [{ ok: true, status: 200 }, "Web app reachable"],
    [{ ok: false, status: 503 }, "Web app answered HTTP 503"],
  ])(
    "diagnostics call only /api/ping and show the result",
    async (reply, text) => {
      const fetchMock = jest.fn().mockResolvedValue(reply);
      global.fetch = fetchMock as unknown as typeof fetch;
      render(
        <MessageList
          messages={messages}
          isLoading={false}
          error="Something failed"
        />,
      );

      fireEvent.click(screen.getByRole("button", { name: "Run Diagnostics" }));

      expect(await screen.findByText(text)).toBeInTheDocument();
      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(fetchMock.mock.calls[0][0]).toBe("/api/ping");
    },
  );

  it("diagnostics report a network failure", async () => {
    global.fetch = jest
      .fn()
      .mockRejectedValue(new TypeError("offline")) as unknown as typeof fetch;
    render(
      <MessageList
        messages={messages}
        isLoading={false}
        error="Something failed"
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Run Diagnostics" }));

    expect(
      await screen.findByText("Web app not reachable"),
    ).toBeInTheDocument();
  });
});
