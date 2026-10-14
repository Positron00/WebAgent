import React from "react";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { AppProvider } from "@/contexts/AppContext";
import { ChatProvider, useChat } from "@/contexts/ChatContext";
import { apiClient, ApiError } from "@/utils/apiClient";

jest.mock("@/utils/apiClient", () => {
  const actual =
    jest.requireActual<typeof import("@/utils/apiClient")>("@/utils/apiClient");
  return {
    ...actual,
    apiClient: { sendChatMessage: jest.fn() },
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

function reply(content: string, security?: unknown) {
  return {
    choices: [
      { message: { role: "assistant", content }, finish_reason: "stop" },
    ],
    route: { route: "agentic", decidedBy: "model", reason: "test" },
    ...(security ? { security } : {}),
  };
}

function Probe() {
  const {
    state,
    pendingConsent,
    sendMessage,
    sendWithConsent,
    editPendingMessage,
    setEditPendingHandler,
    approveAndRerun,
  } = useChat();
  const [draft, setDraft] = React.useState("");
  const firstAssistant = state.messages.find((m) => m.role === "assistant");

  React.useEffect(() => {
    setEditPendingHandler((text) => setDraft(text));
  }, [setEditPendingHandler]);

  return (
    <div>
      <input
        aria-label="draft"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
      />
      <button onClick={() => sendMessage("risky question")}>send</button>
      {pendingConsent && (
        <>
          <span data-testid="consent-open">consent</span>
          <button onClick={() => sendWithConsent()}>send anyway</button>
          <button onClick={() => editPendingMessage()}>edit pending</button>
        </>
      )}
      {firstAssistant?.id && (
        <button onClick={() => approveAndRerun(firstAssistant.id!)}>
          approve rerun
        </button>
      )}
      <ul>
        {state.messages.map((m, i) => (
          <li key={`${m.id}-${i}`}>
            {m.role}
            {m.refusal ? `:refusal:${m.refusal.code}` : `:${m.content}`}
          </li>
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
    </AppProvider>,
  );
}

describe("ChatContext guard flows", () => {
  beforeEach(() => {
    sendChatMessage.mockReset();
    window.localStorage.clear();
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ defaultAgenticMode: "off" }),
    }) as unknown as typeof fetch;
  });

  it("opens consent dialog on consent_required without an error banner", async () => {
    sendChatMessage.mockRejectedValueOnce(
      new ApiError("Consent required", 409, {
        code: "consent_required",
        reason: "Please confirm",
        consent_token: "tok-9",
      }),
    );

    renderChat();
    fireEvent.click(screen.getByText("send"));

    expect(await screen.findByTestId("consent-open")).toBeInTheDocument();
    expect(screen.getByText("user:risky question")).toBeInTheDocument();
    expect(screen.queryByText(/error:/)).not.toBeInTheDocument();
    await waitFor(() =>
      expect(screen.queryByText("loading")).not.toBeInTheDocument(),
    );
  });

  it("resends with consent token when the user chooses Send anyway", async () => {
    sendChatMessage
      .mockRejectedValueOnce(
        new ApiError("Consent required", 409, {
          code: "consent_required",
          reason: "Please confirm",
          consent_token: "tok-9",
        }),
      )
      .mockResolvedValueOnce(reply("allowed answer"));

    renderChat();
    fireEvent.click(screen.getByText("send"));
    await screen.findByTestId("consent-open");
    fireEvent.click(screen.getByText("send anyway"));

    await screen.findByText("assistant:allowed answer");
    const lastCall = sendChatMessage.mock.calls[1];
    expect(lastCall[6]).toMatchObject({ consentToken: "tok-9" });
  });

  it("returns pending text to the input on Edit message", async () => {
    sendChatMessage.mockRejectedValueOnce(
      new ApiError("Consent required", 409, {
        code: "consent_required",
        reason: "Please confirm",
        consent_token: "tok-9",
      }),
    );

    renderChat();
    fireEvent.click(screen.getByText("send"));
    await screen.findByTestId("consent-open");
    fireEvent.click(screen.getByText("edit pending"));

    expect(screen.getByLabelText("draft")).toHaveValue("risky question");
    expect(screen.queryByText("user:risky question")).not.toBeInTheDocument();
  });

  it("shows a refusal placeholder for declined responses", async () => {
    sendChatMessage.mockRejectedValueOnce(
      new ApiError("Declined", 403, { code: "declined", reason: "Blocked" }),
    );

    renderChat();
    fireEvent.click(screen.getByText("send"));

    expect(
      await screen.findByText("assistant:refusal:declined"),
    ).toBeInTheDocument();
    expect(screen.getByText("user:risky question")).toBeInTheDocument();
    expect(screen.queryByText(/error:/)).not.toBeInTheDocument();
  });

  it("approve and re-run inserts a new answer below the original", async () => {
    const security = {
      tainted: false,
      shadow: false,
      dropped: [],
      skipped: [],
      approvals: [
        {
          action: "tool::execute_code",
          reason: "Needs approval",
          approvalToken: "appr-1",
        },
      ],
    };

    sendChatMessage
      .mockResolvedValueOnce(reply("first", security))
      .mockResolvedValueOnce(reply("after approval"));

    renderChat();
    fireEvent.click(screen.getByText("send"));
    await screen.findByText("assistant:first");
    fireEvent.click(screen.getByText("approve rerun"));

    expect(
      await screen.findByText("assistant:after approval"),
    ).toBeInTheDocument();
    const items = screen.getAllByRole("listitem").map((li) => li.textContent);
    expect(items).toEqual([
      "user:risky question",
      "assistant:first",
      "assistant:after approval",
    ]);
    expect(sendChatMessage.mock.calls[1][6]).toMatchObject({
      forceAgentic: true,
      approvalTokens: ["appr-1"],
    });
  });
});
