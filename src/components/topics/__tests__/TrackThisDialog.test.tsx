import React from "react";
import { act, fireEvent, render, screen, waitFor } from "@/test-utils";
import { TrackThisDialog } from "../TrackThisDialog";
import { MessageList } from "@/components/chat/MessageList";
import type { ChatMessage } from "@/types/chat";

const TASK = "0b8d4c2a-1f3e-4a5b-8c6d-7e8f9a0b1c2d";
const fetchMock = jest.fn();
const realDateTimeFormat = Intl.DateTimeFormat;

beforeEach(() => {
  fetchMock.mockReset();
  global.fetch = fetchMock as unknown as typeof fetch;
  // The browser's time zone, as Intl reports it
  jest
    .spyOn(Intl, "DateTimeFormat")
    .mockImplementation(
      (...args: ConstructorParameters<typeof Intl.DateTimeFormat>) => {
        const real = new realDateTimeFormat(...args);
        return Object.assign(Object.create(real), {
          resolvedOptions: () => ({
            ...real.resolvedOptions(),
            timeZone: "Europe/Paris",
          }),
        }) as Intl.DateTimeFormat;
      },
    );
});
afterEach(() => jest.restoreAllMocks());

const reply = (status: number, body: unknown) =>
  ({
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  }) as Response;

// 2026-10-14 is a Wednesday
const WEDNESDAY = new Date(2026, 9, 14, 12, 0);

function renderDialog(
  props: Partial<React.ComponentProps<typeof TrackThisDialog>> = {},
) {
  const onClose = jest.fn();
  const onCreated = jest.fn();
  render(
    <TrackThisDialog
      taskId={TASK}
      query="Compare Postgres and MySQL for analytics in 2026"
      title={"A".repeat(120)}
      onClose={onClose}
      onCreated={onCreated}
      now={WEDNESDAY}
      {...props}
    />,
  );
  return { onClose, onCreated };
}

describe("TrackThisDialog", () => {
  it("prefills the question, a name cut to 80 characters and the defaults", () => {
    renderDialog();

    expect(
      screen.getByRole("dialog", { name: "Track this question" }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("Question")).toHaveValue(
      "Compare Postgres and MySQL for analytics in 2026",
    );
    expect(screen.getByLabelText("Name")).toHaveValue("A".repeat(80));
    expect(screen.getByLabelText("Frequency")).toHaveValue("week");
    expect(screen.getByLabelText("Weekday")).toHaveValue("wed");
    expect(screen.getByLabelText("Time")).toHaveValue("07:30");
    expect(screen.getByLabelText("Something changes")).toBeChecked();
    expect(screen.getByLabelText("Email (optional)")).toHaveValue("");
    expect(
      screen.getByText("We'll send a confirmation link first."),
    ).toBeInTheDocument();
  });

  it("shows the browser's time zone read-only", () => {
    renderDialog();
    const tz = screen.getByLabelText("Time zone");
    expect(tz).toHaveValue("Europe/Paris");
    expect(tz).toHaveAttribute("readonly");
  });

  it("uses the question as the name when there is no title", () => {
    renderDialog({ title: undefined, query: "What changed in EU AI rules?" });
    expect(screen.getByLabelText("Name")).toHaveValue(
      "What changed in EU AI rules?",
    );
  });

  it("hides the weekday for a daily topic and posts the schedule with the time zone", async () => {
    fetchMock.mockResolvedValue(reply(201, { topic_id: "t" }));
    const { onCreated } = renderDialog({ title: "DB comparison" });

    fireEvent.change(screen.getByLabelText("Frequency"), {
      target: { value: "day" },
    });
    expect(screen.queryByLabelText("Weekday")).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Time"), {
      target: { value: "18:05" },
    });
    fireEvent.click(screen.getByLabelText("Every run"));
    fireEvent.click(screen.getByRole("button", { name: "Create topic" }));

    await waitFor(() => expect(onCreated).toHaveBeenCalled());
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/topics");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({
      name: "DB comparison",
      query: "Compare Postgres and MySQL for analytics in 2026",
      schedule: { every: "day", time: "18:05", tz: "Europe/Paris" },
      notify: { when: "always" },
      created_from_task: TASK,
    });
    expect(screen.getByRole("status")).toHaveTextContent("DB comparison");
    expect(screen.getByRole("link", { name: "Topics" })).toHaveAttribute(
      "href",
      "/topics",
    );
  });

  it("sends the weekday for a weekly topic", async () => {
    fetchMock.mockResolvedValue(reply(201, {}));
    renderDialog();
    fireEvent.change(screen.getByLabelText("Weekday"), {
      target: { value: "mon" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Create topic" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    expect(JSON.parse(fetchMock.mock.calls[0][1].body).schedule).toEqual({
      every: "week",
      weekday: "mon",
      time: "07:30",
      tz: "Europe/Paris",
    });
  });

  it.each([
    [
      409,
      { code: "topic_limit" },
      "You have 10 active topics; pause or delete one first.",
    ],
    [
      409,
      { detail: { code: "topic_limit" } },
      "You have 10 active topics; pause or delete one first.",
    ],
    [
      403,
      {
        code: "declined",
        outcome: "decline",
        reason: "It asks for personal data.",
      },
      "This question can't be scheduled: It asks for personal data.",
    ],
    [
      409,
      { code: "consent_required", reason: "Sensitive topic" },
      "This question can't be scheduled: Sensitive topic.",
    ],
  ])("shows a %i %j in words", async (status, body, message) => {
    fetchMock.mockResolvedValue(reply(status, body));
    const { onCreated } = renderDialog();
    fireEvent.click(screen.getByRole("button", { name: "Create topic" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(message);
    expect(onCreated).not.toHaveBeenCalled();
    // The form stays open so the user can change it
    expect(screen.getByRole("button", { name: "Create topic" })).toBeEnabled();
  });

  it("sends no subscriber call without an email", async () => {
    fetchMock.mockResolvedValue(reply(201, { topic_id: "t1" }));
    const { onCreated } = renderDialog();
    fireEvent.click(screen.getByRole("button", { name: "Create topic" }));
    await waitFor(() => expect(onCreated).toHaveBeenCalled());
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(JSON.parse(fetchMock.mock.calls[0][1].body).notify).toEqual({
      when: "changes",
    });
  });

  it("adds the email with a PATCH after the topic is created", async () => {
    const TOPIC = "6f1d3b0e-6c1b-4f7e-9a51-2b7a1c0d9e11";
    fetchMock
      .mockResolvedValueOnce(reply(201, { topic_id: TOPIC }))
      .mockResolvedValueOnce(reply(200, { topic_id: TOPIC }));
    const { onCreated } = renderDialog({ title: "DB comparison" });
    fireEvent.change(screen.getByLabelText("Email (optional)"), {
      target: { value: " me@example.org " },
    });
    fireEvent.click(screen.getByRole("button", { name: "Create topic" }));

    await waitFor(() => expect(onCreated).toHaveBeenCalled());
    const [createUrl, createInit] = fetchMock.mock.calls[0];
    expect(createUrl).toBe("/api/topics");
    // The address isn't part of the create body
    expect(JSON.parse(createInit.body)).not.toHaveProperty("add_subscriber");
    expect(JSON.parse(createInit.body).notify).toEqual({ when: "changes" });
    const [url, init] = fetchMock.mock.calls[1];
    expect(url).toBe(`/api/topics/${TOPIC}`);
    expect(init.method).toBe("PATCH");
    expect(JSON.parse(init.body)).toEqual({ add_subscriber: "me@example.org" });
    expect(screen.getByRole("status")).toHaveTextContent(
      "A confirmation link was sent to me@example.org.",
    );
  });

  it.each([
    [{ code: "email_disabled" }, "Email isn't turned on for this WebAgent"],
    [{ code: "subscriber_limit" }, "already has 5 email addresses"],
  ])(
    "says the topic was made when adding the email fails with %j",
    async (body, message) => {
      fetchMock
        .mockResolvedValueOnce(reply(201, { topic_id: "t1" }))
        .mockResolvedValueOnce(reply(409, body));
      const { onCreated } = renderDialog();
      fireEvent.change(screen.getByLabelText("Email (optional)"), {
        target: { value: "me@example.org" },
      });
      fireEvent.click(screen.getByRole("button", { name: "Create topic" }));
      await waitFor(() => expect(onCreated).toHaveBeenCalled());
      const alert = screen.getByRole("alert");
      expect(alert).toHaveTextContent(
        "The topic was created, but the email address wasn't added",
      );
      expect(alert).toHaveTextContent(message);
      expect(screen.getByRole("status")).not.toHaveTextContent(
        "confirmation link",
      );
    },
  );

  it("won't submit an address that isn't valid", () => {
    renderDialog();
    fireEvent.change(screen.getByLabelText("Email (optional)"), {
      target: { value: "me@nowhere" },
    });
    expect(
      screen.getByText("That email address isn't valid."),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Create topic" })).toBeDisabled();
  });

  it("won't submit an empty question", () => {
    renderDialog();
    fireEvent.change(screen.getByLabelText("Question"), {
      target: { value: "  " },
    });
    expect(screen.getByRole("button", { name: "Create topic" })).toBeDisabled();
  });

  it("traps focus inside the dialog and closes on Escape", () => {
    const { onClose } = renderDialog();
    const name = screen.getByLabelText("Name");
    const cancel = screen.getByRole("button", { name: "Cancel" });
    const create = screen.getByRole("button", { name: "Create topic" });

    expect(name).toHaveFocus();
    create.focus();
    fireEvent.keyDown(document, { key: "Tab" });
    expect(name).toHaveFocus();
    fireEvent.keyDown(document, { key: "Tab", shiftKey: true });
    expect(create).toHaveFocus();
    expect(cancel).toBeInTheDocument();

    fireEvent.keyDown(document, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});

describe("Track this in the chat", () => {
  const question: ChatMessage = {
    id: "u1",
    role: "user",
    content: "compare them again",
  };
  const agentic = (task?: ChatMessage["task"]): ChatMessage => ({
    id: "a1",
    role: "assistant",
    content: "Report body",
    route: { route: "agentic", decidedBy: "rule", reason: "research" },
    ...(task ? { task } : {}),
  });

  it("shows on a completed research answer and prefills from standalone_query and the title", () => {
    render(
      <MessageList
        messages={[
          question,
          agentic({
            id: TASK,
            title: "Postgres vs MySQL",
            standaloneQuery: "Compare Postgres and MySQL for analytics",
          }),
        ]}
        isLoading={false}
      />,
    );
    act(() => {
      fireEvent.click(
        screen.getByRole("button", {
          name: "Track this question on a schedule",
        }),
      );
    });
    expect(screen.getByLabelText("Question")).toHaveValue(
      "Compare Postgres and MySQL for analytics",
    );
    expect(screen.getByLabelText("Name")).toHaveValue("Postgres vs MySQL");
  });

  it("falls back to the user's question without a standalone_query", () => {
    render(
      <MessageList
        messages={[question, agentic({ id: TASK })]}
        isLoading={false}
      />,
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Track this question on a schedule" }),
    );
    expect(screen.getByLabelText("Question")).toHaveValue("compare them again");
  });

  it("doesn't show on direct answers or research answers without a task", () => {
    const direct: ChatMessage = {
      id: "a2",
      role: "assistant",
      content: "Short answer",
      route: { route: "direct", decidedBy: "rule", reason: "simple" },
      task: { id: TASK },
    };
    render(
      <MessageList
        messages={[question, direct, agentic()]}
        isLoading={false}
      />,
    );
    expect(
      screen.queryByRole("button", { name: /Track this/ }),
    ).not.toBeInTheDocument();
  });
});
