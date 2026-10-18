import React from "react";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { TopicList } from "../TopicList";
import { parseTopicList, type TopicSummary } from "@/types/topics";

const ID = "6f1d3b0e-6c1b-4f7e-9a51-2b7a1c0d9e11";
const NOW = new Date("2026-10-18T10:00:00Z");
const fetchMock = jest.fn();

const reply = (status: number, body: unknown = {}) =>
  ({
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  }) as Response;

function topic(overrides: Record<string, unknown> = {}): TopicSummary {
  const [parsed] = parseTopicList({
    topics: [
      {
        topic_id: ID,
        name: "Database news",
        query: "What's new in Postgres?",
        schedule: {
          every: "week",
          weekday: "mon",
          time: "07:30",
          tz: "Europe/Paris",
        },
        notify: { when: "changes" },
        subscriber_count: 0,
        state: "active",
        next_due: "2026-10-19T05:30:00Z",
        last_run: {
          run_id: "r1",
          status: "completed",
          completed_at: "2026-10-12T05:35:00Z",
        },
        last_new_sources: 3,
        skipped: null,
        ...overrides,
      },
    ],
  });
  return parsed;
}

beforeEach(() => {
  fetchMock.mockReset();
  fetchMock.mockResolvedValue(reply(200));
  global.fetch = fetchMock as unknown as typeof fetch;
});

describe("TopicList", () => {
  it("shows an empty state that links to the chat", () => {
    render(<TopicList topics={[]} onChanged={jest.fn()} />);
    expect(
      screen.getByRole("link", {
        name: "Track a research answer to see it here.",
      }),
    ).toHaveAttribute("href", "/");
  });

  it("shows the name, the schedule in words, the state, and the last run's new sources", () => {
    render(<TopicList topics={[topic()]} onChanged={jest.fn()} now={NOW} />);
    const row = screen.getAllByRole("row")[1];
    expect(
      within(row).getByRole("link", { name: "Database news" }),
    ).toHaveAttribute("href", `/topics/${ID}`);
    expect(
      within(row).getByText("Mondays 07:30 Europe/Paris"),
    ).toBeInTheDocument();
    expect(within(row).getByText("Active")).toBeInTheDocument();
    expect(within(row).getByText(/\(3 new sources\)/)).toBeInTheDocument();
  });

  it.each([
    ["active", "Active"],
    ["paused", "Paused"],
    ["held_failures", "Held: failures"],
    ["held_budget", "Held: budget"],
  ])("shows the %s state as text", (state, label) => {
    render(
      <TopicList topics={[topic({ state })]} onChanged={jest.fn()} now={NOW} />,
    );
    expect(screen.getByText(label)).toBeInTheDocument();
  });

  it.each([
    ["daily_cap", "daily limit"],
    ["budget", "budget"],
  ])("shows a %s skip", (reason, words) => {
    render(
      <TopicList
        topics={[topic({ skipped: { reason, at: "2026-10-12T05:30:00Z" } })]}
        onChanged={jest.fn()}
        now={NOW}
      />,
    );
    expect(
      screen.getByText(new RegExp(`^Skipped on .+: ${words}$`)),
    ).toBeInTheDocument();
  });

  it("pauses an active topic with PATCH status paused", async () => {
    const onChanged = jest.fn();
    render(<TopicList topics={[topic()]} onChanged={onChanged} now={NOW} />);
    fireEvent.click(
      screen.getByRole("button", { name: "Pause Database news" }),
    );
    await waitFor(() => expect(onChanged).toHaveBeenCalled());
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`/api/topics/${ID}`);
    expect(init.method).toBe("PATCH");
    expect(JSON.parse(init.body)).toEqual({ status: "paused" });
  });

  it.each(["paused", "held_failures", "held_budget"])(
    "resumes a %s topic with POST resume",
    async (state) => {
      const onChanged = jest.fn();
      render(
        <TopicList
          topics={[topic({ state })]}
          onChanged={onChanged}
          now={NOW}
        />,
      );
      fireEvent.click(
        screen.getByRole("button", { name: "Resume Database news" }),
      );
      await waitFor(() => expect(onChanged).toHaveBeenCalled());
      expect(fetchMock.mock.calls[0][0]).toBe(`/api/topics/${ID}/resume`);
      expect(fetchMock.mock.calls[0][1].method).toBe("POST");
    },
  );

  it("shows a resume refused by the cap in words", async () => {
    fetchMock.mockResolvedValue(reply(409, { code: "topic_limit" }));
    render(
      <TopicList
        topics={[topic({ state: "paused" })]}
        onChanged={jest.fn()}
        now={NOW}
      />,
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Resume Database news" }),
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "You have 10 active topics; pause or delete one first.",
    );
  });

  it("runs now, then disables Run now until the run shows up", async () => {
    const onChanged = jest.fn();
    render(<TopicList topics={[topic()]} onChanged={onChanged} now={NOW} />);
    const button = screen.getByRole("button", {
      name: "Run Database news now",
    });
    expect(button).toBeEnabled();
    fireEvent.click(button);
    await waitFor(() => expect(onChanged).toHaveBeenCalled());
    expect(fetchMock.mock.calls[0][0]).toBe(`/api/topics/${ID}/run`);
    expect(fetchMock.mock.calls[0][1].method).toBe("POST");
    expect(button).toBeDisabled();
  });

  it.each([
    ["a run is due", { next_due: "2026-10-18T09:59:00Z" }],
    [
      "a run is running",
      { last_run: { status: "running", started_at: "2026-10-18T09:58:00Z" } },
    ],
  ])("disables Run now while %s", (_why, overrides) => {
    render(
      <TopicList topics={[topic(overrides)]} onChanged={jest.fn()} now={NOW} />,
    );
    expect(
      screen.getByRole("button", { name: "Run Database news now" }),
    ).toBeDisabled();
  });

  it("asks before deleting, naming the topic, and cancels without a request", () => {
    render(<TopicList topics={[topic()]} onChanged={jest.fn()} now={NOW} />);
    fireEvent.click(
      screen.getByRole("button", { name: "Delete Database news" }),
    );
    const dialog = screen.getByRole("dialog", {
      name: "Delete “Database news”?",
    });
    expect(dialog).toHaveTextContent("Database news");
    fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("deletes after the confirmation", async () => {
    fetchMock.mockResolvedValue(reply(204));
    const onChanged = jest.fn();
    render(<TopicList topics={[topic()]} onChanged={onChanged} now={NOW} />);
    fireEvent.click(
      screen.getByRole("button", { name: "Delete Database news" }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Delete topic" }));
    await waitFor(() => expect(onChanged).toHaveBeenCalled());
    expect(fetchMock.mock.calls[0][0]).toBe(`/api/topics/${ID}`);
    expect(fetchMock.mock.calls[0][1].method).toBe("DELETE");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
});

describe("parseTopicList", () => {
  it("skips malformed items and keeps the rest", () => {
    const parsed = parseTopicList({
      topics: [
        { name: "no id" },
        null,
        {
          topic_id: ID,
          name: "x",
          schedule: { every: "day", time: "07:30", tz: "UTC" },
        },
      ],
    });
    expect(parsed).toHaveLength(1);
    expect(parsed[0]).toMatchObject({
      topicId: ID,
      state: "active",
      running: false,
    });
  });

  it("returns an empty list for an unexpected reply", () => {
    expect(parseTopicList(null)).toEqual([]);
    expect(parseTopicList({ topics: "nope" })).toEqual([]);
  });
});
