import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { SubscriberList } from "../SubscriberList";
import {
  isPlausibleEmail,
  parseTopicSummary,
  topicErrorMessage,
  type TopicSubscriber,
} from "@/types/topics";

const ID = "6f1d3b0e-6c1b-4f7e-9a51-2b7a1c0d9e11";
const fetchMock = jest.fn();

const reply = (status: number, body: unknown = {}) =>
  ({
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  }) as Response;

beforeEach(() => {
  fetchMock.mockReset();
  fetchMock.mockResolvedValue(reply(200));
  global.fetch = fetchMock as unknown as typeof fetch;
});

/** Subscribers as the definition (GET /api/topics/{id}) lists them. */
function subscribers(...entries: unknown[]): TopicSubscriber[] {
  const topic = parseTopicSummary({
    topic_id: ID,
    name: "Database news",
    schedule: { every: "day", time: "07:30", tz: "UTC" },
    notify: { when: "changes", subscribers: entries },
  });
  return topic?.subscribers ?? [];
}

function renderList(list: TopicSubscriber[]) {
  const onChanged = jest.fn();
  render(
    <SubscriberList topicId={ID} subscribers={list} onChanged={onChanged} />,
  );
  return { onChanged };
}

describe("subscriber parsing", () => {
  it("reads addresses with confirmed or pending, and skips bad entries", () => {
    expect(
      subscribers(
        {
          email: "a@example.org",
          added_at: "2026-10-01T00:00:00Z",
          confirmed_at: "2026-10-01T01:00:00Z",
        },
        {
          email: "b@example.org",
          added_at: "2026-10-17T00:00:00Z",
          confirmed_at: null,
        },
        { email: 3 },
        "junk",
      ),
    ).toEqual([
      {
        email: "a@example.org",
        confirmed: true,
        addedAt: "2026-10-01T00:00:00Z",
        confirmedAt: "2026-10-01T01:00:00Z",
      },
      {
        email: "b@example.org",
        confirmed: false,
        addedAt: "2026-10-17T00:00:00Z",
      },
    ]);
  });

  it("leaves subscribers out when the reply has none (the list view)", () => {
    const topic = parseTopicSummary({
      topic_id: ID,
      name: "n",
      schedule: { every: "day", time: "07:30", tz: "UTC" },
      subscriber_count: 2,
    });
    expect(topic?.subscribers).toBeUndefined();
    expect(topic?.subscriberCount).toBe(2);
  });

  it.each([
    [409, { code: "subscriber_limit" }, /already has 5 email addresses/],
    [
      409,
      { detail: { code: "subscriber_limit" } },
      /already has 5 email addresses/,
    ],
    [409, { detail: "subscriber_limit" }, /already has 5 email addresses/],
    [409, { error: "email_disabled" }, /Email isn't turned on/],
    [503, { detail: { code: "email_disabled" } }, /Email isn't turned on/],
    [422, { code: "invalid_email" }, /isn't valid/],
  ])("words a %i %j", (status, body, message) => {
    expect(topicErrorMessage(status, body)).toMatch(message);
  });

  it.each([
    ["a@example.org", true],
    [" a@example.org ", true],
    ["a@b", false],
    ["a@@b.org", false],
    ["@b.org", false],
    ["a b@c.org", false],
    ["a@.org", false],
    [`${"a".repeat(250)}@b.org`, false],
  ])("checks %p as %p", (value, ok) =>
    expect(isPlausibleEmail(value)).toBe(ok),
  );
});

describe("SubscriberList", () => {
  it("shows each address as confirmed or waiting", () => {
    renderList(
      subscribers(
        { email: "a@example.org", confirmed_at: "2026-10-01T01:00:00Z" },
        { email: "b@example.org", added_at: "2026-10-17T00:00:00Z" },
      ),
    );
    const items = screen.getAllByRole("listitem");
    expect(items[0]).toHaveTextContent("a@example.org");
    expect(items[0]).toHaveTextContent("Confirmed");
    expect(items[1]).toHaveTextContent("b@example.org");
    expect(items[1]).toHaveTextContent("Waiting for confirmation");
    expect(
      screen.getByText("We'll send a confirmation link first."),
    ).toBeInTheDocument();
  });

  it("adds an address with add_subscriber and reloads", async () => {
    const { onChanged } = renderList([]);
    expect(screen.getByText("Nobody gets emails yet.")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Add an email address"), {
      target: { value: " new@example.org " },
    });
    fireEvent.click(screen.getByRole("button", { name: "Add" }));

    await waitFor(() => expect(onChanged).toHaveBeenCalled());
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`/api/topics/${ID}`);
    expect(init.method).toBe("PATCH");
    expect(JSON.parse(init.body)).toEqual({
      add_subscriber: "new@example.org",
    });
    expect(screen.getByRole("status")).toHaveTextContent(
      "A confirmation link was sent to new@example.org.",
    );
    expect(screen.getByLabelText("Add an email address")).toHaveValue("");
  });

  it("removes an address with remove_subscriber", async () => {
    const { onChanged } = renderList(
      subscribers({
        email: "a@example.org",
        confirmed_at: "2026-10-01T01:00:00Z",
      }),
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Remove a@example.org" }),
    );
    await waitFor(() => expect(onChanged).toHaveBeenCalled());
    const [, init] = fetchMock.mock.calls[0];
    expect(init.method).toBe("PATCH");
    expect(JSON.parse(init.body)).toEqual({
      remove_subscriber: "a@example.org",
    });
  });

  it.each([
    [
      { code: "subscriber_limit" },
      "This topic already has 5 email addresses; remove one first.",
    ],
    [
      { code: "email_disabled" },
      "Email isn't turned on for this WebAgent, so addresses can't be added.",
    ],
  ])("shows %j in words", async (body, message) => {
    fetchMock.mockResolvedValue(reply(409, body));
    const { onChanged } = renderList([]);
    fireEvent.change(screen.getByLabelText("Add an email address"), {
      target: { value: "x@example.org" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Add" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(message);
    expect(onChanged).not.toHaveBeenCalled();
  });

  it("won't send an address that isn't valid", () => {
    renderList([]);
    fireEvent.change(screen.getByLabelText("Add an email address"), {
      target: { value: "not-an-address" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Add" }));
    expect(screen.getByRole("alert")).toHaveTextContent(
      "That email address isn't valid.",
    );
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("disables adding at 5 addresses", () => {
    renderList(
      subscribers(
        ...[1, 2, 3, 4, 5].map((n) => ({ email: `u${n}@example.org` })),
      ),
    );
    expect(screen.getByLabelText("Add an email address")).toBeDisabled();
    expect(screen.getByRole("button", { name: "Add" })).toBeDisabled();
    expect(
      screen.getByText("A topic can have at most 5 addresses."),
    ).toBeInTheDocument();
  });
});
