import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { ResearchFailed, failureText } from "../ResearchFailed";

describe("ResearchFailed", () => {
  it.each([
    ["timeout", "The research took too long and was stopped."],
    ["budget", "The research used its token budget and was stopped."],
    ["unreachable", "The research service isn't reachable right now."],
    ["failed", "Deep research failed: Together returned 503."],
  ] as const)("shows the %s text in an alert", (kind, text) => {
    render(
      <ResearchFailed
        kind={kind}
        message="Together returned 503."
        busy={false}
        onRetry={jest.fn()}
        onAnswerDirectly={jest.fn()}
      />,
    );

    expect(screen.getByRole("alert")).toHaveTextContent(text);
  });

  it("calls the two actions from their buttons", () => {
    const onRetry = jest.fn();
    const onAnswerDirectly = jest.fn();
    render(
      <ResearchFailed
        kind="timeout"
        message=""
        busy={false}
        onRetry={onRetry}
        onAnswerDirectly={onAnswerDirectly}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    fireEvent.click(
      screen.getByRole("button", { name: "Answer directly instead" }),
    );

    expect(onRetry).toHaveBeenCalledTimes(1);
    expect(onAnswerDirectly).toHaveBeenCalledTimes(1);
  });

  it("disables the buttons while a request is running", () => {
    render(
      <ResearchFailed
        kind="budget"
        message=""
        busy={true}
        onRetry={jest.fn()}
        onAnswerDirectly={jest.fn()}
      />,
    );

    expect(screen.getByRole("button", { name: "Try again" })).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "Answer directly instead" }),
    ).toBeDisabled();
  });
});

describe("failureText", () => {
  it("keeps at most 200 characters of the reason", () => {
    const text = failureText("failed", "x".repeat(500));
    expect(text).toBe(`Deep research failed: ${"x".repeat(199)}….`);
  });

  it("handles an empty reason and the cancelled kind", () => {
    expect(failureText("failed", "  ")).toBe(
      "Deep research failed: unknown error.",
    );
    expect(failureText("cancelled", "")).toBe("Research stopped.");
  });
});
