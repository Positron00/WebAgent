import React from "react";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { ResearchStatus, formatElapsed } from "../ResearchStatus";

describe("ResearchStatus", () => {
  beforeEach(() => jest.useFakeTimers());
  afterEach(() => jest.useRealTimers());

  it("shows the step, the latest log line and the elapsed time, ticking each second", () => {
    render(
      <ResearchStatus
        progress={{
          step: "Web and internal research",
          latestLog: "Searching the web",
          elapsedMs: 0,
        }}
        startedAt={Date.now() - 75_000}
        reducedMotion={false}
        onStop={jest.fn()}
      />,
    );

    expect(
      screen.getByText("Thinking… · Web and internal research"),
    ).toBeInTheDocument();
    expect(screen.getByText("Searching the web")).toBeInTheDocument();
    expect(screen.getByText("1:15")).toBeInTheDocument();

    act(() => {
      jest.advanceTimersByTime(1000);
    });
    expect(screen.getByText("1:16")).toBeInTheDocument();
  });

  it("puts only the step text in the live region", () => {
    render(
      <ResearchStatus
        progress={{ step: "Planning", latestLog: "log", elapsedMs: 0 }}
        startedAt={Date.now()}
        reducedMotion={false}
        onStop={jest.fn()}
      />,
    );

    expect(screen.getByRole("status").textContent).toBe("Thinking… · Planning");
  });

  it('says "Working" before the first progress report and has no log line', () => {
    const { container } = render(
      <ResearchStatus
        startedAt={Date.now()}
        reducedMotion={true}
        onStop={jest.fn()}
      />,
    );

    expect(screen.getByText("Thinking… · Working")).toBeInTheDocument();
    expect(container.querySelector("p")).toBeNull();
    expect(container.querySelector(".animate-spin")).toBeNull();
  });

  it("calls onStop from the Stop button", () => {
    const onStop = jest.fn();
    render(
      <ResearchStatus
        startedAt={Date.now()}
        reducedMotion={false}
        onStop={onStop}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Stop research" }));

    expect(onStop).toHaveBeenCalledTimes(1);
  });
});

describe("formatElapsed", () => {
  it("formats as m:ss", () => {
    expect(formatElapsed(0)).toBe("0:00");
    expect(formatElapsed(9_999)).toBe("0:09");
    expect(formatElapsed(605_000)).toBe("10:05");
    expect(formatElapsed(-50)).toBe("0:00");
  });
});
