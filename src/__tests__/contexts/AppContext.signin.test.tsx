import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import { AppProvider, useApp } from "@/contexts/AppContext";

let mockPath = "/login";
jest.mock("next/navigation", () => ({ usePathname: () => mockPath }));

function Consumer() {
  const { agenticMode, modeReady } = useApp();
  return (
    <div>
      <span data-testid="mode">{agenticMode}</span>
      <span data-testid="ready">{String(modeReady)}</span>
    </div>
  );
}

const tree = () => (
  <AppProvider>
    <Consumer />
  </AppProvider>
);

describe("AppContext default mode and sign-in", () => {
  let signedIn: boolean;
  let fetchMock: jest.Mock;

  beforeEach(() => {
    window.localStorage.clear();
    signedIn = false;
    fetchMock = jest.fn((url: RequestInfo | URL) => {
      if (String(url).includes("/api/settings")) {
        return Promise.resolve(
          signedIn
            ? {
                ok: true,
                status: 200,
                json: async () => ({ defaultAgenticMode: "always" }),
              }
            : { ok: false, status: 401, json: async () => ({}) },
        );
      }
      return Promise.resolve({ ok: true, status: 200, json: async () => ({}) });
    });
    global.fetch = fetchMock as unknown as typeof fetch;
  });

  const settingsCalls = () =>
    fetchMock.mock.calls.filter(([url]) =>
      String(url).includes("/api/settings"),
    );

  it("does not ask for settings on the sign-in page", async () => {
    mockPath = "/login";
    render(tree());
    await waitFor(() =>
      expect(screen.getByTestId("ready")).toHaveTextContent("true"),
    );
    expect(settingsCalls()).toHaveLength(0);
    expect(screen.getByTestId("mode")).toHaveTextContent("off");
  });

  it("loads the operator default after sign-in moves to the app", async () => {
    mockPath = "/login";
    const { rerender } = render(tree());
    await waitFor(() =>
      expect(screen.getByTestId("ready")).toHaveTextContent("true"),
    );

    signedIn = true;
    mockPath = "/";
    rerender(tree());

    await waitFor(() =>
      expect(screen.getByTestId("mode")).toHaveTextContent("always"),
    );
    expect(window.localStorage.getItem("agentic_mode")).toBeNull();
  });

  it("tries again after a 401 instead of settling on off", async () => {
    mockPath = "/";
    const { rerender } = render(tree());
    await waitFor(() => expect(settingsCalls()).toHaveLength(1));
    expect(screen.getByTestId("mode")).toHaveTextContent("off");

    signedIn = true;
    mockPath = "/topics";
    rerender(tree());

    await waitFor(() =>
      expect(screen.getByTestId("mode")).toHaveTextContent("always"),
    );
    expect(settingsCalls()).toHaveLength(2);
  });
});
