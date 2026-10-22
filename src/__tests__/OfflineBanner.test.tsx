import React from "react";
import { render, screen } from "@testing-library/react";
import OfflineBanner from "@/components/OfflineBanner";

let mockOffline = false;
jest.mock("@/contexts/AppContext", () => ({
  useApp: () => ({ isOffline: mockOffline }),
}));

describe("OfflineBanner", () => {
  it("is not in the page while online, so screen readers never announce it", () => {
    mockOffline = false;
    render(<OfflineBanner />);
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("shows an alert while offline", () => {
    mockOffline = true;
    render(<OfflineBanner />);
    expect(screen.getByRole("alert")).toHaveTextContent(
      "You are currently offline",
    );
  });
});
