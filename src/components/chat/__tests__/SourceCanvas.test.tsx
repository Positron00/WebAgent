import React from "react";
import { render, screen } from "@testing-library/react";
import { SourceCanvas } from "../SourceCanvas";
import type { Source } from "@/types/chat";

const source = (id: string, url?: string): Source => ({
  id,
  title: `Source ${id}`,
  url,
  domain: "example.com",
});

describe("SourceCanvas links (IMP-062)", () => {
  it("links to http(s) sources in a new tab without an opener", () => {
    render(
      <SourceCanvas
        sources={[source("1", "https://example.com/a")]}
        isVisible
      />,
    );

    const link = screen.getByRole("link", { name: /Visit Source/ });
    expect(link).toHaveAttribute("href", "https://example.com/a");
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("shows other URLs as plain text, not links", () => {
    render(
      <SourceCanvas sources={[source("1", "javascript:alert(1)")]} isVisible />,
    );

    expect(screen.queryByRole("link")).not.toBeInTheDocument();
    expect(screen.getByText("javascript:alert(1)")).toBeInTheDocument();
  });

  it("shows no link section when a source has no URL", () => {
    render(<SourceCanvas sources={[source("1")]} isVisible />);

    expect(screen.queryByRole("link")).not.toBeInTheDocument();
    expect(screen.getByText("Source 1")).toBeInTheDocument();
  });

  it("shows the placeholder when there are no sources", () => {
    render(<SourceCanvas sources={[]} isVisible />);

    expect(screen.getByText("No sources to display")).toBeInTheDocument();
  });
});
