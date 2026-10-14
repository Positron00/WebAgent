import { safeUrl } from "@/utils/safeUrl";

describe("safeUrl (IMP-062)", () => {
  it.each(["https://example.com/page?q=1", "http://example.org/"])(
    "keeps the http(s) link %s",
    (url) => {
      expect(safeUrl(url)).toBe(url);
    },
  );

  it.each([
    "javascript:alert(1)",
    " JavaScript:alert(1)",
    "data:text/html,<script>alert(1)</script>",
    "vbscript:msgbox(1)",
    "file:///etc/passwd",
    "ftp://example.com/file",
    "/relative/path",
    "example.com",
    "",
  ])("rejects %p", (url) => {
    expect(safeUrl(url)).toBeNull();
  });

  it("rejects values that are not strings", () => {
    expect(safeUrl(undefined)).toBeNull();
    expect(safeUrl(null)).toBeNull();
    expect(safeUrl(42)).toBeNull();
  });
});
