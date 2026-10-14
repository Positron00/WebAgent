/**
 * The web app never writes the user's text to the browser console (IMP-070).
 * Found in the 3.5.0 review: voice input logged every spoken transcript.
 */
import fs from "fs";
import path from "path";

const SRC = path.resolve(__dirname, "../..");
// Names that hold the user's words: the question, a transcript, message content.
const USER_TEXT = /\b(transcript|query|question|content|userMessage|input)\b/;
const CONSOLE_CALL = /console\.(log|info|debug|warn|error)\(([^;]*)\)/g;

function sourceFiles(dir: string): string[] {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      return entry.name === "__tests__" ? [] : sourceFiles(full);
    }
    return /\.(ts|tsx)$/.test(entry.name) && !/\.test\.tsx?$/.test(entry.name)
      ? [full]
      : [];
  });
}

describe("browser console", () => {
  it("no console call passes the user's text", () => {
    const offenders: string[] = [];
    for (const file of sourceFiles(SRC)) {
      const text = fs.readFileSync(file, "utf8");
      for (const match of text.matchAll(CONSOLE_CALL)) {
        // Only the arguments after the first (the message string) can carry data.
        const args = match[2].split(",").slice(1).join(",");
        if (USER_TEXT.test(args)) {
          const line = text.slice(0, match.index).split("\n").length;
          offenders.push(`${path.relative(SRC, file)}:${line}`);
        }
      }
    }
    expect(offenders).toEqual([]);
  });
});
