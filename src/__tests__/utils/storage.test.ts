import { ChatMessage } from "@/types/chat";
import { storage, STORAGE_QUOTA_NOTICE } from "@/utils/storage";

function makeMessages(count: number): ChatMessage[] {
  return Array.from({ length: count }, (_, i) => ({
    id: `msg-${i}`,
    role: (i % 2 === 0 ? "user" : "assistant") as "user" | "assistant",
    content: `message ${i}`,
    timestamp: new Date(0).toISOString(),
  }));
}

describe("storage.saveMessages (IMP-092)", () => {
  const originalLocalStorage = window.localStorage;

  beforeEach(() => {
    window.localStorage.clear();
  });

  afterEach(() => {
    Object.defineProperty(window, "localStorage", {
      value: originalLocalStorage,
      writable: true,
      configurable: true,
    });
    window.localStorage.clear();
  });

  it("persists messages when localStorage has room", () => {
    const messages = makeMessages(4);
    const result = storage.saveMessages(messages);

    expect(result.quotaTrimmed).toBe(false);
    expect(result.persisted).toHaveLength(4);
    expect(storage.getMessages()).toHaveLength(4);
  });

  it("drops the oldest half and retries once on QuotaExceededError", () => {
    const messages = makeMessages(6);
    const backing: Record<string, string> = {};
    let callCount = 0;

    const quotaLocalStorage = {
      getItem: (key: string) => backing[key] ?? null,
      setItem: (key: string, value: string) => {
        callCount += 1;
        if (callCount === 1) {
          throw new DOMException("quota", "QuotaExceededError");
        }
        backing[key] = value;
      },
      removeItem: (key: string) => {
        delete backing[key];
      },
      clear: () => {
        Object.keys(backing).forEach((key) => delete backing[key]);
      },
      key: (index: number) => Object.keys(backing)[index] ?? null,
      get length() {
        return Object.keys(backing).length;
      },
    };

    Object.defineProperty(window, "localStorage", {
      value: quotaLocalStorage,
      writable: true,
      configurable: true,
    });

    const result = storage.saveMessages(messages);

    expect(callCount).toBe(2);
    expect(result.quotaTrimmed).toBe(true);
    expect(result.persisted).toHaveLength(3);
    expect(storage.getMessages()).toHaveLength(3);
    expect(storage.getMessages()[0].id).toBe("msg-3");
  });

  it("exports the user-facing quota notice copy", () => {
    expect(STORAGE_QUOTA_NOTICE).toBe(
      "Older messages were removed to save space.",
    );
  });
});
