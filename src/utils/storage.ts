import { ChatMessage } from "@/types/chat";
import { CHAT_ACCESSIBILITY_COLORS } from "@/config/chatTheme";

const STORAGE_KEYS = {
  MESSAGES: "chat_messages",
  THEME: "theme_preference",
  ACCESSIBILITY: "accessibility_settings",
} as const;

const MAX_MESSAGES = 50;

export const STORAGE_QUOTA_NOTICE =
  "Older messages were removed to save space.";

const isClient = typeof window !== "undefined";

export interface SaveMessagesResult {
  persisted: ChatMessage[];
  quotaTrimmed: boolean;
}

function isQuotaExceededError(error: unknown): boolean {
  if (error instanceof DOMException && error.name === "QuotaExceededError") {
    return true;
  }
  return (
    typeof error === "object" &&
    error !== null &&
    "name" in error &&
    (error as { name: string }).name === "QuotaExceededError"
  );
}

function writeMessages(messages: ChatMessage[]): void {
  localStorage.setItem(STORAGE_KEYS.MESSAGES, JSON.stringify(messages));
}

export interface AccessibilitySettings {
  reducedMotion: boolean;
  highContrast: boolean;
  fontSize: "small" | "normal" | "large";
  promptStyle:
    | "balanced"
    | "creative"
    | "precise"
    | "helpful"
    | "verbose"
    | "concise";
  knowledgeFocus:
    | "general"
    | "medical"
    | "legal"
    | "science"
    | "technology"
    | "business"
    | "history"
    | "nature";
  citeSources: boolean;
  responseTextColor: string;
  queryTextColor: string;
  responseBackgroundColor: string;
  queryBackgroundColor: string;
}

const DEFAULT_ACCESSIBILITY: AccessibilitySettings = {
  reducedMotion: false,
  highContrast: false,
  fontSize: "normal",
  promptStyle: "balanced",
  knowledgeFocus: "general",
  citeSources: true,
  ...CHAT_ACCESSIBILITY_COLORS,
};

export type ThemePreference = "light" | "dark" | "system";

export const storage = {
  getMessages(): ChatMessage[] {
    if (!isClient) return [];
    try {
      const messages = localStorage.getItem(STORAGE_KEYS.MESSAGES);
      return messages ? JSON.parse(messages) : [];
    } catch (error) {
      console.error("Error reading messages from storage:", error);
      return [];
    }
  },

  saveMessages(messages: ChatMessage[]): SaveMessagesResult {
    const trimmedMessages = messages.slice(-MAX_MESSAGES);
    if (!isClient) {
      return { persisted: trimmedMessages, quotaTrimmed: false };
    }
    try {
      writeMessages(trimmedMessages);
      return { persisted: trimmedMessages, quotaTrimmed: false };
    } catch (error) {
      if (!isQuotaExceededError(error)) {
        console.error("Error saving messages to storage:", error);
        return { persisted: trimmedMessages, quotaTrimmed: false };
      }
      const droppedOldestHalf = trimmedMessages.slice(
        Math.floor(trimmedMessages.length / 2),
      );
      try {
        writeMessages(droppedOldestHalf);
        return { persisted: droppedOldestHalf, quotaTrimmed: true };
      } catch (retryError) {
        console.error("Error saving messages after quota trim:", retryError);
        return { persisted: droppedOldestHalf, quotaTrimmed: true };
      }
    }
  },

  getTheme(): ThemePreference {
    if (!isClient) return "system";
    try {
      const theme = localStorage.getItem(STORAGE_KEYS.THEME);
      return (theme as ThemePreference) || "system";
    } catch (error) {
      console.error("Error reading theme from storage:", error);
      return "system";
    }
  },

  saveTheme(theme: ThemePreference): void {
    if (!isClient) return;
    try {
      localStorage.setItem(STORAGE_KEYS.THEME, theme);
    } catch (error) {
      console.error("Error saving theme to storage:", error);
    }
  },

  getAccessibilitySettings(): AccessibilitySettings {
    if (!isClient) return DEFAULT_ACCESSIBILITY;
    try {
      const settings = localStorage.getItem(STORAGE_KEYS.ACCESSIBILITY);
      if (!settings) return DEFAULT_ACCESSIBILITY;
      const stored = JSON.parse(settings) as AccessibilitySettings & {
        agentic?: boolean;
      };
      // The mode lives under its own key (utils/agenticMode.ts); drop a stray legacy flag.
      const { agentic: _removed, ...rest } = stored;
      const parsed: AccessibilitySettings = {
        ...DEFAULT_ACCESSIBILITY,
        ...rest,
      };
      // Migrate legacy blue bubble theme to neutral chat colors
      if (
        parsed.queryBackgroundColor?.toUpperCase() === "#1E3A8A" &&
        parsed.responseBackgroundColor === "#111827"
      ) {
        return { ...parsed, ...CHAT_ACCESSIBILITY_COLORS };
      }
      return parsed;
    } catch (error) {
      console.error(
        "Error reading accessibility settings from storage:",
        error,
      );
      return DEFAULT_ACCESSIBILITY;
    }
  },

  saveAccessibilitySettings(settings: AccessibilitySettings): void {
    if (!isClient) return;
    try {
      localStorage.setItem(
        STORAGE_KEYS.ACCESSIBILITY,
        JSON.stringify(settings),
      );
    } catch (error) {
      console.error("Error saving accessibility settings to storage:", error);
    }
  },

  clearMessages(): void {
    if (!isClient) return;
    try {
      localStorage.removeItem(STORAGE_KEYS.MESSAGES);
    } catch (error) {
      console.error("Error clearing messages from storage:", error);
    }
  },
};
