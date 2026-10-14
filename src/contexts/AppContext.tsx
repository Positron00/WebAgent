"use client";

import React, { createContext, useContext, useEffect, useState } from "react";
import {
  storage,
  AccessibilitySettings,
  ThemePreference,
} from "@/utils/storage";
import { CHAT_ACCESSIBILITY_COLORS } from "@/config/chatTheme";
import { AgenticMode, isAgenticMode } from "@/types/routing";
import {
  getSavedAgenticMode,
  migrateLegacyAgenticSetting,
  saveAgenticMode,
} from "@/utils/agenticMode";

// Define notification type
export interface Notification {
  message: string;
  type: "info" | "success" | "warning" | "error";
  id?: string;
}

export interface AppContextType {
  theme: ThemePreference;
  setTheme: (theme: ThemePreference) => void;
  accessibility: AccessibilitySettings;
  setAccessibility: (settings: AccessibilitySettings) => void;
  agenticMode: AgenticMode;
  setAgenticMode: (mode: AgenticMode) => void;
  modeReady: boolean;
  isOffline: boolean;
  showNotification: (notification: Notification) => void;
  hideNotification: (id?: string) => void;
  isMenuOpen: boolean;
  setMenuOpen: (isOpen: boolean) => void;
  showModal: (content: React.ReactNode) => void;
  hideModal: () => void;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

const SETTINGS_TIMEOUT_MS = 3000;

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<ThemePreference>("system");
  const [accessibility, setAccessibility] = useState<AccessibilitySettings>({
    reducedMotion: false,
    highContrast: false,
    fontSize: "normal",
    promptStyle: "balanced",
    knowledgeFocus: "general",
    citeSources: true,
    ...CHAT_ACCESSIBILITY_COLORS,
  });
  const [agenticMode, setAgenticModeState] = useState<AgenticMode>("off");
  const [modeReady, setModeReady] = useState(false);
  const [isOffline, setIsOffline] = useState(false);
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [isMenuOpen, setMenuOpen] = useState(false);
  const [modalContent, setModalContent] = useState<React.ReactNode | null>(
    null,
  );

  useEffect(() => {
    migrateLegacyAgenticSetting();

    const saved = getSavedAgenticMode();
    if (saved) {
      setAgenticModeState(saved);
      setModeReady(true);
      setTheme(storage.getTheme());
      setAccessibility(storage.getAccessibilitySettings());
      setIsOffline(!navigator.onLine);
      return;
    }

    setTheme(storage.getTheme());
    setAccessibility(storage.getAccessibilitySettings());
    setIsOffline(!navigator.onLine);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), SETTINGS_TIMEOUT_MS);

    fetch("/api/settings", { cache: "no-store", signal: controller.signal })
      .then(async (res) => {
        if (!res.ok) throw new Error(`settings ${res.status}`);
        const data = (await res.json()) as { defaultAgenticMode?: string };
        const value = data.defaultAgenticMode?.trim().toLowerCase();
        if (isAgenticMode(value)) {
          setAgenticModeState(value);
        } else {
          setAgenticModeState("off");
        }
      })
      .catch(() => {
        setAgenticModeState("off");
      })
      .finally(() => {
        clearTimeout(timeoutId);
        setModeReady(true);
      });
  }, []);

  const setAgenticMode = (mode: AgenticMode) => {
    setAgenticModeState(mode);
    saveAgenticMode(mode);
  };

  useEffect(() => {
    storage.saveTheme(theme);

    document.documentElement.setAttribute("data-theme", theme);
    if (theme === "system") {
      const isDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
      document.documentElement.classList.toggle("dark", isDark);
    } else {
      document.documentElement.classList.toggle("dark", theme === "dark");
    }
  }, [theme]);

  useEffect(() => {
    storage.saveAccessibilitySettings(accessibility);

    document.documentElement.style.fontSize =
      accessibility.fontSize === "small"
        ? "90%"
        : accessibility.fontSize === "large"
          ? "110%"
          : "100%";

    document.documentElement.classList.toggle(
      "reduce-motion",
      accessibility.reducedMotion,
    );

    document.documentElement.classList.toggle(
      "high-contrast",
      accessibility.highContrast,
    );
  }, [accessibility]);

  useEffect(() => {
    function handleOnline() {
      setIsOffline(false);
    }

    function handleOffline() {
      setIsOffline(true);
    }

    const checkConnection = () => {
      fetch("/api/ping", {
        method: "HEAD",
        cache: "no-cache",
        headers: { "Cache-Control": "no-cache" },
      })
        .then(() => setIsOffline(false))
        .catch(() => {
          // CSP connect-src is 'self' only; do not probe third-party hosts.
          setIsOffline(true);
        });
    };

    checkConnection();

    const connectionInterval = setInterval(checkConnection, 30000);

    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);

    const handleVisibilityChange = () => {
      if (document.visibilityState === "visible") {
        checkConnection();
      }
    };
    document.addEventListener("visibilitychange", handleVisibilityChange);

    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      clearInterval(connectionInterval);
    };
  }, []);

  const showNotification = (notification: Notification) => {
    const id = notification.id || Date.now().toString();
    const newNotification = { ...notification, id };

    setNotifications((prev) => [...prev, newNotification]);

    if (notification.type !== "error") {
      setTimeout(() => {
        hideNotification(id);
      }, 5000);
    }
  };

  const hideNotification = (id?: string) => {
    if (id) {
      setNotifications((prev) =>
        prev.filter((notification) => notification.id !== id),
      );
    } else {
      setNotifications([]);
    }
  };

  const showModal = (content: React.ReactNode) => {
    setModalContent(content);
  };

  const hideModal = () => {
    setModalContent(null);
  };

  const value = {
    theme,
    setTheme,
    accessibility,
    setAccessibility,
    agenticMode,
    setAgenticMode,
    modeReady,
    isOffline,
    showNotification,
    hideNotification,
    isMenuOpen,
    setMenuOpen,
    showModal,
    hideModal,
  };

  return (
    <AppContext.Provider value={value}>
      {children}
      {modalContent && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg max-w-lg w-full max-h-[90vh] overflow-auto">
            {modalContent}
          </div>
        </div>
      )}
    </AppContext.Provider>
  );
}

export function useApp() {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error("useApp must be used within an AppProvider");
  }
  return context;
}
