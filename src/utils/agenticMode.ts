import { AgenticMode, isAgenticMode } from '@/types/routing';

const STORAGE_KEY = 'agentic_mode';
const ACCESSIBILITY_KEY = 'accessibility_settings';

const isClient = typeof window !== 'undefined';

export function getSavedAgenticMode(): AgenticMode | null {
  if (!isClient) return null;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const trimmed = raw.trim().toLowerCase();
    return isAgenticMode(trimmed) ? trimmed : null;
  } catch {
    return null;
  }
}

export function saveAgenticMode(mode: AgenticMode): void {
  if (!isClient) return;
  try {
    localStorage.setItem(STORAGE_KEY, mode);
  } catch (error) {
    console.error('Error saving agentic mode:', error);
  }
}

export function migrateLegacyAgenticSetting(): void {
  if (!isClient) return;
  try {
    const raw = localStorage.getItem(ACCESSIBILITY_KEY);
    if (!raw) return;

    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(raw) as Record<string, unknown>;
    } catch {
      return;
    }

    if (!('agentic' in parsed)) return;

    const legacyAgentic = parsed.agentic;
    if (legacyAgentic === true && getSavedAgenticMode() === null) {
      saveAgenticMode('auto');
    }

    delete parsed.agentic;
    localStorage.setItem(ACCESSIBILITY_KEY, JSON.stringify(parsed));
  } catch (error) {
    console.error('Error migrating legacy agentic setting:', error);
  }
}
