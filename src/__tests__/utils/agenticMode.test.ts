import {
  getSavedAgenticMode,
  saveAgenticMode,
  migrateLegacyAgenticSetting,
} from '@/utils/agenticMode';
import { storage } from '@/utils/storage';

describe('agenticMode storage', () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it('migrates legacy agentic true to auto when no saved mode', () => {
    window.localStorage.setItem(
      'accessibility_settings',
      JSON.stringify({ agentic: true, citeSources: true })
    );

    migrateLegacyAgenticSetting();

    expect(getSavedAgenticMode()).toBe('auto');
    const stored = JSON.parse(window.localStorage.getItem('accessibility_settings') || '{}');
    expect(stored.agentic).toBeUndefined();
  });

  it('does not save mode when legacy agentic is false', () => {
    window.localStorage.setItem(
      'accessibility_settings',
      JSON.stringify({ agentic: false, citeSources: true })
    );

    migrateLegacyAgenticSetting();

    expect(getSavedAgenticMode()).toBeNull();
    const stored = JSON.parse(window.localStorage.getItem('accessibility_settings') || '{}');
    expect(stored.agentic).toBeUndefined();
  });

  it('does nothing when agentic key is missing', () => {
    window.localStorage.setItem(
      'accessibility_settings',
      JSON.stringify({ citeSources: false })
    );

    migrateLegacyAgenticSetting();

    expect(getSavedAgenticMode()).toBeNull();
  });

  it('is idempotent on second run', () => {
    window.localStorage.setItem(
      'accessibility_settings',
      JSON.stringify({ agentic: true })
    );
    migrateLegacyAgenticSetting();
    saveAgenticMode('always');

    migrateLegacyAgenticSetting();

    expect(getSavedAgenticMode()).toBe('always');
  });

  it('saveAgenticMode persists user choice only via explicit save', () => {
    expect(getSavedAgenticMode()).toBeNull();
    saveAgenticMode('auto');
    expect(getSavedAgenticMode()).toBe('auto');
    expect(window.localStorage.getItem('agentic_mode')).toBe('auto');
  });

  it('storage drops a stray agentic key and still migrates the old bubble colours', () => {
    window.localStorage.setItem(
      'accessibility_settings',
      JSON.stringify({
        agentic: true,
        citeSources: false,
        queryBackgroundColor: '#1e3a8a',
        responseBackgroundColor: '#111827',
      })
    );

    const settings = storage.getAccessibilitySettings();

    expect(settings).not.toHaveProperty('agentic');
    expect(settings.citeSources).toBe(false);
    expect(settings.queryBackgroundColor.toUpperCase()).not.toBe('#1E3A8A');
  });
});

