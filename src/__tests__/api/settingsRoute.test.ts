/**
 * @jest-environment node
 */
import fs from 'fs';
import os from 'os';
import path from 'path';

jest.mock('@/utils/logger', () => ({
  logger: { debug: jest.fn(), info: jest.fn(), warn: jest.fn(), error: jest.fn() },
}));

describe('GET /api/settings', () => {
  let tmpDir: string;
  let originalCwd: string;
  let getSettings: () => Promise<Response>;
  let readDefaultFromFile: () => string | undefined | null;
  let clearSettingsCache: () => void;

  beforeAll(async () => {
    const mod = await import('@/app/api/settings/route');
    const helpers = await import('@/utils/defaultAgenticMode');
    getSettings = mod.GET;
    readDefaultFromFile = helpers.readDefaultFromFile;
    clearSettingsCache = helpers.clearSettingsCache;
  });

  beforeEach(() => {
    originalCwd = process.cwd();
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'webagent-settings-'));
    process.chdir(tmpDir);
    clearSettingsCache();
    delete process.env.WEBAGENT_DEFAULT_AGENTIC_MODE;
  });

  afterEach(() => {
    process.chdir(originalCwd);
    fs.rmSync(tmpDir, { recursive: true, force: true });
    clearSettingsCache();
  });

  function writeEnv(contents: string) {
    fs.writeFileSync(path.join(tmpDir, '.env.local'), contents, 'utf8');
    clearSettingsCache();
  }

  it('reads default from .env.local', async () => {
    writeEnv('WEBAGENT_DEFAULT_AGENTIC_MODE=auto\n');
    expect(readDefaultFromFile()).toBe('auto');

    const response = await getSettings();
    expect(await response.json()).toEqual({ defaultAgenticMode: 'auto' });
  });

  it('uses updated file value after mtime changes', async () => {
    writeEnv('WEBAGENT_DEFAULT_AGENTIC_MODE=off\n');
    expect(readDefaultFromFile()).toBe('off');

    writeEnv('WEBAGENT_DEFAULT_AGENTIC_MODE=always\n');
    expect(readDefaultFromFile()).toBe('always');
  });

  it('falls back to process.env when file is missing', async () => {
    process.env.WEBAGENT_DEFAULT_AGENTIC_MODE = 'auto';
    expect(readDefaultFromFile()).toBeNull();

    const response = await getSettings();
    expect(await response.json()).toEqual({ defaultAgenticMode: 'auto' });
  });

  it('returns off for unknown values', async () => {
    writeEnv('WEBAGENT_DEFAULT_AGENTIC_MODE=banana\n');
    const response = await getSettings();
    expect(await response.json()).toEqual({ defaultAgenticMode: 'off' });
  });

  it('strips quotes and trailing comments', async () => {
    writeEnv('WEBAGENT_DEFAULT_AGENTIC_MODE="auto" # default mode\n');
    expect(readDefaultFromFile()).toBe('auto');
  });
});
