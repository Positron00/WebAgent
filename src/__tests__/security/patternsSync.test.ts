import fs from 'fs';
import path from 'path';

describe('security patterns sync', () => {
  it('matches the backend patterns.json copy', () => {
    const webPath = path.join(process.cwd(), 'src/config/security/patterns.json');
    const backendPath = path.join(process.cwd(), 'backend/config/security/patterns.json');
    const web = fs.readFileSync(webPath, 'utf8');
    const backend = fs.readFileSync(backendPath, 'utf8');
    expect(web).toBe(backend);
  });
});
