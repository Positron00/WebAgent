import patterns from '@/config/security/patterns.json';

const USER_TURN_GROUPS = ['injection_phrase', 'role_marker'] as const;

function compilePattern(pattern: string): RegExp {
  if (pattern.startsWith('(?i)')) {
    return new RegExp(pattern.slice(4), 'i');
  }
  return new RegExp(pattern);
}

function patternsForGroup(group: string): RegExp[] {
  const list = (patterns as Record<string, string[]>)[group] ?? [];
  return list.map((pattern) => compilePattern(pattern));
}

/** Fast local scan for injection phrases and role markers on a user turn (SEC-091). */
export function scanUserTurn(text: string): string[] {
  const hits: string[] = [];
  for (const group of USER_TURN_GROUPS) {
    for (const regex of patternsForGroup(group)) {
      if (regex.test(text)) {
        hits.push(group);
        break;
      }
    }
  }
  return hits;
}
