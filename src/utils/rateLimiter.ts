import { CHAT_SETTINGS } from "@/config/chat";
import { logger } from "@/utils/logger";

const SWEEP_INTERVAL_MS = 5 * 60 * 1000;

/** Fixed-window limiter with one bucket of request times per key (a session ID, or 'shared'). */
export class RateLimiter {
  private buckets = new Map<string, number[]>();
  private lastSweep = Date.now();

  constructor(
    private readonly windowMs: number = CHAT_SETTINGS.rateLimitWindowMs,
    private readonly maxRequestsPerWindow: number = CHAT_SETTINGS.rateLimitPerMinute,
  ) {}

  private recent(key: string, now: number): number[] {
    const windowStart = now - this.windowMs;
    const kept = (this.buckets.get(key) ?? []).filter((t) => t > windowStart);
    if (kept.length > 0) {
      this.buckets.set(key, kept);
    } else {
      this.buckets.delete(key);
    }
    return kept;
  }

  private sweep(now: number): void {
    if (now - this.lastSweep < SWEEP_INTERVAL_MS) return;
    this.lastSweep = now;
    Array.from(this.buckets.keys()).forEach((key) => this.recent(key, now));
  }

  isRateLimited(key: string): boolean {
    const now = Date.now();
    this.sweep(now);
    const limited = this.recent(key, now).length >= this.maxRequestsPerWindow;
    if (limited) {
      logger.warn("Rate limit reached", {
        maxRequests: this.maxRequestsPerWindow,
        windowMs: this.windowMs,
      });
    }
    return limited;
  }

  addRequest(key: string): void {
    const now = Date.now();
    this.buckets.set(key, [...this.recent(key, now), now]);
  }

  getTimeUntilNextAllowed(key: string): number {
    const now = Date.now();
    const times = this.recent(key, now);
    if (times.length < this.maxRequestsPerWindow) return 0;
    const oldestCounted = times[times.length - this.maxRequestsPerWindow];
    return Math.max(0, oldestCounted + this.windowMs - now);
  }

  getRemainingRequests(key: string): number {
    return Math.max(
      0,
      this.maxRequestsPerWindow - this.recent(key, Date.now()).length,
    );
  }

  bucketCount(): number {
    return this.buckets.size;
  }

  clear(): void {
    this.buckets.clear();
  }
}

export const rateLimiter = new RateLimiter();
