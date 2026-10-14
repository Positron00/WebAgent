import { logger } from "@/utils/logger";

/**
 * API error with HTTP status and optional response payload.
 */
export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public data?: unknown,
    public requestId?: string,
    public endpoint?: string,
    public timestamp: number = Date.now(),
  ) {
    super(message);
    this.name = "ApiError";

    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, ApiError);
    }

    logger.error("API Error", {
      message,
      status,
      requestId,
      endpoint,
      timestamp: new Date(timestamp).toISOString(),
      data: typeof data === "object" ? JSON.stringify(data) : data,
    });
  }
}
