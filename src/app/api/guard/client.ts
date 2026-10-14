import { ChatRequestMessage } from "@/types/api";
import { GuardRefusal, normalizeGuardRefusal } from "@/types/security";
import { logger } from "@/utils/logger";
import { backendClientHeaders } from "@/app/api/task/backend";

const BACKEND_URL = process.env.WEBAGENT_BACKEND_URL || "http://127.0.0.1:8000";
const GUARD_TIMEOUT_MS = 3000;

export type GuardCall =
  | { kind: "ok"; outcome: "allow" | "verify" }
  | { kind: "refused"; status: 403 | 409; body: GuardRefusal }
  | { kind: "unreachable" };

function guardMessages(
  messages: ChatRequestMessage[],
): { role: string; content: string }[] {
  return messages
    .filter((m) => m.role !== "system")
    .map((m) => ({
      role: m.role,
      content: typeof m.content === "string" ? m.content : "",
    }));
}

export async function callGuard(
  messages: ChatRequestMessage[],
  hasImage: boolean,
  consentToken: string | null,
  incoming: Request,
): Promise<GuardCall> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...backendClientHeaders(incoming),
  };

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), GUARD_TIMEOUT_MS);

  try {
    const response = await fetch(`${BACKEND_URL}/api/guard/input`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        messages: guardMessages(messages),
        has_image: hasImage,
        ...(consentToken ? { consent_token: consentToken } : {}),
      }),
      signal: controller.signal,
      cache: "no-store",
    });

    clearTimeout(timeoutId);
    const body = (await response.json().catch(() => ({}))) as Record<
      string,
      unknown
    >;

    if (response.status === 200) {
      const outcome = body.outcome;
      if (outcome === "allow" || outcome === "verify") {
        return { kind: "ok", outcome };
      }
      logger.error("Guard returned unexpected 200 outcome", { outcome, body });
      return { kind: "unreachable" };
    }

    if (response.status === 403 || response.status === 409) {
      const refusal = normalizeGuardRefusal(body, response.status);
      if (refusal) {
        return { kind: "refused", status: response.status, body: refusal };
      }
      logger.error("Guard refusal with unparseable body", {
        status: response.status,
        body,
      });
      return { kind: "unreachable" };
    }

    if (response.status === 503) {
      return { kind: "unreachable" };
    }

    logger.error("Guard returned unexpected status", {
      status: response.status,
      body,
    });
    return { kind: "unreachable" };
  } catch (error) {
    clearTimeout(timeoutId);
    logger.warn("Guard unreachable", {
      error: error instanceof Error ? error.message : String(error),
    });
    return { kind: "unreachable" };
  }
}
