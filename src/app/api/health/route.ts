import { NextResponse } from "next/server";
import { clientConfig } from "@/config/clientConfig";
import { ApiClient } from "@/utils/apiClient";

export const dynamic = "force-dynamic";

/** Open health check: status, version and time only; details are in the server logs (IMP-063). */
export async function GET() {
  const healthy = ApiClient.getInstance().isHealthy();
  return NextResponse.json(
    {
      status: healthy ? "healthy" : "degraded",
      version: clientConfig.version,
      timestamp: new Date().toISOString(),
    },
    { headers: { "Cache-Control": "no-store, max-age=0" } },
  );
}
