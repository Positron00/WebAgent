import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

/** Open connectivity check: says the web app is up, and nothing about its config (IMP-063). */
export async function GET() {
  return NextResponse.json({
    status: "ok",
    timestamp: new Date().toISOString(),
  });
}

export async function HEAD() {
  return new NextResponse(null, { status: 200 });
}
