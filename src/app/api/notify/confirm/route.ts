import { handleNotifyPost } from "../shared";

// Public (no sign-in, REC-134): forwards {token} to the backend with the API key and no session.
// POST only; Next.js answers 405 to every other method. See handleNotifyPost for the reply.
export const dynamic = "force-dynamic";

export async function POST(req: Request) {
  return handleNotifyPost(req, "confirm");
}
