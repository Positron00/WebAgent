export type ResearchFailureKind =
  | "timeout"
  | "budget"
  | "cancelled"
  | "failed"
  | "unreachable";

export const RESEARCH_TIMEOUT_MESSAGE =
  "The research took too long and was stopped.";

/** A research run that ended without an answer; its message is shown to the user as is. */
export class ResearchFailed extends Error {
  constructor(
    public kind: ResearchFailureKind,
    message: string,
  ) {
    super(message);
    this.name = "ResearchFailed";
  }
}
