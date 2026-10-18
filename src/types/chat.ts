export type Role = "user" | "assistant" | "system";

export interface Source {
  id: string;
  title: string;
  url?: string;
  snippet?: string;
  relevance?: number; // 0-1 score of relevance
  domain?: string;
}

import type { RouteInfo } from "@/types/routing";
import type { GuardRefusal, SecuritySummary } from "@/types/security";

/** The completed research task behind an answer; "Track this" turns it into a topic (REC-116). */
export interface ResearchTaskRef {
  id: string;
  title?: string;
  /** The supervisor's self-contained version of the question (REC-110). */
  standaloneQuery?: string;
}

export interface ChatMessage {
  id?: string;
  role: Role;
  content: string;
  timestamp?: string;
  sources?: Source[]; // Sources used for this message
  logs?: string[]; // Workflow progress logs
  route?: RouteInfo;
  imageAttached?: boolean;
  security?: SecuritySummary;
  refusal?: GuardRefusal;
  task?: ResearchTaskRef;
}

export interface ChatState {
  messages: ChatMessage[];
  isLoading: boolean;
  error?: string;
}
