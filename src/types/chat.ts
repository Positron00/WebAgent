export type Role = 'user' | 'assistant' | 'system';

export interface Source {
  id: string;
  title: string;
  url?: string;
  snippet?: string;
  relevance?: number; // 0-1 score of relevance
  domain?: string;
}

import type { RouteInfo } from '@/types/routing';

export interface ChatMessage {
  id?: string;
  role: Role;
  content: string;
  timestamp?: string;
  sources?: Source[]; // Sources used for this message
  logs?: string[]; // Workflow progress logs
  route?: RouteInfo;
  imageAttached?: boolean;
}

export interface ChatState {
  messages: ChatMessage[];
  isLoading: boolean;
  error?: string;
} 