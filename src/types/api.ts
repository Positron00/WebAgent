export interface MessageContent {
  type: 'text' | 'image_url';
  text?: string;
  // OpenAI-style chat format, which Together expects: {url: 'data:image/png;base64,...'}
  image_url?: { url: string };
}

export interface ChatRequestMessage {
  role: 'user' | 'assistant' | 'system';
  content: string | MessageContent[];
}

import type { RouteInfo } from '@/types/routing';
import type { Source } from '@/types/chat';

export interface ChatResponseMessage {
  content: string;
  role: string;
  logs?: string[];
  sources?: Source[];
}

export interface ChatCompletionRequest {
  model: string;
  messages: ChatRequestMessage[];
  max_tokens: number;
  temperature: number;
  top_p: number;
  frequency_penalty: number;
  presence_penalty: number;
}

export interface ChatCompletionResponse {
  choices: {
    message: {
      content: string;
      role: string;
      logs?: string[];
      sources?: Source[];
    };
    finish_reason: string;
  }[];
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  route?: RouteInfo;
} 