'use client';

import React, { useRef, useEffect, useCallback } from 'react';
import { ChatMessage } from '@/types/chat';
import { useApp } from '@/contexts/AppContext';
import ReactMarkdown from 'react-markdown';
import { InformationCircleIcon, BeakerIcon } from '@heroicons/react/24/outline';
import { CHAT_THEME } from '@/config/chatTheme';

interface MessageListProps {
  messages: ChatMessage[];
  isLoading: boolean;
  error?: string;
}

// Function to check if a message contains code blocks
export const hasCodeBlocks = (content: string): boolean => {
  // Check for markdown code blocks (```code```)
  const codeBlockRegex = /```[\s\S]*?```/g;
  return codeBlockRegex.test(content);
};

// Function to extract code blocks from a message
export const extractCodeBlocks = (content: string): { language: string, code: string }[] => {
  const codeBlockRegex = /```(\w*)\n([\s\S]*?)```/g;
  const codeBlocks: { language: string, code: string }[] = [];
  
  let match;
  while ((match = codeBlockRegex.exec(content)) !== null) {
    codeBlocks.push({
      language: match[1] || 'text',
      code: match[2].trim()
    });
  }
  
  return codeBlocks;
};

// Add academic-style citation references to the text
const addCitationReferences = (content: string, sourcesCount: number): string => {
  if (sourcesCount === 0) return content;
  
  // Create array of available citation numbers
  const citationNumbers = Array.from({ length: sourcesCount }, (_, i) => i + 1);
  let modifiedContent = content;
  
  // Find sentences to cite
  const sentences = modifiedContent.match(/[^.!?]+[.!?]+/g) || [];
  
  if (sentences.length === 0) {
    // If we couldn't parse sentences, just add a citation at the end
    return modifiedContent + ` [${citationNumbers[0]}]`;
  }
  
  // Always cite at least 3 sentences if possible, or all sentences if fewer than 3
  const sentencesToCite = Math.min(
    Math.max(3, Math.ceil(sentences.length * 0.4)), 
    sentences.length
  );
  
  // Select sentences to cite, prioritizing the final sentence
  // and then evenly distributed throughout the text
  const selectedIndices = new Set<number>();
  
  // Always cite the last sentence if available
  if (sentences.length > 0) {
    selectedIndices.add(sentences.length - 1);
  }
  
  // Distribute remaining citations throughout the text
  while (selectedIndices.size < sentencesToCite && selectedIndices.size < citationNumbers.length) {
    // Find an index that hasn't been selected yet
    const index = Math.floor(Math.random() * sentences.length);
    if (!selectedIndices.has(index)) {
      selectedIndices.add(index);
    }
  }
  
  // Apply citations
  const indices = Array.from(selectedIndices).sort((a, b) => a - b);
  for (let i = 0; i < indices.length && i < citationNumbers.length; i++) {
    const sentenceIndex = indices[i];
    const citation = citationNumbers[i];
    
    // Skip sentences that already have citations
    if (sentences[sentenceIndex] && !sentences[sentenceIndex].includes('[')) {
      const citedSentence = sentences[sentenceIndex].replace(/([.!?]+)$/, ` [${citation}]$1`);
      
      // Escape special regex characters in the original sentence
      const escapeRegex = (string: string): string => string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const originalSentenceEscaped = escapeRegex(sentences[sentenceIndex]);
      
      // Replace just this occurrence
      modifiedContent = modifiedContent.replace(
        new RegExp(`${originalSentenceEscaped}(?![^.!?]*\\[)`), 
        citedSentence
      );
      
      // Update our record of the sentence
      sentences[sentenceIndex] = citedSentence;
    }
  }
  
  // If we somehow didn't add any citations, add one to the last sentence
  if (!modifiedContent.includes('[')) {
    modifiedContent = modifiedContent.replace(/([.!?]+)$/, ` [${citationNumbers[0]}]$1`);
  }
  
  return modifiedContent;
};

// Enhanced message formatting for article-like appearance
const formatArticleContent = (content: string): string => {
  // Check if this message has multiple sections by counting headings
  const headingCount = (content.match(/^#+\s+.+/gm) || []).length;
  const hasNumberedSections = (content.match(/^\d+\.\s+.+/gm) || []).length > 2;
  const hasSectionKeywords = content.toLowerCase().includes('conclusion') || 
                          content.toLowerCase().includes('summary') || 
                          content.toLowerCase().includes('next steps') ||
                          content.toLowerCase().includes('recommendation');
  
  // Only add introduction if there are multiple sections in the response
  const hasMultipleSections = headingCount > 1 || hasNumberedSections || hasSectionKeywords;
  
  // Check if this is a longer, more structured response that needs an introduction
  const needsIntroduction = content.length > 250 && 
                          hasMultipleSections &&
                          (content.includes('# ') || content.includes('## ') || 
                           content.includes('1. ') || content.includes('2. ') ||
                           content.split('\n\n').length > 3);

  // Process markdown to improve article structure
  return content
    // Ensure headings are properly formatted
    .replace(/^(?!#)(.+?):\s*$/gm, '## $1\n')
    // Add proper newlines after headings
    .replace(/^(#+\s.*?)$/gm, '$1\n\n')
    // Format lists with proper spacing
    .replace(/(\d+\.\s.*?)\n(\d+\.)/g, '$1\n\n$2')
    .replace(/([*\-•].*)\n([*\-•])/g, '$1\n\n$2')
    // Ensure paragraphs have proper spacing
    .replace(/([^\s])\n([^\s])/g, '$1\n\n$2')
    // Add introduction section only if needed and not already present
    .replace(/^(?!#)(.+?)(?:\n\n|\n(?=#))/m, (match, p1) => {
      // Only add Introduction if needed and if it doesn't already contain "introduction"
      if (needsIntroduction && 
          !content.toLowerCase().includes('# introduction') && 
          !content.toLowerCase().includes('## introduction') &&
          !p1.toLowerCase().includes('introduction')) {
        return `## Introduction\n\n${p1}\n\n`;
      }
      return match;
    })
    // Add conclusion if there isn't one and content is long enough and has multiple sections
    .replace(/(.+)$/m, (match) => {
      if (content.length > 200 && hasMultipleSections && !content.includes('# Conclusion') && !content.includes('## Conclusion')) {
        // Find the last paragraph that might be a conclusion
        const lastParagraphs = content.split('\n\n').slice(-2);
        if (lastParagraphs.some((p: string) => p.toLowerCase().includes('conclusion') || 
                                     p.toLowerCase().includes('summary') || 
                                     p.toLowerCase().includes('in sum'))) {
          // If there's a paragraph that looks like a conclusion, format it properly
          return match.replace(/(.+?\n\n)(.+)$/m, '$1## Conclusion\n\n$2');
        }
      }
      return match;
    })
    // Make sure there are not too many consecutive newlines
    .replace(/\n{3,}/g, '\n\n');
};

type ConversationTurn = {
  user: ChatMessage;
  assistant?: ChatMessage;
};

function groupMessagesIntoTurns(messages: ChatMessage[]): ConversationTurn[] {
  const turns: ConversationTurn[] = [];
  let pending: ConversationTurn | null = null;

  for (const message of messages) {
    if (message.role === 'user') {
      if (pending) {
        turns.push(pending);
      }
      pending = { user: message };
      continue;
    }
    if (message.role === 'assistant') {
      if (pending) {
        pending.assistant = message;
        turns.push(pending);
        pending = null;
      }
    }
  }

  if (pending) {
    turns.push(pending);
  }

  return turns;
}

export function MessageList({ messages, isLoading, error }: MessageListProps) {
  const { accessibility } = useApp();
  const latestTurnRef = useRef<HTMLDivElement>(null);
  const turns = groupMessagesIntoTurns(messages);

  // Keep the latest question visible at the top of the scroll area (Perplexity-style)
  useEffect(() => {
    if (!latestTurnRef.current || turns.length === 0) return;
    latestTurnRef.current.scrollIntoView({
      block: 'start',
      behavior: accessibility.reducedMotion ? 'auto' : 'smooth',
    });
  }, [turns.length, isLoading, accessibility.reducedMotion]);

  // Apply custom colors or high contrast theme if enabled
  const getMessageClassName = (role: string) => {
    const sans =
      'font-sans antialiased break-words max-w-3xl w-full mx-auto text-[15px] leading-[1.65]';
    if (role === 'user') {
      return `${sans} py-3.5 px-5 rounded-2xl text-center shrink-0`;
    }
    return `font-sans antialiased break-words w-full max-w-none py-4 px-1 text-left chat-prose chat-answer text-[15px] leading-[1.65]`;
  };

  // Get styles for messages based on accessibility settings
  const getMessageStyles = (role: string) => {
    // If high contrast is enabled, use those colors instead of custom colors
    if (accessibility.highContrast) {
      if (role === 'user') {
        return {
          backgroundColor: '#1a1a1a',
          color: CHAT_THEME.textPrimary,
        };
      }
      return {
        backgroundColor: CHAT_THEME.pageBackground,
        color: CHAT_THEME.textPrimary,
      };
    }
    
    // Otherwise use custom colors
    if (role === 'user') {
      return { 
        backgroundColor: accessibility.queryBackgroundColor,
        color: accessibility.queryTextColor
      };
    } else {
      // For assistant messages
      return { 
        backgroundColor: accessibility.responseBackgroundColor,
        color: accessibility.responseTextColor
      };
    }
  };

  // Render article-formatted assistant messages
  const renderArticleContent = (content: string, hasSources: boolean) => {
    // Format content as an article
    const articleContent = formatArticleContent(content);
    
    return (
      <div className="relative">
        <ReactMarkdown
          components={{
            p: ({ node, ...props }) => <p {...props} />,
            h1: ({ node, ...props }) => <h1 {...props} />,
            h2: ({ node, ...props }) => <h2 {...props} />,
            h3: ({ node, ...props }) => <h3 {...props} />,
            h4: ({ node, ...props }) => <h4 className="text-base font-semibold mb-3 mt-4" {...props} />,
            ul: ({ node, ...props }) => <ul {...props} />,
            ol: ({ node, ...props }) => <ol {...props} />,
            li: ({ node, ...props }) => <li {...props} />,
            blockquote: ({ node, ...props }) => (
              <blockquote
                className="border-l-2 border-white/20 pl-4 my-4 text-[#c9c9c9]"
                {...props}
              />
            ),
            strong: ({ node, ...props }) => <strong {...props} />,
            em: ({ node, ...props }) => <em className="italic text-[#d4d4d4]" {...props} />,
            table: ({ node, ...props }) => (
              <table className="w-full mb-4 border-collapse border border-white/10" {...props} />
            ),
            th: ({ node, ...props }) => (
              <th className="border border-white/10 p-2 bg-white/5 text-left" {...props} />
            ),
            td: ({ node, ...props }) => <td className="border border-white/10 p-2" {...props} />,
            code: ({ node, className, ...props }) => {
              const isInline = !className || !className.includes('language-');
              return isInline ? (
                <code {...props} />
              ) : (
                <code className={`block ${className}`} {...props} />
              );
            },
          }}
        >
          {articleContent}
        </ReactMarkdown>

        {hasSources && accessibility.citeSources && (
          <div
            className="absolute top-0 right-0 text-[#8e8e8e] text-xs px-2 py-1 rounded-full border border-white/10 flex items-center"
          >
            <InformationCircleIcon className="w-3 h-3 mr-1" />
            <span>Cited Sources</span>
          </div>
        )}
      </div>
    );
  };

  // Render workflow logs
  const renderWorkflowLogs = (logs?: string[]) => {
    if (!logs || logs.length === 0 || !accessibility.agentic) {
      return null;
    }

    return (
      <div className="mt-3 text-gray-500 text-xs border-t border-gray-800 pt-2">
        <div className="flex items-center mb-1">
          <BeakerIcon className="w-3 h-3 mr-1" />
          <span className="text-gray-400 font-semibold">Workflow Logs</span>
        </div>
        <div className="text-left font-mono pl-1">
          {logs.map((log, index) => (
            <div key={index} className="mb-1 whitespace-pre-wrap opacity-75">{log}</div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="w-full text-left mb-4">
      <div
        className="w-full flex-1 min-h-[50vh] max-h-[70vh] rounded-xl border overflow-y-auto px-2 py-3 font-sans"
        style={{
          backgroundColor: CHAT_THEME.pageBackground,
          color: CHAT_THEME.textPrimary,
          borderColor: CHAT_THEME.borderSubtle,
        }}
        role="log"
        aria-live="polite"
        aria-label="Chat Messages"
      >
        {messages.length === 0 ? (
          <div
            className="h-full flex flex-col items-center justify-center p-8 text-[15px]"
            style={{ color: CHAT_THEME.textMuted }}
          >
            <span>Start a conversation...</span>
            <span>Responses will appear here</span>
          </div>
        ) : (
          <>
            {turns.map((turn, index) => {
              const { user: userMessage, assistant: assistantMessage } = turn;
              const isLatestTurn = index === turns.length - 1;

              let processedContent = assistantMessage?.content;
              const hasSources = !!(
                assistantMessage?.sources && assistantMessage.sources.length > 0
              );
              if (
                hasSources &&
                assistantMessage &&
                assistantMessage.sources &&
                accessibility.citeSources
              ) {
                processedContent = addCitationReferences(
                  assistantMessage.content,
                  assistantMessage.sources.length
                );
              }

              return (
                <div
                  key={userMessage.id}
                  ref={isLatestTurn ? latestTurnRef : undefined}
                  className="mb-8 pb-8 border-b last:border-b-0 last:mb-0 last:pb-0"
                  style={{ borderColor: CHAT_THEME.borderSubtle }}
                >
                  <div className="mb-5 flex justify-center">
                    <div
                      className={getMessageClassName('user')}
                      role="note"
                      aria-label="user's message"
                      style={getMessageStyles('user')}
                    >
                      <div className="chat-prose chat-question text-center [&_p]:text-center [&_p]:mb-0">
                        <ReactMarkdown>{userMessage.content}</ReactMarkdown>
                      </div>
                    </div>
                  </div>

                  {assistantMessage && (
                    <div className="relative w-full text-left">
                      <div
                        className={getMessageClassName('assistant')}
                        role="article"
                        aria-label="assistant's response"
                        style={getMessageStyles('assistant')}
                      >
                        {processedContent &&
                          renderArticleContent(processedContent, hasSources)}
                        {renderWorkflowLogs(assistantMessage.logs)}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </>
        )}
        {isLoading && (
          <div 
            className="flex items-center justify-start space-x-2 ml-8 mt-4 text-[#8e8e8e] text-[15px]"
            role="status"
            aria-label="Loading response"
          >
            <div 
              className={`${
                accessibility.reducedMotion ? '' : 'animate-spin'
              } rounded-full h-4 w-4 border-2 border-gray-300 border-t-transparent`}
              aria-hidden="true"
            />
            <span>Thinking...</span>
          </div>
        )}
        {error && (
          <div className="flex flex-col items-start ml-8 mt-4">
            <div 
              className="text-left text-red-400 p-2 bg-red-900/50 rounded w-full mb-2"
              role="alert"
              aria-live="assertive"
            >
              {error}
            </div>
            <button
              onClick={() => {
                console.log("Diagnostic information:");
                console.log("- Navigator online:", navigator.onLine);
                console.log("- User Agent:", navigator.userAgent);
                console.log("- Current error:", error);
                
                // Check local storage for API key (don't log the actual key)
                const hasApiKey = localStorage.getItem('TOGETHER_API_KEY') !== null;
                console.log("- API key present in localStorage:", hasApiKey);
                
                // Check if we can reach the API ping endpoint
                fetch('/api/ping')
                  .then(response => {
                    console.log("- API ping response:", response.status, response.statusText);
                    return response.text();
                  })
                  .then(text => console.log("- API ping response body:", text))
                  .catch(err => console.log("- API ping error:", err));
                
                // Try a direct test of chat API to isolate the issue
                fetch('/api/chat', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({
                    messages: [{ role: 'user', content: 'Hello! This is a diagnostic test.' }],
                    promptStyle: 'balanced',
                    knowledgeFocus: 'general',
                    citeSources: false
                  })
                })
                .then(response => {
                  console.log("- Direct API test response status:", response.status, response.statusText);
                  console.log("- Direct API test response headers:", Object.fromEntries([...response.headers.entries()]));
                  if (!response.ok) return response.text().then(text => { throw new Error(text) });
                  return response.json();
                })
                .then(data => console.log("- Direct API test successful:", data))
                .catch(err => console.log("- Direct API test failed:", err.toString()));
                
                // Try to get logs from console
                try {
                  // Access recent console logs
                  console.log("- Check browser console for all recent error logs");
                } catch (err) {
                  console.log("- Error accessing logs:", err);
                }
                
                alert("Diagnostic information has been logged to the console. Please open the browser developer tools to view it.");
              }}
              className="text-xs bg-blue-800 hover:bg-blue-700 text-white py-1 px-2 rounded transition-colors text-left"
            >
              Run Diagnostics
            </button>
          </div>
        )}
      </div>
    </div>
  );
} 