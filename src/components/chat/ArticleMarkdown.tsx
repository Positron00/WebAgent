import React from "react";
import ReactMarkdown from "react-markdown";

// The Markdown renderer for research answers, shared by the chat and stored topic reports.
// It uses react-markdown without rehype-raw, so raw HTML in a report is never rendered (RNFR-06).

// Enhanced message formatting for article-like appearance
export const formatArticleContent = (content: string): string => {
  // Check if this message has multiple sections by counting headings
  const headingCount = (content.match(/^#+\s+.+/gm) || []).length;
  const hasNumberedSections = (content.match(/^\d+\.\s+.+/gm) || []).length > 2;
  const hasSectionKeywords =
    content.toLowerCase().includes("conclusion") ||
    content.toLowerCase().includes("summary") ||
    content.toLowerCase().includes("next steps") ||
    content.toLowerCase().includes("recommendation");

  // Only add introduction if there are multiple sections in the response
  const hasMultipleSections =
    headingCount > 1 || hasNumberedSections || hasSectionKeywords;

  // Check if this is a longer, more structured response that needs an introduction
  const needsIntroduction =
    content.length > 250 &&
    hasMultipleSections &&
    (content.includes("# ") ||
      content.includes("## ") ||
      content.includes("1. ") ||
      content.includes("2. ") ||
      content.split("\n\n").length > 3);

  // Process markdown to improve article structure
  return (
    content
      // Ensure headings are properly formatted
      .replace(/^(?!#)(.+?):\s*$/gm, "## $1\n")
      // Add proper newlines after headings
      .replace(/^(#+\s.*?)$/gm, "$1\n\n")
      // Format lists with proper spacing
      .replace(/(\d+\.\s.*?)\n(\d+\.)/g, "$1\n\n$2")
      .replace(/([*\-•].*)\n([*\-•])/g, "$1\n\n$2")
      // Ensure paragraphs have proper spacing
      .replace(/([^\s])\n([^\s])/g, "$1\n\n$2")
      // Add introduction section only if needed and not already present
      .replace(/^(?!#)(.+?)(?:\n\n|\n(?=#))/m, (match, p1) => {
        // Only add Introduction if needed and if it doesn't already contain "introduction"
        if (
          needsIntroduction &&
          !content.toLowerCase().includes("# introduction") &&
          !content.toLowerCase().includes("## introduction") &&
          !p1.toLowerCase().includes("introduction")
        ) {
          return `## Introduction\n\n${p1}\n\n`;
        }
        return match;
      })
      // Add conclusion if there isn't one and content is long enough and has multiple sections
      .replace(/(.+)$/m, (match) => {
        if (
          content.length > 200 &&
          hasMultipleSections &&
          !content.includes("# Conclusion") &&
          !content.includes("## Conclusion")
        ) {
          // Find the last paragraph that might be a conclusion
          const lastParagraphs = content.split("\n\n").slice(-2);
          if (
            lastParagraphs.some(
              (p: string) =>
                p.toLowerCase().includes("conclusion") ||
                p.toLowerCase().includes("summary") ||
                p.toLowerCase().includes("in sum"),
            )
          ) {
            // If there's a paragraph that looks like a conclusion, format it properly
            return match.replace(/(.+?\n\n)(.+)$/m, "$1## Conclusion\n\n$2");
          }
        }
        return match;
      })
      // Make sure there are not too many consecutive newlines
      .replace(/\n{3,}/g, "\n\n")
  );
};

/** Renders an answer or a stored report as an article. */
export function ArticleMarkdown({ content }: { content: string }) {
  return (
    <ReactMarkdown
      components={{
        p: ({ node, ...props }) => <p {...props} />,
        h1: ({ node, ...props }) => <h1 {...props} />,
        h2: ({ node, ...props }) => <h2 {...props} />,
        h3: ({ node, ...props }) => <h3 {...props} />,
        h4: ({ node, ...props }) => (
          <h4 className="text-base font-semibold mb-3 mt-4" {...props} />
        ),
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
        em: ({ node, ...props }) => (
          <em className="italic text-[#d4d4d4]" {...props} />
        ),
        table: ({ node, ...props }) => (
          <table
            className="w-full mb-4 border-collapse border border-white/10"
            {...props}
          />
        ),
        th: ({ node, ...props }) => (
          <th
            className="border border-white/10 p-2 bg-white/5 text-left"
            {...props}
          />
        ),
        td: ({ node, ...props }) => (
          <td className="border border-white/10 p-2" {...props} />
        ),
        code: ({ node, className, ...props }) => {
          const isInline = !className || !className.includes("language-");
          return isInline ? (
            <code {...props} />
          ) : (
            <code className={`block ${className}`} {...props} />
          );
        },
      }}
    >
      {formatArticleContent(content)}
    </ReactMarkdown>
  );
}
