/** Shared chat typography and colors (Perplexity-inspired dark UI). */
export const CHAT_THEME = {
  pageBackground: '#191919',
  textPrimary: '#E5E5E5',
  textSecondary: '#A3A3A3',
  textMuted: '#8E8E8E',
  userBackground: '#2A2A2A',
  responseBackground: '#191919',
  inputBackground: '#252525',
  codeBackground: '#2D2D2D',
  borderSubtle: 'rgba(255, 255, 255, 0.08)',
} as const;

export const CHAT_ACCESSIBILITY_COLORS = {
  responseTextColor: CHAT_THEME.textPrimary,
  queryTextColor: CHAT_THEME.textPrimary,
  responseBackgroundColor: CHAT_THEME.responseBackground,
  queryBackgroundColor: CHAT_THEME.userBackground,
};
