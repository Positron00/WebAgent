/**
 * Browser compatibility detection utility
 * 
 * This utility helps detect browser capabilities and provides fallbacks
 * for unsupported features. It's designed to be lightweight and easy to use.
 */

interface BrowserCapabilities {
  supportsWebP: boolean;
  supportsIntersectionObserver: boolean;
  supportsLocalStorage: boolean;
  supportsServiceWorker: boolean;
  supportsFetch: boolean;
  supportsWebSockets: boolean;
  isIOS: boolean;
  isSafari: boolean;
  isFirefox: boolean;
  isChrome: boolean;
  isEdge: boolean;
  isMobile: boolean;
  browserName: string;
  browserVersion: string;
}

/**
 * Detect browser capabilities
 * @returns Object with browser capability flags
 */
export function detectBrowserCapabilities(): BrowserCapabilities {
  // Default capabilities for server-side rendering
  if (typeof window === 'undefined') {
    return {
      supportsWebP: true,
      supportsIntersectionObserver: true,
      supportsLocalStorage: true,
      supportsServiceWorker: true,
      supportsFetch: true,
      supportsWebSockets: true,
      isIOS: false,
      isSafari: false,
      isFirefox: false,
      isChrome: false,
      isEdge: false,
      isMobile: false,
      browserName: 'server',
      browserVersion: 'n/a',
    };
  }

  // Browser detection
  const ua = navigator.userAgent;
  const isIOS = /iPad|iPhone|iPod/.test(ua) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
  const isSafari = /^((?!chrome|android).)*safari/i.test(ua);
  const isFirefox = ua.toLowerCase().indexOf('firefox') > -1;
  const isEdge = ua.indexOf('Edg') > -1;
  const isChrome = ua.toLowerCase().indexOf('chrome') > -1 && !isEdge;
  const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(ua);

  // Extract browser name and version
  let browserName = 'unknown';
  let browserVersion = 'unknown';

  if (isEdge) {
    browserName = 'Edge';
    const edgeMatch = ua.match(/Edg\/([0-9]+\.[0-9]+)/);
    browserVersion = edgeMatch ? edgeMatch[1] : 'unknown';
  } else if (isChrome) {
    browserName = 'Chrome';
    const chromeMatch = ua.match(/Chrome\/([0-9]+\.[0-9]+)/);
    browserVersion = chromeMatch ? chromeMatch[1] : 'unknown';
  } else if (isFirefox) {
    browserName = 'Firefox';
    const firefoxMatch = ua.match(/Firefox\/([0-9]+\.[0-9]+)/);
    browserVersion = firefoxMatch ? firefoxMatch[1] : 'unknown';
  } else if (isSafari) {
    browserName = 'Safari';
    const safariMatch = ua.match(/Version\/([0-9]+\.[0-9]+)/);
    browserVersion = safariMatch ? safariMatch[1] : 'unknown';
  }

  // Feature detection
  const supportsWebP = (() => {
    try {
      return document.createElement('canvas')
        .toDataURL('image/webp')
        .indexOf('data:image/webp') === 0;
    } catch (e) {
      return false;
    }
  })();

  const supportsIntersectionObserver = typeof IntersectionObserver !== 'undefined';
  const supportsLocalStorage = (() => {
    try {
      localStorage.setItem('test', 'test');
      localStorage.removeItem('test');
      return true;
    } catch (e) {
      return false;
    }
  })();
  const supportsServiceWorker = 'serviceWorker' in navigator;
  const supportsFetch = 'fetch' in window;
  const supportsWebSockets = 'WebSocket' in window;

  return {
    supportsWebP,
    supportsIntersectionObserver,
    supportsLocalStorage,
    supportsServiceWorker,
    supportsFetch,
    supportsWebSockets,
    isIOS,
    isSafari,
    isFirefox,
    isChrome,
    isEdge,
    isMobile,
    browserName,
    browserVersion,
  };
}

/**
 * Get a user-friendly browser compatibility message
 * @returns Message string or null if browser is compatible
 */
export function getBrowserCompatibilityMessage(): string | null {
  const capabilities = detectBrowserCapabilities();
  
  // Critical features that must be supported
  if (!capabilities.supportsFetch) {
    return 'Your browser does not support modern web features. Please upgrade to a newer browser.';
  }
  
  // Warnings for specific browsers with known issues
  if (capabilities.isSafari && parseFloat(capabilities.browserVersion) < 14) {
    return 'You are using an older version of Safari. Some features may not work correctly. Consider upgrading for the best experience.';
  }
  
  if (capabilities.isIOS && !capabilities.supportsWebP) {
    return 'Your iOS device may have limited support for some image features. Please update to the latest iOS version for the best experience.';
  }
  
  return null; // No compatibility issues
}

/**
 * Log browser capabilities for debugging
 */
export function logBrowserCapabilities(): void {
  const capabilities = detectBrowserCapabilities();
  console.log('Browser Capabilities:', capabilities);
}

/**
 * Check if the current browser is supported
 * @returns True if the browser is supported
 */
export function isBrowserSupported(): boolean {
  const capabilities = detectBrowserCapabilities();
  
  // Minimum requirements for the application to function
  return capabilities.supportsFetch && capabilities.supportsLocalStorage;
}

/**
 * Apply polyfills for older browsers
 */
export function applyPolyfills(): void {
  // Only run in browser
  if (typeof window === 'undefined') return;
  
  // Add any necessary polyfills here
  // Example: Object.fromEntries polyfill
  if (!Object.fromEntries) {
    Object.fromEntries = function fromEntries(entries: Iterable<[string, any]>) {
      if (!entries || !entries[Symbol.iterator]) {
        throw new Error('Object.fromEntries requires a single iterable argument');
      }
      
      const obj: Record<string, any> = {};
      
      for (const [key, value] of entries) {
        obj[key] = value;
      }
      
      return obj;
    };
  }
} 