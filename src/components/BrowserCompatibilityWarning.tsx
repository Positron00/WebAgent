import { useEffect, useState } from 'react';
import { getBrowserCompatibilityMessage, isBrowserSupported, applyPolyfills } from '@/utils/browserCompatibility';

/**
 * Component to display browser compatibility warnings
 * 
 * This component checks browser compatibility and displays a warning
 * message if the browser is not fully supported.
 */
export default function BrowserCompatibilityWarning() {
  const [message, setMessage] = useState<string | null>(null);
  const [isSupported, setIsSupported] = useState(true);

  useEffect(() => {
    // Apply polyfills for older browsers
    applyPolyfills();
    
    // Check browser compatibility
    const compatMessage = getBrowserCompatibilityMessage();
    setMessage(compatMessage);
    setIsSupported(isBrowserSupported());
  }, []);

  // If browser is supported or no message, don't render anything
  if (isSupported && !message) {
    return null;
  }

  // Critical compatibility issue - browser not supported
  if (!isSupported) {
    return (
      <div className="fixed inset-x-0 top-0 z-50 p-4 bg-red-600 text-white text-center">
        <p className="font-medium">
          <span className="mr-2">⚠️</span>
          Your browser is not supported. Please upgrade to a modern browser for the best experience.
        </p>
      </div>
    );
  }

  // Warning message - browser supported but with limitations
  return (
    <div className="fixed inset-x-0 top-0 z-50 p-3 bg-amber-100 text-amber-800 text-center">
      <p className="text-sm">
        <span className="mr-2">ℹ️</span>
        {message}
      </p>
    </div>
  );
} 