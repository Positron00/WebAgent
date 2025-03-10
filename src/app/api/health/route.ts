import { NextResponse } from 'next/server';
import { clientConfig } from '@/config/clientConfig';
import { logger } from '@/utils/logger';
import { ApiClient } from '@/utils/apiClient';

// Add this export to declare the route as dynamic
export const dynamic = 'force-dynamic';

/**
 * Health check endpoint for monitoring system status
 * 
 * Returns:
 * - Basic health information
 * - Version and environment details
 * - API client metrics
 * - System uptime
 * - Browser compatibility information
 */
export async function GET(req: Request) {
  const startTime = Date.now();
  const requestId = `health_${Date.now()}`;
  
  try {
    logger.info('Health check requested', { requestId }, requestId, 'health-api');
    
    // Get API client metrics
    const apiClient = ApiClient.getInstance();
    const apiMetrics = apiClient.getMetrics();
    const apiHealthy = apiClient.isHealthy();
    
    // Get logger statistics
    const logStats = logger.getStats(1); // Last hour
    
    // Get system uptime
    const uptime = process.uptime();
    
    // Memory usage (if available)
    let memory = null;
    if (typeof process.memoryUsage === 'function') {
      const memoryUsage = process.memoryUsage();
      memory = {
        rss: Math.round(memoryUsage.rss / 1024 / 1024), // MB
        heapTotal: Math.round(memoryUsage.heapTotal / 1024 / 1024), // MB
        heapUsed: Math.round(memoryUsage.heapUsed / 1024 / 1024), // MB
        external: Math.round(memoryUsage.external / 1024 / 1024), // MB
      };
    }
    
    // Get browser information from request headers
    const userAgent = req.headers.get('user-agent') || 'unknown';
    
    // Detect browser type
    const browserInfo = {
      userAgent,
      isBot: /bot|crawler|spider|crawling/i.test(userAgent),
      isMobile: /mobile|android|iphone|ipad|ipod/i.test(userAgent),
      browser: getBrowserNameFromUserAgent(userAgent),
    };
    
    // Combine all health information
    const healthInfo = {
      status: apiHealthy ? 'healthy' : 'degraded',
      version: clientConfig.version,
      environment: clientConfig.environment,
      timestamp: new Date().toISOString(),
      uptime: {
        seconds: Math.round(uptime),
        formatted: formatUptime(uptime)
      },
      api: {
        healthy: apiHealthy,
        metrics: {
          requestCount: apiMetrics.requestCount,
          errorCount: apiMetrics.errorCount,
          errorRate: apiMetrics.requestCount > 0 
            ? (apiMetrics.errorCount / apiMetrics.requestCount) 
            : 0,
          averageResponseTime: apiMetrics.averageResponseTime,
          consecutiveErrors: apiMetrics.consecutiveErrors,
          statusCodes: apiMetrics.statusCodes || {}
        }
      },
      logs: {
        errorRate: logStats.errorRate,
        recentLogs: logStats.total,
        recentErrors: logStats.byLevel.error
      },
      client: {
        browser: browserInfo,
        supportedBrowsers: [
          { name: 'Chrome', minVersion: '90' },
          { name: 'Firefox', minVersion: '90' },
          { name: 'Safari', minVersion: '14' },
          { name: 'Edge', minVersion: '90' }
        ]
      },
      memory
    };
    
    // Calculate response time
    const responseTime = Date.now() - startTime;
    
    // Log health check result
    logger.info('Health check completed', 
      { responseTime, status: healthInfo.status }, 
      requestId, 
      'health-api'
    );
    
    // Return health information
    return NextResponse.json(healthInfo, {
      status: apiHealthy ? 200 : 200, // Still return 200 even if degraded
      headers: {
        'X-Response-Time': responseTime.toString(),
        'Cache-Control': 'no-store, max-age=0'
      }
    });
  } catch (error) {
    // Log error
    logger.error('Health check failed', 
      { error }, 
      requestId, 
      'health-api'
    );
    
    // Return error response
    return NextResponse.json(
      { 
        status: 'error',
        error: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date().toISOString()
      },
      { 
        status: 500,
        headers: {
          'Cache-Control': 'no-store, max-age=0'
        }
      }
    );
  }
}

/**
 * Format uptime in a human-readable format
 */
function formatUptime(uptime: number): string {
  const days = Math.floor(uptime / (24 * 60 * 60));
  const hours = Math.floor((uptime % (24 * 60 * 60)) / (60 * 60));
  const minutes = Math.floor((uptime % (60 * 60)) / 60);
  const seconds = Math.floor(uptime % 60);
  
  const parts = [];
  if (days > 0) parts.push(`${days}d`);
  if (hours > 0) parts.push(`${hours}h`);
  if (minutes > 0) parts.push(`${minutes}m`);
  if (seconds > 0 || parts.length === 0) parts.push(`${seconds}s`);
  
  return parts.join(' ');
}

/**
 * Extract browser name from user agent string
 */
function getBrowserNameFromUserAgent(userAgent: string): string {
  if (/edg/i.test(userAgent)) return 'Edge';
  if (/firefox/i.test(userAgent)) return 'Firefox';
  if (/chrome/i.test(userAgent)) return 'Chrome';
  if (/safari/i.test(userAgent)) return 'Safari';
  if (/msie|trident/i.test(userAgent)) return 'Internet Explorer';
  if (/opera|opr/i.test(userAgent)) return 'Opera';
  
  return 'Unknown';
} 