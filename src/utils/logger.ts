import { clientConfig } from '@/config/clientConfig';

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

interface LogEntry {
  timestamp: string;
  level: LogLevel;
  message: string;
  data?: any;
  requestId?: string;
  userId?: string;
  sessionId?: string;
  component?: string;
  duration?: number;
  version?: string;
  environment?: string;
}

/**
 * Enhanced Logger with improved observability features
 * 
 * Features:
 * - Structured logging
 * - Log rotation
 * - Color-coded console output
 * - Log filtering
 * - Error rate tracking
 * - Support for request IDs and session tracking
 * - Configurable log level based on environment
 */
class Logger {
  private static instance: Logger;
  private logs: LogEntry[] = [];
  private readonly maxLogs = 2000;
  private sessionId: string;
  private startTime: number;
  private currentLogLevel: LogLevel;
  
  private logLevelPriority: Record<LogLevel, number> = {
    debug: 0,
    info: 1,
    warn: 2,
    error: 3
  };

  private constructor() {
    this.sessionId = this.generateSessionId();
    this.startTime = Date.now();
    
    // Set log level based on environment
    this.currentLogLevel = clientConfig.isProduction ? 'info' : 'debug';
    
    // Log initialization
    this.info('Logger initialized', {
      sessionId: this.sessionId,
      environment: clientConfig.environment,
      version: clientConfig.version
    });
  }

  static getInstance(): Logger {
    if (!Logger.instance) {
      Logger.instance = new Logger();
    }
    return Logger.instance;
  }
  
  private generateSessionId(): string {
    return `sess_${Date.now()}_${Math.random().toString(36).substring(2, 10)}`;
  }

  private formatTimestamp(): string {
    return new Date().toISOString();
  }
  
  private shouldLog(level: LogLevel): boolean {
    return this.logLevelPriority[level] >= this.logLevelPriority[this.currentLogLevel];
  }
  
  setLogLevel(level: LogLevel): void {
    this.currentLogLevel = level;
    this.info(`Log level set to ${level}`, { previousLevel: this.currentLogLevel });
  }

  private addLog(level: LogLevel, message: string, data?: any, requestId?: string, component?: string, duration?: number) {
    // Skip if below current log level
    if (!this.shouldLog(level)) {
      return;
    }
    
    // Ensure data is serializable
    let cleanData = data;
    if (data instanceof Error) {
      cleanData = {
        name: data.name,
        message: data.message,
        stack: data.stack
      };
    }
    
    const logEntry: LogEntry = {
      timestamp: this.formatTimestamp(),
      level,
      message,
      data: cleanData,
      requestId,
      sessionId: this.sessionId,
      component,
      duration,
      version: clientConfig.version,
      environment: clientConfig.environment
    };

    // Add to internal logs with rotation
    this.logs.push(logEntry);
    if (this.logs.length > this.maxLogs) {
      this.logs = this.logs.slice(this.logs.length - this.maxLogs);
    }

    // Only log to console in development or if forced
    if (!clientConfig.isProduction || level === 'error') {
      // Console output with color
      const coloredLevel = this.getColoredLevel(level);
      const requestIdStr = requestId ? ` [${requestId}]` : '';
      const componentStr = component ? ` (${component})` : '';
      const durationStr = duration ? ` [${duration}ms]` : '';
      
      console.log(
        `${logEntry.timestamp} ${coloredLevel}${requestIdStr}${componentStr}${durationStr}: ${message}`,
        cleanData ? cleanData : ''
      );
    }

    // If error, also log to error monitoring service (if available)
    if (level === 'error' && typeof window !== 'undefined') {
      // Integration point for error monitoring services
      console.error(message, cleanData);
      
      // Record error metrics
      this.recordErrorMetrics(message, cleanData);
    }
  }
  
  private recordErrorMetrics(message: string, data?: any): void {
    // This is where you'd integrate with your error tracking/metrics system
    // For now we'll just add it to a window-level counter that could be reported
    try {
      if (typeof window !== 'undefined') {
        window.__errorMetrics = window.__errorMetrics || {
          count: 0,
          byType: {}
        };
        
        window.__errorMetrics.count++;
        
        // Categorize errors by type
        const errorType = data && data.name ? data.name : 'UnknownError';
        window.__errorMetrics.byType[errorType] = (window.__errorMetrics.byType[errorType] || 0) + 1;
      }
    } catch (e) {
      // Don't let metrics recording cause further errors
      console.error('Error recording metrics', e);
    }
  }

  private getColoredLevel(level: LogLevel): string {
    const colors = {
      debug: '\x1b[36m', // Cyan
      info: '\x1b[32m',  // Green
      warn: '\x1b[33m',  // Yellow
      error: '\x1b[31m', // Red
      reset: '\x1b[0m',  // Reset
    };

    return `${colors[level]}${level.toUpperCase()}${colors.reset}`;
  }

  /**
   * Log a debug message
   * @param message The message to log
   * @param data Additional data to include
   * @param requestId Optional request ID for tracing
   * @param component Component name for better categorization
   */
  debug(message: string, data?: any, requestId?: string, component?: string) {
    this.addLog('debug', message, data, requestId, component);
  }

  /**
   * Log an info message
   * @param message The message to log
   * @param data Additional data to include
   * @param requestId Optional request ID for tracing
   * @param component Component name for better categorization
   */
  info(message: string, data?: any, requestId?: string, component?: string) {
    this.addLog('info', message, data, requestId, component);
  }

  /**
   * Log a warning message
   * @param message The message to log
   * @param data Additional data to include
   * @param requestId Optional request ID for tracing
   * @param component Component name for better categorization
   */
  warn(message: string, data?: any, requestId?: string, component?: string) {
    this.addLog('warn', message, data, requestId, component);
  }

  /**
   * Log an error message
   * @param message The message to log
   * @param data Additional data to include
   * @param requestId Optional request ID for tracing
   * @param component Component name for better categorization
   */
  error(message: string, data?: any, requestId?: string, component?: string) {
    this.addLog('error', message, data, requestId, component);
  }
  
  /**
   * Time a specific operation and log its duration
   * @param operationName Name of the operation to time
   * @param level Log level to use
   * @param callback Function to execute and time
   * @param component Optional component name
   * @param requestId Optional request ID
   */
  async time<T>(
    operationName: string,
    level: LogLevel,
    callback: () => Promise<T>,
    component?: string,
    requestId?: string
  ): Promise<T> {
    const startTime = Date.now();
    try {
      const result = await callback();
      const duration = Date.now() - startTime;
      this.addLog(level, `${operationName} completed`, { duration }, requestId, component, duration);
      return result;
    } catch (error) {
      const duration = Date.now() - startTime;
      this.addLog('error', `${operationName} failed`, { error, duration }, requestId, component, duration);
      throw error;
    }
  }

  /**
   * Get filtered logs
   * @param level Optional log level filter
   * @param limit Maximum number of logs to return
   * @param requestId Optional request ID filter
   * @param component Optional component filter
   * @returns Filtered log entries
   */
  getLogs(level?: LogLevel, limit = 100, requestId?: string, component?: string): LogEntry[] {
    let filteredLogs = this.logs;
    
    if (level) {
      filteredLogs = filteredLogs.filter(log => log.level === level);
    }
    
    if (requestId) {
      filteredLogs = filteredLogs.filter(log => log.requestId === requestId);
    }
    
    if (component) {
      filteredLogs = filteredLogs.filter(log => log.component === component);
    }
    
    return filteredLogs.slice(-limit);
  }

  /**
   * Get logs for a specific request chain
   * @param requestId Request ID to filter by
   * @returns All logs for the specified request
   */
  getRequestLogs(requestId: string): LogEntry[] {
    return this.logs.filter(log => log.requestId === requestId);
  }

  /**
   * Clear all logs
   */
  clearLogs() {
    this.logs = [];
    this.info('Logs cleared');
  }

  /**
   * Get error rate over the last N logs
   * @param sampleSize Number of recent logs to consider
   * @returns Percentage of logs that are errors
   */
  getErrorRate(sampleSize = 100): number {
    const recentLogs = this.logs.slice(-sampleSize);
    const totalLogs = recentLogs.length;
    const errorLogs = recentLogs.filter(log => log.level === 'error').length;
    return totalLogs > 0 ? errorLogs / totalLogs : 0;
  }
  
  /**
   * Get statistics about logged operations
   * @param hours Number of hours to look back
   */
  getStats(hours = 1): Record<string, any> {
    const startTime = Date.now() - (hours * 60 * 60 * 1000);
    const recentLogs = this.logs.filter(log => new Date(log.timestamp).getTime() > startTime);
    
    return {
      total: recentLogs.length,
      byLevel: {
        debug: recentLogs.filter(log => log.level === 'debug').length,
        info: recentLogs.filter(log => log.level === 'info').length,
        warn: recentLogs.filter(log => log.level === 'warn').length,
        error: recentLogs.filter(log => log.level === 'error').length,
      },
      errorRate: this.getErrorRate(),
      uniqueComponents: [...new Set(recentLogs.filter(log => log.component).map(log => log.component))],
      uniqueRequests: [...new Set(recentLogs.filter(log => log.requestId).map(log => log.requestId))],
      sessionDuration: Date.now() - this.startTime
    };
  }
}

// Add type definition for window error metrics
declare global {
  interface Window {
    __errorMetrics?: {
      count: number;
      byType: Record<string, number>;
    };
  }
}

export const logger = Logger.getInstance(); 