import { BackendResponse, HealthStatus, DebugAnalysisResponse, AnalysisProgress } from '../types';

// Modal Error interface defined locally since it's causing import issues
interface ModalError extends Error {
  type: 'COLD_START_TIMEOUT' | 'ANALYSIS_TIMEOUT' | 'MODEL_ERROR' | 
        'RATE_LIMIT' | 'INVALID_REPO' | 'SERVER_ERROR' | 'NETWORK_ERROR';
  stage: 'HEALTH_CHECK' | 'INITIALIZATION' | 'FETCH' | 'ANALYSIS' | 'RECOMMENDATIONS';
  retryable: boolean;
  retryDelay?: number;
  details?: string;
}

// API Endpoints configuration
const API_ENDPOINTS = {
  development: {
    analyzeRepository: "https://adamskog--analyze-repository.modal.run",
    analyzeFile: "https://adamskog--analyze-file.modal.run",
    debugAnalysis: "https://adamskog--debug-analysis.modal.run",
    healthCheck: "https://adamskog--health.modal.run"
  },
  production: {
    analyzeRepository: "https://adamskog--analyze-repository.modal.run",
    analyzeFile: "https://adamskog--analyze-file.modal.run",
    debugAnalysis: "https://adamskog--debug-analysis.modal.run",
    healthCheck: "https://adamskog--health.modal.run"
  }
};

// Timeout configuration optimized for Modal
const TIMEOUTS = {
  HEALTH_CHECK: 10_000,      // 10 seconds
  COLD_START: 60_000,        // 1 minute
  WARM_START: 30_000,        // 30 seconds  
  SMALL_REPO: 120_000,       // 2 minutes (< 10 files)
  MEDIUM_REPO: 300_000,      // 5 minutes (10-30 files)
  LARGE_REPO: 600_000,       // 10 minutes (30+ files)
};

// Error recovery strategies - properly typed
const ERROR_RECOVERY: Record<ModalError['type'], { retry: boolean; delay?: number; maxAttempts?: number; fallback?: string }> = {
  COLD_START_TIMEOUT: { retry: true, delay: 30000, maxAttempts: 2 },
  ANALYSIS_TIMEOUT: { retry: true, delay: 60000, maxAttempts: 1 },
  MODEL_ERROR: { retry: false, fallback: 'debug_mode' },
  RATE_LIMIT: { retry: true, delay: 300000, maxAttempts: 3 },
  NETWORK_ERROR: { retry: true, delay: 5000, maxAttempts: 3 },
  INVALID_REPO: { retry: false },
  SERVER_ERROR: { retry: true, delay: 10000, maxAttempts: 2 }
};

interface RequestOptions {
  timeout?: number;
  retryAttempts?: number;
  onProgress?: (progress: AnalysisProgress) => void;
  signal?: AbortSignal;
}

class ModalApiClient {
  private baseUrls: Record<string, string>;
  private activeRequests = new Map<string, AbortController>();

  constructor() {
    const environment = process.env.NODE_ENV === 'production' ? 'production' : 'development';
    this.baseUrls = API_ENDPOINTS[environment];
  }

  private createModalError(
    type: ModalError['type'],
    stage: ModalError['stage'],
    message: string,
    details?: string,
    retryable: boolean = false
  ): ModalError {
    const error = new Error(message) as ModalError;
    error.type = type;
    error.stage = stage;
    error.retryable = retryable;
    error.details = details;
    
    const recovery = ERROR_RECOVERY[type];
    if (recovery && recovery.delay) {
      error.retryDelay = recovery.delay;
    }
    
    return error;
  }

  private async makeRequest<T>(
    url: string, 
    options: RequestInit & RequestOptions,
    stage: ModalError['stage']
  ): Promise<T> {
    const { timeout = TIMEOUTS.MEDIUM_REPO, retryAttempts = 1, onProgress, ...fetchOptions } = options;
    
    let lastError: ModalError | null = null;
    
    for (let attempt = 0; attempt < retryAttempts; attempt++) {
      const controller = new AbortController();
      const requestId = `${Date.now()}-${Math.random()}`;
      
      // Store controller for potential cancellation
      this.activeRequests.set(requestId, controller);
      
      try {
        // Add debugging information
        console.log(`Making request to: ${url}`);
        console.log(`Request options:`, { method: fetchOptions.method, headers: fetchOptions.headers });
        
        // Create timeout promise
        const timeoutPromise = new Promise<never>((_, reject) => {
          setTimeout(() => {
            controller.abort();
            reject(this.createModalError(
              'ANALYSIS_TIMEOUT',
              stage,
              `Request timed out after ${timeout}ms`,
              `Attempt ${attempt + 1} of ${retryAttempts}`,
              attempt < retryAttempts - 1
            ));
          }, timeout);
        });

        // Create fetch promise with better error handling
        const fetchPromise = fetch(url, {
          ...fetchOptions,
          signal: controller.signal,
          mode: 'cors', // Explicitly set CORS mode
          headers: {
            'Content-Type': 'application/json',
            ...fetchOptions.headers,
          },
        }).catch((fetchError) => {
          console.error('Fetch error:', fetchError);
          
          // Provide more specific error messages based on the type of fetch error
          if (fetchError.name === 'TypeError' && fetchError.message === 'Failed to fetch') {
            throw this.createModalError(
              'NETWORK_ERROR',
              stage,
              'Network connection failed. This could be due to CORS policy, network connectivity, or the Modal service being unavailable.',
              `URL: ${url}\nError: ${fetchError.message}\nThis often indicates CORS issues or the Modal endpoint is not accessible.`,
              true
            );
          }
          
          throw fetchError;
        });

        // Race between fetch and timeout
        const response = await Promise.race([fetchPromise, timeoutPromise]);
        
        console.log(`Response status: ${response.status}`);
        
        if (!response.ok) {
          const errorText = await response.text();
          console.log(`Error response:`, errorText);
          
          let errorData: any = {};
          
          try {
            errorData = JSON.parse(errorText);
          } catch {
            errorData = { error: errorText };
          }

          throw this.createModalError(
            this.mapHttpStatusToErrorType(response.status),
            stage,
            errorData.error || `HTTP ${response.status}: ${response.statusText}`,
            errorData.details,
            response.status >= 500 // Retry on server errors
          );
        }

        const data = await response.json();
        console.log('Response data received successfully');
        this.activeRequests.delete(requestId);
        return data;

      } catch (error: any) {
        this.activeRequests.delete(requestId);
        
        if (error instanceof Error && 'type' in error) {
          lastError = error as ModalError;
        } else if (error.name === 'AbortError') {
          lastError = this.createModalError(
            'ANALYSIS_TIMEOUT',
            stage,
            'Request was cancelled',
            'User cancelled or timeout',
            false
          );
        } else {
          lastError = this.createModalError(
            'NETWORK_ERROR',
            stage,
            error.message || 'Network error',
            error.stack,
            true
          );
        }

        console.error(`Request attempt ${attempt + 1} failed:`, lastError);

        // If this was the last attempt or error is not retryable, throw
        if (attempt === retryAttempts - 1 || !lastError.retryable) {
          throw lastError;
        }

        // Wait before retry
        if (lastError.retryDelay) {
          onProgress?.({
            stage: 'INITIALIZATION',
            progress: 0,
            message: `Retrying in ${lastError.retryDelay / 1000}s (attempt ${attempt + 2}/${retryAttempts})...`,
          });
          await this.sleep(lastError.retryDelay);
        }
      }
    }

    throw lastError || new Error('Unknown error in retry loop');
  }

  private mapHttpStatusToErrorType(status: number): ModalError['type'] {
    switch (status) {
      case 400:
        return 'INVALID_REPO';
      case 429:
        return 'RATE_LIMIT';
      case 500:
      case 502:
      case 503:
        return 'SERVER_ERROR';
      case 504:
        return 'ANALYSIS_TIMEOUT';
      default:
        return 'SERVER_ERROR';
    }
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  private estimateTimeout(repositoryUrl: string): number {
    // Simple heuristic based on repository URL patterns
    if (repositoryUrl.includes('small') || repositoryUrl.includes('demo')) {
      return TIMEOUTS.SMALL_REPO;
    }
    if (repositoryUrl.includes('large') || repositoryUrl.includes('framework')) {
      return TIMEOUTS.LARGE_REPO;
    }
    return TIMEOUTS.MEDIUM_REPO;
  }

  async healthCheck(): Promise<HealthStatus> {
    try {
      const health = await this.makeRequest<HealthStatus>(
        this.baseUrls.healthCheck,
        {
          method: 'GET',
          timeout: TIMEOUTS.HEALTH_CHECK,
          retryAttempts: 2,
        },
        'HEALTH_CHECK'
      );
      
      return health;
    } catch (error: any) {
      throw this.createModalError(
        'SERVER_ERROR',
        'HEALTH_CHECK',
        'Health check failed',
        error.message,
        true
      );
    }
  }

  async analyzeRepository(
    repositoryUrl: string, 
    options: RequestOptions = {}
  ): Promise<BackendResponse> {
    const { onProgress } = options;
    
    // Step 1: Health check with cold start detection
    onProgress?.({
      stage: 'HEALTH_CHECK',
      progress: 5,
      message: 'Checking security analyzer status...',
    });

    try {
      const health = await this.healthCheck();
      
      if (health.status === 'initializing') {
        onProgress?.({
          stage: 'INITIALIZATION',
          progress: 10,
          message: 'Security analyzer is starting up (cold start detected)...',
        });
        
        // Wait for warm-up
        await this.sleep(30000);
      }
    } catch (error) {
      onProgress?.({
        stage: 'INITIALIZATION',
        progress: 15,
        message: 'Health check failed, proceeding with analysis...',
      });
    }

    // Step 2: Start repository analysis
    onProgress?.({
      stage: 'FETCH',
      progress: 20,
      message: 'Fetching repository files...',
    });

    const timeout = this.estimateTimeout(repositoryUrl);
    
    try {
      const result = await this.makeRequest<BackendResponse>(
        this.baseUrls.analyzeRepository,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ repository_url: repositoryUrl }),
          timeout,
          retryAttempts: 2,
          onProgress,
          ...options,
        },
        'ANALYSIS'
      );

      onProgress?.({
        stage: 'COMPLETE',
        progress: 100,
        message: 'Analysis complete!',
      });

      return result;

    } catch (error) {
      if (error instanceof Error && 'type' in error) {
        throw error;
      }
      
      throw this.createModalError(
        'SERVER_ERROR',
        'ANALYSIS',
        'Repository analysis failed',
        error instanceof Error ? error.message : 'Unknown error',
        false
      );
    }
  }

  async analyzeFile(
    filePath: string,
    fileContent: string,
    language: string,
    options: RequestOptions = {}
  ): Promise<any> {
    return this.makeRequest(
      this.baseUrls.analyzeFile,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          file_path: filePath,
          file_content: fileContent,
          language,
        }),
        timeout: TIMEOUTS.WARM_START,
        retryAttempts: 1,
        ...options,
      },
      'ANALYSIS'
    );
  }

  async debugAnalysis(
    filePath: string,
    fileContent: string,
    language: string,
    options: RequestOptions = {}
  ): Promise<DebugAnalysisResponse> {
    return this.makeRequest<DebugAnalysisResponse>(
      this.baseUrls.debugAnalysis,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          file_path: filePath,
          file_content: fileContent,
          language,
        }),
        timeout: TIMEOUTS.WARM_START,
        retryAttempts: 1,
        ...options,
      },
      'ANALYSIS'
    );
  }

  cancelAnalysis(requestId?: string): void {
    if (requestId && this.activeRequests.has(requestId)) {
      this.activeRequests.get(requestId)?.abort();
      this.activeRequests.delete(requestId);
    } else {
      // Cancel all active requests
      this.activeRequests.forEach(controller => controller.abort());
      this.activeRequests.clear();
    }
  }

  getActiveRequestCount(): number {
    return this.activeRequests.size;
  }
}

// Export singleton instance
export const modalApiClient = new ModalApiClient();
export default modalApiClient;
export type { ModalError }; 