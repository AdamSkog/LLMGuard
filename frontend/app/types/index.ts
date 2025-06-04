// Backend Response Types matching Modal API
export interface SecurityIssue {
  file_path: string;
  line_number?: number;
  severity: "HIGH" | "MEDIUM" | "LOW";
  vulnerability_type: string;
  description: string;
  recommendation: string;
  code_snippet?: string;
}

export interface FileAnalysis {
  file_path: string;
  language: string;
  issues: SecurityIssue[];
  analysis_status: "SUCCESS" | "SKIPPED" | "ERROR" | "TRUNCATED";
  error_message?: string;
}

export interface RepositoryItem {
  type: "file" | "directory";
  language?: string;
  size?: number;
  skipped?: boolean;
  reason?: "too_large" | "unsupported_type";
}

export interface AnalysisSummary {
  severity_breakdown: {
    HIGH: number;
    MEDIUM: number;
    LOW: number;
  };
  vulnerability_types: Record<string, number>;
  risk_score: number; // 0-100 scale
  recommendations: string[]; // AI-generated recommendations
  fetch_time: number; // Time to fetch repository (seconds)
}

export interface BackendResponse {
  repository_url: string;
  total_files_scanned: number;
  files_with_issues: number;
  total_issues: number;
  file_analyses: FileAnalysis[];
  repository_structure: Record<string, RepositoryItem>;
  analysis_summary: AnalysisSummary;
}

// Modal-specific Error Types
export interface ModalErrorResponse {
  error: string;
  details?: string;
  type?: "TIMEOUT" | "RATE_LIMIT" | "INVALID_REPO" | "SERVER_ERROR" | "MODEL_ERROR";
}

export interface ModalError extends Error {
  type: 'COLD_START_TIMEOUT' | 'ANALYSIS_TIMEOUT' | 'MODEL_ERROR' | 
        'RATE_LIMIT' | 'INVALID_REPO' | 'SERVER_ERROR' | 'NETWORK_ERROR';
  stage: 'HEALTH_CHECK' | 'INITIALIZATION' | 'FETCH' | 'ANALYSIS' | 'RECOMMENDATIONS';
  retryable: boolean;
  retryDelay?: number;
  details?: string;
}

// Debug endpoint response
export interface DebugAnalysisResponse {
  file_path: string;
  language: string;
  prompt: string;
  raw_model_output: string;
  parsed_issues_count: number;
  parsed_issues: SecurityIssue[];
  contains_no_issues: boolean;
}

// Health check response
export interface HealthStatus {
  status: "healthy" | "unhealthy" | "initializing";
  base_model: string;
  lora_adapters: string;
  gpu_available: string;
}

// Progress tracking types
export interface AnalysisProgress {
  stage: 'HEALTH_CHECK' | 'INITIALIZATION' | 'FETCH' | 'ANALYSIS' | 'RECOMMENDATIONS' | 'COMPLETE' | 'ERROR';
  progress: number; // 0-100
  currentFile?: string;
  filesProcessed?: number;
  totalFiles?: number;
  estimatedTimeRemaining?: number;
  message: string;
}

// UI State Types
export interface AnalysisState {
  isAnalyzing: boolean;
  progress: AnalysisProgress;
  result: BackendResponse | null;
  error: ModalError | null;
  requestId?: string;
}

// Filter and display options
export interface FilterOptions {
  severity: ("HIGH" | "MEDIUM" | "LOW")[];
  vulnerabilityTypes: string[];
  fileTypes: string[];
  showOnlyWithIssues: boolean;
}

export interface ViewOptions {
  groupBy: "file" | "severity" | "type";
  sortBy: "name" | "issues" | "severity";
  sortOrder: "asc" | "desc";
  expandedFiles: Set<string>;
}

// Chart data types
export interface ChartData {
  name: string;
  value: number;
  color?: string;
}

export interface RiskGaugeProps {
  riskScore: number;
  size?: "sm" | "md" | "lg";
  animated?: boolean;
} 