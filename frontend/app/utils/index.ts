import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { SecurityIssue, FileAnalysis, FilterOptions } from '../types';

// Tailwind CSS class merging utility
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Severity helpers
export const getSeverityColor = (severity: string) => {
  switch (severity.toUpperCase()) {
    case 'HIGH':
      return 'bg-red-500/20 border-red-500/50 text-red-400';
    case 'MEDIUM':
      return 'bg-yellow-500/20 border-yellow-500/50 text-yellow-400';
    case 'LOW':
      return 'bg-green-500/20 border-green-500/50 text-green-400';
    default:
      return 'bg-gray-500/20 border-gray-500/50 text-gray-400';
  }
};

export const getSeverityBadgeColor = (severity: string) => {
  switch (severity.toUpperCase()) {
    case 'HIGH':
      return 'bg-red-500 text-white';
    case 'MEDIUM':
      return 'bg-yellow-500 text-white';
    case 'LOW':
      return 'bg-green-500 text-white';
    default:
      return 'bg-gray-500 text-white';
  }
};

export const getSeverityPriority = (severity: string): number => {
  switch (severity.toUpperCase()) {
    case 'HIGH':
      return 3;
    case 'MEDIUM':
      return 2;
    case 'LOW':
      return 1;
    default:
      return 0;
  }
};

// Risk score helpers
export const getRiskScoreColor = (riskScore: number) => {
  if (riskScore >= 80) return 'text-red-400';
  if (riskScore >= 60) return 'text-orange-400';
  if (riskScore >= 40) return 'text-yellow-400';
  if (riskScore >= 20) return 'text-blue-400';
  return 'text-green-400';
};

export const getRiskScoreLabel = (riskScore: number) => {
  if (riskScore >= 80) return 'Critical Risk';
  if (riskScore >= 60) return 'High Risk';
  if (riskScore >= 40) return 'Medium Risk';
  if (riskScore >= 20) return 'Low Risk';
  return 'Minimal Risk';
};

// File helpers
export const getFileIcon = (language: string) => {
  switch (language.toLowerCase()) {
    case 'python':
      return '🐍';
    case 'javascript':
    case 'typescript':
      return '📜';
    case 'java':
      return '☕';
    case 'cpp':
    case 'c':
      return '⚡';
    case 'csharp':
      return '#️⃣';
    case 'php':
      return '🐘';
    case 'ruby':
      return '💎';
    case 'swift':
      return '🦅';
    case 'go':
      return '🐹';
    case 'kotlin':
      return '🎯';
    case 'fortran':
      return '🔬';
    default:
      return '📄';
  }
};

export const getLanguageColor = (language: string) => {
  switch (language.toLowerCase()) {
    case 'python':
      return 'text-yellow-400';
    case 'javascript':
      return 'text-yellow-300';
    case 'typescript':
      return 'text-blue-400';
    case 'java':
      return 'text-orange-400';
    case 'cpp':
    case 'c':
      return 'text-blue-300';
    case 'csharp':
      return 'text-purple-400';
    case 'php':
      return 'text-indigo-400';
    case 'ruby':
      return 'text-red-400';
    case 'swift':
      return 'text-orange-300';
    case 'go':
      return 'text-cyan-400';
    case 'kotlin':
      return 'text-purple-300';
    case 'fortran':
      return 'text-green-400';
    default:
      return 'text-gray-400';
  }
};

// Filtering helpers
export const filterFileAnalyses = (
  fileAnalyses: FileAnalysis[],
  filters: FilterOptions
): FileAnalysis[] => {
  return fileAnalyses.filter((fileAnalysis) => {
    // Filter by file types
    if (filters.fileTypes.length > 0) {
      const fileExtension = fileAnalysis.file_path.split('.').pop()?.toLowerCase();
      if (!fileExtension || !filters.fileTypes.includes(fileExtension)) {
        return false;
      }
    }

    // Filter by "show only with issues"
    if (filters.showOnlyWithIssues && fileAnalysis.issues.length === 0) {
      return false;
    }

    // Filter issues within the file
    const filteredIssues = fileAnalysis.issues.filter((issue) => {
      // Filter by severity
      if (!filters.severity.includes(issue.severity)) {
        return false;
      }

      // Filter by vulnerability types
      if (filters.vulnerabilityTypes.length > 0) {
        if (!filters.vulnerabilityTypes.includes(issue.vulnerability_type)) {
          return false;
        }
      }

      return true;
    });

    // Return file analysis with filtered issues
    return filteredIssues.length > 0 || !filters.showOnlyWithIssues;
  }).map((fileAnalysis) => ({
    ...fileAnalysis,
    issues: fileAnalysis.issues.filter((issue) => {
      return filters.severity.includes(issue.severity) &&
        (filters.vulnerabilityTypes.length === 0 || 
         filters.vulnerabilityTypes.includes(issue.vulnerability_type));
    }),
  }));
};

// Sorting helpers
export const sortFileAnalyses = (
  fileAnalyses: FileAnalysis[],
  sortBy: 'name' | 'issues' | 'severity',
  sortOrder: 'asc' | 'desc'
): FileAnalysis[] => {
  const sorted = [...fileAnalyses].sort((a, b) => {
    let comparison = 0;

    switch (sortBy) {
      case 'name':
        comparison = a.file_path.localeCompare(b.file_path);
        break;
      case 'issues':
        comparison = a.issues.length - b.issues.length;
        break;
      case 'severity':
        const aMaxSeverity = Math.max(...a.issues.map(i => getSeverityPriority(i.severity)), 0);
        const bMaxSeverity = Math.max(...b.issues.map(i => getSeverityPriority(i.severity)), 0);
        comparison = aMaxSeverity - bMaxSeverity;
        break;
    }

    return sortOrder === 'desc' ? -comparison : comparison;
  });

  return sorted;
};

// Time formatting
export const formatTime = (seconds: number): string => {
  if (seconds < 60) {
    return `${Math.round(seconds)}s`;
  }
  if (seconds < 3600) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.round(seconds % 60);
    return `${minutes}m ${remainingSeconds}s`;
  }
  const hours = Math.floor(seconds / 3600);
  const remainingMinutes = Math.floor((seconds % 3600) / 60);
  return `${hours}h ${remainingMinutes}m`;
};

export const formatRelativeTime = (timestamp: number): string => {
  const now = Date.now();
  const diff = now - timestamp;
  const seconds = Math.floor(diff / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (days > 0) return `${days} day${days > 1 ? 's' : ''} ago`;
  if (hours > 0) return `${hours} hour${hours > 1 ? 's' : ''} ago`;
  if (minutes > 0) return `${minutes} minute${minutes > 1 ? 's' : ''} ago`;
  return 'Just now';
};

// Repository URL helpers
export const extractRepoName = (repositoryUrl: string): string => {
  const match = repositoryUrl.match(/github\.com\/([^\/]+)\/([^\/]+)/);
  if (match) {
    return `${match[1]}/${match[2]}`;
  }
  return repositoryUrl;
};

export const isValidGitHubUrl = (url: string): boolean => {
  const githubUrlRegex = /^https:\/\/github\.com\/[a-zA-Z0-9\-_.]+\/[a-zA-Z0-9\-_.]+\/?$/;
  return githubUrlRegex.test(url);
};

// Progress helpers
export const getProgressColor = (progress: number): string => {
  if (progress < 25) return 'bg-blue-500';
  if (progress < 50) return 'bg-green-500';
  if (progress < 75) return 'bg-yellow-500';
  return 'bg-purple-500';
};

export const getStageIcon = (stage: string): string => {
  switch (stage) {
    case 'HEALTH_CHECK':
      return '🔍';
    case 'INITIALIZATION':
      return '🚀';
    case 'FETCH':
      return '📥';
    case 'ANALYSIS':
      return '🧠';
    case 'RECOMMENDATIONS':
      return '💡';
    case 'COMPLETE':
      return '✅';
    case 'ERROR':
      return '❌';
    default:
      return '⚙️';
  }
};

// Data aggregation helpers
export const aggregateIssuesByType = (issues: SecurityIssue[]): Record<string, number> => {
  return issues.reduce((acc, issue) => {
    acc[issue.vulnerability_type] = (acc[issue.vulnerability_type] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);
};

export const aggregateIssuesBySeverity = (issues: SecurityIssue[]) => {
  return issues.reduce((acc, issue) => {
    acc[issue.severity] = (acc[issue.severity] || 0) + 1;
    return acc;
  }, { HIGH: 0, MEDIUM: 0, LOW: 0 });
};

// Chart data helpers
export const createChartData = (data: Record<string, number>, colors?: string[]) => {
  return Object.entries(data).map(([name, value], index) => ({
    name,
    value,
    color: colors?.[index % colors.length] || `hsl(${(index * 137.508) % 360}, 70%, 60%)`,
  }));
};

// Local storage helpers
export const getStorageItem = <T>(key: string, defaultValue: T): T => {
  try {
    const item = localStorage.getItem(key);
    return item ? JSON.parse(item) : defaultValue;
  } catch {
    return defaultValue;
  }
};

export const setStorageItem = <T>(key: string, value: T): void => {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch (error) {
    console.warn('Failed to save to localStorage:', error);
  }
};

// Debounce helper
export const debounce = <T extends (...args: any[]) => any>(
  func: T,
  wait: number
): ((...args: Parameters<T>) => void) => {
  let timeout: NodeJS.Timeout;
  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}; 