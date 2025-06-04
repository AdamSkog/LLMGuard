import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FileAnalysis, SecurityIssue } from '../../types';
import { Badge } from '../ui/Badge';
import { CodeBlock } from '../ui/CodeBlock';
import { cn, getFileIcon, getLanguageColor, getSeverityPriority } from '../../utils';

interface FileAnalysisViewProps {
  fileAnalyses: FileAnalysis[];
  className?: string;
}

interface FileItemProps {
  fileAnalysis: FileAnalysis;
  isExpanded: boolean;
  onToggle: () => void;
}

const FileItem: React.FC<FileItemProps> = ({ fileAnalysis, isExpanded, onToggle }) => {
  const { file_path, language, issues, analysis_status, error_message } = fileAnalysis;
  const fileName = file_path.split('/').pop() || file_path;
  const fileIcon = getFileIcon(language);
  const languageColor = getLanguageColor(language);
  
  // Sort issues by severity
  const sortedIssues = [...issues].sort((a, b) => 
    getSeverityPriority(b.severity) - getSeverityPriority(a.severity)
  );

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'SUCCESS': return 'text-green-400';
      case 'ERROR': return 'text-red-400';
      case 'SKIPPED': return 'text-yellow-400';
      case 'TRUNCATED': return 'text-orange-400';
      default: return 'text-gray-400';
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white/5 border border-white/10 rounded-lg overflow-hidden"
    >
      {/* File Header */}
      <div 
        className="p-4 cursor-pointer hover:bg-white/5 transition-colors"
        onClick={onToggle}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <span className="text-xl">{fileIcon}</span>
            <div>
              <h3 className="font-medium text-white">{fileName}</h3>
              <p className="text-sm text-gray-400">{file_path}</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-3">
            <Badge variant="language" className={languageColor}>
              {language}
            </Badge>
            <Badge variant="status" status={analysis_status}>
              {analysis_status}
            </Badge>
            {issues.length > 0 && (
              <span className="text-sm text-red-400 font-medium">
                {issues.length} issue{issues.length !== 1 ? 's' : ''}
              </span>
            )}
            <motion.div
              animate={{ rotate: isExpanded ? 180 : 0 }}
              transition={{ duration: 0.2 }}
              className="text-gray-400"
            >
              ▼
            </motion.div>
          </div>
        </div>
      </div>

      {/* Expanded Content */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="border-t border-white/10"
          >
            {/* Error Message */}
            {error_message && analysis_status === 'ERROR' && (
              <div className="p-4 bg-red-500/10 border-b border-red-500/20">
                <p className="text-red-400 text-sm">
                  <strong>Error:</strong> {error_message}
                </p>
              </div>
            )}

            {/* Issues List */}
            {sortedIssues.length > 0 ? (
              <div className="p-4">
                <h4 className="text-sm font-medium text-gray-300 mb-3">
                  Security Issues ({sortedIssues.length})
                </h4>
                <div className="space-y-4">
                  {sortedIssues.map((issue, index) => (
                    <IssueItem key={index} issue={issue} />
                  ))}
                </div>
              </div>
            ) : analysis_status === 'SUCCESS' ? (
              <div className="p-4 text-center text-green-400">
                <span className="text-xl mr-2">✅</span>
                No security issues found in this file
              </div>
            ) : null}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

const IssueItem: React.FC<{ issue: SecurityIssue }> = ({ issue }) => {
  const [showCode, setShowCode] = useState(false);

  return (
    <div className="bg-gray-800/50 rounded-lg p-4">
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1">
          <div className="flex items-center space-x-2 mb-2">
            <Badge variant="severity" severity={issue.severity} animated>
              {issue.severity}
            </Badge>
            <span className="text-sm text-gray-400">
              {issue.vulnerability_type}
            </span>
            {issue.line_number && (
              <span className="text-xs text-blue-400">
                Line {issue.line_number}
              </span>
            )}
          </div>
          <h5 className="font-medium text-white mb-2">{issue.vulnerability_type}</h5>
          <p className="text-gray-300 text-sm mb-3">{issue.description}</p>
        </div>
      </div>

      {/* Recommendation */}
      <div className="mb-3 p-3 bg-blue-500/10 border border-blue-500/20 rounded">
        <p className="text-blue-300 text-sm">
          <strong className="text-blue-400">💡 Recommendation:</strong>{' '}
          {issue.recommendation}
        </p>
      </div>

      {/* Code Snippet */}
      {issue.code_snippet && (
        <div>
          <button
            onClick={() => setShowCode(!showCode)}
            className="text-sm text-gray-400 hover:text-gray-300 mb-2 flex items-center"
          >
            <span className="mr-1">{showCode ? '▼' : '▶'}</span>
            {showCode ? 'Hide' : 'Show'} code snippet
          </button>
          
          <AnimatePresence>
            {showCode && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
              >
                <CodeBlock
                  code={issue.code_snippet}
                  language="javascript" // Default fallback
                  highlightLines={issue.line_number ? [issue.line_number] : []}
                  maxHeight="200px"
                />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
};

export const FileAnalysisView: React.FC<FileAnalysisViewProps> = ({
  fileAnalyses,
  className,
}) => {
  const [expandedFiles, setExpandedFiles] = useState<Set<string>>(new Set());

  const toggleFile = (filePath: string) => {
    const newExpanded = new Set(expandedFiles);
    if (newExpanded.has(filePath)) {
      newExpanded.delete(filePath);
    } else {
      newExpanded.add(filePath);
    }
    setExpandedFiles(newExpanded);
  };

  // Sort files by number of issues (desc) and then by name
  const sortedFiles = [...fileAnalyses].sort((a, b) => {
    const issuesDiff = b.issues.length - a.issues.length;
    if (issuesDiff !== 0) return issuesDiff;
    return a.file_path.localeCompare(b.file_path);
  });

  if (fileAnalyses.length === 0) {
    return (
      <div className={cn(
        'bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-8 text-center',
        className
      )}>
        <div className="text-4xl mb-4">📁</div>
        <h3 className="text-lg font-semibold text-white mb-2">No Files Analyzed</h3>
        <p className="text-gray-400">No file analysis data available.</p>
      </div>
    );
  }

  return (
    <div className={cn(
      'bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-6',
      className
    )}>
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-semibold text-white flex items-center">
          <span className="mr-2">📋</span>
          File Analysis Details
        </h3>
        <div className="text-sm text-gray-400">
          {fileAnalyses.length} files analyzed
        </div>
      </div>

      <div className="space-y-4">
        {sortedFiles.map((fileAnalysis) => (
          <FileItem
            key={fileAnalysis.file_path}
            fileAnalysis={fileAnalysis}
            isExpanded={expandedFiles.has(fileAnalysis.file_path)}
            onToggle={() => toggleFile(fileAnalysis.file_path)}
          />
        ))}
      </div>
    </div>
  );
}; 