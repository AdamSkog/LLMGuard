import React from 'react';
import { motion } from 'framer-motion';
import { BackendResponse } from '../../types';
import { Badge } from '../ui/Badge';
import { cn, getRiskScoreColor, getRiskScoreLabel, formatTime, extractRepoName } from '../../utils';

interface SecuritySummaryProps {
  result: BackendResponse;
  className?: string;
}

export const SecuritySummary: React.FC<SecuritySummaryProps> = ({
  result,
  className,
}) => {
  const { analysis_summary: summary } = result;
  const riskScore = summary.risk_score;
  const severityBreakdown = summary.severity_breakdown;
  
  const totalIssues = Object.values(severityBreakdown).reduce((sum, count) => sum + count, 0);
  const repoName = extractRepoName(result.repository_url);

  // Calculate completion rate
  const successfulFiles = result.file_analyses.filter(
    f => f.analysis_status === 'SUCCESS'
  ).length;
  const completionRate = result.total_files_scanned > 0 
    ? (successfulFiles / result.total_files_scanned) * 100 
    : 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        'bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-6',
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-white mb-1">Security Analysis Summary</h2>
          <p className="text-gray-400 text-sm">📁 {repoName}</p>
        </div>
        
        <div className="text-right">
          <div className={cn(
            'text-3xl font-bold mb-1',
            getRiskScoreColor(riskScore)
          )}>
            {riskScore}/100
          </div>
          <div className="text-sm text-gray-400">
            {getRiskScoreLabel(riskScore)}
          </div>
        </div>
      </div>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.1 }}
          className="bg-gray-800/50 rounded-lg p-4 text-center"
        >
          <div className="text-2xl font-bold text-red-400">{totalIssues}</div>
          <div className="text-sm text-gray-400">Total Issues</div>
        </motion.div>
        
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="bg-gray-800/50 rounded-lg p-4 text-center"
        >
          <div className="text-2xl font-bold text-orange-400">{result.files_with_issues}</div>
          <div className="text-sm text-gray-400">Affected Files</div>
        </motion.div>
        
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="bg-gray-800/50 rounded-lg p-4 text-center"
        >
          <div className="text-2xl font-bold text-blue-400">{result.total_files_scanned}</div>
          <div className="text-sm text-gray-400">Files Scanned</div>
        </motion.div>
        
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="bg-gray-800/50 rounded-lg p-4 text-center"
        >
          <div className="text-2xl font-bold text-green-400">{completionRate.toFixed(1)}%</div>
          <div className="text-sm text-gray-400">Success Rate</div>
        </motion.div>
      </div>

      {/* Severity Breakdown */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-white mb-3">Issue Severity Breakdown</h3>
        <div className="space-y-3">
          {Object.entries(severityBreakdown).map(([severity, count], index) => {
            const percentage = totalIssues > 0 ? (count / totalIssues) * 100 : 0;
            
            return (
              <motion.div
                key={severity}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.5 + index * 0.1 }}
                className="flex items-center justify-between"
              >
                <div className="flex items-center space-x-3">
                  <Badge variant="severity" severity={severity as any} animated>
                    {severity}
                  </Badge>
                  <span className="text-gray-300">{count} issues</span>
                </div>
                
                <div className="flex items-center space-x-2">
                  <div className="w-24 bg-gray-700 rounded-full h-2">
                    <motion.div
                      className={cn(
                        'h-2 rounded-full',
                        severity === 'HIGH' && 'bg-red-500',
                        severity === 'MEDIUM' && 'bg-yellow-500',
                        severity === 'LOW' && 'bg-green-500'
                      )}
                      initial={{ width: 0 }}
                      animate={{ width: `${percentage}%` }}
                      transition={{ delay: 0.7 + index * 0.1, duration: 0.5 }}
                    />
                  </div>
                  <span className="text-sm text-gray-400 w-12 text-right">
                    {percentage.toFixed(1)}%
                  </span>
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* Analysis Metadata */}
      <div className="border-t border-gray-700 pt-4">
        <h3 className="text-lg font-semibold text-white mb-3">Analysis Details</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-400">Fetch Time:</span>
            <span className="text-gray-300">{formatTime(summary.fetch_time)}</span>
          </div>
          
          <div className="flex justify-between">
            <span className="text-gray-400">Vulnerability Types:</span>
            <span className="text-gray-300">{Object.keys(summary.vulnerability_types).length}</span>
          </div>
          
          <div className="flex justify-between">
            <span className="text-gray-400">Analysis Time:</span>
            <span className="text-gray-300">{new Date().toLocaleTimeString()}</span>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="mt-4 flex flex-wrap gap-2">
        <Badge variant="language" className="text-xs">
          🤖 AI-Powered Analysis
        </Badge>
        <Badge variant="language" className="text-xs">
          🔒 Qwen3 Security Model
        </Badge>
        {summary.fetch_time < 30 && (
          <Badge variant="language" className="text-xs">
            ⚡ Fast Analysis
          </Badge>
        )}
        {completionRate === 100 && (
          <Badge variant="language" className="text-xs">
            ✅ Complete Scan
          </Badge>
        )}
      </div>
    </motion.div>
  );
}; 