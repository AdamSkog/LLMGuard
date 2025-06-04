import React from 'react';
import { motion } from 'framer-motion';
import { cn, getProgressColor, getStageIcon } from '../../utils';
import { AnalysisProgress } from '../../types';

interface ProgressBarProps {
  progress: AnalysisProgress;
  className?: string;
  showDetails?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

const sizeClasses = {
  sm: 'h-2',
  md: 'h-3',
  lg: 'h-4',
};

export const ProgressBar: React.FC<ProgressBarProps> = ({
  progress,
  className,
  showDetails = true,
  size = 'md',
}) => {
  const progressPercentage = Math.max(0, Math.min(100, progress.progress));
  const stageIcon = getStageIcon(progress.stage);
  
  return (
    <div className={cn('w-full', className)}>
      {showDetails && (
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center space-x-2">
            <span className="text-lg">{stageIcon}</span>
            <span className="text-sm font-medium text-gray-300">
              {progress.message}
            </span>
          </div>
          <span className="text-sm text-gray-400">
            {progressPercentage}%
          </span>
        </div>
      )}
      
      <div className={cn(
        'bg-gray-700 rounded-full overflow-hidden',
        sizeClasses[size]
      )}>
        <motion.div
          className={cn(
            'h-full rounded-full transition-colors duration-300',
            getProgressColor(progressPercentage)
          )}
          initial={{ width: 0 }}
          animate={{ width: `${progressPercentage}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
        />
      </div>
      
      {showDetails && progress.currentFile && (
        <div className="mt-2 text-xs text-gray-500">
          {progress.filesProcessed !== undefined && progress.totalFiles !== undefined && (
            <span>
              Analyzing file {progress.filesProcessed} of {progress.totalFiles}:{' '}
            </span>
          )}
          <span className="text-gray-400">{progress.currentFile}</span>
        </div>
      )}
      
      {showDetails && progress.estimatedTimeRemaining && (
        <div className="mt-1 text-xs text-gray-500">
          Estimated time remaining: {Math.ceil(progress.estimatedTimeRemaining / 1000)}s
        </div>
      )}
    </div>
  );
}; 