import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAnalysisProgress, useIsAnalyzing } from '../../store/analysisStore';
import { ProgressBar } from '../ui/ProgressBar';
import { Button } from '../ui/Button';
import { cn, getStageIcon, formatTime } from '../../utils';
import modalApiClient from '../../services/modalApi';

interface ProgressTrackerProps {
  onCancel?: () => void;
  className?: string;
}

const STAGE_DESCRIPTIONS = {
  HEALTH_CHECK: 'Checking analyzer availability...',
  INITIALIZATION: 'Starting security analyzer (cold start may take 30-60s)...',
  FETCH: 'Fetching repository files...',
  ANALYSIS: 'Analyzing files for security vulnerabilities...',
  RECOMMENDATIONS: 'Generating AI recommendations...',
  COMPLETE: 'Analysis complete!',
  ERROR: 'Analysis failed',
};

const STAGE_ORDER = [
  'HEALTH_CHECK',
  'INITIALIZATION', 
  'FETCH',
  'ANALYSIS',
  'RECOMMENDATIONS',
  'COMPLETE'
];

export const ProgressTracker: React.FC<ProgressTrackerProps> = ({
  onCancel,
  className,
}) => {
  const progress = useAnalysisProgress();
  const isAnalyzing = useIsAnalyzing();
  const [startTime, setStartTime] = useState<number | null>(null);
  const [elapsedTime, setElapsedTime] = useState(0);

  // Track elapsed time
  useEffect(() => {
    if (isAnalyzing && !startTime) {
      setStartTime(Date.now());
    } else if (!isAnalyzing) {
      setStartTime(null);
      setElapsedTime(0);
    }
  }, [isAnalyzing, startTime]);

  useEffect(() => {
    if (!startTime) return;

    const interval = setInterval(() => {
      setElapsedTime((Date.now() - startTime) / 1000);
    }, 1000);

    return () => clearInterval(interval);
  }, [startTime]);

  const handleCancel = () => {
    modalApiClient.cancelAnalysis();
    onCancel?.();
  };

  const getCurrentStageIndex = () => {
    return STAGE_ORDER.indexOf(progress.stage);
  };

  const isColdStart = progress.stage === 'INITIALIZATION' && progress.message.includes('cold start');

  if (!isAnalyzing && progress.stage !== 'COMPLETE' && progress.stage !== 'ERROR') {
    return null;
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className={cn(
        'bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-6',
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-white flex items-center">
          <span className="mr-2 text-xl">{getStageIcon(progress.stage)}</span>
          Security Analysis Progress
        </h3>
        
        <div className="flex items-center space-x-4">
          {startTime && (
            <div className="text-sm text-gray-400">
              Elapsed: {formatTime(elapsedTime)}
            </div>
          )}
          
          {isAnalyzing && (
            <Button
              variant="outline"
              size="sm"
              onClick={handleCancel}
              className="text-red-400 border-red-400/50 hover:bg-red-400/10"
            >
              Cancel
            </Button>
          )}
        </div>
      </div>

      {/* Main Progress Bar */}
      <ProgressBar
        progress={progress}
        size="lg"
        className="mb-6"
      />

      {/* Stage Indicator */}
      <div className="mb-6">
        <div className="flex items-center justify-between text-sm text-gray-400 mb-2">
          <span>Analysis Stages</span>
          <span>{getCurrentStageIndex() + 1} of {STAGE_ORDER.length}</span>
        </div>
        
        <div className="flex items-center space-x-2">
          {STAGE_ORDER.map((stage, index) => {
            const isActive = stage === progress.stage;
            const isCompleted = index < getCurrentStageIndex();
            const stageIcon = getStageIcon(stage);
            
            return (
              <div
                key={stage}
                className={cn(
                  'flex-1 flex items-center justify-center py-2 px-3 rounded-lg text-xs transition-colors',
                  isActive && 'bg-blue-500/20 border border-blue-500/50 text-blue-400',
                  isCompleted && 'bg-green-500/20 border border-green-500/50 text-green-400',
                  !isActive && !isCompleted && 'bg-gray-700/50 text-gray-500'
                )}
              >
                <span className="mr-1">{stageIcon}</span>
                <span className="hidden sm:inline">
                  {stage.replace('_', ' ')}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Cold Start Warning */}
      <AnimatePresence>
        {isColdStart && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4 mb-4"
          >
            <div className="flex items-start">
              <span className="text-yellow-400 mr-2">⚡</span>
              <div>
                <p className="text-yellow-400 font-medium text-sm">Cold Start Detected</p>
                <p className="text-yellow-300/80 text-xs mt-1">
                  The security analyzer is starting up for the first time. This may take 30-60 seconds 
                  while the AI model loads into memory. Subsequent analyses will be much faster.
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Active Request Count */}
      {modalApiClient.getActiveRequestCount() > 0 && (
        <div className="text-xs text-gray-500 text-center">
          Active requests: {modalApiClient.getActiveRequestCount()}
        </div>
      )}

      {/* Current Stage Details */}
      <div className="mt-4 p-3 bg-gray-800/50 rounded-lg">
        <div className="text-sm text-gray-300">
          <strong>{progress.stage.replace('_', ' ')}:</strong>{' '}
          {STAGE_DESCRIPTIONS[progress.stage] || progress.message}
        </div>
        
        {progress.currentFile && (
          <div className="text-xs text-gray-500 mt-2">
            <strong>Current file:</strong> {progress.currentFile}
          </div>
        )}
        
        {progress.filesProcessed !== undefined && progress.totalFiles !== undefined && (
          <div className="text-xs text-gray-500 mt-1">
            <strong>Files processed:</strong> {progress.filesProcessed} of {progress.totalFiles}
          </div>
        )}
      </div>
    </motion.div>
  );
}; 