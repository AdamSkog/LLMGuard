'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Toaster, toast } from 'react-hot-toast';
import Navigation from './components/Navigation';
import { Button } from './components/ui/Button';
import { ProgressTracker } from './components/analysis/ProgressTracker';
import { SecuritySummary } from './components/analysis/SecuritySummary';
import { useAnalysisStore } from './store/analysisStore';
import { isValidGitHubUrl } from './utils';
import modalApiClient from './services/modalApi';

interface RepositoryInfo {
  name: string;
  description: string | null;
  stars: number;
  forks: number;
  watchers: number;
  open_issues: number;
  language: string | null;
  created_at: string;
  updated_at: string;
  license: string | null;
}

interface SecurityIssue {
  type: string;
  severity: string;
  location: string;
  description: string;
  recommendation: string;
}

interface DependencyIssue {
  package: string;
  version: string;
  vulnerabilities: string[];
}

interface CodeMetrics {
  total_lines: number;
  complexity: number;
  test_coverage: number;
}

interface AnalysisSummary {
  total_issues: number;
  critical_issues: number;
  high_issues: number;
  medium_issues: number;
  recommendations: string[];
}

interface AnalysisResult {
  repository_info: RepositoryInfo;
  analysis: {
    security_issues: SecurityIssue[];
    dependency_issues: DependencyIssue[];
    code_metrics: CodeMetrics;
    summary: AnalysisSummary;
  };
}

export default function Home() {
  const [url, setUrl] = useState('');
  
  const {
    isAnalyzing,
    result,
    error,
    startAnalysis,
    updateProgress,
    setResult,
    setError,
    clearError,
    resetAnalysis,
    cancelAnalysis,
  } = useAnalysisStore();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!isValidGitHubUrl(url)) {
      toast.error('Please enter a valid GitHub repository URL');
      return;
    }

    clearError();
    startAnalysis(url);

    try {
      const analysisResult = await modalApiClient.analyzeRepository(url, {
        onProgress: updateProgress,
      });
      
      setResult(analysisResult);
      toast.success('Analysis completed successfully!');
    } catch (err: any) {
      console.error('Analysis failed:', err);
      setError(err);
      
      // Show user-friendly error messages with actionable suggestions
      if (err.type === 'NETWORK_ERROR') {
        toast.error('Connection failed. Check your internet connection and try again.');
      } else if (err.type === 'COLD_START_TIMEOUT') {
        toast.error('The security analyzer is starting up. Please try again in a moment.');
      } else if (err.type === 'ANALYSIS_TIMEOUT') {
        toast.error('Analysis timed out. The repository may be too large (>50 files) or the service may be busy.');
      } else if (err.type === 'INVALID_REPO') {
        toast.error('Invalid repository URL or repository not accessible.');
      } else if (err.type === 'RATE_LIMIT') {
        toast.error('Rate limit exceeded. Please try again later.');
      } else {
        toast.error(err.message || 'Analysis failed. Please try again.');
      }
    }
  };

  const handleTestConnection = async () => {
    toast('Testing connection to Modal service...', { icon: '🔍' });
    
    try {
      const health = await modalApiClient.healthCheck();
      toast.success(`Connection successful! Service status: ${health.status}`);
      console.log('Health check result:', health);
    } catch (err: any) {
      console.error('Health check failed:', err);
      toast.error(`Connection test failed: ${err.message}`);
    }
  };

  const handleCancel = () => {
    cancelAnalysis();
    toast('Analysis cancelled', { icon: '⏹️' });
  };

  const handleReset = () => {
    resetAnalysis();
    setUrl('');
  };

  return (
    <>
      <Navigation />
      <Toaster 
        position="top-right"
        toastOptions={{
          style: {
            background: 'rgba(17, 24, 39, 0.95)',
            color: '#fff',
            border: '1px solid rgba(255, 255, 255, 0.1)',
          },
        }}
      />
      
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-blue-900 text-white">
        {/* Animated background elements */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute w-96 h-96 bg-blue-500/10 rounded-full blur-3xl -top-48 -left-48 animate-blob"></div>
          <div className="absolute w-96 h-96 bg-purple-500/10 rounded-full blur-3xl top-48 -right-48 animate-blob animation-delay-2000"></div>
          <div className="absolute w-96 h-96 bg-green-500/10 rounded-full blur-3xl -bottom-48 left-48 animate-blob animation-delay-4000"></div>
        </div>

        <main className="relative min-h-screen pt-32 pb-8 px-8">
          <div className="max-w-7xl mx-auto">
            
            {/* Hero Section */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="text-center mb-12"
            >
              <h1 className="text-6xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
                🔐 LLMGuard Security Analyzer
              </h1>
              <p className="text-xl text-gray-300 max-w-4xl mx-auto leading-relaxed">
                Advanced AI-powered security analysis using fine-tuned Qwen3 model. 
                Identify vulnerabilities, get actionable recommendations, and secure your codebase with confidence.
              </p>
              
              <div className="mt-6 flex justify-center space-x-4 text-sm text-gray-400">
                <span className="flex items-center">🤖 AI-Powered</span>
                <span className="flex items-center">⚡ Real-time Analysis</span>
                <span className="flex items-center">🎯 50+ File Types</span>
                <span className="flex items-center">🔒 Security Focused</span>
              </div>
            </motion.div>

            {/* Analysis Form */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="max-w-4xl mx-auto mb-8"
            >
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="flex gap-4">
                  <input
                    type="text"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    placeholder="Enter GitHub repository URL (e.g., https://github.com/owner/repo)"
                    className="flex-1 p-4 rounded-lg bg-white/5 border border-white/10 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent backdrop-blur-sm"
                    required
                    disabled={isAnalyzing}
                  />
                  
                  <div className="flex gap-2">
                    {!isAnalyzing ? (
                      <>
                        <Button
                          type="submit"
                          size="lg"
                          disabled={!url.trim()}
                          className="px-8"
                        >
                          🚀 Analyze Repository
                        </Button>
                        <Button
                          type="button"
                          variant="outline"
                          size="lg"
                          onClick={handleTestConnection}
                          className="px-6"
                        >
                          🔍 Test Connection
                        </Button>
                      </>
                    ) : (
                      <Button
                        type="button"
                        variant="danger"
                        size="lg"
                        onClick={handleCancel}
                        className="px-8"
                      >
                        ⏹️ Cancel
                      </Button>
                    )}
                  </div>
                </div>
                
                {/* Quick Examples */}
                <div className="text-center">
                  <p className="text-sm text-gray-500 mb-2">Try these example repositories:</p>
                  <div className="flex flex-wrap justify-center gap-2">
                    {[
                      'https://github.com/AdamSkog/Hadoop-DocuSearch',
                      'https://github.com/octocat/Hello-World',
                      'https://github.com/adamskog/sample-python-app'
                    ].map((exampleUrl) => (
                      <button
                        key={exampleUrl}
                        type="button"
                        onClick={() => !isAnalyzing && setUrl(exampleUrl)}
                        disabled={isAnalyzing}
                        className="text-xs text-blue-400 hover:text-blue-300 disabled:opacity-50 disabled:cursor-not-allowed px-2 py-1 rounded border border-blue-400/30 hover:border-blue-400/50 transition-colors"
                      >
                        {exampleUrl.split('/').slice(-2).join('/')}
                      </button>
                    ))}
                  </div>
                </div>
              </form>
            </motion.div>

            {/* Progress Tracker */}
            <AnimatePresence>
              {(isAnalyzing || result || error) && (
                <div className="max-w-4xl mx-auto mb-8">
                  <ProgressTracker onCancel={handleCancel} />
                </div>
              )}
            </AnimatePresence>

            {/* Error Display */}
            <AnimatePresence>
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="max-w-4xl mx-auto mb-8"
                >
                  <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-6">
                    <div className="flex items-start">
                      <span className="text-red-400 mr-3 text-xl">❌</span>
                      <div className="flex-1">
                        <h3 className="text-red-400 font-semibold mb-2">Analysis Failed</h3>
                        <p className="text-red-300/80 mb-3">{error.message}</p>
                        
                        {error.details && (
                          <details className="text-sm text-red-300/60">
                            <summary className="cursor-pointer hover:text-red-300/80">
                              Technical Details
                            </summary>
                            <pre className="mt-2 p-2 bg-red-900/20 rounded text-xs overflow-auto">
                              {error.details}
                            </pre>
                          </details>
                        )}
                        
                        <div className="mt-4 flex space-x-3">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleSubmit(new Event('submit') as any)}
                            disabled={isAnalyzing}
                          >
                            🔄 Retry Analysis
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={handleReset}
                          >
                            🔄 Start Over
                          </Button>
                        </div>
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Results Display */}
            <AnimatePresence>
              {result && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.5 }}
                  className="space-y-8"
                >
                  {/* Security Summary */}
                  <SecuritySummary result={result} />

                  {/* AI Recommendations */}
                  {result.analysis_summary.recommendations.length > 0 && (
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.3 }}
                      className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-6"
                    >
                      <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
                        <span className="mr-2">💡</span>
                        AI Security Recommendations
                      </h3>
                      <div className="space-y-3">
                        {result.analysis_summary.recommendations.map((recommendation, index) => (
                          <motion.div
                            key={index}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: 0.4 + index * 0.1 }}
                            className="flex items-start p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg"
                          >
                            <span className="text-blue-400 mr-3 font-bold">{index + 1}.</span>
                            <p className="text-blue-100 leading-relaxed">{recommendation}</p>
                          </motion.div>
                        ))}
                      </div>
                    </motion.div>
                  )}

                  {/* Action Buttons */}
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.5 }}
                    className="text-center space-x-4"
                  >
                    <Button
                      variant="outline"
                      onClick={handleReset}
                    >
                      🔄 Analyze Another Repository
                    </Button>
                    <Button
                      variant="ghost"
                      onClick={() => {
                        // TODO: Implement export functionality
                        toast.success('Export feature coming soon!');
                      }}
                    >
                      📄 Export Report
                    </Button>
                  </motion.div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Features Grid (shown when no analysis is running) */}
            {!isAnalyzing && !result && !error && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.4 }}
                className="mt-16"
              >
                <h2 className="text-3xl font-bold text-center text-white mb-8">
                  Advanced Security Analysis Features
                </h2>
                
                <div className="grid md:grid-cols-3 gap-8">
                  <motion.div
                    whileHover={{ scale: 1.02 }}
                    className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-6 text-center"
                  >
                    <div className="text-4xl mb-4">🤖</div>
                    <h3 className="text-xl font-semibold text-white mb-3">AI-Powered Detection</h3>
                    <p className="text-gray-400 leading-relaxed">
                      Fine-tuned Qwen3 model trained specifically on security vulnerabilities 
                      across multiple programming languages.
                    </p>
                  </motion.div>

                  <motion.div
                    whileHover={{ scale: 1.02 }}
                    className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-6 text-center"
                  >
                    <div className="text-4xl mb-4">⚡</div>
                    <h3 className="text-xl font-semibold text-white mb-3">Real-time Analysis</h3>
                    <p className="text-gray-400 leading-relaxed">
                      Fast, comprehensive scanning with progress tracking and detailed 
                      insights for immediate action.
                    </p>
                  </motion.div>

                  <motion.div
                    whileHover={{ scale: 1.02 }}
                    className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-6 text-center"
                  >
                    <div className="text-4xl mb-4">🎯</div>
                    <h3 className="text-xl font-semibold text-white mb-3">Multi-Language Support</h3>
                    <p className="text-gray-400 leading-relaxed">
                      Supports Python, JavaScript, Java, C++, PHP, Ruby, Swift, Go, 
                      Kotlin, and more programming languages.
                    </p>
                  </motion.div>
                </div>
              </motion.div>
            )}
          </div>
        </main>
      </div>
    </>
  );
}
