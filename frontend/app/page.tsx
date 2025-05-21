'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import Navigation from './components/Navigation';

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

const API_URL = `${process.env.NEXT_PUBLIC_API}/api/analyze`;

export default function Home() {
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to analyze repository');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'critical':
        return 'bg-red-500/20 border-red-500/50 text-red-400';
      case 'high':
        return 'bg-orange-500/20 border-orange-500/50 text-orange-400';
      case 'medium':
        return 'bg-yellow-500/20 border-yellow-500/50 text-yellow-400';
      case 'low':
        return 'bg-green-500/20 border-green-500/50 text-green-400';
      default:
        return 'bg-gray-500/20 border-gray-500/50 text-gray-400';
    }
  };

  return (
    <>
      <Navigation />
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-blue-900 text-white">
        {/* Animated background elements */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute w-96 h-96 bg-blue-500/10 rounded-full blur-3xl -top-48 -left-48 animate-blob"></div>
          <div className="absolute w-96 h-96 bg-purple-500/10 rounded-full blur-3xl top-48 -right-48 animate-blob animation-delay-2000"></div>
          <div className="absolute w-96 h-96 bg-green-500/10 rounded-full blur-3xl -bottom-48 left-48 animate-blob animation-delay-4000"></div>
        </div>

        <main className="relative min-h-screen pt-32 pb-8 px-8">
          <div className="max-w-7xl mx-auto">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="text-center mb-16"
            >
              <h1 className="text-6xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
                GitHub Repository Security Analyzer
              </h1>
              <p className="text-xl text-gray-300 max-w-3xl mx-auto">
                Analyze your GitHub repository for security vulnerabilities, code quality, and best practices.
                Get instant insights and actionable recommendations.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="max-w-3xl mx-auto"
            >
              <form onSubmit={handleSubmit} className="mb-8">
                <div className="flex gap-4">
                  <input
                    type="text"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    placeholder="Enter GitHub repository URL"
                    className="flex-1 p-4 rounded-lg bg-white/5 border border-white/10 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent backdrop-blur-sm"
                    required
                  />
                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    type="submit"
                    disabled={loading}
                    className="px-8 py-4 rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 text-white font-semibold hover:from-blue-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl transition-all"
                  >
                    {loading ? (
                      <div className="flex items-center">
                        <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Analyzing...
                      </div>
                    ) : 'Analyze'}
                  </motion.button>
                </div>
              </form>
            </motion.div>

            {error && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="max-w-3xl mx-auto p-4 mb-8 bg-red-500/20 border border-red-500/50 text-red-200 rounded-lg backdrop-blur-sm"
              >
                {error}
              </motion.div>
            )}

            {result && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.5 }}
                className="space-y-8"
              >
                <section className="p-8 bg-white/5 rounded-xl backdrop-blur-sm border border-white/10">
                  <h2 className="text-2xl font-semibold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
                    Repository Information
                  </h2>
                  <div className="grid grid-cols-2 gap-8">
                    <div className="space-y-4">
                      <div>
                        <span className="text-gray-400">Name</span>
                        <p className="text-xl text-white">{result.repository_info.name}</p>
                      </div>
                      <div>
                        <span className="text-gray-400">Description</span>
                        <p className="text-xl text-white">{result.repository_info.description || 'N/A'}</p>
                      </div>
                      <div>
                        <span className="text-gray-400">Language</span>
                        <p className="text-xl text-white">{result.repository_info.language || 'N/A'}</p>
                      </div>
                      <div>
                        <span className="text-gray-400">License</span>
                        <p className="text-xl text-white">{result.repository_info.license || 'N/A'}</p>
                      </div>
                    </div>
                    <div className="space-y-4">
                      <div>
                        <span className="text-gray-400">Stars</span>
                        <p className="text-xl text-white">{result.repository_info.stars}</p>
                      </div>
                      <div>
                        <span className="text-gray-400">Forks</span>
                        <p className="text-xl text-white">{result.repository_info.forks}</p>
                      </div>
                      <div>
                        <span className="text-gray-400">Watchers</span>
                        <p className="text-xl text-white">{result.repository_info.watchers}</p>
                      </div>
                      <div>
                        <span className="text-gray-400">Open Issues</span>
                        <p className="text-xl text-white">{result.repository_info.open_issues}</p>
                      </div>
                    </div>
                  </div>
                </section>

                <section className="p-8 bg-white/5 rounded-xl backdrop-blur-sm border border-white/10">
                  <h2 className="text-2xl font-semibold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
                    Security Summary
                  </h2>
                  <div className="grid grid-cols-4 gap-6">
                    <div className="p-6 bg-red-500/20 rounded-lg border border-red-500/50">
                      <div className="text-4xl font-bold text-red-400 mb-2">{result.analysis.summary.critical_issues}</div>
                      <div className="text-sm text-red-300">Critical Issues</div>
                    </div>
                    <div className="p-6 bg-orange-500/20 rounded-lg border border-orange-500/50">
                      <div className="text-4xl font-bold text-orange-400 mb-2">{result.analysis.summary.high_issues}</div>
                      <div className="text-sm text-orange-300">High Issues</div>
                    </div>
                    <div className="p-6 bg-yellow-500/20 rounded-lg border border-yellow-500/50">
                      <div className="text-4xl font-bold text-yellow-400 mb-2">{result.analysis.summary.medium_issues}</div>
                      <div className="text-sm text-yellow-300">Medium Issues</div>
                    </div>
                    <div className="p-6 bg-blue-500/20 rounded-lg border border-blue-500/50">
                      <div className="text-4xl font-bold text-blue-400 mb-2">{result.analysis.summary.total_issues}</div>
                      <div className="text-sm text-blue-300">Total Issues</div>
                    </div>
                  </div>
                </section>

                <section className="p-8 bg-white/5 rounded-xl backdrop-blur-sm border border-white/10">
                  <h2 className="text-2xl font-semibold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
                    Security Issues
                  </h2>
                  <div className="space-y-4">
                    {result.analysis.security_issues.map((issue, index) => (
                      <motion.div
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.1 }}
                        key={index}
                        className="p-6 bg-white/5 rounded-lg border border-white/10 backdrop-blur-sm"
                      >
                        <div className="flex justify-between items-start mb-4">
                          <h3 className="text-xl font-semibold text-white">{issue.type}</h3>
                          <span className={`px-3 py-1 rounded-full ${getSeverityColor(issue.severity)}`}>
                            {issue.severity}
                          </span>
                        </div>
                        <p className="text-gray-300 mb-4">{issue.description}</p>
                        <div className="flex justify-between items-center">
                          <p className="text-sm text-gray-400">Location: {issue.location}</p>
                          <p className="text-sm text-blue-400">Recommendation: {issue.recommendation}</p>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </section>

                <section className="p-8 bg-white/5 rounded-xl backdrop-blur-sm border border-white/10">
                  <h2 className="text-2xl font-semibold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
                    Code Metrics
                  </h2>
                  <div className="grid grid-cols-3 gap-6">
                    <div className="p-6 bg-blue-500/20 rounded-lg border border-blue-500/50">
                      <div className="text-3xl font-bold text-blue-400 mb-2">{result.analysis.code_metrics.total_lines}</div>
                      <div className="text-sm text-blue-300">Total Lines</div>
                    </div>
                    <div className="p-6 bg-purple-500/20 rounded-lg border border-purple-500/50">
                      <div className="text-3xl font-bold text-purple-400 mb-2">{result.analysis.code_metrics.complexity}</div>
                      <div className="text-sm text-purple-300">Complexity Score</div>
                    </div>
                    <div className="p-6 bg-green-500/20 rounded-lg border border-green-500/50">
                      <div className="text-3xl font-bold text-green-400 mb-2">{result.analysis.code_metrics.test_coverage}%</div>
                      <div className="text-sm text-green-300">Test Coverage</div>
                    </div>
                  </div>
                </section>
              </motion.div>
            )}
          </div>
        </main>
      </div>
    </>
  );
}
