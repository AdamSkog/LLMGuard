'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export default function Navigation() {
  const [isAboutOpen, setIsAboutOpen] = useState(false);

  return (
    <>
      <nav className="fixed top-0 left-0 right-0 z-50 bg-transparent">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-20">
            <div className="flex items-center">
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                className="flex items-center space-x-2"
              >
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                  <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-500 to-purple-600">
                  LLMGuard
                </span>
              </motion.div>
            </div>
            <div className="flex items-center space-x-6">
              <motion.a
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                href="#features"
                className="text-gray-300 hover:text-white transition-colors"
              >
                Features
              </motion.a>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setIsAboutOpen(true)}
                className="px-6 py-2 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 text-white font-medium hover:from-blue-600 hover:to-purple-700 transition-all shadow-lg hover:shadow-xl"
              >
                About Us
              </motion.button>
            </div>
          </div>
        </div>
      </nav>

      <AnimatePresence>
        {isAboutOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsAboutOpen(false)}
              className="fixed inset-0 bg-black/50 z-50 backdrop-blur-sm"
            />
            <motion.div
              initial={{ opacity: 0, scale: 0.9, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9, y: 20 }}
              className="fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 z-50 w-full max-w-lg bg-gradient-to-br from-gray-900 to-gray-800 rounded-xl shadow-2xl p-8 border border-white/10"
            >
              <div className="relative">
                <button
                  onClick={() => setIsAboutOpen(false)}
                  className="absolute top-0 right-0 text-gray-400 hover:text-white transition-colors"
                >
                  <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
                <div className="text-center">
                  <h2 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400 mb-4">
                    About LLMGuard
                  </h2>
                  <p className="text-gray-300 mb-6">
                    LLMGuard is a powerful LLM application analyzer that helps developers identify security vulnerabilities,
                    code quality issues, and potential improvements in their codebase. Our mission is to make LLM security
                    accessible and actionable for everyone.
                  </p>
                  <div className="grid grid-cols-3 gap-4 mb-6">
                    <div className="p-4 bg-white/5 rounded-lg border border-white/10 text-center">
                      <h3 className="font-semibold text-blue-400">Security First</h3>
                      <p className="text-sm text-gray-400">Advanced vulnerability detection</p>
                    </div>
                    <div className="p-4 bg-white/5 rounded-lg border border-white/10 text-center">
                      <h3 className="font-semibold text-purple-400">Extensive</h3>
                      <p className="text-sm text-gray-400">Deep code analysis</p>
                    </div>
                    <div className="p-4 bg-white/5 rounded-lg border border-white/10 text-center">
                      <h3 className="font-semibold text-green-400">Real-time</h3>
                      <p className="text-sm text-gray-400">Instant feedback</p>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  );
} 