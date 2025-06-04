import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { BackendResponse, ModalError, AnalysisProgress, FilterOptions, ViewOptions } from '../types';

interface AnalysisState {
  // Analysis state
  isAnalyzing: boolean;
  progress: AnalysisProgress;
  result: BackendResponse | null;
  error: ModalError | null;
  requestId?: string;

  // UI state
  filters: FilterOptions;
  viewOptions: ViewOptions;
  debugMode: boolean;
  
  // History
  analysisHistory: Array<{
    id: string;
    repositoryUrl: string;
    timestamp: number;
    result: BackendResponse;
  }>;

  // Actions
  startAnalysis: (repositoryUrl: string) => void;
  updateProgress: (progress: AnalysisProgress) => void;
  setResult: (result: BackendResponse) => void;
  setError: (error: ModalError) => void;
  clearError: () => void;
  resetAnalysis: () => void;
  cancelAnalysis: () => void;
  
  // History actions
  addToHistory: (repositoryUrl: string, result: BackendResponse) => void;
  clearHistory: () => void;
  
  // Filter actions
  updateFilters: (filters: Partial<FilterOptions>) => void;
  resetFilters: () => void;
  
  // View actions
  updateViewOptions: (options: Partial<ViewOptions>) => void;
  toggleFileExpansion: (filePath: string) => void;
  
  // Settings
  toggleDebugMode: () => void;
}

const initialProgress: AnalysisProgress = {
  stage: 'HEALTH_CHECK',
  progress: 0,
  message: 'Ready to analyze...',
};

const initialFilters: FilterOptions = {
  severity: ['HIGH', 'MEDIUM', 'LOW'],
  vulnerabilityTypes: [],
  fileTypes: [],
  showOnlyWithIssues: false,
};

const initialViewOptions: ViewOptions = {
  groupBy: 'file',
  sortBy: 'name',
  sortOrder: 'asc',
  expandedFiles: new Set(),
};

export const useAnalysisStore = create<AnalysisState>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial state
        isAnalyzing: false,
        progress: initialProgress,
        result: null,
        error: null,
        requestId: undefined,
        
        filters: initialFilters,
        viewOptions: initialViewOptions,
        debugMode: false,
        analysisHistory: [],

        // Analysis actions
        startAnalysis: (repositoryUrl: string) => {
          const requestId = `${Date.now()}-${Math.random()}`;
          set({
            isAnalyzing: true,
            result: null,
            error: null,
            requestId,
            progress: {
              stage: 'HEALTH_CHECK',
              progress: 0,
              message: 'Starting analysis...',
            },
          });
        },

        updateProgress: (progress: AnalysisProgress) => {
          set({ progress });
        },

        setResult: (result: BackendResponse) => {
          const { addToHistory } = get();
          set({
            isAnalyzing: false,
            result,
            error: null,
            progress: {
              stage: 'COMPLETE',
              progress: 100,
              message: 'Analysis complete!',
            },
          });
          addToHistory(result.repository_url, result);
        },

        setError: (error: ModalError) => {
          set({
            isAnalyzing: false,
            error,
            progress: {
              stage: 'ERROR',
              progress: 0,
              message: `Error: ${error.message}`,
            },
          });
        },

        clearError: () => {
          set({ error: null });
        },

        resetAnalysis: () => {
          set({
            isAnalyzing: false,
            result: null,
            error: null,
            requestId: undefined,
            progress: initialProgress,
          });
        },

        cancelAnalysis: () => {
          set({
            isAnalyzing: false,
            progress: {
              stage: 'ERROR',
              progress: 0,
              message: 'Analysis cancelled',
            },
          });
        },

        // History actions
        addToHistory: (repositoryUrl: string, result: BackendResponse) => {
          const history = get().analysisHistory;
          const newEntry = {
            id: `${Date.now()}-${Math.random()}`,
            repositoryUrl,
            timestamp: Date.now(),
            result,
          };
          
          // Keep only last 10 analyses
          const updatedHistory = [newEntry, ...history.slice(0, 9)];
          set({ analysisHistory: updatedHistory });
        },

        clearHistory: () => {
          set({ analysisHistory: [] });
        },

        // Filter actions
        updateFilters: (newFilters: Partial<FilterOptions>) => {
          set({
            filters: { ...get().filters, ...newFilters },
          });
        },

        resetFilters: () => {
          set({ filters: initialFilters });
        },

        // View actions
        updateViewOptions: (newOptions: Partial<ViewOptions>) => {
          const currentOptions = get().viewOptions;
          set({
            viewOptions: {
              ...currentOptions,
              ...newOptions,
              // Handle Set objects properly
              expandedFiles: newOptions.expandedFiles || currentOptions.expandedFiles,
            },
          });
        },

        toggleFileExpansion: (filePath: string) => {
          const { viewOptions } = get();
          const newExpandedFiles = new Set(viewOptions.expandedFiles);
          
          if (newExpandedFiles.has(filePath)) {
            newExpandedFiles.delete(filePath);
          } else {
            newExpandedFiles.add(filePath);
          }
          
          set({
            viewOptions: {
              ...viewOptions,
              expandedFiles: newExpandedFiles,
            },
          });
        },

        // Settings
        toggleDebugMode: () => {
          set({ debugMode: !get().debugMode });
        },
      }),
      {
        name: 'llmguard-analysis-store',
        // Don't persist certain volatile state
        partialize: (state) => ({
          analysisHistory: state.analysisHistory,
          filters: state.filters,
          viewOptions: {
            ...state.viewOptions,
            expandedFiles: [], // Don't persist expanded files
          },
          debugMode: state.debugMode,
        }),
      }
    ),
    {
      name: 'analysis-store',
    }
  )
);

// Selector hooks for better performance
export const useAnalysisProgress = () => useAnalysisStore((state) => state.progress);
export const useAnalysisResult = () => useAnalysisStore((state) => state.result);
export const useAnalysisError = () => useAnalysisStore((state) => state.error);
export const useIsAnalyzing = () => useAnalysisStore((state) => state.isAnalyzing);
export const useFilters = () => useAnalysisStore((state) => state.filters);
export const useViewOptions = () => useAnalysisStore((state) => state.viewOptions);
export const useDebugMode = () => useAnalysisStore((state) => state.debugMode);
export const useAnalysisHistory = () => useAnalysisStore((state) => state.analysisHistory); 