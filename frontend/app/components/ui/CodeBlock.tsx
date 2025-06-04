import React from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { motion } from 'framer-motion';
import { cn } from '../../utils';

interface CodeBlockProps {
  code: string;
  language: string;
  fileName?: string;
  highlightLines?: number[];
  className?: string;
  showLineNumbers?: boolean;
  maxHeight?: string;
}

export const CodeBlock: React.FC<CodeBlockProps> = ({
  code,
  language,
  fileName,
  highlightLines = [],
  className,
  showLineNumbers = true,
  maxHeight = '400px',
}) => {
  // Custom style with line highlighting
  const customStyle = {
    ...atomDark,
    'pre[class*="language-"]': {
      ...atomDark['pre[class*="language-"]'],
      background: 'rgb(17 24 39)', // gray-900
      margin: 0,
      maxHeight,
      overflow: 'auto',
    },
    'code[class*="language-"]': {
      ...atomDark['code[class*="language-"]'],
      background: 'transparent',
    },
  };

  // Line props function to highlight specific lines
  const lineProps = (lineNumber: number) => {
    const isHighlighted = highlightLines.includes(lineNumber);
    return {
      style: {
        backgroundColor: isHighlighted ? 'rgba(239, 68, 68, 0.2)' : 'transparent',
        borderLeft: isHighlighted ? '3px solid rgb(239, 68, 68)' : 'none',
        paddingLeft: isHighlighted ? '8px' : '11px',
        display: 'block',
        margin: '0',
      },
    };
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        'bg-gray-900 rounded-lg overflow-hidden border border-gray-700',
        className
      )}
    >
      {fileName && (
        <div className="px-4 py-2 bg-gray-800 border-b border-gray-700 text-sm text-gray-300 font-mono">
          📄 {fileName}
        </div>
      )}
      
      <div className="relative">
        <SyntaxHighlighter
          language={language}
          style={customStyle}
          showLineNumbers={showLineNumbers}
          lineProps={lineProps}
          customStyle={{
            margin: 0,
            padding: '16px',
            background: 'rgb(17 24 39)',
            fontSize: '14px',
            lineHeight: '1.5',
          }}
          codeTagProps={{
            style: {
              fontFamily: 'Consolas, Monaco, "Andale Mono", "Ubuntu Mono", monospace',
            },
          }}
        >
          {code}
        </SyntaxHighlighter>
        
        {highlightLines.length > 0 && (
          <div className="absolute top-2 right-2 bg-red-500/20 border border-red-500/50 rounded px-2 py-1 text-xs text-red-400">
            🔍 {highlightLines.length} issue{highlightLines.length !== 1 ? 's' : ''} highlighted
          </div>
        )}
      </div>
    </motion.div>
  );
}; 