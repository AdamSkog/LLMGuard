import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import { motion } from 'framer-motion';
import { cn, getRiskScoreColor, getRiskScoreLabel } from '../../utils';

interface RiskGaugeProps {
  riskScore: number;
  size?: 'sm' | 'md' | 'lg';
  animated?: boolean;
  className?: string;
}

const sizeConfigs = {
  sm: { width: 120, height: 120, innerRadius: 25, outerRadius: 50 },
  md: { width: 200, height: 200, innerRadius: 40, outerRadius: 80 },
  lg: { width: 300, height: 300, innerRadius: 60, outerRadius: 120 },
};

export const RiskGauge: React.FC<RiskGaugeProps> = ({
  riskScore,
  size = 'md',
  animated = true,
  className,
}) => {
  const config = sizeConfigs[size];
  
  // Create data for the gauge
  const data = [
    { name: 'Risk', value: riskScore, color: getRiskScoreColor(riskScore).replace('text-', '') },
    { name: 'Safe', value: 100 - riskScore, color: 'gray-600' },
  ];

  // Define color mapping
  const getColorValue = (colorName: string) => {
    const colorMap: Record<string, string> = {
      'red-400': '#f87171',
      'orange-400': '#fb923c',
      'yellow-400': '#facc15',
      'blue-400': '#60a5fa',
      'green-400': '#4ade80',
      'gray-600': '#4b5563',
    };
    return colorMap[colorName] || '#4b5563';
  };

  const GaugeComponent = animated ? motion.div : 'div';
  const animationProps = animated
    ? {
        initial: { scale: 0.8, opacity: 0 },
        animate: { scale: 1, opacity: 1 },
        transition: { duration: 0.5 },
      }
    : {};

  return (
    <GaugeComponent
      className={cn('relative flex items-center justify-center', className)}
      {...animationProps}
    >
      <ResponsiveContainer width={config.width} height={config.height}>
        <PieChart>
          <Pie
            data={data}
            startAngle={180}
            endAngle={0}
            innerRadius={config.innerRadius}
            outerRadius={config.outerRadius}
            dataKey="value"
            stroke="none"
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={getColorValue(entry.color)} />
            ))}
          </Pie>
        </PieChart>
      </ResponsiveContainer>
      
      {/* Center content */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <div className={cn(
          'font-bold mb-1',
          getRiskScoreColor(riskScore),
          size === 'sm' && 'text-lg',
          size === 'md' && 'text-2xl',
          size === 'lg' && 'text-4xl'
        )}>
          {riskScore}
        </div>
        <div className={cn(
          'text-gray-400 text-center leading-tight',
          size === 'sm' && 'text-xs',
          size === 'md' && 'text-sm',
          size === 'lg' && 'text-base'
        )}>
          {getRiskScoreLabel(riskScore)}
        </div>
      </div>
    </GaugeComponent>
  );
}; 