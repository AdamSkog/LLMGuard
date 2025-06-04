import React from 'react';
import { motion } from 'framer-motion';
import { cn, getSeverityBadgeColor } from '../../utils';

interface BadgeProps {
  children: React.ReactNode;
  variant?: 'default' | 'severity' | 'status' | 'language';
  severity?: 'HIGH' | 'MEDIUM' | 'LOW';
  status?: 'SUCCESS' | 'ERROR' | 'SKIPPED' | 'TRUNCATED';
  className?: string;
  animated?: boolean;
}

const badgeVariants = {
  default: 'bg-gray-500 text-white',
  severity: '', // Will be set based on severity prop
  status: '', // Will be set based on status prop
  language: 'bg-blue-500/20 border border-blue-500/50 text-blue-400',
};

const statusColors = {
  SUCCESS: 'bg-green-500 text-white',
  ERROR: 'bg-red-500 text-white',
  SKIPPED: 'bg-yellow-500 text-white',
  TRUNCATED: 'bg-orange-500 text-white',
};

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = 'default',
  severity,
  status,
  className,
  animated = false,
}) => {
  let badgeColor = badgeVariants[variant];

  if (variant === 'severity' && severity) {
    badgeColor = getSeverityBadgeColor(severity);
  }

  if (variant === 'status' && status) {
    badgeColor = statusColors[status];
  }

  const BadgeComponent = animated ? motion.span : 'span';
  const animationProps = animated
    ? {
        initial: { scale: 0.8, opacity: 0 },
        animate: { scale: 1, opacity: 1 },
        whileHover: { scale: 1.05 },
        transition: { duration: 0.2 },
      }
    : {};

  return (
    <BadgeComponent
      className={cn(
        'inline-flex items-center px-2 py-1 rounded-full text-xs font-medium',
        badgeColor,
        className
      )}
      {...animationProps}
    >
      {children}
    </BadgeComponent>
  );
}; 