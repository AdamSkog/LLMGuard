import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LLMGuard - AI-Powered Security Analysis",
  description: "Advanced GitHub repository security analysis using fine-tuned Qwen3 AI model. Identify vulnerabilities, get actionable recommendations, and secure your codebase.",
  keywords: ["security analysis", "AI", "Qwen3", "GitHub", "vulnerability detection", "code security"],
  authors: [{ name: "LLMGuard Team" }],
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  themeColor: '#1f2937',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
