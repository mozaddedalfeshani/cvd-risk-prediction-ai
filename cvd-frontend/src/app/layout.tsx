import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "CVD Risk Assessment - AI-Powered Cardiovascular Disease Prediction",
  description: "Clinical-grade AI model for cardiovascular disease risk assessment with 95.91% accuracy. Designed for healthcare professionals.",
  keywords: "CVD, cardiovascular disease, risk assessment, AI, machine learning, healthcare, clinical decision support",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.className} antialiased`}>
        {children}
      </body>
    </html>
  );
}
