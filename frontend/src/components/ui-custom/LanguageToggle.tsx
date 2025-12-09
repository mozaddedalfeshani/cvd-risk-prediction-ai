"use client";

import { motion } from "framer-motion";
import { useLanguage } from "@/components/providers/LanguageProvider";
import { Globe } from "lucide-react";

export default function LanguageToggle() {
  const { language, setLanguage } = useLanguage();

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="fixed bottom-6 right-6 z-50"
    >
      <div className="bg-white/80 backdrop-blur-md p-1.5 rounded-full shadow-lg border border-gray-200 flex items-center gap-1">
        <button
          onClick={() => setLanguage('bn')}
          className={`px-3 py-1.5 rounded-full text-sm font-bold transition-all ${
            language === 'bn'
              ? 'bg-green-600 text-white shadow-sm'
              : 'text-gray-500 hover:bg-gray-100'
          }`}
        >
          🇧🇩 BN
        </button>
        <button
          onClick={() => setLanguage('en')}
          className={`px-3 py-1.5 rounded-full text-sm font-bold transition-all ${
            language === 'en'
              ? 'bg-blue-600 text-white shadow-sm'
              : 'text-gray-500 hover:bg-gray-100'
          }`}
        >
          🇺🇸 EN
        </button>
      </div>
    </motion.div>
  );
}
