"use client";

import { motion } from "framer-motion";
import { Activity, Shield, Zap, Brain } from "lucide-react";
import { useLanguage } from "@/components/providers/LanguageProvider";

export default function FeaturesSection() {
  const { t } = useLanguage();

  const features = [
    {
      icon: <Brain className="w-8 h-8 text-purple-600" />,
      title: t.features.dual_ai,
      description: t.features.dual_ai_desc,
      color: "bg-purple-100",
    },
    {
      icon: <Zap className="w-8 h-8 text-yellow-600" />,
      title: t.features.instant,
      description: t.features.instant_desc,
      color: "bg-yellow-100",
    },
    {
      icon: <Shield className="w-8 h-8 text-green-600" />,
      title: t.features.privacy,
      description: t.features.privacy_desc,
      color: "bg-green-100",
    },
    {
      icon: <Activity className="w-8 h-8 text-red-600" />,
      title: t.features.insights,
      description: t.features.insights_desc,
      color: "bg-red-100",
    },
  ];

  return (
    <section className="py-20 bg-white/50 backdrop-blur-sm">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
            {t.features.title}
          </h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            {t.features.subtitle}
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              className="p-6 bg-white rounded-2xl shadow-lg hover:shadow-xl transition-shadow border border-gray-100"
            >
              <div className={`w-16 h-16 ${feature.color} rounded-2xl flex items-center justify-center mb-6`}>
                {feature.icon}
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-3">{feature.title}</h3>
              <p className="text-gray-600 leading-relaxed">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
