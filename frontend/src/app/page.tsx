"use client";

import { useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import CVDAssessmentFormDual from "@/components/CVDAssessmentFormDual";
import PredictionResult from "@/components/PredictionResult";
import { motion } from "framer-motion";
import { Activity, ShieldCheck } from "lucide-react";

interface PredictionResult {
  model_used: {
    type: string;
    name: string;
    accuracy: number;
    features_used: number;
  };
  prediction: {
    risk_level: string;
    risk_code: number;
    confidence: number;
    probabilities: {
      LOW: number;
      INTERMEDIARY: number;
      HIGH: number;
    };
  };
  clinical_interpretation: {
    risk_category: string;
    confidence_level: string;
    recommendations: {
      recommendation: string;
      follow_up: string;
      lifestyle: string;
    };
  };
}

export default function Home() {
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  type PredictPayload = { model_type: 'full'|'quick'; patient_data: Record<string, number|string> };

  const handlePrediction = async (patientData: PredictPayload) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch("http://localhost:5001/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(patientData),
      });

      const data = await response.json();

      if (data.success) {
        setPrediction(data.result);
      } else {
        setError(data.error || "Prediction failed");
      }
    } catch {
      setError(
        "Failed to connect to the backend. Please ensure the Flask server is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setPrediction(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50 p-4 md:p-8 font-sans">
      <div className="max-w-5xl mx-auto">
        {/* Fun Header */}
        <motion.div
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ type: "spring", stiffness: 100 }}
          className="text-center mb-12"
        >
          <motion.div
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ repeat: Infinity, duration: 1.5 }}
            className="text-6xl mb-4 inline-block cursor-default"
          >
            ❤️
          </motion.div>
          <h1 className="text-4xl md:text-6xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-pink-600 mb-4 tracking-tight drop-shadow-sm">
            Check Your Heart Health
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto font-medium">
            A super smart AI that helps you understand your heart risks! 🤖✨
          </p>
          
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.5 }}
            className="inline-flex items-center gap-2 bg-white border-2 border-purple-200 text-purple-700 px-6 py-2 rounded-full mt-6 shadow-lg"
          >
            <span className="text-2xl">🛡️</span>
            <span className="font-bold">Dual AI Power</span>
            <span className="w-2 h-2 bg-purple-400 rounded-full mx-2" />
            <span className="font-bold">95.9% Accurate!</span>
          </motion.div>
        </motion.div>

        {/* Main Content */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, type: "spring" }}
        >
          {!prediction ? (
            <Card className="border-none shadow-2xl bg-white/60 backdrop-blur-xl rounded-[2.5rem] overflow-hidden ring-4 ring-white/50">
              <CardHeader className="border-b border-gray-100 bg-white/50 p-8 text-center">
                <CardTitle className="flex items-center justify-center gap-3 text-3xl font-bold text-gray-800">
                  <span className="text-4xl">📋</span>
                  Start Your Assessment
                </CardTitle>
                <CardDescription className="text-lg mt-2 font-medium text-gray-500">
                  Fill out the simple form below to get your instant heart health report!
                </CardDescription>
              </CardHeader>
              <CardContent className="p-6 md:p-10">
                <CVDAssessmentFormDual
                  onSubmit={handlePrediction}
                  loading={loading}
                  error={error}
                />
              </CardContent>
            </Card>
          ) : (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="space-y-8"
            >
              <PredictionResult prediction={prediction} />
              <div className="text-center">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={resetForm}
                  className="px-10 py-4 text-xl font-bold text-white bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full shadow-xl hover:shadow-2xl transition-all"
                >
                  🔄 Check Another Patient
                </motion.button>
              </div>
            </motion.div>
          )}
        </motion.div>

        {/* Footer */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
          className="mt-20 text-center text-gray-400 text-sm font-medium"
        >
          <p>
            Made with 💖 for a healthier world. (Remember: This is an AI tool, always see a doctor! 👨‍⚕️)
          </p>
        </motion.div>
      </div>
    </div>
  );
}
