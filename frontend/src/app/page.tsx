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
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            CVD Risk Assessment
          </h1>
          <p className="text-lg text-gray-600">
            AI-Powered Cardiovascular Disease Risk Prediction
          </p>
          <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-2 rounded-lg mt-4 inline-block">
            <span className="font-semibold">Dual Model System</span> •
            Full Accuracy (95.91%) & Quick Assessment (86.79%)
          </div>
        </div>

        {/* Main Content */}
        {!prediction ? (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <span className="bg-blue-100 p-2 rounded-lg">🏥</span>
                Patient Assessment Form
              </CardTitle>
              <CardDescription>
                Please fill in the patient information below to assess
                cardiovascular disease risk. All fields are required for
                accurate prediction.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <CVDAssessmentFormDual
                onSubmit={handlePrediction}
                loading={loading}
                error={error}
              />
            </CardContent>
          </Card>
        ) : (
          <div className="space-y-6">
            <PredictionResult prediction={prediction} />
            <div className="text-center">
              <Button onClick={resetForm} variant="outline" size="lg">
                Assess Another Patient
              </Button>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="mt-12 text-center text-gray-500 text-sm">
          <p>
            This tool is for clinical decision support only. Always consult with
            healthcare professionals for medical decisions.
          </p>
        </div>
      </div>
    </div>
  );
}
