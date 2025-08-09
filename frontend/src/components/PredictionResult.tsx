'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

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

interface PredictionResultProps {
  prediction: PredictionResult;
}

export default function PredictionResult({ prediction }: PredictionResultProps) {
  const { model_used, prediction: predictionResult, clinical_interpretation } = prediction;
  
  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'LOW':
        return 'text-green-700 bg-green-100 border-green-300';
      case 'INTERMEDIARY':
        return 'text-yellow-700 bg-yellow-100 border-yellow-300';
      case 'HIGH':
        return 'text-red-700 bg-red-100 border-red-300';
      default:
        return 'text-gray-700 bg-gray-100 border-gray-300';
    }
  };

  const getRiskIcon = (riskLevel: string) => {
    switch (riskLevel) {
      case 'LOW':
        return '✅';
      case 'INTERMEDIARY':
        return '⚠️';
      case 'HIGH':
        return '🚨';
      default:
        return '❓';
    }
  };

  const getRiskDescription = (riskLevel: string) => {
    switch (riskLevel) {
      case 'LOW':
        return 'The patient has a low risk of developing cardiovascular disease. Continue with regular preventive care and healthy lifestyle habits.';
      case 'INTERMEDIARY':
        return 'The patient has an intermediate risk of developing cardiovascular disease. Consider enhanced monitoring and lifestyle modifications.';
      case 'HIGH':
        return 'The patient has a high risk of developing cardiovascular disease. Immediate medical attention and intervention strategies are recommended.';
      default:
        return 'Risk level assessment complete.';
    }
  };

  const getRecommendations = (riskLevel: string) => {
    switch (riskLevel) {
      case 'LOW':
        return [
          'Continue regular physical activity',
          'Maintain healthy diet',
          'Regular health check-ups',
          'Monitor blood pressure annually',
          'Avoid smoking and excessive alcohol'
        ];
      case 'INTERMEDIARY':
        return [
          'Increase physical activity to 150+ minutes/week',
          'Follow heart-healthy diet (DASH or Mediterranean)',
          'Monitor blood pressure more frequently',
          'Consider cholesterol management',
          'Stress reduction techniques',
          'Weight management if applicable'
        ];
      case 'HIGH':
        return [
          'Immediate consultation with cardiologist',
          'Consider medication for blood pressure/cholesterol',
          'Structured exercise program under supervision',
          'Smoking cessation if applicable',
          'Diabetes management if applicable',
          'Regular cardiac monitoring',
          'Emergency action plan discussion'
        ];
      default:
        return [];
    }
  };

  return (
    <div className="space-y-6">
      {/* Main Result Card */}
      <Card className={`border-2 ${getRiskColor(predictionResult.risk_level)}`}>
        <CardHeader>
          <CardTitle className="flex items-center gap-3 text-2xl">
            <span className="text-3xl">{getRiskIcon(predictionResult.risk_level)}</span>
            <div>
              <div className={`text-xl font-bold ${getRiskColor(predictionResult.risk_level)}`}>
                {predictionResult.risk_level} RISK
              </div>
              <div className="text-sm font-normal text-gray-600">
                Confidence: {(predictionResult.confidence * 100).toFixed(1)}%
              </div>
            </div>
          </CardTitle>
          <CardDescription className="text-base">
            {getRiskDescription(predictionResult.risk_level)}
          </CardDescription>
        </CardHeader>
      </Card>

      {/* Probability Breakdown */}
      <Card>
        <CardHeader>
          <CardTitle>Risk Probability Breakdown</CardTitle>
          <CardDescription>
            Detailed probability distribution across all risk levels
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {Object.entries(predictionResult.probabilities).map(([level, probability]) => (
              <div key={level} className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className={`font-medium ${level === predictionResult.risk_level ? 'font-bold' : ''}`}>
                    {getRiskIcon(level)} {level} Risk
                  </span>
                  <span className={`${level === predictionResult.risk_level ? 'font-bold' : ''}`}>
                    {(probability * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div
                    className={`h-3 rounded-full transition-all duration-500 ${
                      level === 'LOW' ? 'bg-green-500' :
                      level === 'INTERMEDIARY' ? 'bg-yellow-500' : 'bg-red-500'
                    } ${level === predictionResult.risk_level ? 'ring-2 ring-offset-1 ring-gray-400' : ''}`}
                    style={{ width: `${probability * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Model Information */}
      <Card>
        <CardHeader>
          <CardTitle>Model Information</CardTitle>
          <CardDescription>
            Details about the prediction model used
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-sm text-gray-600">Model Used</div>
              <div className="font-semibold">{model_used.name}</div>
            </div>
            <div>
              <div className="text-sm text-gray-600">Accuracy</div>
              <div className="font-semibold text-green-600">{(model_used.accuracy * 100).toFixed(2)}%</div>
            </div>
            <div>
              <div className="text-sm text-gray-600">Features Analyzed</div>
              <div className="font-semibold">{model_used.features_used}</div>
            </div>
            <div>
              <div className="text-sm text-gray-600">Model Type</div>
              <div className="font-semibold">{model_used.type === 'full' ? 'Comprehensive' : 'Quick Assessment'}</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Recommendations */}
      <Card>
        <CardHeader>
          <CardTitle>Clinical Recommendations</CardTitle>
          <CardDescription>
            Suggested actions based on the risk assessment
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {getRecommendations(predictionResult.risk_level).map((recommendation, index) => (
              <div key={index} className="flex items-start gap-3">
                <span className="text-blue-500 font-bold mt-1">•</span>
                <span>{recommendation}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Model Information */}
      <Card>
        <CardHeader>
          <CardTitle>Model Information</CardTitle>
          <CardDescription>
            Details about the AI model used for this assessment
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
            <div className="bg-blue-50 p-4 rounded-lg">
              <div className="text-2xl font-bold text-blue-600">
                {(model_used.accuracy * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600">Model Accuracy</div>
            </div>
            <div className="bg-green-50 p-4 rounded-lg">
              <div className="text-2xl font-bold text-green-600">XGBoost</div>
              <div className="text-sm text-gray-600">Algorithm Used</div>
            </div>
            <div className="bg-purple-50 p-4 rounded-lg">
              <div className="text-2xl font-bold text-purple-600">Clinical</div>
              <div className="text-sm text-gray-600">Grade Model</div>
            </div>
          </div>
          
          <div className="mt-4 p-4 bg-gray-50 rounded-lg">
            <div className="text-sm text-gray-600">
              <strong>Disclaimer:</strong> This AI-powered assessment is intended for clinical decision support only. 
              It should not replace professional medical judgment. Always consult with qualified healthcare 
              professionals for medical decisions and treatment plans.
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}