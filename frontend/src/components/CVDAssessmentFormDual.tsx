"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
// Select UI not used in current UX
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

interface CVDAssessmentFormDualProps {
  onSubmit: (data: { model_type: 'full'|'quick'; patient_data: Record<string, number|string> }) => void;
  loading: boolean;
  error: string | null;
}

interface ModelOption {
  id: string;
  name: string;
  description: string;
  accuracy: string;
  features: number;
  time_required: string;
  recommended_for: string;
}

export default function CVDAssessmentFormDual({
  onSubmit,
  loading,
  error,
}: CVDAssessmentFormDualProps) {
  const DERIVED_FIELDS = useState<string[]>([
    'Age_Group',
    'BMI_Category',
    'Pulse_Pressure',
    'Cholesterol_HDL_Ratio',
    'LDL_HDL_Ratio',
    'Multiple_Risk_Factors',
    'Waist-to-Height Ratio',
  ])[0];
  const [selectedModel, setSelectedModel] = useState<string>('full');
  const [availableModels, setAvailableModels] = useState<ModelOption[]>([]);
  const [requiredFeatures, setRequiredFeatures] = useState<string[]>([]);
  const [allServerFeatures, setAllServerFeatures] = useState<string[]>([]);
  const [featureCategories, setFeatureCategories] = useState<Record<string, string[]>>({});
  const [formData, setFormData] = useState<Record<string, string>>({});

  // Load available models on component mount
  useEffect(() => {
    fetchAvailableModels();
  }, []);

  // Load required features when model selection changes
  useEffect(() => {
    if (selectedModel) {
      fetchModelFeatures(selectedModel);
    }
  }, [selectedModel]);

  const fetchAvailableModels = async () => {
    try {
      const response = await fetch(`http://localhost:5001/api/models`);
      const data = await response.json();
      setAvailableModels(data.available_models || []);
      if (data.default_model) {
        setSelectedModel(data.default_model);
      }
    } catch (error) {
      console.error('Error fetching models:', error);
    }
  };

  const fetchModelFeatures = async (modelType: string) => {
    try {
      const response = await fetch(`http://localhost:5001/api/features/${modelType}`);
      const data = await response.json();
      const serverFeatures: string[] = data.required_features || [];
      setAllServerFeatures(serverFeatures);
      // Hide derived fields from the UI; they will be auto-computed on submit
      const displayFeatures = serverFeatures.filter((f: string) => !DERIVED_FIELDS.includes(f));
      setRequiredFeatures(displayFeatures);
      setFeatureCategories(data.categories || {});
      
      // Reset form data when switching models
      const newFormData: Record<string, string> = {};
      displayFeatures.forEach((feature: string) => {
        newFormData[feature] = '';
      });
      setFormData(newFormData);
    } catch (error) {
      console.error('Error fetching features:', error);
    }
  };

  const loadExampleData = async (riskType: string) => {
    try {
      const response = await fetch(
        `http://localhost:5001/api/example/${selectedModel}?risk=${riskType}`
      );
      const data = await response.json();
      if (data.example_data) {
        const newFormData: Record<string, string> = {};
        Object.entries(data.example_data).forEach(([key, value]) => {
          newFormData[key] = String(value);
        });
        setFormData(newFormData);
      }
    } catch (error) {
      console.error('Error loading example:', error);
    }
  };

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    // Validate required fields
    const missingFields = requiredFeatures.filter(field => !formData[field]);
    if (missingFields.length > 0) {
      alert(`Please fill in all required fields: ${missingFields.join(', ')}`);
      return;
    }

    // Convert string values to appropriate types
    const processedData: Record<string, number|string> = {};
    Object.entries(formData).forEach(([key, value]) => {
      if (value.trim()) {
        processedData[key] = isNaN(Number(value)) ? value : Number(value);
      }
    });

    // Auto-compute derived features expected by the model
    const getNum = (k: string) => (processedData[k] as number | undefined);

    // Age_Group
    if (allServerFeatures.includes('Age_Group') && processedData['Age'] !== undefined) {
      const age = Number(processedData['Age']);
      processedData['Age_Group'] = age < 35 ? 1 : age < 45 ? 2 : age < 55 ? 3 : age < 65 ? 4 : 5;
    }
    // BMI_Category (assumes BMI provided)
    if (allServerFeatures.includes('BMI_Category') && processedData['BMI'] !== undefined) {
      const bmi = Number(processedData['BMI']);
      processedData['BMI_Category'] = bmi < 18.5 ? 1 : bmi < 25 ? 2 : bmi < 30 ? 3 : 4;
    }
    // Pulse_Pressure
    if (allServerFeatures.includes('Pulse_Pressure') && getNum('Systolic BP') && getNum('Diastolic BP')) {
      processedData['Pulse_Pressure'] = Number(getNum('Systolic BP')) - Number(getNum('Diastolic BP'));
    }
    // Cholesterol_HDL_Ratio
    if (allServerFeatures.includes('Cholesterol_HDL_Ratio') && getNum('Total Cholesterol (mg/dL)') && getNum('HDL (mg/dL)')) {
      processedData['Cholesterol_HDL_Ratio'] = Number(getNum('Total Cholesterol (mg/dL)')) / Number(getNum('HDL (mg/dL)'));
    }
    // LDL_HDL_Ratio
    if (allServerFeatures.includes('LDL_HDL_Ratio') && getNum('Estimated LDL (mg/dL)') && getNum('HDL (mg/dL)')) {
      processedData['LDL_HDL_Ratio'] = Number(getNum('Estimated LDL (mg/dL)')) / Number(getNum('HDL (mg/dL)'));
    }
    // Multiple_Risk_Factors
    if (allServerFeatures.includes('Multiple_Risk_Factors')) {
      const riskFlags = ['Smoking Status', 'Diabetes Status', 'Family History of CVD']
        .map(k => Number(processedData[k] || 0))
        .filter(v => v === 1).length;
      processedData['Multiple_Risk_Factors'] = riskFlags;
    }
    // Waist-to-Height Ratio
    if (allServerFeatures.includes('Waist-to-Height Ratio') && getNum('Abdominal Circumference (cm)') && getNum('Height (m)')) {
      processedData['Waist-to-Height Ratio'] = Number(getNum('Abdominal Circumference (cm)')) / (Number(getNum('Height (m)')) * 100);
    }

    onSubmit({
      model_type: selectedModel as 'full'|'quick',
      patient_data: processedData
    });
  };

  const renderInputField = (field: string) => {
    return (
      <div key={field} className="space-y-2">
        <Label htmlFor={field} className="text-sm font-medium">
          {field}
        </Label>
        <Input
          id={field}
          type={field.includes('Sex') || field.includes('Status') || field.includes('Category') || field.includes('Level') ? 'number' : 'number'}
          step={field.includes('Height') || field.includes('Ratio') ? '0.01' : '1'}
          value={formData[field] || ''}
          onChange={(e) => handleInputChange(field, e.target.value)}
          placeholder={`Enter ${field.toLowerCase()}`}
          className="w-full"
        />
      </div>
    );
  };

  const selectedModelInfo = availableModels.find(m => m.id === selectedModel);

  return (
    <div className="space-y-6">
      {/* Model Selection */}
      <Card>
        <CardHeader>
          <CardTitle>Choose Assessment Type</CardTitle>
          <CardDescription>
            Select the model that best fits your needs
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {availableModels.map((model) => (
              <div
                key={model.id}
                className={`p-4 border rounded-lg cursor-pointer transition-all ${
                  selectedModel === model.id
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => setSelectedModel(model.id)}
              >
                <div className="flex items-center space-x-2 mb-2">
                  <input
                    type="radio"
                    checked={selectedModel === model.id}
                    onChange={() => setSelectedModel(model.id)}
                    className="text-blue-600"
                  />
                  <h3 className="font-semibold">{model.name}</h3>
                </div>
                <p className="text-sm text-gray-600 mb-2">{model.description}</p>
                <div className="text-sm space-y-1">
                  <div className="flex justify-between">
                    <span>Accuracy:</span>
                    <span className="font-medium text-green-600">{model.accuracy}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Features:</span>
                    <span className="font-medium">{model.features}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Time:</span>
                    <span className="font-medium">{model.time_required}</span>
                  </div>
                </div>
                <p className="text-xs text-gray-500 mt-2">{model.recommended_for}</p>
              </div>
            ))}
          </div>

          {selectedModelInfo && (
            <div className="p-3 bg-gray-50 rounded-lg">
              <p className="text-sm">
                <strong>Selected:</strong> {selectedModelInfo.name} - {selectedModelInfo.accuracy} accuracy
                with {selectedModelInfo.features} features
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Example Data Buttons */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Start</CardTitle>
          <CardDescription>
            Load example data to test the selected model
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2">
            <Button
              type="button"
              variant="outline"
              onClick={() => loadExampleData('low')}
              className="flex-1"
            >
              Load Low Risk Example
            </Button>
            <Button
              type="button"
              variant="outline"
              onClick={() => loadExampleData('high')}
              className="flex-1"
            >
              Load High Risk Example
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Assessment Form */}
      <Card>
        <CardHeader>
          <CardTitle>
            {selectedModelInfo?.name || 'CVD Risk Assessment'}
          </CardTitle>
          <CardDescription>
            Please fill in all required fields for {selectedModelInfo?.name.toLowerCase() || 'assessment'}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">
            {Object.entries(featureCategories).map(([category, fields]) => (
              <div key={category}>
                <h3 className="text-lg font-semibold mb-3 text-gray-800 border-b pb-1">
                  {category}
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {(fields as string[])
                    .filter((f) => !DERIVED_FIELDS.includes(f))
                    .map(renderInputField)}
                </div>
              </div>
            ))}

            {error && (
              <div className="p-3 bg-red-50 border border-red-200 rounded-md">
                <p className="text-red-700 text-sm">{error}</p>
              </div>
            )}

            <div className="flex space-x-4 pt-4">
              <Button
                type="submit"
                disabled={loading || requiredFeatures.length === 0}
                className="flex-1"
              >
                {loading ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                    Analyzing...
                  </>
                ) : (
                  `Predict Risk (${selectedModelInfo?.accuracy || 'N/A'})`
                )}
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
