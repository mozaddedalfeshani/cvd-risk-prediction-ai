"use client";

import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

interface CVDAssessmentFormProps {
  onSubmit: (data: { model_type?: 'full'; patient_data: Record<string, number|string> }) => void;
  loading: boolean;
  error: string | null;
}

export default function CVDAssessmentForm({
  onSubmit,
  loading,
  error,
}: CVDAssessmentFormProps) {
  const [formData, setFormData] = useState<Record<string, string>>({
    // Demographics
    Sex: "",
    Age: "",
    "Weight (kg)": "",
    "Height (m)": "",
    BMI: "",

    // Vital Signs
    "Systolic BP": "",
    "Diastolic BP": "",
    "Blood Pressure Category": "",

    // Lab Values
    "Total Cholesterol (mg/dL)": "",
    "HDL (mg/dL)": "",
    "Estimated LDL (mg/dL)": "",
    "Fasting Blood Sugar (mg/dL)": "",

    // Risk Factors
    "Smoking Status": "",
    "Diabetes Status": "",
    "Family History of CVD": "",
    "Physical Activity Level": "",

    // Additional Measurements
    "Abdominal Circumference (cm)": "",
    "Waist-to-Height Ratio": "",
    "CVD Risk Score": "",
  });

  const [currentStep, setCurrentStep] = useState<number>(0);

  const handleInputChange = (field: string, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }));

    // Auto-calculate derived fields
    if (field === "Weight (kg)" || field === "Height (m)") {
      calculateBMI();
    }
    if (field === "Systolic BP" || field === "Diastolic BP") {
      calculatePulsePressure();
    }
    if (field === "Total Cholesterol (mg/dL)" || field === "HDL (mg/dL)") {
      calculateCholesterolRatio();
    }
  };

  const calculateBMI = () => {
    const weight = parseFloat(formData["Weight (kg)"]);
    const height = parseFloat(formData["Height (m)"]);
    if (weight && height) {
      const bmi = weight / (height * height);
      setFormData((prev) => ({ ...prev, BMI: bmi.toFixed(1) }));
    }
  };

  const calculatePulsePressure = () => {
    const systolic = parseFloat(formData["Systolic BP"]);
    const diastolic = parseFloat(formData["Diastolic BP"]);
    if (systolic && diastolic) {
      const pulse = systolic - diastolic;
      setFormData((prev) => ({ ...prev, Pulse_Pressure: pulse.toString() }));
    }
  };

  const calculateCholesterolRatio = () => {
    const total = parseFloat(formData["Total Cholesterol (mg/dL)"]);
    const hdl = parseFloat(formData["HDL (mg/dL)"]);
    if (total && hdl) {
      const ratio = total / hdl;
      setFormData((prev) => ({
        ...prev,
        Cholesterol_HDL_Ratio: ratio.toFixed(2),
      }));
    }
  };

  const processFormData = () => {
    const processed: Record<string, number|string> = { ...formData };

    // Convert string values to numbers
    const numericFields = [
      "Age",
      "Weight (kg)",
      "Height (m)",
      "BMI",
      "Systolic BP",
      "Diastolic BP",
      "Total Cholesterol (mg/dL)",
      "HDL (mg/dL)",
      "Estimated LDL (mg/dL)",
      "Fasting Blood Sugar (mg/dL)",
      "Abdominal Circumference (cm)",
      "Waist-to-Height Ratio",
      "CVD Risk Score",
    ];

    numericFields.forEach((field) => {
      const val = processed[field];
      if (val !== undefined && val !== null && val !== "") {
        processed[field] = typeof val === "number" ? val : parseFloat(val as string);
      }
    });

    // Calculate derived fields
    const weight = processed["Weight (kg)"] as number | undefined;
    const height = processed["Height (m)"] as number | undefined;
    const systolic = processed["Systolic BP"] as number | undefined;
    const diastolic = processed["Diastolic BP"] as number | undefined;
    const total_chol = processed["Total Cholesterol (mg/dL)"] as number | undefined;
    const hdl = processed["HDL (mg/dL)"] as number | undefined;
    const ldl = processed["Estimated LDL (mg/dL)"] as number | undefined;
    const age = processed["Age"] as number | undefined;
    const bmi = processed["BMI"] as number | undefined;
    const abdominal = processed["Abdominal Circumference (cm)"] as number | undefined;

    // Auto-calculate derived metrics
    if (weight && height) {
      processed["BMI"] = weight / (height * height);
    }

    if (systolic && diastolic) {
      processed["Pulse_Pressure"] = systolic - diastolic;
    }

    if (total_chol && hdl) {
      processed["Cholesterol_HDL_Ratio"] = total_chol / hdl;
    }

    if (ldl && hdl) {
      processed["LDL_HDL_Ratio"] = ldl / hdl;
    }

    if (abdominal && height) {
      processed["Waist-to-Height Ratio"] = abdominal / (height * 100);
    }

    // Age groups: 1=25-34, 2=35-44, 3=45-54, 4=55-64, 5=65+
    if (age) {
      if (age < 35) processed["Age_Group"] = 1;
      else if (age < 45) processed["Age_Group"] = 2;
      else if (age < 55) processed["Age_Group"] = 3;
      else if (age < 65) processed["Age_Group"] = 4;
      else processed["Age_Group"] = 5;
    }

    // BMI categories: 1=Underweight, 2=Normal, 3=Overweight, 4=Obese
    if (bmi) {
      if (bmi < 18.5) processed["BMI_Category"] = 1;
      else if (bmi < 25) processed["BMI_Category"] = 2;
      else if (bmi < 30) processed["BMI_Category"] = 3;
      else processed["BMI_Category"] = 4;
    }

    // Calculate multiple risk factors
    const riskFactors = [
      processed["Smoking Status"],
      processed["Diabetes Status"],
      processed["Family History of CVD"],
    ].filter((factor) => factor === 1).length;

    processed["Multiple_Risk_Factors"] = riskFactors;

    return processed;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const processedData = processFormData();
    onSubmit({ model_type: 'full', patient_data: processedData });
  };

  // Step configuration for wizard UI
  const steps = useMemo(
    () => [
      {
        key: "demographics",
        title: "Demographics",
        description: "Basic patient information",
      },
      {
        key: "vitals",
        title: "Vital Signs",
        description: "Blood pressure and cardiovascular measurements",
      },
      {
        key: "labs",
        title: "Laboratory Values",
        description: "Cholesterol and blood sugar measurements",
      },
      {
        key: "risk",
        title: "Risk Factors",
        description: "Lifestyle and genetic risk factors",
      },
      {
        key: "additional",
        title: "Additional Measurements",
        description: "Additional clinical measurements",
      },
      {
        key: "review",
        title: "Review & Submit",
        description: "Confirm details and assess risk",
      },
    ],
    []
  );

  // Minimal validation per step (ensure essential fields are present)
  const isStepValid = (stepIndex: number): boolean => {
    switch (steps[stepIndex]?.key) {
      case "demographics":
        return (
          formData["Sex"] !== "" &&
          !!formData["Age"] &&
          !!formData["Weight (kg)"] &&
          !!formData["Height (m)"]
        );
      case "vitals":
        return (
          !!formData["Systolic BP"] &&
          !!formData["Diastolic BP"] &&
          formData["Blood Pressure Category"] !== ""
        );
      case "labs":
        return (
          !!formData["Total Cholesterol (mg/dL)"] &&
          !!formData["HDL (mg/dL)"] &&
          !!formData["Estimated LDL (mg/dL)"] &&
          !!formData["Fasting Blood Sugar (mg/dL)"]
        );
      case "risk":
        return (
          formData["Smoking Status"] !== "" &&
          formData["Diabetes Status"] !== "" &&
          formData["Family History of CVD"] !== "" &&
          formData["Physical Activity Level"] !== ""
        );
      case "additional":
        return (
          !!formData["Abdominal Circumference (cm)"] &&
          !!formData["Waist-to-Height Ratio"] &&
          !!formData["CVD Risk Score"]
        );
      case "review":
        return true;
      default:
        return true;
    }
  };

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      if (!isStepValid(currentStep)) return;
      setCurrentStep((s) => s + 1);
    }
  };

  const handleBack = () => {
    if (currentStep > 0) setCurrentStep((s) => s - 1);
  };

  const loadExampleData = async (type: "low_risk" | "high_risk") => {
    try {
      const response = await fetch(
        `http://localhost:5001/api/example?type=${type}`
      );
      const data = await response.json();

      if (data.data) {
        // Convert the example data to form format
        const exampleFormData: Record<string, string> = {};
        Object.entries(data.data).forEach(([key, value]) => {
          exampleFormData[key] = value?.toString() || "";
        });
        setFormData(exampleFormData);
      }
    } catch (err) {
      console.error("Failed to load example data:", err);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-8">
      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
          <div className="font-semibold">Error:</div>
          <div>{error}</div>
        </div>
      )}

      {/* Stepper Header */}
      <Card className="border-2">
        <CardHeader>
          <CardTitle className="text-xl flex items-center gap-3">
            <span className="text-2xl">🧭</span>
            <span>
              Step {currentStep + 1} of {steps.length}: {steps[currentStep].title}
            </span>
          </CardTitle>
          <CardDescription>{steps[currentStep].description}</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div
              className="h-3 rounded-full bg-blue-600 transition-all duration-500"
              style={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
            />
          </div>
          <div className="flex flex-wrap gap-2 mt-3 text-sm text-gray-600">
            {steps.map((s, i) => (
              <div
                key={s.key}
                className={`px-3 py-1 rounded-full border ${
                  i === currentStep
                    ? "bg-blue-50 border-blue-300 text-blue-700"
                    : i < currentStep
                    ? "bg-green-50 border-green-300 text-green-700"
                    : "bg-gray-50 border-gray-300"
                }`}
              >
                {i + 1}. {s.title}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Example Data Buttons (always accessible) */}
      <div className="flex flex-wrap gap-4 mb-2">
        <Button
          type="button"
          variant="outline"
          onClick={() => loadExampleData("low_risk")}
          className="bg-green-50 border-green-200 hover:bg-green-100"
          size="lg"
        >
          Load Low Risk Example
        </Button>
        <Button
          type="button"
          variant="outline"
          onClick={() => loadExampleData("high_risk")}
          className="bg-red-50 border-red-200 hover:bg-red-100"
          size="lg"
        >
          Load High Risk Example
        </Button>
      </div>

      {/* Step Content */}
      {steps[currentStep].key === "demographics" && (
        <Card>
          <CardHeader>
            <CardTitle className="text-xl">Demographics</CardTitle>
            <CardDescription>Basic patient information</CardDescription>
          </CardHeader>
          <CardContent className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div>
              <Label htmlFor="sex">Sex</Label>
              <Select onValueChange={(value) => handleInputChange("Sex", value)}>
                <SelectTrigger>
                  <SelectValue placeholder="Select sex" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="0">Female</SelectItem>
                  <SelectItem value="1">Male</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label htmlFor="age">Age (years)</Label>
              <Input
                id="age"
                type="number"
                value={formData["Age"]}
                onChange={(e) => handleInputChange("Age", e.target.value)}
                placeholder="e.g., 45"
              />
            </div>
            <div>
              <Label htmlFor="weight">Weight (kg)</Label>
              <Input
                id="weight"
                type="number"
                step="0.1"
                value={formData["Weight (kg)"]}
                onChange={(e) => handleInputChange("Weight (kg)", e.target.value)}
                placeholder="e.g., 70.5"
              />
            </div>
            <div>
              <Label htmlFor="height">Height (m)</Label>
              <Input
                id="height"
                type="number"
                step="0.01"
                value={formData["Height (m)"]}
                onChange={(e) => handleInputChange("Height (m)", e.target.value)}
                placeholder="e.g., 1.75"
              />
            </div>
            <div>
              <Label htmlFor="bmi">BMI (auto-calculated)</Label>
              <Input
                id="bmi"
                type="number"
                step="0.1"
                value={formData["BMI"]}
                onChange={(e) => handleInputChange("BMI", e.target.value)}
                placeholder="Calculated automatically"
                readOnly
              />
            </div>
          </CardContent>
        </Card>
      )}

      {steps[currentStep].key === "vitals" && (
        <Card>
          <CardHeader>
            <CardTitle className="text-xl">Vital Signs</CardTitle>
            <CardDescription>
              Blood pressure and cardiovascular measurements
            </CardDescription>
          </CardHeader>
          <CardContent className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div>
              <Label htmlFor="systolic">Systolic BP (mmHg)</Label>
              <Input
                id="systolic"
                type="number"
                value={formData["Systolic BP"]}
                onChange={(e) => handleInputChange("Systolic BP", e.target.value)}
                placeholder="e.g., 120"
              />
            </div>
            <div>
              <Label htmlFor="diastolic">Diastolic BP (mmHg)</Label>
              <Input
                id="diastolic"
                type="number"
                value={formData["Diastolic BP"]}
                onChange={(e) =>
                  handleInputChange("Diastolic BP", e.target.value)
                }
                placeholder="e.g., 80"
              />
            </div>
            <div>
              <Label htmlFor="bp-category">Blood Pressure Category</Label>
              <Select
                onValueChange={(value) =>
                  handleInputChange("Blood Pressure Category", value)
                }
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select category" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1">Normal</SelectItem>
                  <SelectItem value="2">Elevated</SelectItem>
                  <SelectItem value="3">Hypertension Stage 1</SelectItem>
                  <SelectItem value="4">Hypertension Stage 2</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardContent>
        </Card>
      )}

      {steps[currentStep].key === "labs" && (
        <Card>
          <CardHeader>
            <CardTitle className="text-xl">Laboratory Values</CardTitle>
            <CardDescription>
              Cholesterol and blood sugar measurements
            </CardDescription>
          </CardHeader>
          <CardContent className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div>
              <Label htmlFor="total-chol">Total Cholesterol (mg/dL)</Label>
              <Input
                id="total-chol"
                type="number"
                value={formData["Total Cholesterol (mg/dL)"]}
                onChange={(e) =>
                  handleInputChange("Total Cholesterol (mg/dL)", e.target.value)
                }
                placeholder="e.g., 200"
              />
            </div>
            <div>
              <Label htmlFor="hdl">HDL Cholesterol (mg/dL)</Label>
              <Input
                id="hdl"
                type="number"
                value={formData["HDL (mg/dL)"]}
                onChange={(e) => handleInputChange("HDL (mg/dL)", e.target.value)}
                placeholder="e.g., 50"
              />
            </div>
            <div>
              <Label htmlFor="ldl">LDL Cholesterol (mg/dL)</Label>
              <Input
                id="ldl"
                type="number"
                value={formData["Estimated LDL (mg/dL)"]}
                onChange={(e) =>
                  handleInputChange("Estimated LDL (mg/dL)", e.target.value)
                }
                placeholder="e.g., 130"
              />
            </div>
            <div>
              <Label htmlFor="glucose">Fasting Blood Sugar (mg/dL)</Label>
              <Input
                id="glucose"
                type="number"
                value={formData["Fasting Blood Sugar (mg/dL)"]}
                onChange={(e) =>
                  handleInputChange("Fasting Blood Sugar (mg/dL)", e.target.value)
                }
                placeholder="e.g., 100"
              />
            </div>
          </CardContent>
        </Card>
      )}

      {steps[currentStep].key === "risk" && (
        <Card>
          <CardHeader>
            <CardTitle className="text-xl">Risk Factors</CardTitle>
            <CardDescription>Lifestyle and genetic risk factors</CardDescription>
          </CardHeader>
          <CardContent className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div>
              <Label htmlFor="smoking">Smoking Status</Label>
              <Select
                onValueChange={(value) =>
                  handleInputChange("Smoking Status", value)
                }
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select status" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="0">Non-smoker</SelectItem>
                  <SelectItem value="1">Smoker</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label htmlFor="diabetes">Diabetes Status</Label>
              <Select
                onValueChange={(value) =>
                  handleInputChange("Diabetes Status", value)
                }
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select status" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="0">No Diabetes</SelectItem>
                  <SelectItem value="1">Has Diabetes</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label htmlFor="family-history">Family History of CVD</Label>
              <Select
                onValueChange={(value) =>
                  handleInputChange("Family History of CVD", value)
                }
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select history" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="0">No Family History</SelectItem>
                  <SelectItem value="1">Has Family History</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label htmlFor="activity">Physical Activity Level</Label>
              <Select
                onValueChange={(value) =>
                  handleInputChange("Physical Activity Level", value)
                }
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select level" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="0">Low</SelectItem>
                  <SelectItem value="1">Moderate</SelectItem>
                  <SelectItem value="2">High</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardContent>
        </Card>
      )}

      {steps[currentStep].key === "additional" && (
        <Card>
          <CardHeader>
            <CardTitle className="text-xl">Additional Measurements</CardTitle>
            <CardDescription>Additional clinical measurements</CardDescription>
          </CardHeader>
          <CardContent className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div>
              <Label htmlFor="waist">Abdominal Circumference (cm)</Label>
              <Input
                id="waist"
                type="number"
                step="0.1"
                value={formData["Abdominal Circumference (cm)"]}
                onChange={(e) =>
                  handleInputChange(
                    "Abdominal Circumference (cm)",
                    e.target.value
                  )
                }
                placeholder="e.g., 90"
              />
            </div>
            <div>
              <Label htmlFor="waist-height-ratio">Waist-to-Height Ratio</Label>
              <Input
                id="waist-height-ratio"
                type="number"
                step="0.01"
                value={formData["Waist-to-Height Ratio"]}
                onChange={(e) =>
                  handleInputChange("Waist-to-Height Ratio", e.target.value)
                }
                placeholder="e.g., 0.5"
              />
            </div>
            <div>
              <Label htmlFor="cvd-score">CVD Risk Score</Label>
              <Input
                id="cvd-score"
                type="number"
                step="0.1"
                value={formData["CVD Risk Score"]}
                onChange={(e) =>
                  handleInputChange("CVD Risk Score", e.target.value)
                }
                placeholder="e.g., 15.5"
              />
            </div>
          </CardContent>
        </Card>
      )}

      {steps[currentStep].key === "review" && (
        <Card>
          <CardHeader>
            <CardTitle className="text-xl">Review & Submit</CardTitle>
            <CardDescription>
              Please review the entered details before assessing risk.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.entries(formData).map(([key, value]) => (
                <div key={key} className="p-3 bg-gray-50 rounded-md">
                  <div className="text-xs text-gray-500">{key}</div>
                  <div className="text-base font-medium">{String(value || "-")}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Navigation Controls */}
      <div className="flex items-center justify-between pt-2">
        <Button
          type="button"
          variant="outline"
          onClick={handleBack}
          disabled={currentStep === 0}
          size="lg"
        >
          ← Previous
        </Button>

        {currentStep < steps.length - 1 ? (
          <Button
            type="button"
            onClick={handleNext}
            disabled={!isStepValid(currentStep)}
            size="lg"
            className="bg-blue-600 hover:bg-blue-700 text-white"
          >
            Next →
          </Button>
        ) : (
          <Button
            type="submit"
            disabled={loading || !isStepValid(steps.length - 2)}
            className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 text-lg"
            size="lg"
          >
            {loading ? "Analyzing..." : "Assess CVD Risk"}
          </Button>
        )}
      </div>
    </form>
  );
}
