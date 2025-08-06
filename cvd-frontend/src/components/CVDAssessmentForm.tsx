"use client";

import { useState } from "react";
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
  onSubmit: (data: any) => void;
  loading: boolean;
  error: string | null;
}

export default function CVDAssessmentForm({
  onSubmit,
  loading,
  error,
}: CVDAssessmentFormProps) {
  const [formData, setFormData] = useState({
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
    const processed = { ...formData };

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
      if (processed[field]) {
        processed[field] = parseFloat(processed[field]);
      }
    });

    // Calculate derived fields
    const weight = processed["Weight (kg)"];
    const height = processed["Height (m)"];
    const systolic = processed["Systolic BP"];
    const diastolic = processed["Diastolic BP"];
    const total_chol = processed["Total Cholesterol (mg/dL)"];
    const hdl = processed["HDL (mg/dL)"];
    const ldl = processed["Estimated LDL (mg/dL)"];
    const age = processed["Age"];
    const bmi = processed["BMI"];
    const abdominal = processed["Abdominal Circumference (cm)"];

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
    onSubmit(processedData);
  };

  const loadExampleData = async (type: "low_risk" | "high_risk") => {
    try {
      const response = await fetch(
        `http://localhost:5001/api/example?type=${type}`
      );
      const data = await response.json();

      if (data.data) {
        // Convert the example data to form format
        const exampleFormData: any = {};
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

      {/* Example Data Buttons */}
      <div className="flex gap-4 mb-6">
        <Button
          type="button"
          variant="outline"
          onClick={() => loadExampleData("low_risk")}
          className="bg-green-50 border-green-200 hover:bg-green-100">
          Load Low Risk Example
        </Button>
        <Button
          type="button"
          variant="outline"
          onClick={() => loadExampleData("high_risk")}
          className="bg-red-50 border-red-200 hover:bg-red-100">
          Load High Risk Example
        </Button>
      </div>

      {/* Demographics Section */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Demographics</CardTitle>
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
              required
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
              required
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
              required
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

      {/* Vital Signs Section */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Vital Signs</CardTitle>
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
              required
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
              required
            />
          </div>
          <div>
            <Label htmlFor="bp-category">Blood Pressure Category</Label>
            <Select
              onValueChange={(value) =>
                handleInputChange("Blood Pressure Category", value)
              }>
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

      {/* Lab Values Section */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Laboratory Values</CardTitle>
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
              required
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
              required
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
              required
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
              required
            />
          </div>
        </CardContent>
      </Card>

      {/* Risk Factors Section */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Risk Factors</CardTitle>
          <CardDescription>Lifestyle and genetic risk factors</CardDescription>
        </CardHeader>
        <CardContent className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div>
            <Label htmlFor="smoking">Smoking Status</Label>
            <Select
              onValueChange={(value) =>
                handleInputChange("Smoking Status", value)
              }>
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
              }>
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
              }>
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
              }>
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

      {/* Additional Measurements Section */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Additional Measurements</CardTitle>
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
              required
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
              required
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
              required
            />
          </div>
        </CardContent>
      </Card>

      {/* Submit Button */}
      <div className="flex justify-center pt-6">
        <Button
          type="submit"
          disabled={loading}
          className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 text-lg"
          size="lg">
          {loading ? "Analyzing..." : "Assess CVD Risk"}
        </Button>
      </div>
    </form>
  );
}
