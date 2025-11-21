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
import { motion, AnimatePresence } from "framer-motion";
import { ChevronLeft, ChevronRight } from "lucide-react";

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
  const [direction, setDirection] = useState(0);

  const handleInputChange = (field: string, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }));

    // Auto-calculate derived fields
    if (field === "Weight (kg)" || field === "Height (m)") {
      calculateBMI(field === "Weight (kg)" ? value : formData["Weight (kg)"], field === "Height (m)" ? value : formData["Height (m)"]);
    }
    if (field === "Systolic BP" || field === "Diastolic BP") {
      calculatePulsePressure();
    }
    if (field === "Total Cholesterol (mg/dL)" || field === "HDL (mg/dL)") {
      calculateCholesterolRatio();
    }
  };

  const calculateBMI = (weightStr: string, heightStr: string) => {
    const weight = parseFloat(weightStr);
    const height = parseFloat(heightStr);
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
        title: "About You",
        description: "Let's get to know you! 👋",
        emoji: "👤",
        color: "bg-blue-100 text-blue-600",
      },
      {
        key: "vitals",
        title: "Heart Stats",
        description: "How's your heart beating? 💓",
        emoji: "💓",
        color: "bg-red-100 text-red-600",
      },
      {
        key: "labs",
        title: "Lab Check",
        description: "Sugar and stuff! 🧪",
        emoji: "🧪",
        color: "bg-purple-100 text-purple-600",
      },
      {
        key: "risk",
        title: "Lifestyle",
        description: "Your daily habits! 🏃",
        emoji: "🏃",
        color: "bg-green-100 text-green-600",
      },
      {
        key: "additional",
        title: "Extra Info",
        description: "Just a few more things! 📏",
        emoji: "📏",
        color: "bg-orange-100 text-orange-600",
      },
      {
        key: "review",
        title: "Ready?",
        description: "Let's check everything! ✅",
        emoji: "✅",
        color: "bg-teal-100 text-teal-600",
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
      setDirection(1);
      setCurrentStep((s) => s + 1);
    }
  };

  const handleBack = () => {
    if (currentStep > 0) {
      setDirection(-1);
      setCurrentStep((s) => s - 1);
    }
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

  const variants = {
    enter: (direction: number) => ({
      x: direction > 0 ? 100 : -100,
      opacity: 0,
      scale: 0.8,
    }),
    center: {
      x: 0,
      opacity: 1,
      scale: 1,
    },
    exit: (direction: number) => ({
      x: direction < 0 ? 100 : -100,
      opacity: 0,
      scale: 0.8,
    })
  };

  return (
    <div className="space-y-8">
      {/* Error Display */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.9 }}
            className="bg-red-100 border-2 border-red-200 text-red-700 px-6 py-4 rounded-2xl shadow-sm flex items-center gap-3"
          >
            <span className="text-2xl">🚫</span>
            <div>
              <div className="font-bold">Oops! Something went wrong:</div>
              <div>{error}</div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Fun Stepper Header */}
      <div className="relative px-4">
        <div className="flex justify-between mb-8">
          {steps.map((step, index) => (
            <div
              key={step.key}
              className="flex flex-col items-center relative z-10"
            >
              <motion.div
                initial={false}
                animate={{
                  scale: index === currentStep ? 1.3 : 1,
                  y: index === currentStep ? -5 : 0,
                }}
                className={`w-12 h-12 md:w-14 md:h-14 rounded-full flex items-center justify-center text-2xl md:text-3xl shadow-md transition-all duration-300 cursor-default border-4 ${
                  index <= currentStep
                    ? "border-white bg-gradient-to-br from-yellow-200 to-orange-300"
                    : "border-gray-100 bg-gray-100 grayscale opacity-50"
                }`}
              >
                {step.emoji}
              </motion.div>
              {index === currentStep && (
                <motion.div
                  layoutId="step-label"
                  className="absolute top-16 w-32 text-center"
                >
                  <span className="text-sm font-bold text-gray-800 bg-white px-3 py-1 rounded-full shadow-sm border border-gray-100">
                    {step.title}
                  </span>
                </motion.div>
              )}
            </div>
          ))}
        </div>
        {/* Progress Bar Background */}
        <div className="absolute top-6 md:top-7 left-0 w-full h-3 bg-gray-100 rounded-full -z-0" />
        {/* Active Progress Bar */}
        <motion.div
          className="absolute top-6 md:top-7 left-0 h-3 bg-gradient-to-r from-yellow-400 to-orange-500 rounded-full -z-0"
          initial={{ width: "0%" }}
          animate={{ width: `${(currentStep / (steps.length - 1)) * 100}%` }}
          transition={{ type: "spring", stiffness: 100, damping: 20 }}
        />
      </div>

      {/* Example Data Buttons */}
      <div className="flex flex-wrap gap-3 justify-center md:justify-end">
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          type="button"
          onClick={() => loadExampleData("low_risk")}
          className="bg-green-100 text-green-700 px-4 py-2 rounded-full text-sm font-semibold hover:bg-green-200 transition-colors border border-green-200"
        >
          🌱 Try Healthy Example
        </motion.button>
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          type="button"
          onClick={() => loadExampleData("high_risk")}
          className="bg-red-100 text-red-700 px-4 py-2 rounded-full text-sm font-semibold hover:bg-red-200 transition-colors border border-red-200"
        >
          🍔 Try Risk Example
        </motion.button>
      </div>

      {/* Step Content */}
      <Card className="overflow-hidden border-none shadow-2xl bg-white/90 backdrop-blur-md rounded-3xl">
        <CardHeader className={`${steps[currentStep].color} bg-opacity-20 border-b border-gray-100 p-6`}>
          <CardTitle className="text-2xl md:text-3xl flex items-center gap-3">
            <motion.span
              key={currentStep}
              initial={{ scale: 0, rotate: -180 }}
              animate={{ scale: 1, rotate: 0 }}
              transition={{ type: "spring", stiffness: 200, damping: 15 }}
            >
              {steps[currentStep].emoji}
            </motion.span>
            {steps[currentStep].title}
          </CardTitle>
          <CardDescription className="text-lg opacity-90 font-medium">
            {steps[currentStep].description}
          </CardDescription>
        </CardHeader>
        <CardContent className="p-6 md:p-8 min-h-[400px]">
          <AnimatePresence mode="wait" custom={direction}>
            <motion.div
              key={currentStep}
              custom={direction}
              variants={variants}
              initial="enter"
              animate="center"
              exit="exit"
              transition={{
                x: { type: "spring", stiffness: 300, damping: 30 },
                opacity: { duration: 0.2 },
                scale: { duration: 0.2 },
              }}
              className="h-full"
            >
              {steps[currentStep].key === "demographics" && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <Label htmlFor="sex" className="text-lg">Sex ⚧️</Label>
                    <Select onValueChange={(value) => handleInputChange("Sex", value)} value={formData["Sex"]}>
                      <SelectTrigger className="h-12 text-lg rounded-xl border-2 focus:border-blue-400">
                        <SelectValue placeholder="Select sex" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="0">👩 Female</SelectItem>
                        <SelectItem value="1">👨 Male</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="age" className="text-lg">Age 🎂</Label>
                    <Input
                      id="age"
                      type="number"
                      value={formData["Age"]}
                      onChange={(e) => handleInputChange("Age", e.target.value)}
                      placeholder="e.g., 45"
                      className="h-12 text-lg rounded-xl border-2 focus:border-blue-400"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="weight" className="text-lg">Weight (kg) ⚖️</Label>
                    <Input
                      id="weight"
                      type="number"
                      step="0.1"
                      value={formData["Weight (kg)"]}
                      onChange={(e) => handleInputChange("Weight (kg)", e.target.value)}
                      placeholder="e.g., 70.5"
                      className="h-12 text-lg rounded-xl border-2 focus:border-blue-400"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="height" className="text-lg">Height (m) 📏</Label>
                    <Input
                      id="height"
                      type="number"
                      step="0.01"
                      value={formData["Height (m)"]}
                      onChange={(e) => handleInputChange("Height (m)", e.target.value)}
                      placeholder="e.g., 1.75"
                      className="h-12 text-lg rounded-xl border-2 focus:border-blue-400"
                    />
                  </div>
                  <div className="space-y-2 md:col-span-2">
                    <Label htmlFor="bmi" className="text-lg">BMI (Auto) 🤖</Label>
                    <Input
                      id="bmi"
                      type="number"
                      value={formData["BMI"]}
                      readOnly
                      className="h-12 text-lg rounded-xl border-2 bg-gray-50 font-bold text-blue-600"
                    />
                  </div>
                </div>
              )}

              {steps[currentStep].key === "vitals" && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <Label htmlFor="systolic" className="text-lg">Systolic BP ⬆️</Label>
                    <Input
                      id="systolic"
                      type="number"
                      value={formData["Systolic BP"]}
                      onChange={(e) => handleInputChange("Systolic BP", e.target.value)}
                      placeholder="e.g., 120"
                      className="h-12 text-lg rounded-xl border-2 focus:border-red-400"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="diastolic" className="text-lg">Diastolic BP ⬇️</Label>
                    <Input
                      id="diastolic"
                      type="number"
                      value={formData["Diastolic BP"]}
                      onChange={(e) => handleInputChange("Diastolic BP", e.target.value)}
                      placeholder="e.g., 80"
                      className="h-12 text-lg rounded-xl border-2 focus:border-red-400"
                    />
                  </div>
                  <div className="space-y-2 md:col-span-2">
                    <Label htmlFor="bp-category" className="text-lg">BP Category 📊</Label>
                    <Select
                      onValueChange={(value) =>
                        handleInputChange("Blood Pressure Category", value)
                      }
                      value={formData["Blood Pressure Category"]}
                    >
                      <SelectTrigger className="h-12 text-lg rounded-xl border-2 focus:border-red-400">
                        <SelectValue placeholder="Select category" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="1">💚 Normal</SelectItem>
                        <SelectItem value="2">💛 Elevated</SelectItem>
                        <SelectItem value="3">🧡 Hypertension Stage 1</SelectItem>
                        <SelectItem value="4">❤️ Hypertension Stage 2</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              )}

              {steps[currentStep].key === "labs" && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <Label htmlFor="total-chol" className="text-lg">Total Cholesterol 🩸</Label>
                    <Input
                      id="total-chol"
                      type="number"
                      value={formData["Total Cholesterol (mg/dL)"]}
                      onChange={(e) =>
                        handleInputChange("Total Cholesterol (mg/dL)", e.target.value)
                      }
                      placeholder="e.g., 200"
                      className="h-12 text-lg rounded-xl border-2 focus:border-purple-400"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="hdl" className="text-lg">HDL (Good) 😇</Label>
                    <Input
                      id="hdl"
                      type="number"
                      value={formData["HDL (mg/dL)"]}
                      onChange={(e) => handleInputChange("HDL (mg/dL)", e.target.value)}
                      placeholder="e.g., 50"
                      className="h-12 text-lg rounded-xl border-2 focus:border-purple-400"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="ldl" className="text-lg">LDL (Bad) 😈</Label>
                    <Input
                      id="ldl"
                      type="number"
                      value={formData["Estimated LDL (mg/dL)"]}
                      onChange={(e) =>
                        handleInputChange("Estimated LDL (mg/dL)", e.target.value)
                      }
                      placeholder="e.g., 130"
                      className="h-12 text-lg rounded-xl border-2 focus:border-purple-400"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="glucose" className="text-lg">Blood Sugar 🍬</Label>
                    <Input
                      id="glucose"
                      type="number"
                      value={formData["Fasting Blood Sugar (mg/dL)"]}
                      onChange={(e) =>
                        handleInputChange("Fasting Blood Sugar (mg/dL)", e.target.value)
                      }
                      placeholder="e.g., 100"
                      className="h-12 text-lg rounded-xl border-2 focus:border-purple-400"
                    />
                  </div>
                </div>
              )}

              {steps[currentStep].key === "risk" && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <Label htmlFor="smoking" className="text-lg">Smoking? 🚬</Label>
                    <Select
                      onValueChange={(value) =>
                        handleInputChange("Smoking Status", value)
                      }
                      value={formData["Smoking Status"]}
                    >
                      <SelectTrigger className="h-12 text-lg rounded-xl border-2 focus:border-green-400">
                        <SelectValue placeholder="Select status" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="0">🚭 Non-smoker</SelectItem>
                        <SelectItem value="1">🚬 Smoker</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="diabetes" className="text-lg">Diabetes? 💉</Label>
                    <Select
                      onValueChange={(value) =>
                        handleInputChange("Diabetes Status", value)
                      }
                      value={formData["Diabetes Status"]}
                    >
                      <SelectTrigger className="h-12 text-lg rounded-xl border-2 focus:border-green-400">
                        <SelectValue placeholder="Select status" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="0">❌ No Diabetes</SelectItem>
                        <SelectItem value="1">✅ Has Diabetes</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="family-history" className="text-lg">Family History? 👪</Label>
                    <Select
                      onValueChange={(value) =>
                        handleInputChange("Family History of CVD", value)
                      }
                      value={formData["Family History of CVD"]}
                    >
                      <SelectTrigger className="h-12 text-lg rounded-xl border-2 focus:border-green-400">
                        <SelectValue placeholder="Select history" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="0">❌ No History</SelectItem>
                        <SelectItem value="1">✅ Has History</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="activity" className="text-lg">Activity Level 🏃</Label>
                    <Select
                      onValueChange={(value) =>
                        handleInputChange("Physical Activity Level", value)
                      }
                      value={formData["Physical Activity Level"]}
                    >
                      <SelectTrigger className="h-12 text-lg rounded-xl border-2 focus:border-green-400">
                        <SelectValue placeholder="Select level" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="0">🛋️ Low</SelectItem>
                        <SelectItem value="1">🚶 Moderate</SelectItem>
                        <SelectItem value="2">🏃 High</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              )}

              {steps[currentStep].key === "additional" && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <Label htmlFor="waist" className="text-lg">Waist Size (cm) 👖</Label>
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
                      className="h-12 text-lg rounded-xl border-2 focus:border-orange-400"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="waist-height-ratio" className="text-lg">Waist-Height Ratio ➗</Label>
                    <Input
                      id="waist-height-ratio"
                      type="number"
                      step="0.01"
                      value={formData["Waist-to-Height Ratio"]}
                      onChange={(e) =>
                        handleInputChange("Waist-to-Height Ratio", e.target.value)
                      }
                      placeholder="e.g., 0.5"
                      className="h-12 text-lg rounded-xl border-2 focus:border-orange-400"
                    />
                  </div>
                  <div className="space-y-2 md:col-span-2">
                    <Label htmlFor="cvd-score" className="text-lg">CVD Score 💯</Label>
                    <Input
                      id="cvd-score"
                      type="number"
                      step="0.1"
                      value={formData["CVD Risk Score"]}
                      onChange={(e) =>
                        handleInputChange("CVD Risk Score", e.target.value)
                      }
                      placeholder="e.g., 15.5"
                      className="h-12 text-lg rounded-xl border-2 focus:border-orange-400"
                    />
                  </div>
                </div>
              )}

              {steps[currentStep].key === "review" && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {Object.entries(formData).map(([key, value]) => (
                    <motion.div
                      key={key}
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className="p-4 bg-teal-50 rounded-2xl border border-teal-100"
                    >
                      <div className="text-xs text-teal-600 uppercase tracking-wider font-bold mb-1">{key}</div>
                      <div className="text-lg font-bold text-gray-800">{String(value || "❓")}</div>
                    </motion.div>
                  ))}
                </div>
              )}
            </motion.div>
          </AnimatePresence>
        </CardContent>
      </Card>

      {/* Navigation Controls */}
      <div className="flex items-center justify-between pt-4 px-2">
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          type="button"
          onClick={handleBack}
          disabled={currentStep === 0}
          className={`flex items-center px-6 py-3 rounded-full text-lg font-bold transition-colors ${
            currentStep === 0
              ? "bg-gray-100 text-gray-400 cursor-not-allowed"
              : "bg-white border-2 border-gray-200 text-gray-700 hover:bg-gray-50"
          }`}
        >
          <ChevronLeft className="w-6 h-6 mr-2" />
          Back
        </motion.button>

        {currentStep < steps.length - 1 ? (
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            type="button"
            onClick={handleNext}
            disabled={!isStepValid(currentStep)}
            className={`flex items-center px-8 py-3 rounded-full text-lg font-bold text-white shadow-lg transition-all ${
              !isStepValid(currentStep)
                ? "bg-gray-300 cursor-not-allowed"
                : "bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700"
            }`}
          >
            Next
            <ChevronRight className="w-6 h-6 ml-2" />
          </motion.button>
        ) : (
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            type="submit"
            onClick={handleSubmit}
            disabled={loading || !isStepValid(steps.length - 2)}
            className="flex items-center px-10 py-4 rounded-full text-xl font-bold text-white shadow-xl bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700"
          >
            {loading ? (
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                className="w-6 h-6 border-4 border-white border-t-transparent rounded-full mr-2"
              />
            ) : (
              "🚀 Assess Risk!"
            )}
          </motion.button>
        )}
      </div>
    </div>
  );
}
