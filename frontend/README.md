# 📱 CVD Risk Assessment Frontend

## 🎯 Overview

Modern React/Next.js web application providing an intuitive interface for healthcare professionals to assess cardiovascular disease risk using AI-powered predictions.

## ✨ Features

### 🏥 **Clinical Interface**

- Professional healthcare-focused design
- Comprehensive patient assessment form
- Real-time risk calculation and visualization
- Example patient data for testing

### 🎨 **Modern UI/UX**

- Responsive design for all devices
- Shadcn/UI component library
- Tailwind CSS for styling
- Accessibility-compliant interface

### 🔒 **Production Ready**

- TypeScript for type safety
- ESLint for code quality
- Performance optimized
- SEO optimized

## 📁 Structure

```
frontend/
├── src/
│   ├── app/                    # Next.js app directory
│   │   ├── page.tsx           # Main assessment page
│   │   ├── layout.tsx         # Root layout
│   │   └── globals.css        # Global styles
│   ├── components/
│   │   ├── CVDAssessmentForm.tsx  # Main form component
│   │   ├── PredictionResult.tsx   # Results display
│   │   └── ui/                # Reusable UI components
│   │       ├── button.tsx
│   │       ├── card.tsx
│   │       ├── form.tsx
│   │       ├── input.tsx
│   │       └── select.tsx
│   └── lib/
│       └── utils.ts           # Utility functions
├── public/                    # Static assets
├── package.json              # Dependencies
├── next.config.ts            # Next.js configuration
├── tailwind.config.ts        # Tailwind configuration
├── tsconfig.json             # TypeScript configuration
└── README.md                 # This file
```

## 🚀 Quick Start

### Prerequisites

- Node.js 16+
- npm or yarn

### Installation

```bash
cd frontend
npm install
```

### Development Server

```bash
npm run dev
```

Visit http://localhost:3000 to view the application.

### Production Build

```bash
npm run build
npm start
```

## 🎯 Component Overview

### CVDAssessmentForm

Main form component handling patient data input and validation.

**Features:**

- Comprehensive medical form with validation
- Auto-calculation of derived metrics (BMI, ratios)
- Example data loading for testing
- Real-time field validation
- Professional medical interface

**Key Sections:**

- Demographics (age, sex, height, weight)
- Vital Signs (blood pressure, heart rate)
- Laboratory Values (cholesterol, glucose)
- Risk Factors (smoking, diabetes, family history)
- Additional Measurements (waist circumference)

### PredictionResult

Results display component showing risk assessment.

**Features:**

- Risk level visualization (Low/Intermediate/High)
- Confidence scores and probabilities
- Clinical recommendations
- Model accuracy information
- Printable results format

## 🔧 Configuration

### API Configuration

Update API endpoints in components:

```typescript
// CVDAssessmentForm.tsx
const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://54.160.253.120:5001";

const response = await fetch(`${API_BASE_URL}/api/predict`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify(patientData),
});
```

### Environment Variables

Create `.env.local`:

```bash
NEXT_PUBLIC_API_URL=http://54.160.253.120:5001
NEXT_PUBLIC_APP_NAME="CVD Risk Assessment"
NEXT_PUBLIC_MODEL_VERSION="1.0"
```

## 🎨 UI Components

### Form Components

Built with Shadcn/UI for professional appearance:

```typescript
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { Card } from "@/components/ui/card";
```

### Styling

Tailwind CSS classes for responsive design:

```typescript
<div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
  <div className="max-w-4xl mx-auto">
    <Card className="shadow-lg">{/* Content */}</Card>
  </div>
</div>
```

## 📱 Responsive Design

### Breakpoints

- **Mobile**: < 768px
- **Tablet**: 768px - 1024px
- **Desktop**: > 1024px

### Grid Layout

```typescript
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
  {/* Form fields */}
</div>
```

## 🔒 Form Validation

### Client-Side Validation

```typescript
const validateAge = (age: number) => {
  if (age < 18 || age > 100) {
    return "Age must be between 18 and 100";
  }
  return null;
};

const validateBP = (systolic: number, diastolic: number) => {
  if (systolic <= diastolic) {
    return "Systolic must be greater than diastolic";
  }
  return null;
};
```

### Required Fields

All clinical fields are marked as required:

```typescript
<Input
  type="number"
  required
  min="18"
  max="100"
  value={formData.Age}
  onChange={(e) => handleInputChange("Age", e.target.value)}
/>
```

## 🧪 Testing

### Unit Tests

```bash
npm run test
```

### E2E Tests

```bash
npm run test:e2e
```

### Component Tests

```typescript
// __tests__/CVDAssessmentForm.test.tsx
import { render, screen } from "@testing-library/react";
import CVDAssessmentForm from "@/components/CVDAssessmentForm";

test("renders assessment form", () => {
  render(<CVDAssessmentForm onSubmit={jest.fn()} />);
  expect(screen.getByText("Patient Assessment Form")).toBeInTheDocument();
});
```

## 📊 Performance Optimization

### Code Splitting

```typescript
// Lazy load heavy components
const PredictionResult = dynamic(() => import("./PredictionResult"), {
  loading: () => <div>Loading...</div>,
});
```

### Image Optimization

```typescript
import Image from "next/image";

<Image
  src="/medical-icon.svg"
  alt="Medical Icon"
  width={64}
  height={64}
  priority
/>;
```

## 🌐 Deployment

### Vercel (Recommended)

```bash
npm i -g vercel
vercel
```

### Docker

```dockerfile
FROM node:18-alpine

WORKDIR /app
COPY package*.json ./
RUN npm install --only=production

COPY . .
RUN npm run build

EXPOSE 3000
CMD ["npm", "start"]
```

### Environment Setup

```bash
# Production environment variables
NEXT_PUBLIC_API_URL=https://api.yourapp.com
NEXT_PUBLIC_GA_ID=G-XXXXXXXXXX
NODE_ENV=production
```

## 🎯 Clinical Usage

### User Workflow

1. **Patient Input**: Enter comprehensive medical data
2. **Validation**: Real-time field validation and checks
3. **Prediction**: Submit for AI risk assessment
4. **Results**: View risk level and recommendations
5. **Action**: Use results for clinical decision making

### Risk Categories

- **🟢 LOW RISK**: Routine monitoring recommended
- **🟡 INTERMEDIATE**: Enhanced screening suggested
- **🔴 HIGH RISK**: Immediate clinical intervention

### Example Data

Quick testing with realistic patient scenarios:

- Low risk: Young, healthy patient profile
- High risk: Multiple risk factors present

## 📋 Dependencies

### Core Framework

```json
{
  "next": "^14.0.0",
  "react": "^18.0.0",
  "react-dom": "^18.0.0",
  "typescript": "^5.0.0"
}
```

### UI Components

```json
{
  "@radix-ui/react-select": "^2.0.0",
  "tailwindcss": "^3.3.0",
  "class-variance-authority": "^0.7.0",
  "clsx": "^2.0.0",
  "tailwind-merge": "^2.0.0"
}
```

### Development Tools

```json
{
  "eslint": "^8.0.0",
  "eslint-config-next": "^14.0.0",
  "@types/react": "^18.0.0",
  "@types/node": "^20.0.0"
}
```

## 🔍 Accessibility

### WCAG Compliance

- Semantic HTML structure
- Proper ARIA labels
- Keyboard navigation support
- Screen reader compatibility
- Color contrast compliance

### Implementation

```typescript
<Label htmlFor="age" className="sr-only">
  Patient Age
</Label>
<Input
  id="age"
  aria-describedby="age-help"
  aria-required="true"
  type="number"
/>
```

---

**🎯 Professional healthcare interface for clinical CVD risk assessment!**
