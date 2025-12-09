import AssessmentWrapper from "@/components/check/AssessmentWrapper";
import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Check Your Heart Health ❤️",
  description: "AI-powered cardiovascular risk assessment",
};

export default function CheckPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50 p-4 md:p-8 font-sans">
      <div className="max-w-5xl mx-auto">
        <AssessmentWrapper />
      </div>
    </div>
  );
}
