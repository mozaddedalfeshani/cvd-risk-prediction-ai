import HeroSection from "@/components/home/HeroSection";
import FeaturesSection from "@/components/home/FeaturesSection";
import { Metadata } from "next";

export const metadata: Metadata = {
  title: "CVD Risk AI - Home",
  description: "Advanced AI for cardiovascular health prediction",
};

export default function HomePage() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-white to-blue-50">
      <HeroSection />
      <FeaturesSection />
      
      {/* Simple Footer */}
      <footer className="py-8 text-center text-gray-400 text-sm">
        <p>© 2024 CVD Risk AI. For educational purposes only.</p>
      </footer>
    </main>
  );
}
