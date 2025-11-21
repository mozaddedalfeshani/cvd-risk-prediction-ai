export type Language = 'en' | 'bn';

export const translations = {
  en: {
    hero: {
      badge: "AI-Powered Health Analysis 🤖",
      title: "Predict Your Heart Health In Seconds.",
      subtitle: "Advanced machine learning models analyze your vitals to provide instant, accurate cardiovascular risk assessments.",
      start: "Start Assessment",
      learn: "Learn More"
    },
    features: {
      title: "Why Choose Our AI?",
      subtitle: "We use state-of-the-art technology to help you stay ahead of health risks.",
      dual_ai: "Dual AI Models",
      dual_ai_desc: "Combines two powerful algorithms for maximum prediction accuracy.",
      instant: "Instant Results",
      instant_desc: "Get your comprehensive health report in milliseconds.",
      privacy: "Privacy First",
      privacy_desc: "Your health data is processed securely and never shared.",
      insights: "Clinical Insights",
      insights_desc: "Receive detailed interpretations and lifestyle recommendations."
    },
    form: {
      title: "Start Your Assessment",
      subtitle: "Fill out the simple form below to get your instant heart health report!",
      steps: {
        demographics: "About You",
        vitals: "Heart Stats",
        labs: "Lab Check",
        risk: "Lifestyle",
        additional: "Extra Info",
        review: "Ready?"
      },
      labels: {
        sex: "Sex",
        age: "Age",
        weight: "Weight (kg)",
        height: "Height (m)",
        bmi: "BMI (Auto)",
        systolic: "Systolic BP",
        diastolic: "Diastolic BP",
        bp_category: "BP Category",
        cholesterol: "Total Cholesterol",
        hdl: "HDL (Good)",
        ldl: "LDL (Bad)",
        glucose: "Blood Sugar",
        smoking: "Smoking?",
        diabetes: "Diabetes?",
        family: "Family History?",
        activity: "Activity Level",
        waist: "Waist Size (cm)",
        waist_ratio: "Waist-Height Ratio",
        cvd_score: "CVD Score"
      },
      buttons: {
        back: "Back",
        next: "Next",
        submit: "Assess Risk!",
        loading: "Analyzing..."
      }
    }
  },
  bn: {
    hero: {
      badge: "এআই-চালিত স্বাস্থ্য বিশ্লেষণ 🤖",
      title: "আপনার হার্টের স্বাস্থ্য যাচাই করুন নিমিষেই।",
      subtitle: "উন্নত মেশিন লার্নিং মডেল আপনার তথ্য বিশ্লেষণ করে তাৎক্ষণিক এবং নির্ভুল হৃদরোগের ঝুঁকি নির্ণয় করে।",
      start: "যাচাই শুরু করুন",
      learn: "আরও জানুন"
    },
    features: {
      title: "কেন আমাদের এআই বেছে নেবেন?",
      subtitle: "আমরা অত্যাধুনিক প্রযুক্তি ব্যবহার করি যাতে আপনি স্বাস্থ্য ঝুঁকির এক ধাপ আগে থাকতে পারেন।",
      dual_ai: "ডুয়াল এআই মডেল",
      dual_ai_desc: "সর্বোচ্চ নির্ভুলতার জন্য দুটি শক্তিশালী অ্যালগরিদম একত্রিত করে।",
      instant: "তাৎক্ষণিক ফলাফল",
      instant_desc: "মিলি সেকেন্ডের মধ্যে আপনার বিস্তারিত স্বাস্থ্য রিপোর্ট পান।",
      privacy: "গোপনীয়তা প্রথম",
      privacy_desc: "আপনার স্বাস্থ্য তথ্য সুরক্ষিতভাবে প্রক্রিয়া করা হয় এবং কখনও শেয়ার করা হয় না।",
      insights: "ক্লিনিকাল অন্তর্দৃষ্টি",
      insights_desc: "বিস্তারিত ব্যাখ্যা এবং জীবনযাত্রার পরামর্শ পান।"
    },
    form: {
      title: "আপনার মূল্যায়ন শুরু করুন",
      subtitle: "আপনার তাৎক্ষণিক হার্ট হেলথ রিপোর্ট পেতে নিচের সহজ ফর্মটি পূরণ করুন!",
      steps: {
        demographics: "আপনার সম্পর্কে",
        vitals: "হার্ট স্ট্যাটাস",
        labs: "ল্যাব চেক",
        risk: "জীবনধারা",
        additional: "অতিরিক্ত তথ্য",
        review: "প্রস্তুত?"
      },
      labels: {
        sex: "লিঙ্গ",
        age: "বয়স",
        weight: "ওজন (কেজি)",
        height: "উচ্চতা (মিটার)",
        bmi: "বিএমআই (অটো)",
        systolic: "সিস্টোলিক বিপি",
        diastolic: "ডায়াস্টোলিক বিপি",
        bp_category: "বিপি ক্যাটাগরি",
        cholesterol: "মোট কোলেস্টেরল",
        hdl: "এইচডিএল (ভাল)",
        ldl: "এলডিএল (খারাপ)",
        glucose: "রক্তের চিনি",
        smoking: "ধূমপান?",
        diabetes: "ডায়াবেটিস?",
        family: "পারিবারিক ইতিহাস?",
        activity: "কর্মতৎপরতার স্তর",
        waist: "কোমরের মাপ (সেমি)",
        waist_ratio: "কোমর-উচ্চতা অনুপাত",
        cvd_score: "সিভিডি স্কোর"
      },
      buttons: {
        back: "পেছনে",
        next: "পরবর্তী",
        submit: "ঝুঁকি যাচাই করুন!",
        loading: "বিশ্লেষণ করা হচ্ছে..."
      }
    }
  }
};
