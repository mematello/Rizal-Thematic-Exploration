"use client";

import { useEffect, useState } from "react";

const TIPS = [
    "Did you know? Noli Me Tangere translates to 'Touch Me Not' and refers to John 20:17.",
    "Rizal dedicated El Filibusterismo to the execution of the three martyr priests: Gomburza.",
    "Maria Clara is widely thought to be modeled after Rizal's childhood sweetheart, Leonor Rivera.",
    "Simoun is actually Crisostomo Ibarra in disguise, returned for revenge.",
    "Sisa often symbolizes the suffering motherland in Rizal's allegories."
];

export function SkeletonLoader() {
    const [tip, setTip] = useState("");

    useEffect(() => {
        setTip(TIPS[Math.floor(Math.random() * TIPS.length)]);
    }, []);

    return (
        <div className="w-full max-w-4xl mx-auto p-4 space-y-6 animate-pulse">
            {/* Search Stats Skeleton */}
            <div className="h-4 bg-gray-200 rounded w-1/4 mb-6"></div>

            {/* Result Card Skeletons */}
            {[1, 2, 3].map((i) => (
                <div key={i} className="bg-white rounded-r-lg shadow-sm border-l-4 border-gray-200 p-4">
                    <div className="flex gap-4 mb-4">
                        <div className="h-4 bg-gray-200 rounded w-1/4"></div>
                        <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                    </div>
                    <div className="space-y-2 mb-4">
                        <div className="h-3 bg-gray-200 rounded w-full"></div>
                        <div className="h-3 bg-gray-200 rounded w-full"></div>
                        <div className="h-3 bg-gray-200 rounded w-2/3"></div>
                    </div>
                    <div className="flex gap-2">
                        <div className="h-6 bg-gray-200 rounded w-16"></div>
                        <div className="h-6 bg-gray-200 rounded w-16"></div>
                    </div>
                </div>
            ))}

            {/* Tip */}
            <div className="text-center mt-8 text-brand-brown/60 text-sm font-crimson italic">
                Loading... {tip}
            </div>
        </div>
    );
}
