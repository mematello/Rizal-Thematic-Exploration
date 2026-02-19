"use client";

import { motion } from "framer-motion";
import { BookOpen, User, Tag, FileText, BarChart3 } from "lucide-react";

const stats = [
    { label: "Lexical model usage", value: 40, color: "bg-brand-gold" },
    { label: "Semantic model usage", value: 30, color: "bg-brand-navy" },
    { label: "Lexical + Semantic model usage", value: 70, color: "bg-fili-accent" },
];

export function HeroSection() {
    return (
        <section className="relative w-full py-12 md:py-24 overflow-hidden">
            {/* Dynamic Background Glow */}
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-full pointer-events-none opacity-20">
                <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] rounded-full bg-brand-gold blur-[120px]" />
                <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] rounded-full bg-brand-navy blur-[120px]" />
            </div>
            <div className="max-w-7xl mx-auto px-4 grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
                {/* Left Side: Text Content */}
                <motion.div
                    initial={{ opacity: 0, x: -30 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.6, ease: "easeOut" }}
                    className="flex flex-col space-y-6"
                >
                    <div className="inline-flex items-center space-x-2 bg-brand-gold/10 px-3 py-1 rounded-full w-fit">
                        <span className="w-2 h-2 rounded-full bg-brand-gold animate-pulse" />
                        <span className="text-[10px] font-bold tracking-widest text-brand-gold uppercase">Proyektong Pananaliksik</span>
                    </div>

                    <h2 className="text-4xl md:text-5xl lg:text-7xl font-serif text-brand-navy leading-[1.05] font-black tracking-tight">
                        Lexical and Semantic Embedding of <span className="text-brand-gold italic">XLM</span>
                    </h2>

                    <p className="text-lg md:text-xl text-brand-text/70 font-serif leading-relaxed max-w-xl">
                        Explore characters, themes, and related sentences with ease using our intelligent model — showcasing the novelty of combining lexical and semantic understanding for the Filipino language.
                    </p>

                    <div className="flex items-center space-x-6 pt-4">
                        <div className="flex -space-x-3">
                            {[1, 2, 3, 4].map((i) => (
                                <div key={i} className="w-10 h-10 rounded-full border-2 border-brand-cream bg-brand-paper flex items-center justify-center shadow-sm">
                                    {i === 1 && <User className="w-5 h-5 text-brand-gold" />}
                                    {i === 2 && <Tag className="w-5 h-5 text-brand-navy" />}
                                    {i === 3 && <BookOpen className="w-5 h-5 text-noli-accent" />}
                                    {i === 4 && <FileText className="w-5 h-5 text-fili-accent" />}
                                </div>
                            ))}
                        </div>
                        <p className="text-sm font-medium text-brand-text-light tracking-wide uppercase">
                            Integrated Intelligence
                        </p>
                    </div>
                </motion.div>

                {/* Right Side: Animated Info Bar & Illustration */}
                <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.8, ease: "easeOut", delay: 0.2 }}
                    className="relative bg-brand-paper p-8 rounded-3xl shadow-2xl border border-brand-gold/10 overflow-hidden"
                >
                    {/* Decorative Background Icons */}
                    <div className="absolute inset-0 opacity-[0.03] pointer-events-none select-none">
                        <User className="absolute top-10 right-10 w-32 h-32 rotate-12" />
                        <BookOpen className="absolute bottom-10 left-10 w-40 h-40 -rotate-12" />
                    </div>

                    <div className="relative z-10 flex flex-col space-y-10">
                        {/* Illustration Mockup (Since generation failed) */}
                        <div className="flex justify-center mb-4">
                            <div className="relative">
                                <motion.div
                                    animate={{ y: [0, -10, 0] }}
                                    transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
                                    className="bg-brand-cream p-10 rounded-full shadow-inner border border-brand-gold/5"
                                >
                                    <BookOpen className="w-24 h-24 text-brand-gold/80 stroke-[1]" />
                                </motion.div>
                                {/* Floating Icons */}
                                <motion.div
                                    animate={{ x: [0, 5, 0], y: [0, -5, 0], rotate: [0, 10, 0] }}
                                    transition={{ duration: 3, repeat: Infinity, ease: "easeInOut", delay: 0.5 }}
                                    className="absolute -top-4 -right-4 bg-white p-3 rounded-xl shadow-lg border border-brand-gold/10"
                                >
                                    <User className="w-6 h-6 text-brand-navy" />
                                </motion.div>
                                <motion.div
                                    animate={{ x: [0, -5, 0], y: [0, 5, 0], rotate: [0, -10, 0] }}
                                    transition={{ duration: 3.5, repeat: Infinity, ease: "easeInOut", delay: 1 }}
                                    className="absolute bottom-0 -left-6 bg-white p-3 rounded-xl shadow-lg border border-brand-gold/10"
                                >
                                    <Tag className="w-6 h-6 text-noli-accent" />
                                </motion.div>
                                <motion.div
                                    animate={{ scale: [1, 1.1, 1] }}
                                    transition={{ duration: 5, repeat: Infinity, ease: "easeInOut" }}
                                    className="absolute top-1/2 -right-8 transform -translate-y-1/2 bg-white p-3 rounded-xl shadow-lg border border-brand-gold/10"
                                >
                                    <BarChart3 className="w-6 h-6 text-fili-accent" />
                                </motion.div>
                            </div>
                        </div>

                        {/* Stats Bar Section */}
                        <div className="space-y-6">
                            <div className="flex justify-between items-end">
                                <h4 className="text-xl font-serif text-brand-navy font-bold">Model Efficiency</h4>
                                <p className="text-xs font-bold tracking-widest text-brand-gold opacity-70">REAL-TIME TELEMETRY</p>
                            </div>

                            <div className="space-y-4">
                                {stats.map((stat, idx) => (
                                    <div key={idx} className="space-y-2">
                                        <div className="flex justify-between text-sm">
                                            <span className="text-brand-text font-medium">{stat.label}</span>
                                            <span className="font-bold text-brand-navy">{stat.value}%</span>
                                        </div>
                                        <div className="w-full h-2 bg-brand-cream rounded-full overflow-hidden">
                                            <motion.div
                                                initial={{ width: 0 }}
                                                animate={{ width: `${stat.value}%` }}
                                                transition={{ duration: 1.5, ease: "circOut", delay: 0.5 + idx * 0.2 }}
                                                className={`h-full ${stat.color} shadow-[0_0_10px_rgba(0,0,0,0.1)]`}
                                            />
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </motion.div>
            </div>
        </section>
    );
}
