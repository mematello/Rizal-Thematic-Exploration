"use client";

import { motion, AnimatePresence } from "framer-motion";
import { X, Quote } from "lucide-react";

interface SanggunianModalProps {
    isOpen: boolean;
    onClose: () => void;
    sourceText: string;
    book: string;
    chapterNumber: number;
    mode: 'buod' | 'full';
    referenceData: {
        reference_text: string;
        alignment_status?: string;
        matched_characters?: string[];
        mode?: string;
    } | null;
    isLoading: boolean;
    error: string | null;
}

export function SanggunianModal({
    isOpen,
    onClose,
    sourceText,
    book,
    chapterNumber,
    mode,
    referenceData,
    isLoading,
    error
}: SanggunianModalProps) {
    const isNoli = book.toLowerCase().includes('noli');
    
    return (
        <AnimatePresence>
            {isOpen && (
                <motion.div 
                    initial={{ opacity: 0 }} 
                    animate={{ opacity: 1 }} 
                    exit={{ opacity: 0 }} 
                    className="fixed inset-0 z-[100] bg-white flex flex-col"
                >
                    <div className="border-b border-brand-gold/30 bg-brand-cream/50 py-6">
                        <div className="max-w-7xl mx-auto px-6 text-center space-y-2 relative">
                            <h2 className="text-2xl font-serif text-brand-navy font-bold">Kaugnay na Sanggunian</h2>
                            <p className="text-lg text-brand-text">{isNoli ? "Noli Me Tangere" : "El Filibusterismo"} · Kabanata {chapterNumber}</p>
                            <div className="flex items-center justify-center gap-2 pt-2">
                                <span className="px-3 py-1 bg-brand-navy text-white text-[10px] font-bold uppercase rounded-full">
                                    Mula sa {mode === 'buod' ? 'Buod' : 'Buong Kwento'}
                                </span>
                                <span className="text-brand-gold">→</span>
                                <span className="px-3 py-1 bg-brand-gold text-white text-[10px] font-bold uppercase rounded-full">
                                    Patungo sa {mode === 'buod' ? 'Buong Kwento' : 'Buod'}
                                </span>
                            </div>
                            <button 
                                onClick={onClose} 
                                className="absolute top-0 right-0 p-2 hover:bg-black/5 rounded-full"
                            >
                                <X size={24} className="text-brand-navy" />
                            </button>
                        </div>
                    </div>
                    
                    <div className="flex-1 overflow-hidden flex flex-col md:flex-row">
                        {/* Source Section */}
                        <div className="flex-1 overflow-y-auto p-8 bg-brand-paper/30 border-b md:border-b-0 md:border-r border-brand-gold/10">
                            <div className="max-w-xl mx-auto space-y-6">
                                <h3 className="text-xs font-bold uppercase tracking-widest text-brand-navy/40 border-b border-brand-gold/10 pb-2">
                                    Pinagmulang Teksto
                                </h3>
                                <p className="font-serif text-brand-text leading-loose text-justify text-xl italic">
                                    &quot;{sourceText}&quot;
                                </p>
                            </div>
                        </div>
                        
                        {/* Target Section */}
                        <div className="w-full md:w-[500px] bg-white overflow-y-auto p-10">
                            <h3 className="text-xs font-bold uppercase tracking-widest text-brand-gold mb-8">
                                Nahanap na Kaugnayan
                            </h3>
                            
                            {isLoading ? (
                                <div className="flex flex-col items-center py-20 space-y-4">
                                    <div className="w-10 h-10 border-2 border-brand-gold border-t-transparent rounded-full animate-spin" />
                                    <p className="text-sm font-serif italic text-brand-text/60 text-center">
                                        Sinisiyasat ng AI ang kabilang bersyon...
                                    </p>
                                </div>
                            ) : error ? (
                                <div className="py-10 text-center space-y-4">
                                    <div className="w-12 h-12 bg-red-50 text-red-400 rounded-full flex items-center justify-center mx-auto">
                                        <X size={24} />
                                    </div>
                                    <p className="text-red-600 font-serif">{error}</p>
                                </div>
                            ) : referenceData && (
                                <div className="space-y-8">
                                    <div className="p-6 bg-brand-cream/20 rounded-xl border border-brand-gold/10">
                                        <div className="mb-4 flex justify-between items-start">
                                            <div>
                                                <span className="text-[10px] font-bold text-brand-navy/40 uppercase tracking-tighter">
                                                    Lokasyon sa {mode === 'buod' ? 'Buong Kwento' : 'Buod'}
                                                </span>
                                            </div>
                                            {referenceData.alignment_status && (
                                                <span className={`text-[9px] uppercase font-bold px-2 py-0.5 rounded-full ${referenceData.alignment_status === 'precise' ? 'bg-green-100 text-green-700' : 'bg-amber-100 text-amber-700'}`}>
                                                    {referenceData.alignment_status}
                                                </span>
                                            )}
                                        </div>
                                        <p className="text-xl font-serif leading-relaxed text-brand-text">
                                            &quot;{referenceData.reference_text}&quot;
                                        </p>
                                        
                                        {referenceData.matched_characters && referenceData.matched_characters.length > 0 && (
                                            <div className="mt-6 pt-4 border-t border-brand-gold/10">
                                                <span className="text-[10px] font-bold text-brand-navy/40 uppercase tracking-tighter block mb-2">
                                                    Tauhan
                                                </span>
                                                <div className="flex flex-wrap gap-2">
                                                    {referenceData.matched_characters.map(c => (
                                                        <span key={c} className="bg-brand-gold/10 text-brand-navy px-2 py-1 rounded text-xs font-bold">
                                                            {c}
                                                        </span>
                                                    ))}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                    <div className="pt-6 border-t border-brand-gold/10">
                                        <p className="text-xs text-brand-text/50 leading-relaxed italic">
                                            Ang resultang ito ay nabuo sa pamamagitan ng Triple-Signal Segmentation at Hybrid Scoring na may Dynamic Position Window.
                                        </p>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </motion.div>
            )}
        </AnimatePresence>
    );
}
