"use client";

import { useRouter } from "next/navigation";
import { SearchBar } from "@/components/SearchBar";

const THEMES = [
    { id: 'hustisya', label: 'Hustisya at Katarungan', icon: '⚖️', desc: 'Ang kawalan ng sistema ng hustisya sa kolonyal na pamahalaan.' },
    { id: 'pag-ibig', label: 'Pag-ibig at Sawi', icon: '💔', desc: 'Mga kwento ng pag-irog nina Maria Clara, Ibarra, at iba pa.' },
    { id: 'himagsikan', label: 'Rebolusyon', icon: '🔥', desc: 'Ang unti-unting paggising ng damdaming makabayan.' },
    { id: 'edukasyon', label: 'Edukasyon', icon: '📚', desc: 'Ang halaga ng karunungan para sa kinabukasan ng bayan.' },
    { id: 'relihiyon', label: 'Relihiyon at Prayle', icon: '⛪', desc: 'Ang impluwensya at abuso ng mga prayle.' },
    { id: 'kahirapan', label: 'Kahirapan', icon: '🌾', desc: 'Ang buhay ng mga indio at ang sapilitang paggawa.' },
];

export default function ExplorePage() {
    const router = useRouter();

    const handleSearch = (query: string) => {
        router.push(`/search?q=${encodeURIComponent(query)}`);
    };

    return (
        <div className="min-h-screen bg-brand-cream flex flex-col items-center p-4 md:p-8">
            <header className="w-full max-w-4xl flex justify-between items-center mb-8">
                <h1 className="font-crimson font-bold text-3xl text-brand-brown">
                    Tuklasin ang mga Tema
                </h1>
                <button onClick={() => router.push('/')} className="text-sm font-bold text-brand-blue uppercase hover:underline">
                    Bumalik sa Home
                </button>
            </header>

            <div className="w-full max-w-4xl grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                {THEMES.map((theme) => (
                    <button
                        key={theme.id}
                        onClick={() => handleSearch(theme.id)}
                        className="group text-left bg-white p-6 rounded-xl shadow-sm hover:shadow-md transition-all border border-brand-brown/5 hover:border-brand-brown/20"
                    >
                        <div className="text-4xl mb-4 group-hover:scale-110 transition-transform duration-300">
                            {theme.icon}
                        </div>
                        <h3 className="font-roboto font-bold text-xl text-brand-brown mb-2 group-hover:text-brand-blue">
                            {theme.label}
                        </h3>
                        <p className="font-crimson text-brand-text/80 leading-relaxed">
                            {theme.desc}
                        </p>
                    </button>
                ))}
            </div>
        </div>
    );
}
