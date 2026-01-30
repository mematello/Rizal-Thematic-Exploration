"use client";

import { SearchBar } from "@/components/SearchBar";
import { useRouter } from "next/navigation";
import Image from "next/image";

export default function Home() {
  const router = useRouter();

  const handleSearch = (query: string) => {
    router.push(`/search?q=${encodeURIComponent(query)}`);
  };

  return (
    <main className="min-h-screen flex flex-col items-center justify-center p-4 bg-brand-cream relative overflow-hidden">
      {/* Background decoration (optional) */}
      <div className="absolute inset-0 opacity-5 pointer-events-none bg-[url('/paper-texture.png')] mix-blend-multiply"></div>

      <div className="z-10 w-full max-w-2xl text-center space-y-8 animate-in fade-in zoom-in-95 duration-700">

        {/* Logo / Title */}
        <div className="space-y-2">
          <h1 className="font-crimson font-bold text-5xl md:text-6xl text-brand-brown tracking-tight">
            Noli & Fili
          </h1>
          <p className="font-roboto text-brand-text/80 text-lg uppercase tracking-widest">
            Semantikong Paggalugad
          </p>
        </div>

        {/* Hero Search */}
        <div className="w-full">
          <SearchBar
            variant="hero"
            onSearch={handleSearch}
            placeholder="Tanungin tungkol sa hustisya, pag-ibig, o himagsikan..."
          />
        </div>

        {/* Quick Links / Hints */}
        <div className="flex flex-wrap justify-center gap-3 text-sm font-crimson text-brand-brown/60">
          <span>Subukan:</span>
          <button onClick={() => handleSearch("katarungan")} className="hover:text-brand-blue hover:underline">Katarungan</button>
          <span>•</span>
          <button onClick={() => handleSearch("edukasyon")} className="hover:text-brand-blue hover:underline">Edukasyon</button>
          <span>•</span>
          <button onClick={() => handleSearch("Sisa")} className="hover:text-brand-blue hover:underline">Sisa</button>
          <span>•</span>
          <button onClick={() => handleSearch("Simoun")} className="hover:text-brand-blue hover:underline">Simoun</button>
        </div>

      </div>

      {/* Footer */}
      <footer className="absolute bottom-4 text-xs font-roboto text-brand-brown/40 uppercase tracking-widest">
        Rizal Thematic Exploration System
      </footer>
    </main>
  );
}
