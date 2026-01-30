"use client";

import { useSearchParams, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { SearchBar } from "@/components/SearchBar";
import { FilterBar } from "@/components/FilterBar";
import { ResultCard } from "@/components/ResultCard";
import { SkeletonLoader } from "@/components/SkeletonLoader";
import { useRizalSearch } from "@/hooks/useRizalSearch";

import { Suspense } from "react";

function SearchContent() {
    const searchParams = useSearchParams();
    const router = useRouter();
    const initialQuery = searchParams.get("q") || "";

    const [matchFilter, setMatchFilter] = useState<'all' | 'noli' | 'fili'>('all');
    const [minScore, setMinScore] = useState(0);

    const { data, isLoading, error } = useRizalSearch(initialQuery);

    const handleSearch = (newQuery: string) => {
        router.push(`/search?q=${encodeURIComponent(newQuery)}`);
    };

    const results = data?.results;
    const noliResults = results?.noli || [];
    const filiResults = results?.fili || [];

    // Filter logic
    const filteredNoli = noliResults.filter(r => r.scores?.final >= minScore && (matchFilter === 'all' || matchFilter === 'noli'));
    const filteredFili = filiResults.filter(r => r.scores?.final >= minScore && (matchFilter === 'all' || matchFilter === 'fili'));

    return (
        <div className="min-h-screen bg-gray-50 flex flex-col">
            {/* Sticky Header */}
            <header className="bg-brand-cream border-b border-brand-brown/10 sticky top-0 z-40">
                <div className="max-w-6xl mx-auto px-4 py-3 flex items-center gap-4">
                    <div
                        onClick={() => router.push('/')}
                        className="hidden md:block font-crimson font-bold text-xl text-brand-brown cursor-pointer"
                    >
                        RizalEx
                    </div>
                    <div className="flex-1">
                        <SearchBar
                            variant="persistent"
                            defaultValue={initialQuery}
                            onSearch={handleSearch}
                            isLoading={isLoading}
                        />
                    </div>
                </div>

                {/* Filters */}
                <FilterBar
                    matchFilter={matchFilter}
                    onMatchFilterChange={setMatchFilter}
                    minScore={minScore}
                    onMinScoreChange={setMinScore}
                />
            </header>

            {/* Main Content */}
            <main className="flex-1 max-w-6xl w-full mx-auto px-4 py-6">
                {isLoading ? (
                    <SkeletonLoader />
                ) : error ? (
                    <div className="text-center py-20 text-red-500">
                        Nagkaroon ng problema sa paghahanap. Subukang muli.
                    </div>
                ) : (
                    <div className="grid md:grid-cols-2 gap-6">

                        {/* Noli Column */}
                        {(matchFilter === 'all' || matchFilter === 'noli') && (
                            <section className="block">
                                <div className="flex items-center gap-2 mb-4 pb-2 border-b-2 border-noli-gold">
                                    <span className="font-roboto font-bold text-brand-brown uppercase tracking-wider">
                                        Noli Me Tangere
                                    </span>
                                    <span className="bg-noli-gold/20 text-noli-gold text-xs px-2 py-0.5 rounded-full font-bold">
                                        {filteredNoli.length}
                                    </span>
                                </div>
                                <div className="space-y-4">
                                    {filteredNoli.map((result) => (
                                        <ResultCard
                                            key={result.id}
                                            {...result}
                                            novel="noli"
                                        />
                                    ))}
                                    {filteredNoli.length === 0 && (
                                        <div className="text-center py-10 text-gray-400 italic font-crimson">
                                            Walang nakitang resulta sa Noli.
                                        </div>
                                    )}
                                </div>
                            </section>
                        )}

                        {/* Fili Column */}
                        {(matchFilter === 'all' || matchFilter === 'fili') && (
                            <section className="block">
                                <div className="flex items-center gap-2 mb-4 pb-2 border-b-2 border-fili-magenta">
                                    <span className="font-roboto font-bold text-brand-brown uppercase tracking-wider">
                                        El Filibusterismo
                                    </span>
                                    <span className="bg-fili-magenta/20 text-fili-magenta text-xs px-2 py-0.5 rounded-full font-bold">
                                        {filteredFili.length}
                                    </span>
                                </div>
                                <div className="space-y-4">
                                    {filteredFili.map((result) => (
                                        <ResultCard
                                            key={result.id}
                                            {...result}
                                            novel="fili"
                                        />
                                    ))}
                                    {filteredFili.length === 0 && (
                                        <div className="text-center py-10 text-gray-400 italic font-crimson">
                                            Walang nakitang resulta sa Fili.
                                        </div>
                                    )}
                                </div>
                            </section>
                        )}
                    </div>
                )}
            </main>
        </div>
    );
}

export default function SearchPage() {
    return (
        <Suspense fallback={<SkeletonLoader />}>
            <SearchContent />
        </Suspense>
    );
}
