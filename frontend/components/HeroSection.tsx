"use client";

import { ArrowDown, ArrowUpRight, BookOpen } from "lucide-react";
import { useNovelStore } from "@/store/novelStore";

export function HeroSection({ novel }: { novel: "noli" | "fili" | "both" }) {
    const { setSelectedNovel } = useNovelStore();
    return (
        <section className="literary-hero" aria-labelledby="library-welcome">
            <div className="literary-hero-copy">
                <span className="literary-eyebrow"><BookOpen size={14} /> ANG IYONG AKLATANG RIZAL</span>
                <h1 id="library-welcome">Buksan ang kuwento.<br /><em>Tuklasin ang kahulugan.</em></h1>
                <p>Dalawang nobela. Maraming pananaw. Basahin ang mga kabanata, kilalanin ang mga tauhan, at tuklasin ang mga salita ni Rizal—sa sarili mong bilis.</p>
                <a className="literary-text-link" href="#library"><ArrowDown size={16} /> Pumili ng kabanata</a>
            </div>
            <div className="literary-books" aria-label="Piliin ang iyong nobela">
                {([{ id: 'noli', title: 'Noli Me Tangere', year: '1887', letter: 'N' }, { id: 'fili', title: 'El Filibusterismo', year: '1891', letter: 'F' }] as const).map(book => (
                    <button key={book.id} type="button" className={`literary-book literary-book-${book.id}`} aria-pressed={novel === book.id} onClick={() => setSelectedNovel(book.id)} aria-label={`Piliin ang ${book.title}`}>
                        <span className="book-cover"><span className="book-author">JOSÉ RIZAL</span><span className="book-monogram" aria-hidden="true">{book.letter}</span><span className="book-cover-title">{book.title}</span><span className="book-year">{book.year}</span></span>
                        <span className="book-caption"><span>{book.title}<small>{novel === book.id ? 'NAPILING NOBELA' : 'TUKLASIN ANG NOBELA'}</small></span><ArrowUpRight size={18} /></span>
                    </button>
                ))}
            </div>
        </section>
    );
}
