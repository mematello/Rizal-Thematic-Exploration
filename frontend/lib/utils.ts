import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

const LOWER_WORDS = new Set([
  'ng', 'mga', 'sa', 'at', 'o', 'ni', 'kay', 'nang', 'de', 'da', 'di', 'na', 'ang', 'ay', 'ukol'
]);

const PROPER_NOUNS = new Set([
  'maynila', 'zayb', 'tales', 'tiago', 'ibarra', 'basilio', 'simoun', 'clara', 'maria', 'elias', 'sisa', 'damaso', 'victis', 'vae', 'quidquid', 'latet', 'adparebit', 'nil', 'inultum', 'remanebit'
]);

/**
 * Converts chapter titles into Title Case matching the standard formatting of buod summaries.
 * Handles Tagalog grammatical particles, enclitic 't, hyphenated reduplications, and punctuation.
 */
export function formatChapterTitle(title?: string | null): string {
  if (!title) return '';
  let cleaned = title.trim();
  // Strip trailing period(s) if not ellipsis (common in 19th-century typesetting)
  cleaned = cleaned.replace(/\.+([\"\'“”»]?)$/, '$1').trim();

  // Split preserving delimiters (spaces, hyphens, quotes, brackets, em-dashes)
  const tokens = cleaned.split(/(\s+|[—\-\–\"\'“”«»’‘/\\()]+)/);
  const words: string[] = [];
  let wordIdx = 0;
  let prevDelim = '';

  for (const token of tokens) {
    if (!token) continue;
    if (!/[a-zA-Z\u00C0-\u024F]/.test(token)) {
      words.push(token);
      prevDelim = token;
      if (/[:;.]/.test(token)) wordIdx = 0;
      continue;
    }

    const wordLower = token.toLowerCase();

    // Enclitic 't / ’t (e.g. Kayamana't, Katuwira't)
    if (wordLower === 't' && (prevDelim.includes("'") || prevDelim.includes("’"))) {
      words.push('t');
      wordIdx++;
      prevDelim = '';
      continue;
    }

    const isAfterHyphen = prevDelim.includes('-') && !prevDelim.includes('—') && wordIdx > 0;

    if (isAfterHyphen && !PROPER_NOUNS.has(wordLower)) {
      words.push(wordLower);
    } else if (wordIdx > 0 && LOWER_WORDS.has(wordLower)) {
      words.push(wordLower);
    } else {
      const match = token.match(/^([^a-zA-Z\u00C0-\u024F]*)([a-zA-Z\u00C0-\u024F])(.*)$/);
      if (match) {
        const [, lead, first, rest] = match;
        words.push(lead + first.toUpperCase() + rest.toLowerCase());
      } else {
        words.push(token.charAt(0).toUpperCase() + token.slice(1).toLowerCase());
      }
    }
    wordIdx++;
    prevDelim = '';
  }
  return words.join('');
}

