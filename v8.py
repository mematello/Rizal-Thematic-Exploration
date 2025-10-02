import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import warnings
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich import box

warnings.filterwarnings('ignore')

class QueryAnalyzer:
    """Handles query validation and linguistic analysis (display only)"""
    
    def __init__(self):
        self.MIN_FILIPINO_FREQUENCY = 1e-8
        self.MIN_VALID_WORD_RATIO = 0.5
        
        # Filipino stopwords for filtering
        self.STOPWORDS = {
            'ng', 'sa', 'ang', 'na', 'ay', 'at', 'mga',
            'ko', 'mo', 'niya', 'nila', 'natin', 'namin', 'atin', 'amin',
            'ka', 'siya', 'ako', 'ikaw', 'tayo', 'kayo', 'kami',
            'ito', 'iyan', 'iyon', 'dito', 'diyan', 'doon',
            'pa', 'din', 'rin', 'lang', 'lamang', 'ni', 'nina',
            'dahil', 'kasi', 'kung', 'para', 'pero', 'kaya', 'o',
            'may', 'mayroon', 'wala', 'walang',
            'ano', 'sino', 'saan', 'kailan', 'paano', 'bakit',
            'ngunit', 'subalit', 'kundi', 'datapwat'
        }
    
    def get_word_frequency(self, word, lang='tl'):
        """Get word frequency using wordfreq library"""
        try:
            from wordfreq import word_frequency
            return word_frequency(word.lower(), lang)
        except Exception:
            return 0.0
    
    def is_valid_filipino_word(self, word):
        """Check if a word is valid Filipino"""
        if len(word) < 2:
            return False
        
        freq = self.get_word_frequency(word, 'tl')
        return freq >= self.MIN_FILIPINO_FREQUENCY
    
    def validate_filipino_query(self, query):
        """Validate if query contains valid Filipino words"""
        words = re.findall(r'\b[a-zA-ZÀ-ÿñÑ]+\b', query)
        
        if not words:
            return False, {
                'reason': 'No valid words found in query',
                'total_words': 0,
                'valid_words': 0,
                'invalid_words': [],
                'valid_ratio': 0.0
            }
        
        valid_words = []
        invalid_words = []
        
        for word in words:
            if self.is_valid_filipino_word(word):
                valid_words.append(word)
            else:
                invalid_words.append(word)
        
        total_words = len(words)
        valid_count = len(valid_words)
        valid_ratio = valid_count / total_words if total_words > 0 else 0.0
        is_valid = valid_ratio >= self.MIN_VALID_WORD_RATIO
        
        validation_info = {
            'total_words': total_words,
            'valid_words': valid_count,
            'invalid_words': invalid_words,
            'valid_words_list': valid_words,
            'valid_ratio': valid_ratio,
            'reason': ''
        }
        
        if not is_valid:
            if valid_count == 0:
                validation_info['reason'] = 'No valid Filipino words detected'
            else:
                validation_info['reason'] = f'Only {valid_ratio:.1%} of words are valid Filipino'
        
        return is_valid, validation_info
    
    def analyze_query_words(self, query):
        """Analyze query words: frequencies and stopword status"""
        words = re.findall(r'\b[a-zA-ZÀ-ÿñÑ]+\b', query)
        analysis = []
        
        for word in words:
            word_lower = word.lower()
            freq = self.get_word_frequency(word_lower, 'tl')
            is_stopword = word_lower in self.STOPWORDS
            
            analysis.append({
                'word': word,
                'frequency': freq,
                'is_stopword': is_stopword,
                'is_content_word': not is_stopword and freq < 0.001
            })
        
        return analysis

class CleanNoliSystem:
    """
    Clean CLEAR-inspired hybrid retrieval (Gao et al., 2021)
    Formula: s_final(q,d) = λ_emb · s_emb(q,d) + λ_lex · s_lex(q,d)
    
    Design principles:
    - Semantic similarity is the foundation (XLM-RoBERTa handles morphology)
    - Lexical matching is a precision boost
    - NO stem matching (too unreliable for Filipino)
    - Strict deduplication (no passage appears twice)
    """
    
    def __init__(self):
        self.console = Console()
        
        self.console.print("Loading XLM-RoBERTa model...", style="cyan")
        self.model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
        
        self.console.print("Loading datasets...", style="cyan")
        self.chapters_df = pd.read_csv('chapters.csv')
        self.themes_df = pd.read_csv('themes.csv')
        
        self.chapters_df.columns = self.chapters_df.columns.str.strip()
        self.themes_df.columns = self.themes_df.columns.str.strip()
        
        self.console.print("Initializing query analyzer...", style="cyan")
        self.query_analyzer = QueryAnalyzer()
        
        self.console.print("Computing embeddings...", style="cyan")
        self._compute_embeddings()
        
        # System parameters
        self.MIN_SEMANTIC_THRESHOLD = 0.20
        self.THEMATIC_THRESHOLD = 0.45
        self.CONTEXT_RELEVANCE_THRESHOLD = 0.30
        self.SHORT_SENTENCE_THRESHOLD = 5
        self.SHORT_SENTENCE_PENALTY = 0.08
        self.MAX_CONTEXT_EXPANSION = 5
        
        # Track all used passages globally
        self.used_passages = set()
        
        self.console.print("Clean CLEAR system ready!", style="bold green")
    
    def _compute_embeddings(self):
        """Compute embeddings for chapters and themes"""
        self.chapters_df['combined_text'] = (
            self.chapters_df['chapter_title'].astype(str) + " " + 
            self.chapters_df['sentence_text'].astype(str)
        )
        
        self.chapters_df['sentence_word_count'] = (
            self.chapters_df['sentence_text'].astype(str).apply(lambda x: len(x.split()))
        )
        
        texts = self.chapters_df['combined_text'].tolist()
        self.chapter_embeddings = self.model.encode(texts, show_progress_bar=False)
        
        self.themes_df['theme_text'] = (
            self.themes_df['Tagalog Title'].astype(str) + " " + 
            self.themes_df['Meaning'].astype(str)
        )
        
        theme_texts = self.themes_df['theme_text'].tolist()
        self.theme_embeddings = self.model.encode(theme_texts, show_progress_bar=False)
    
    def _get_passage_id(self, chapter_num, sentence_num):
        """Create unique identifier for passages"""
        return (int(chapter_num), int(sentence_num))
    
    def _compute_dynamic_weights(self, query_length):
        """
        Length-aware dynamic weighting
        Short queries: trust lexical more (precise matching)
        Long queries: trust semantic more (conceptual matching)
        """
        L = query_length
        
        if L <= 1:
            alpha, beta = 0.40, 0.60
        elif L == 2:
            alpha, beta = 0.55, 0.45
        elif L == 3:
            alpha, beta = 0.68, 0.32
        elif L == 4:
            alpha, beta = 0.75, 0.25
        elif L == 5:
            alpha, beta = 0.80, 0.20
        else:
            alpha, beta = 0.85, 0.15
        
        return alpha, beta
    
    def _compute_lexical_score(self, query, sentence_text):
        """
        Compute lexical overlap score
        1.0 = exact match, partial = ratio of overlap
        """
        query_lower = query.lower().strip()
        sentence_lower = sentence_text.lower().strip()
        
        if query_lower == sentence_lower:
            return 1.0
        elif query_lower in sentence_lower:
            return min(1.0, len(query_lower) / len(sentence_lower) * 2)
        else:
            query_words = set(re.findall(r'\b\w+\b', query_lower))
            sentence_words = set(re.findall(r'\b\w+\b', sentence_lower))
            
            if not query_words:
                return 0.0
            
            overlap = query_words & sentence_words
            return len(overlap) / len(query_words)
    
    def _calculate_clear_score(self, semantic_sim, lexical_score, weights, word_count):
        """
        CLEAR hybrid scoring: s_final = λ_emb·s_emb + λ_lex·s_lex - penalty
        """
        lambda_emb, lambda_lex = weights
        
        final_score = (lambda_emb * semantic_sim) + (lambda_lex * lexical_score)
        
        if word_count < self.SHORT_SENTENCE_THRESHOLD:
            penalty = self.SHORT_SENTENCE_PENALTY * (
                self.SHORT_SENTENCE_THRESHOLD - word_count
            ) / self.SHORT_SENTENCE_THRESHOLD
            final_score -= penalty
        
        return max(0.0, min(1.0, final_score))
    
    def _check_context_relevance(self, context_text, query, theme_context=None):
        """Check if context sentence is relevant to query/theme"""
        try:
            query_embedding = self.model.encode([query])
            context_embedding = self.model.encode([context_text])
            query_similarity = cosine_similarity(query_embedding, context_embedding)[0][0]
            
            theme_similarity = 0.0
            if theme_context:
                theme_embedding = self.model.encode([theme_context])
                theme_similarity = cosine_similarity(theme_embedding, context_embedding)[0][0]
            
            return max(query_similarity, theme_similarity) >= self.CONTEXT_RELEVANCE_THRESHOLD
        except:
            return False
    
    def _get_expanded_context(self, chapter_num, sentence_num, query, theme_context=None):
        """Get context sentences with strict deduplication"""
        context = {
            'prev_sentences': [],
            'next_sentences': [],
            'prev_relevant_count': 0,
            'next_relevant_count': 0
        }
        
        chapter_sentences = self.chapters_df[
            self.chapters_df['chapter_number'] == chapter_num
        ].sort_values('sentence_number')
        
        current_idx = None
        for idx, row in chapter_sentences.iterrows():
            if row['sentence_number'] == sentence_num:
                current_idx = idx
                break
        
        if current_idx is None:
            return context
        
        chapter_list = chapter_sentences.index.tolist()
        current_pos = chapter_list.index(current_idx)
        
        # Expand backward
        for i in range(1, self.MAX_CONTEXT_EXPANSION + 1):
            if current_pos - i < 0:
                break
            
            prev_idx = chapter_list[current_pos - i]
            prev_row = self.chapters_df.loc[prev_idx]
            prev_id = self._get_passage_id(prev_row['chapter_number'], prev_row['sentence_number'])
            
            if prev_id in self.used_passages:
                break
            
            is_relevant = self._check_context_relevance(
                prev_row['sentence_text'], query, theme_context
            )
            
            context['prev_sentences'].append({
                'sentence_number': prev_row['sentence_number'],
                'sentence_text': prev_row['sentence_text'],
                'is_relevant': is_relevant,
                'distance': i
            })
            
            if is_relevant:
                context['prev_relevant_count'] += 1
            else:
                break
        
        context['prev_sentences'].reverse()
        
        # Expand forward
        for i in range(1, self.MAX_CONTEXT_EXPANSION + 1):
            if current_pos + i >= len(chapter_list):
                break
            
            next_idx = chapter_list[current_pos + i]
            next_row = self.chapters_df.loc[next_idx]
            next_id = self._get_passage_id(next_row['chapter_number'], next_row['sentence_number'])
            
            if next_id in self.used_passages:
                break
            
            is_relevant = self._check_context_relevance(
                next_row['sentence_text'], query, theme_context
            )
            
            context['next_sentences'].append({
                'sentence_number': next_row['sentence_number'],
                'sentence_text': next_row['sentence_text'],
                'is_relevant': is_relevant,
                'distance': i
            })
            
            if is_relevant:
                context['next_relevant_count'] += 1
            else:
                break
        
        return context
    
    def _mark_passages_as_used(self, chapter_num, sentence_num, context):
        """Mark all passages (main + context) as globally used"""
        self.used_passages.add(self._get_passage_id(chapter_num, sentence_num))
        
        for sent in context.get('prev_sentences', []):
            self.used_passages.add(self._get_passage_id(chapter_num, sent['sentence_number']))
        
        for sent in context.get('next_sentences', []):
            self.used_passages.add(self._get_passage_id(chapter_num, sent['sentence_number']))
    
    def _retrieve_passages(self, query, top_k=9):
        """CLEAR-based hybrid retrieval: semantic + lexical only"""
        self.used_passages = set()
        
        query_words = re.findall(r'\b[a-zA-ZÀ-ÿñÑ]+\b', query)
        query_length = len(query_words)
        weights = self._compute_dynamic_weights(query_length)
        lambda_emb, lambda_lex = weights
        
        query_embedding = self.model.encode([query])
        semantic_similarities = cosine_similarity(query_embedding, self.chapter_embeddings)[0]
        
        results = []
        
        for idx, semantic_sim in enumerate(semantic_similarities):
            if semantic_sim < self.MIN_SEMANTIC_THRESHOLD:
                continue
            
            row = self.chapters_df.iloc[idx]
            passage_id = self._get_passage_id(row['chapter_number'], row['sentence_number'])
            
            if passage_id in self.used_passages:
                continue
            
            lexical_score = self._compute_lexical_score(query, row['sentence_text'])
            
            if lexical_score >= 0.95:
                match_type = 'exact'
            elif lexical_score >= 0.3:
                match_type = 'partial_lexical'
            else:
                match_type = 'semantic'
            
            try:
                context = self._get_expanded_context(
                    row['chapter_number'], row['sentence_number'], query
                )
                has_relevant_context = (
                    context['prev_relevant_count'] > 0 or context['next_relevant_count'] > 0
                )
            except:
                context = {
                    'prev_sentences': [], 'next_sentences': [],
                    'prev_relevant_count': 0, 'next_relevant_count': 0
                }
                has_relevant_context = False
            
            final_score = self._calculate_clear_score(
                semantic_sim, lexical_score, weights, row['sentence_word_count']
            )
            
            results.append({
                'index': idx,
                'chapter_number': row['chapter_number'],
                'chapter_title': row['chapter_title'],
                'sentence_number': row['sentence_number'],
                'sentence_text': row['sentence_text'],
                'semantic_score': semantic_sim,
                'lexical_score': lexical_score,
                'final_score': final_score,
                'match_type': match_type,
                'word_count': row['sentence_word_count'],
                'context': context,
                'has_relevant_context': has_relevant_context,
                'total_context_sentences': len(context['prev_sentences']) + len(context['next_sentences']),
                'weights': {'lambda_emb': lambda_emb, 'lambda_lex': lambda_lex}
            })
        
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        for result in results[:top_k]:
            self._mark_passages_as_used(
                result['chapter_number'], result['sentence_number'], result['context']
            )
        
        chapter_counts = {}
        final_results = []
        
        for result in results:
            ch_num = result['chapter_number']
            count = chapter_counts.get(ch_num, 0)
            
            if count < 3:
                final_results.append(result)
                chapter_counts[ch_num] = count + 1
                
                if len(final_results) >= top_k:
                    break
        
        return final_results[:top_k]
    
    def _get_thematic_classification(self, passages, query):
        """Classify passages by themes"""
        if not passages:
            return passages, False, 0.0
        
        thematic_results = []
        
        for passage in passages:
            sentence_embedding = self.model.encode([passage['sentence_text']])
            theme_similarities = cosine_similarity(sentence_embedding, self.theme_embeddings)[0]
            
            matching_themes = []
            for idx, similarity in enumerate(theme_similarities):
                if similarity >= self.THEMATIC_THRESHOLD:
                    theme_row = self.themes_df.iloc[idx]
                    matching_themes.append({
                        'tagalog_title': theme_row['Tagalog Title'],
                        'meaning': theme_row['Meaning'],
                        'confidence': similarity
                    })
            
            matching_themes.sort(key=lambda x: x['confidence'], reverse=True)
            
            enhanced = passage.copy()
            if matching_themes:
                enhanced['themes'] = matching_themes[:2]
                enhanced['primary_theme'] = matching_themes[0]
                enhanced['has_theme'] = True
            else:
                enhanced['themes'] = []
                enhanced['primary_theme'] = None
                enhanced['has_theme'] = False
            
            thematic_results.append(enhanced)
        
        sentences_with_themes = sum(1 for s in thematic_results if s['has_theme'])
        thematic_coverage = sentences_with_themes / len(thematic_results) if thematic_results else 0
        
        avg_theme_conf = 0.0
        if sentences_with_themes > 0:
            theme_confs = [s['primary_theme']['confidence'] for s in thematic_results if s['has_theme']]
            avg_theme_conf = np.mean(theme_confs)
        
        has_themes = thematic_coverage >= 0.3 and avg_theme_conf >= self.THEMATIC_THRESHOLD
        
        return thematic_results, has_themes, avg_theme_conf
    
    def query(self, user_query):
        """Main query interface"""
        is_valid, validation_info = self.query_analyzer.validate_filipino_query(user_query)
        
        if not is_valid:
            return {
                'type': 'invalid_filipino',
                'validation_info': validation_info,
                'message': f"Invalid Filipino query: {validation_info['reason']}"
            }
        
        query_analysis = self.query_analyzer.analyze_query_words(user_query)
        query_length = len(re.findall(r'\b[a-zA-ZÀ-ÿñÑ]+\b', user_query))
        
        passages = self._retrieve_passages(user_query)
        
        if not passages:
            return {
                'type': 'no_matches',
                'message': "No matches found in Noli Me Tangere",
                'query_analysis': query_analysis,
                'query_length': query_length
            }
        
        thematic_passages, has_themes, avg_theme_conf = self._get_thematic_classification(
            passages, user_query
        )
        
        avg_semantic = np.mean([p['semantic_score'] for p in passages])
        avg_lexical = np.mean([p['lexical_score'] for p in passages])
        avg_final = np.mean([p['final_score'] for p in passages])
        
        exact_count = sum(1 for p in passages if p['match_type'] == 'exact')
        partial_lex_count = sum(1 for p in passages if p['match_type'] == 'partial_lexical')
        semantic_only_count = sum(1 for p in passages if p['match_type'] == 'semantic')
        context_count = sum(1 for p in thematic_passages if p['has_relevant_context'])
        total_context = sum(p.get('total_context_sentences', 0) for p in thematic_passages)
        
        weights = passages[0]['weights'] if passages else {}
        
        return {
            'type': 'success',
            'subtype': 'thematic' if has_themes else 'semantic',
            'results': thematic_passages,
            'query_length': query_length,
            'weights': weights,
            'semantic_confidence': avg_semantic,
            'lexical_confidence': avg_lexical,
            'final_confidence': avg_final,
            'theme_confidence': avg_theme_conf,
            'exact_matches': exact_count,
            'partial_lexical_matches': partial_lex_count,
            'semantic_only_matches': semantic_only_count,
            'context_matches': context_count,
            'total_results': len(thematic_passages),
            'total_context_sentences': total_context,
            'query_analysis': query_analysis
        }
    
    def display_results(self, response, query=""):
        """Display results with rich formatting"""
        result_type = response['type']
        
        if result_type == 'invalid_filipino':
            validation_info = response['validation_info']
            
            none_table = Table(show_header=False, box=box.HEAVY, border_style="red", width=20)
            none_table.add_column("Result", style="bold red", justify="center")
            none_table.add_row("none")
            self.console.print(none_table)
            
            validation_panel = Panel(
                f"{response['message']}\n\n"
                f"Analysis:\n"
                f"  Total words: {validation_info['total_words']}\n"
                f"  Valid Filipino: {validation_info['valid_words']}\n"
                f"  Invalid: {', '.join(validation_info['invalid_words']) if validation_info['invalid_words'] else 'N/A'}\n"
                f"  Valid ratio: {validation_info['valid_ratio']:.1%}",
                title="Invalid Query",
                style="red",
                box=box.ROUNDED
            )
            self.console.print(validation_panel)
            return
        
        if result_type != 'success':
            error_panel = Panel(
                f"{response['message']}",
                title="No Results",
                style="yellow",
                box=box.ROUNDED
            )
            self.console.print(error_panel)
            
            if 'query_analysis' in response:
                self._display_query_analysis(response['query_analysis'])
            return
        
        if 'query_analysis' in response:
            self._display_query_analysis(response['query_analysis'])
        
        results = response['results']
        subtype = response['subtype']
        has_themes = subtype == 'thematic'
        
        weights = response.get('weights', {})
        query_len = response.get('query_length', 0)
        
        metrics_text = (
            f"CLEAR Hybrid (Semantic + Lexical Only) | Query Length: {query_len} words\n"
            f"Weights: λ_emb={weights.get('lambda_emb', 0):.2f} | λ_lex={weights.get('lambda_lex', 0):.2f}\n"
            f"Semantic: {response['semantic_confidence']:.1%} | "
            f"Lexical: {response['lexical_confidence']:.1%} | "
            f"Final: {response['final_confidence']:.1%}\n"
            f"Exact: {response['exact_matches']} | "
            f"Partial: {response.get('partial_lexical_matches', 0)} | "
            f"Semantic-only: {response.get('semantic_only_matches', 0)}\n"
            f"Context: {response.get('context_matches', 0)} | "
            f"Total Context: {response.get('total_context_sentences', 0)} | "
            f"Results: {response['total_results']}"
        )
        if response['theme_confidence'] > 0:
            metrics_text += f"\nThematic: {response['theme_confidence']:.1%}"
        
        header_text = Text(f"Results for: '{query}'\n{metrics_text}", style="bold white")
        header_panel = Panel(
            Align.center(header_text),
            style="bright_blue",
            box=box.DOUBLE,
            padding=(1, 2)
        )
        self.console.print(header_panel)
        
        for i, result in enumerate(results, 1):
            self.console.print(f"\nResult {i}", style="bold cyan")
            
            main_table = Table(
                show_header=True,
                header_style="bold magenta",
                border_style="bright_blue",
                box=box.ROUNDED,
                padding=(0, 1),
                expand=True
            )
            
            main_table.add_column("Book", style="cyan", width=15)
            main_table.add_column("Ch", style="bright_green", width=4, justify="center")
            main_table.add_column("S#", style="yellow", width=4, justify="center")
            main_table.add_column("Semantic", style="bright_cyan", width=9, justify="center")
            main_table.add_column("Lexical", style="bright_yellow", width=9, justify="center")
            main_table.add_column("Final", style="bright_white", width=9, justify="center")
            main_table.add_column("Type", style="bright_green", width=14, justify="center")
            main_table.add_column("Ctx", style="dim white", width=5, justify="center")
            
            match_type = result['match_type']
            if match_type == 'exact':
                type_display = "Exact Match"
            elif match_type == 'partial_lexical':
                type_display = "Partial Lexical"
            else:
                type_display = "Semantic"
            
            context_count = result.get('total_context_sentences', 0)
            
            main_table.add_row(
                "Noli Me Tangere",
                str(result['chapter_number']),
                str(result['sentence_number']),
                f"{result['semantic_score']:.1%}",
                f"{result['lexical_score']:.1%}",
                f"{result['final_score']:.1%}",
                type_display,
                str(context_count) if context_count > 0 else "-"
            )
            
            self.console.print(main_table)
            
            chapter_panel = Panel(
                f"{result['chapter_title']}",
                style="bright_green",
                box=box.SIMPLE
            )
            self.console.print(chapter_panel)
            
            content_panel = Panel(
                result['sentence_text'],
                style="white",
                box=box.ROUNDED,
                padding=(1, 2)
            )
            self.console.print(content_panel)
            
            context = result.get('context', {})
            prev_sentences = context.get('prev_sentences', [])
            next_sentences = context.get('next_sentences', [])
            
            if prev_sentences or next_sentences:
                context_table = Table(
                    title=f"Context ({len(prev_sentences) + len(next_sentences)} sentences)",
                    show_header=True,
                    header_style="bold yellow",
                    border_style="yellow",
                    box=box.SIMPLE,
                    expand=True
                )
                
                context_table.add_column("Position", style="yellow", width=10)
                context_table.add_column("S#", style="bright_yellow", width=4, justify="center")
                context_table.add_column("Dist", style="dim yellow", width=5, justify="center")
                context_table.add_column("Rel", style="bright_green", width=4, justify="center")
                context_table.add_column("Content", style="white", min_width=40)
                
                for sent in prev_sentences:
                    rel_icon = "Y" if sent['is_relevant'] else "N"
                    context_table.add_row(
                        "Previous",
                        str(sent['sentence_number']),
                        f"-{sent['distance']}",
                        rel_icon,
                        sent['sentence_text']
                    )
                
                for sent in next_sentences:
                    rel_icon = "Y" if sent['is_relevant'] else "N"
                    context_table.add_row(
                        "Next",
                        str(sent['sentence_number']),
                        f"+{sent['distance']}",
                        rel_icon,
                        sent['sentence_text']
                    )
                
                self.console.print(context_table)
            
            if has_themes and result.get('has_theme'):
                primary_theme = result['primary_theme']
                
                theme_table = Table(
                    title="Thematic Analysis",
                    show_header=True,
                    header_style="bold magenta",
                    border_style="magenta",
                    box=box.SIMPLE,
                    expand=True
                )
                
                theme_table.add_column("Tagalog Title", style="bright_magenta", width=25)
                theme_table.add_column("Meaning", style="magenta", min_width=40)
                theme_table.add_column("Confidence", style="bright_cyan", width=12, justify="center")
                
                theme_table.add_row(
                    primary_theme['tagalog_title'],
                    primary_theme['meaning'],
                    f"{primary_theme['confidence']:.1%}"
                )
                
                self.console.print(theme_table)
            
            if i < len(results):
                self.console.print("─" * 100, style="dim blue")
        
        chapters_found = len(set(r['chapter_number'] for r in results))
        exact_count = sum(1 for r in results if r['match_type'] == 'exact')
        partial_count = sum(1 for r in results if r['match_type'] == 'partial_lexical')
        semantic_count = sum(1 for r in results if r['match_type'] == 'semantic')
        context_count = sum(1 for r in results if r.get('has_relevant_context', False))
        theme_count = sum(1 for r in results if r.get('has_theme', False))
        total_context = response.get('total_context_sentences', 0)
        
        classification = "Thematic Analysis" if has_themes else "Semantic Search"
        
        summary_parts = [
            f"{classification}",
            f"{len(results)} sentences from {chapters_found} chapters",
            f"{exact_count} exact",
            f"{partial_count} partial",
            f"{semantic_count} semantic",
            f"{context_count} with context",
            f"{total_context} total context"
        ]
        
        if has_themes:
            summary_parts.append(f"{theme_count} with themes")
        
        summary = " | ".join(summary_parts)
        footer_panel = Panel(
            Align.center(Text(summary, style="bold white")),
            style="bright_blue",
            box=box.DOUBLE,
            padding=(1, 2)
        )
        self.console.print(footer_panel)
    
    def _display_query_analysis(self, query_analysis):
        """Display query word analysis"""
        if not query_analysis:
            return
        
        analysis_table = Table(
            title="Query Word Analysis",
            show_header=True,
            header_style="bold cyan",
            border_style="cyan",
            box=box.ROUNDED,
            expand=True
        )
        
        analysis_table.add_column("Word", style="bright_white", width=20)
        analysis_table.add_column("Frequency", style="bright_yellow", width=15, justify="right")
        analysis_table.add_column("Type", style="bright_cyan", width=15, justify="center")
        
        for item in query_analysis:
            if item['is_stopword']:
                word_type = "Stopword"
            elif item['is_content_word']:
                word_type = "Content"
            else:
                word_type = "Common"
            
            analysis_table.add_row(
                item['word'],
                f"{item['frequency']:.6f}",
                word_type
            )
        
        self.console.print("\n")
        self.console.print(analysis_table)
        self.console.print("\n")

if __name__ == "__main__":
    system = CleanNoliSystem()
    
    welcome_panel = Panel(
        Align.center(Text(
            "Noli Me Tangere Clean CLEAR Hybrid System\n"
            "XLM-RoBERTa Semantic + Lexical Matching\n"
            "No Stem Matching | Strict Deduplication | Dynamic Weighting",
            style="bold white"
        )),
        style="bright_green",
        box=box.HEAVY
    )
    system.console.print(welcome_panel)
    
    while True:
        system.console.print("\n" + "─" * 80, style="dim")
        user_input = system.console.input("[bold cyan]Enter query (or 'exit'): [/bold cyan]").strip()
        
        if user_input.lower() == 'exit':
            goodbye_panel = Panel(
                Align.center(Text("Thank you for using the system!", style="bold green")),
                style="bright_green",
                box=box.ROUNDED
            )
            system.console.print(goodbye_panel)
            break
        
        if not user_input:
            system.console.print("[red]Please enter a valid query.[/red]")
            continue
        
        system.console.print(f"[dim]Processing '{user_input}'...[/dim]")
        
        try:
            response = system.query(user_input)
            system.display_results(response, user_input)
        except Exception as e:
            error_panel = Panel(
                f"System error: {str(e)}\nTry a different query.",
                title="Error",
                style="red",
                box=box.ROUNDED
            )
            system.console.print(error_panel)