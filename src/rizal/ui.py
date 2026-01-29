"""
User Interface module.
Handles all input/output, formatting, and plotting.
"""
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich import box
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class RizalUI:
    def __init__(self):
        self.console = Console()

    def print_welcome(self):
        title = Text("Rizal Thematic Exploration Engine", style="bold white")
        subtitle = Text("Dynamic Dual-Formula CLEAR System", style="cyan")
        
        self.console.print(Panel(
            Align.center(title + "\n" + subtitle),
            border_style="green",
            box=box.DOUBLE
        ))

    def print_loading(self, message):
        self.console.print(f"[cyan]{message}[/cyan]")

    def print_error(self, title, message, details=None):
        self.console.print(f"[bold red]{title}[/bold red]")
        self.console.print(f"[red]{message}[/red]")
        if details:
            self.console.print(details)

    def display_results(self, response):
        """Main display router."""
        status = response.get('status')
        if status == 'error':
            self.print_error(response.get('error_type'), response.get('message'))
            self.display_suggestions(response.get('suggestions', []))
        elif status == 'empty':
            self.console.print("[yellow]No matches found.[/yellow]")
            self.display_suggestions(response.get('suggestions', []))
        elif status == 'success':
            self._display_success(response)

    def _display_success(self, response):
        results_by_book = response['results_by_book']
        query_analysis = response.get('query_analysis')
        
        # Display Query Analysis
        self._display_analysis(query_analysis)
        
        # Display Results
        for book_key, data in results_by_book.items():
            self.console.print(f"\n[bold green]== Results for {book_key.upper()} ==[/bold green]")
            
            # Metrics
            metrics = data['metrics']
            m_text = (
                f"Avg Semantic: {metrics['avg_semantic']:.2f} | "
                f"Avg Lexical: {metrics['avg_lexical']:.2f} | "
                f"Avg Final: {metrics['avg_final']:.2f}"
            )
            self.console.print(Panel(m_text, title="Metrics", border_style="blue"))
            
            # Passages
            self._display_passages(data['results'])

        # Next Steps
        if response.get('next_queries'):
            self.console.print("\n[bold cyan]Next Recommended Actions:[/bold cyan]")
            for i, q in enumerate(response['next_queries'], 1):
                self.console.print(f"{i}. {q}")

    def _display_analysis(self, analysis):
        if not analysis: return
        t = Table(title="Query Analysis", box=box.SIMPLE)
        t.add_column("Word")
        t.add_column("Type")
        t.add_column("Weight")
        
        for w in analysis:
            w_type = "Stopword" if w['is_stopword'] else "Content"
            t.add_row(w['word'], w_type, f"{w['semantic_weight']:.2f}")
        
        self.console.print(t)

    def _display_passages(self, passages):
        for p in passages:
            # Theme info
            theme_str = ""
            if p.get('has_theme'):
                t = p['primary_theme']
                theme_str = f"\n[italic magenta]Theme: {t['tagalog_title']} ({t['confidence']:.2f})[/italic magenta]"
            
            # Text
            text = f"[bold]{p['sentence_text']}[/bold]{theme_str}"
            
            # Context
            ctx_before = ""
            if p.get('context'):
                for s in p['context']['prev_sentences']:
                    if s['is_relevant']:
                        ctx_before += f"[dim]{s['sentence_text']}[/dim] "
            
            ctx_after = ""
            if p.get('context'):
                for s in p['context']['next_sentences']:
                    if s['is_relevant']:
                        ctx_after += f" [dim]{s['sentence_text']}[/dim]"
            
            full_text = f"{ctx_before}{text}{ctx_after}"
            
            self.console.print(Panel(
                full_text,
                title=f"Chapter {p['chapter_number']}: {p['chapter_title']}",
                subtitle=f"Score: {p.get('final_score', 0):.3f}",
                border_style="white"
            ))

    def display_suggestions(self, suggestions):
        if not suggestions: return
        self.console.print("\n[bold yellow]Suggestions:[/bold yellow]")
        for s in suggestions:
            self.console.print(f"- {s}")

    def plot_embeddings(self, vectors, labels, title="Embedding Visualization"):
        """Plot embeddings provided by engine."""
        if not vectors or len(vectors) < 2:
            return
            
        arr = np.array(vectors)
        if arr.shape[0] < 4:
            reducer = PCA(n_components=2, random_state=42)
        else:
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(5, arr.shape[0]-1))
            
        coords = reducer.fit_transform(arr)
        
        plt.figure(figsize=(10, 8), dpi=100)
        x, y = coords[:, 0], coords[:, 1]
        plt.scatter(x, y, c='tab:blue', alpha=0.6)
        
        for i, lab in enumerate(labels):
            plt.annotate(lab, (x[i], y[i]), fontsize=9, alpha=0.8)
            
        plt.title(title)
        plt.tight_layout()
        plt.show(block=False) 
        # Using block=False to not freeze CLI, but usually plt.show() blocks.
        # For a CLI loop, we might want to pause or just show and close.
        plt.pause(0.1)
