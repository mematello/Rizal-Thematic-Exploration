"""
Main CLI application logic.
Combines Engine and UI.
"""
from .engine import RizalEngine
from .loader import DataLoader
from .ui import RizalUI
from .config import EMBEDDING_MODEL_NAME

def run_cli():
    """Run the interactive CLI."""
    ui = RizalUI()
    ui.print_welcome()
    
    try:
        ui.print_loading("Initializing engine components...")
        loader = DataLoader(EMBEDDING_MODEL_NAME)
        # Load data (this might take time)
        # Instantiate Engine dependencies
        from .query_analyzer import QueryAnalyzer
        qa = QueryAnalyzer()
        
        ui.print_loading("Loading data/models (this may take a moment)...")
        # Pass the QueryAnalyzer instance to the loader
        loader.load(query_analyzer=qa)
        
        engine = RizalEngine(loader)
        engine.query_analyzer = qa # Ensure sharing same instance/stopwords
        
        ui.console.print("\n[bold green]System Ready![/bold green]\n")
        
        while True:
            query = ui.console.input("[bold cyan]Enter query (or 'exit'):[/bold cyan] ").strip()
            
            if query.lower() in ('exit', 'quit'):
                ui.console.print("[yellow]Goodbye![/yellow]")
                break
                
            if not query:
                continue
                
            ui.print_loading("Analyzing...")
            response = engine.query(query)
            ui.display_results(response)
            
    except KeyboardInterrupt:
        ui.console.print("\n[yellow]Interrupted by user.[/yellow]")
    except Exception as e:
        ui.print_error("Critical System Error", str(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_cli()
