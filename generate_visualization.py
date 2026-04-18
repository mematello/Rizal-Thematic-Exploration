import os

html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Sentence Search Processing</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>
        mermaid.initialize({ startOnLoad: true, theme: 'dark' });
    </script>
    <style>
        body { font-family: Arial, sans-serif; background-color: #1a1a1a; color: #f0f0f0; margin: 40px; }
        .diagram-container { background-color: #2d2d2d; border-radius: 8px; padding: 20px; margin-bottom: 40px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        h1, h2 { color: #4db8ff; }
        h1 { text-align: center; border-bottom: 2px solid #4db8ff; padding-bottom: 10px; }
    </style>
</head>
<body>
    <h1>Multi-Sentence Query Engine Architecture</h1>
    
    <div class="diagram-container">
        <h2>1. Global Input Splitting</h2>
        <div class="mermaid">
        graph TD
            A[User Input: "Query 1. Query 2"] -->|Detected Delimiter '.'| B(Split String)
            B --> C{Validator}
            C -->|Trim Whitespace| D[Query Segment 1: "Query 1"]
            C -->|Trim Whitespace| E[Query Segment 2: "Query 2"]
            C -->|Empty Segment `..`| F[/Ignored/]
            style A fill:#ff5e5e,stroke:#fff,stroke-width:2px,color:#fff
            style B fill:#5ab4ff,stroke:#fff,stroke-width:2px,color:#fff
            style D fill:#68ff68,stroke:#fff,stroke-width:2px,color:#333
            style E fill:#68ff68,stroke:#fff,stroke-width:2px,color:#333
        </div>
    </div>

    <div class="diagram-container">
        <h2>2. Query 1 Processing Pipeline</h2>
        <div class="mermaid">
        graph TD
            Start1[Input Segment: "Query 1"] --> Enc1[SentenceTransformer Embedding]
            Start1 --> Lex1[Lexical Keyword Analysis]
            
            Enc1 -->|Semantic Vector| Retrieve1{Candidate Retrieval}
            Lex1 -->|TF-IDF Overlap| Retrieve1
            
            Retrieve1 -->|Search Space| Noli1[(Noli Me Tangere DB)]
            Retrieve1 -->|Search Space| Fili1[(El Filibusterismo DB)]
            
            Noli1 --> Score1[Score Computation Engine]
            Fili1 --> Score1
            
            Score1 -->|Lam_Lex & Lam_Sem| Math1("Score = (Final_Score) + (Precision * 0.5)")
            Math1 --> Rank1[Ranked List: Query 1]
            
            style Start1 fill:#68ff68,stroke:#fff,stroke-width:2px,color:#333
            style Noli1 fill:#ffc107,stroke:#fff,stroke-width:2px,color:#333
            style Fili1 fill:#ffc107,stroke:#fff,stroke-width:2px,color:#333
            style Math1 fill:#ff5e5e,stroke:#fff,stroke-width:2px,color:#fff
            style Rank1 fill:#5ab4ff,stroke:#fff,stroke-width:2px,color:#fff
        </div>
    </div>

    <div class="diagram-container">
        <h2>3. Query 2 Processing Pipeline</h2>
        <div class="mermaid">
        graph TD
            Start2[Input Segment: "Query 2"] --> Enc2[SentenceTransformer Embedding]
            Start2 --> Lex2[Lexical Keyword Analysis]
            
            Enc2 -->|Semantic Vector| Retrieve2{Candidate Retrieval}
            Lex2 -->|TF-IDF Overlap| Retrieve2
            
            Retrieve2 -->|Search Space| Noli2[(Noli Me Tangere DB)]
            Retrieve2 -->|Search Space| Fili2[(El Filibusterismo DB)]
            
            Noli2 --> Score2[Score Computation Engine]
            Fili2 --> Score2
            
            Score2 -->|Lam_Lex & Lam_Sem| Math2("Score = (Final_Score) + (Precision * 0.5)")
            Math2 --> Rank2[Ranked List: Query 2]
            
            style Start2 fill:#68ff68,stroke:#fff,stroke-width:2px,color:#333
            style Noli2 fill:#ffc107,stroke:#fff,stroke-width:2px,color:#333
            style Fili2 fill:#ffc107,stroke:#fff,stroke-width:2px,color:#333
            style Math2 fill:#ff5e5e,stroke:#fff,stroke-width:2px,color:#fff
            style Rank2 fill:#5ab4ff,stroke:#fff,stroke-width:2px,color:#fff
        </div>
    </div>

</body>
</html>
"""

filepath = os.path.join(os.path.dirname(__file__), "multi_query_visualization.html")

with open(filepath, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Visualization generated successfully at: {filepath}")
print("Please open the HTML file in any web browser to view the diagrams.")
