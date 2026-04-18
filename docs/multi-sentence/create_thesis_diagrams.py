import os
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except IOError:
        return ImageFont.load_default()

def draw_graph(filename, nodes_dict, edges, width=1200, height=900, title=""):
    img = Image.new('RGBA', (width, height), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    font_title = get_font(28)
    font_node = get_font(16)
    font_label = get_font(14)
    
    # draw.text((30, 20), title, fill="#68ff68", font=font_title)
    
    # Draw edges
    for start, end, label in edges:
        sx, sy, sw, sh, _, _ = nodes_dict[start]
        ex, ey, ew, eh, _, _ = nodes_dict[end]
        
        start_pt = (sx + sw//2, sy + sh)
        end_pt = (ex + ew//2, ey)
        
        draw.line([start_pt, end_pt], fill="#888888", width=3)
        if label:
            mx = (start_pt[0] + end_pt[0])//2
            my = (start_pt[1] + end_pt[1])//2
            draw.text((mx + 10, my - 5), label, fill="black", font=font_label)
            
    # Draw nodes
    for k, (x, y, w, h, bg, text) in nodes_dict.items():
        draw.rectangle([x, y, x+w, y+h], fill=bg, outline="white", width=2)
        
        lines = text.split('\n')
        line_heights = [draw.textbbox((0,0), line, font=font_node)[3] for line in lines]
        total_h = sum(line_heights) + (len(lines)-1)*4
        
        cy = y + (h - total_h)//2
        for ln_idx, line in enumerate(lines):
            bbox = draw.textbbox((0,0), line, font=font_node)
            cx = x + (w - bbox[2])//2
            draw.text((cx, cy), line, fill="black" if bg not in ["#2c2c2c", "black"] else "white", font=font_node)
            cy += line_heights[ln_idx] + 4

    img.save(os.path.join(OUT_DIR, filename))
    print(f"Generated {filename}")

# --- DIAGRAM 1: GLOBAL SPLITTING ---
nodes_1 = {
    'input': (300, 100, 600, 80, "#ff5e5e", "User Input:\n\"Pagpapatayo ng paaralan. Pagtutol ng mga prayle\""),
    'detect': (450, 250, 300, 60, "black", "Detector: Period (.) Found"),
    'split': (450, 380, 300, 60, "#5ab4ff", "String Operations\nSplit Strings -> Trim Whitespace"),
    'q1': (150, 520, 400, 80, "#68ff68", "Query 1 Segment:\n\"Pagpapatayo ng paaralan\""),
    'q2': (650, 520, 400, 80, "#68ff68", "Query 2 Segment:\n\"Pagtutol ng mga prayle\""),
    'route1': (200, 680, 300, 60, "#dddddd", "Independent Routing Queue 1"),
    'route2': (700, 680, 300, 60, "#dddddd", "Independent Routing Queue 2"),
}
edges_1 = [
    ('input', 'detect', ""), ('detect', 'split', "Executes Array mapping"),
    ('split', 'q1', "Isolates segment"), ('split', 'q2', "Isolates segment"),
    ('q1', 'route1', "Push to Pipeline Iteration 1"), ('q2', 'route2', "Push to Pipeline Iteration 2")
]
draw_graph("1_global_splitting.png", nodes_1, edges_1, 1200, 850, "1. GLOBAL SPLITTING PROCESS (Executed Once)")

# --- DIAGRAMS 2-4: COMPUTATION RESULTS ---
def generate_scoring_diagram(filename, title, q1_text, q2_text, score1, score2, noli_rank1, fili_rank2):
    nodes = {
        # Q1 Track
        'q1_head': (50, 100, 500, 60, "#68ff68", f"[Query 1 Processing]\n{q1_text}"),
        'q1_db': (50, 220, 500, 60, "#dddddd", "Retrieval: Search Noli Me Tangere & El Filibusterismo DBs"),
        'q1_scoring': (50, 340, 500, 100, "black", f"Feature Scoring Breakdown (Q1)\n---------------------------------\nLexical Similarity: {score1['lex']}%\nSemantic Similarity (Model): {score1['sem']}%"),
        'q1_math': (50, 500, 500, 80, "#ff5e5e", f"Score Computation\nDerived Final Rank = {score1['final']}%"),
        'q1_rank': (50, 650, 500, 100, "#5ab4ff", f"[Top Extracted Ranking - Q1]\nRelevance: {score1['final']}%\nNovel: {noli_rank1}"),
        
        # Q2 Track
        'q2_head': (650, 100, 500, 60, "#68ff68", f"[Query 2 Processing]\n{q2_text}"),
        'q2_db': (650, 220, 500, 60, "#dddddd", "Retrieval: Search Noli Me Tangere & El Filibusterismo DBs"),
        'q2_scoring': (650, 340, 500, 100, "black", f"Feature Scoring Breakdown (Q2)\n---------------------------------\nLexical Similarity: {score2['lex']}%\nSemantic Similarity (Model): {score2['sem']}%"),
        'q2_math': (650, 500, 500, 80, "#ff5e5e", f"Score Computation\nDerived Final Rank = {score2['final']}%"),
        'q2_rank': (650, 650, 500, 100, "#5ab4ff", f"[Top Extracted Ranking - Q2]\nRelevance: {score2['final']}%\nNovel: {fili_rank2}"),
    }
    edges = [
        ('q1_head', 'q1_db', ""), ('q1_db', 'q1_scoring', "Extract Vectors & Tokens"),
        ('q1_scoring', 'q1_math', "Apply Weight Hybrid Equation"), ('q1_math', 'q1_rank', "Independently Sort Highest Match"),
        ('q2_head', 'q2_db', ""), ('q2_db', 'q2_scoring', "Extract Vectors & Tokens"),
        ('q2_scoring', 'q2_math', "Apply Weight Hybrid Equation"), ('q2_math', 'q2_rank', "Independently Sort Highest Match")
    ]
    draw_graph(filename, nodes, edges, 1200, 850, title)

# 2. Example 1 (Paaralan at Pagtutol ng mga Prayle)
generate_scoring_diagram("2_ex1_paaralan.png", "2. RESULT COMPUTATION (Pagpapatayo ng paaralan. Pagtutol ng mga prayle)", 
                         "Pagpapatayo ng paaralan", "Pagtutol ng mga prayle",
                         {'lex': 100, 'sem': 9, 'final': 55}, {'lex': 100, 'sem': 9, 'final': 55},
                         "Noli Me Tangere (Kab 25)", "Noli Me Tangere (Kab 35)")

# 3. Example 2 (Pagkawala ng mga anak. Pagkabaliw ni Sisa)
generate_scoring_diagram("3_ex2_sisa.png", "3. RESULT COMPUTATION (Pagkawala ng mga anak. Pagkabaliw ni Sisa)", 
                         "Pagkawala ng mga anak", "Pagkabaliw ni Sisa",
                         {'lex': 100, 'sem': 1, 'final': 51}, {'lex': 10, 'sem': 4, 'final': 7},
                         "Noli Me Tangere (Kab 24)", "Noli Me Tangere (Kab 39)")

# 4. Example 3 (Pangarap ni Basilio. Pag-aaral ng medisina)
generate_scoring_diagram("4_ex3_basilio.png", "4. RESULT COMPUTATION (Pangarap ni Basilio. Pag-aaral ng medisina)", 
                         "Pangarap ni Basilio", "Pag-aaral ng medisina",
                         {'lex': 100, 'sem': 16, 'final': 58}, {'lex': 100, 'sem': 19, 'final': 60},
                         "El Filibusterismo (Kab 6)", "El Filibusterismo (Kab 6)")

# --- DIAGRAM 5: HOW SCORE IS GENERATED ---
def generate_math_diagram():
    nodes = {
        'input_query': (250, 100, 250, 60, "#68ff68", "Isolated Query Segment"),
        'input_db': (700, 100, 250, 60, "#5ab4ff", "Database Target Sentence"),
        
        'engine_lex': (150, 250, 400, 80, "#dddddd", "[ Lexical Engine ]\nTF-IDF Vector Overlap & Exact Keyword Matches"),
        'engine_sem': (650, 250, 400, 80, "#dddddd", "[ Semantic Engine ]\nSentenceTransformer Cosine Similarity Vectors"),
        
        'score_lex': (250, 400, 200, 60, "black", "Lexical Score (X%)"),
        'score_sem': (750, 400, 200, 60, "black", "Semantic Score (Y%)"),
        
        'math': (350, 550, 500, 100, "#ff5e5e", "Mathematical Fusion Layer\n----------------------------------\nEquation: (Lexical% * 0.5) + (Semantic% * 0.5) = Final%"),
        'output': (450, 720, 300, 60, "#68ff68", "Final Relevance Ranking Score")
    }
    
    edges = [
        ('input_query', 'engine_lex', ""), ('input_query', 'engine_sem', ""),
        ('input_db', 'engine_lex', ""), ('input_db', 'engine_sem', ""),
        ('engine_lex', 'score_lex', "Calculate % Match"),
        ('engine_sem', 'score_sem', "Calculate % Match"),
        ('score_lex', 'math', "Weight * 0.5"),
        ('score_sem', 'math', "Weight * 0.5"),
        ('math', 'output', "Rounded Total Match")
    ]
    draw_graph("5_how_score_generated.png", nodes, edges, 1200, 850, "5. ARCHITECTURE: HOW SCORES ARE GENERATED")

generate_math_diagram()
