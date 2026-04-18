import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def create_diagram(title, noli, fili, filename):
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')
    
    plt.text(0.5, 0.98, f"Map View Structural Logic: {title}", fontsize=18, fontweight='bold', ha='center')
    
    def add_box(x, y, text, w=0.04, h=0.04, color="#eee", fontsize=8):
        box = patches.FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.02", 
                                     edgecolor='black', facecolor=color, lw=1)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight='bold')
        return (x, y - h/2), (x, y + h/2)

    # 1. Novels Layer
    noli_btm, _ = add_box(0.25, 0.88, "Noli Me Tangere\n(Corpus Selection)", w=0.15, h=0.04, color="#c8e6c9", fontsize=12)
    fili_btm, _ = add_box(0.75, 0.88, "El Filibusterismo\n(Corpus Selection)", w=0.15, h=0.04, color="#ffcc80", fontsize=12)
    
    noli_chaps = {}
    for ch, sent in noli: noli_chaps.setdefault(ch, []).append(sent)
    fili_chaps = {}
    for ch, sent in fili: fili_chaps.setdefault(ch, []).append(sent)
    
    noli_sorted_chaps = sorted(noli_chaps.keys())
    fili_sorted_chaps = sorted(fili_chaps.keys())
    
    # 2. Chapters Selection Layer
    add_box(0.5, 0.82, "STEP 1: Extract Distinct Chapters Based on Matches", w=0.25, h=0.03, color="#fff", fontsize=10)
    
    noli_chap_pts = {}
    for i, ch in enumerate(noli_sorted_chaps):
        x = 0.05 + 0.40 * (i / max(1, len(noli_sorted_chaps)-1)) if len(noli_sorted_chaps) > 1 else 0.25
        btm, top = add_box(x, 0.73, f"Ch {ch}", color="#e8f5e9", w=0.035)
        ax.annotate("", xy=top, xytext=noli_btm, arrowprops=dict(arrowstyle="->", color="gray"))
        noli_chap_pts[ch] = btm
        
    fili_chap_pts = {}
    for i, ch in enumerate(fili_sorted_chaps):
        x = 0.55 + 0.40 * (i / max(1, len(fili_sorted_chaps)-1)) if len(fili_sorted_chaps) > 1 else 0.75
        btm, top = add_box(x, 0.73, f"Ch {ch}", color="#fff3e0", w=0.035)
        ax.annotate("", xy=top, xytext=fili_btm, arrowprops=dict(arrowstyle="->", color="gray"))
        fili_chap_pts[ch] = btm
        
    # 3. Sentences Grouped
    add_box(0.5, 0.67, "STEP 2: Group Nodes (Sentences) By Chapter", w=0.25, h=0.03, color="#fff", fontsize=10)
    noli_sents_flat = [(ch, s) for ch in noli_sorted_chaps for s in sorted(noli_chaps[ch])]
    fili_sents_flat = [(ch, s) for ch in fili_sorted_chaps for s in sorted(fili_chaps[ch])]
    
    noli_sent_pts = []
    for i, (ch, s) in enumerate(noli_sents_flat):
        x = 0.02 + 0.46 * (i / max(1, len(noli_sents_flat)-1)) if len(noli_sents_flat) > 1 else 0.25
        btm, top = add_box(x, 0.58, f"S{s}", w=0.03, h=0.025, color="white", fontsize=8)
        ax.annotate("", xy=top, xytext=noli_chap_pts[ch], arrowprops=dict(arrowstyle="->", color="#90caf9"))
        noli_sent_pts.append(((ch, s), btm))
        
    fili_sent_pts = []
    for i, (ch, s) in enumerate(fili_sents_flat):
        x = 0.52 + 0.46 * (i / max(1, len(fili_sents_flat)-1)) if len(fili_sents_flat) > 1 else 0.75
        btm, top = add_box(x, 0.58, f"S{s}", w=0.03, h=0.025, color="white", fontsize=8)
        ax.annotate("", xy=top, xytext=fili_chap_pts[ch], arrowprops=dict(arrowstyle="->", color="#ffab91"))
        fili_sent_pts.append(((ch, s), btm))
        
    # 4. Final Ordered Map
    all_sents = noli_sents_flat + fili_sents_flat
    
    add_box(0.5, 0.45, "STEP 3: Rearrangement & Clustering Layout (Noli Sequence → Fili Sequence)", w=0.35, h=0.04, color="#e1bee7", fontsize=11)
    
    for i, (ch, s) in enumerate(all_sents):
        x = 0.02 + 0.96 * (i / max(1, len(all_sents)-1)) if len(all_sents) > 1 else 0.5
        is_noli = i < len(noli_sents_flat)
        c_color = "#c8e6c9" if is_noli else "#ffcc80"
        tag = "Noli" if is_noli else "Fili"
        _, top = add_box(x, 0.25, f"{i+1}\n{tag}\nCh{ch}\nS{s}", w=0.035, h=0.05, color=c_color, fontsize=8)
        
        src = noli_sent_pts[i][1] if is_noli else fili_sent_pts[i - len(noli_sents_flat)][1]
        rad = 0.15 if (x > 0.5 and is_noli) else (-0.15 if (x < 0.5 and not is_noli) else (0.1 if x > src[0] else -0.1))
        ax.annotate("", xy=top, xytext=src, 
                    arrowprops=dict(arrowstyle="->", color="#ce93d8", alpha=0.5, connectionstyle=f"arc3,rad={rad}"))
                    
    plt.savefig(filename, bbox_inches='tight', dpi=100)
    plt.close()

edukasyon_noli = [(8,12), (19,6), (19,11), (19,8), (19,5), (19,7), (19,9), (19,10), (38,5), (53,1)]
edukasyon_fili = [(2,3), (19,7), (23,18), (27,8), (28,18), (31,7), (32,2), (39,50), (39,58), (39,10)]

kamatayan_noli = [(1,8), (3,9), (8,14), (9,8), (57,14), (58,1), (59,32), (59,13), (64,14), (64,15)]
kamatayan_fili = [(8,2), (10,3), (22,12), (28,28), (38,9), (38,16), (38,1), (39,30), (39,32), (39,55)]

pagmamahal_noli = [(3,9), (6,37), (34,6), (39,2), (50,28), (58,1), (59,29), (59,2), (59,14), (64,14)]
pagmamahal_fili = [(8,2), (10,3), (18,18), (18,16), (25,17), (25,18), (38,9), (38,1), (39,32), (39,64)]

base_dir = r"C:\Users\ianku\.gemini\antigravity\brain\31fa9ee4-aa85-4c5f-b2af-fa7c8e24d71f"
create_diagram("Edukasyon", edukasyon_noli, edukasyon_fili, os.path.join(base_dir, "edukasyon_diagram.png"))
create_diagram("Kamatayan", kamatayan_noli, kamatayan_fili, os.path.join(base_dir, "kamatayan_diagram.png"))
create_diagram("Pagmamahal", pagmamahal_noli, pagmamahal_fili, os.path.join(base_dir, "pagmamahal_diagram.png"))
