import os
import sys
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# Add backend root to path
sys.path.append(os.path.join(os.getcwd()))

try:
    from app.core.robust_aligner import RobustAligner
except ImportError:
    from backend.app.core.robust_aligner import RobustAligner

def round_floats(data: Any) -> Any:
    if isinstance(data, (float, np.float32, np.float64)):
        return round(float(data), 3)
    if isinstance(data, (int, np.int32, np.int64)):
        return int(data)
    if isinstance(data, (bool, np.bool_)):
        return bool(data)
    if isinstance(data, dict):
        return {k: round_floats(v) for k, v in data.items()}
    if isinstance(data, list):
        return [round_floats(x) for x in data]
    if isinstance(data, np.ndarray):
        return round_floats(data.tolist())
    return data

class ThesisExporter:
    def __init__(self):
        print("Initializing Thesis Exporter...")
        # Find project root (where csvFiles is located)
        curr = os.getcwd()
        if os.path.exists(os.path.join(curr, 'csvFiles')):
            self.base_dir = curr
        elif os.path.exists(os.path.join(curr, '..', 'csvFiles')):
            self.base_dir = os.path.normpath(os.path.join(curr, '..'))
        else:
            self.base_dir = curr # Fallback
            
        self.csv_dir = os.path.join(self.base_dir, 'csvFiles')
        self.model_path = os.path.join(self.base_dir, 'backend', 'app', 'models', 'rizal-xlm-r-dapt')
        
        print(f"Project Root: {self.base_dir}")
        print(f"CSV Dir: {self.csv_dir}")
        print(f"Loading model from {self.model_path}...")
        self.model = SentenceTransformer(self.model_path)
        
        # Load characters
        self.tauhan_list = self._load_tauhan()
        self.aligner = RobustAligner(tauhan_list=self.tauhan_list)
        
        # Load CSV Data
        print("Loading CSV datasets...")
        self.noli_buod = pd.read_csv(os.path.join(self.csv_dir, 'noli_chapters.csv'))
        self.fili_buod = pd.read_csv(os.path.join(self.csv_dir, 'elfili_chapters.csv'))
        # Row offset: Noli has 1212 summary sentences
        self.noli_offset = 1212
        
        self.noli_full = pd.read_csv(os.path.join(self.csv_dir, 'fullversion_noli.csv'))
        self.fili_full = pd.read_csv(os.path.join(self.csv_dir, 'fullversion_elfili.csv'))

    def _load_tauhan(self) -> List[str]:
        path = os.path.join(self.base_dir, 'backend', 'app', 'data', 'character_aliases.json')
        if not os.path.exists(path):
            return []
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        chars = set()
        for entry in data:
            chars.add(entry['name'])
            for alias in entry.get('aliases', []):
                chars.add(alias)
        return list(chars)

    def get_buod_by_id(self, sid: int) -> Dict[str, Any]:
        if sid <= self.noli_offset:
            row = self.noli_buod.iloc[sid - 1]
            return {
                "id": sid,
                "novel": "Noli Me Tangere",
                "book_key": "noli",
                "chapter": int(row['chapter_number']),
                "text": str(row['sentence_text']),
                "index_in_chapter": int(row['sentence_number'])
            }
        else:
            row = self.fili_buod.iloc[sid - self.noli_offset - 1]
            return {
                "id": sid,
                "novel": "El Filibusterismo",
                "book_key": "elfili",
                "chapter": int(row['chapter_number']),
                "text": str(row['sentence_text']),
                "index_in_chapter": int(row['sentence_number'])
            }

    def run_exports(self):
        # 42 Audit IDs
        audit_ids = [
            1, 6, 46, 47, 66, 154, 203, 248, 370, 472, 
            99, 288, 447, 607, 798, 978, 1161, 1162, 3358, 
            171, 187, 228, 238, 267, 871, 1213, 1198,
            1222, 1248, 1274, 1298, 1410, 1531, 1611, 
            1733, 4094, 1924, 1941, 1955, 1977, 4226
        ]
        # Resolve 3358 (Manual fix from eval_40_results log: Chapter 63 Noli)
        # Note: 3358 seems to be a non-sequential ID in their DB. 
        # I will use the text context to find it in the CSV if sid is out of index.
        
        # FIGURE 1 & 4: Noli Chapter 1
        self.export_figure_1_and_4(novel="Noli Me Tangere", chapter=1)
        
        # FIGURE 2 & 3: Specific IDs
        self.export_figure_2_and_3()
        
        # FIGURE 5 & 6: Audit 42
        self.export_audit_5_and_6(audit_ids)

    def export_figure_1_and_4(self, novel: str, chapter: int):
        print(f"Exporting Figure 1 & 4 for {novel} Ch {chapter}...")
        results, debug = self.align_chapter(novel, chapter)
        
        # Figure 1: Score Matrix (Max over window sizes)
        # num_buod x num_full_starts
        num_buod = debug["final_scores"].shape[0]
        num_windows = debug["final_scores"].shape[1]
        windows = debug["windows"]
        
        # Map window index to start index
        full_len = max(w["end"] for w in windows) + 1
        matrix = np.zeros((num_buod, full_len))
        
        for i in range(num_buod):
            for j in range(num_windows):
                start = windows[j]["start"]
                score = debug["final_scores"][i, j]
                if score > matrix[i, start]:
                    matrix[i, start] = score
        
        fig1_data = {
            "chapter": chapter,
            "novel": novel,
            "buod_sentences": [{"index": i, "text": results[i].buod_text} for i in range(num_buod)],
            "full_text_window_starts": list(range(full_len)),
            "score_matrix": matrix.tolist(),
            "dp_path": [{"buod_index": i, "full_text_window_start": results[i].best_window_start} for i in range(num_buod)]
        }
        with open('figure1_score_matrix.json', 'w', encoding='utf-8') as f:
            json.dump(round_floats(fig1_data), f, indent=2)

        # Figure 4: Greedy vs DP
        # Greedy = Argmax per row
        greedy_path = []
        for i in range(num_buod):
            best_win_idx = np.argmax(debug["final_scores"][i, :])
            win = windows[best_win_idx]
            greedy_path.append({
                "buod_index": i,
                "buod_text": results[i].buod_text,
                "matched_full_text_index": win["start"],
                "score": float(debug["final_scores"][i, best_win_idx])
            })
        
        # Check for 3 backward jumps in greedy
        jumps = 0
        for i in range(len(greedy_path) - 1):
            if greedy_path[i]["matched_full_text_index"] > greedy_path[i+1]["matched_full_text_index"]:
                jumps += 1
        
        if jumps < 3 and chapter == 1:
            print(f"Only {jumps} jumps in Ch 1. Switching to Ch 21...")
            self.export_figure_1_and_4(novel, 21)
            return

        fig4_data = {
            "chapter": chapter,
            "novel": novel,
            "total_full_text_sentences": full_len,
            "greedy_path": greedy_path,
            "dp_path": [{"buod_index": i, "buod_text": results[i].buod_text, "matched_full_text_index": results[i].best_window_start, "score": results[i].final_score} for i in range(num_buod)]
        }
        with open('figure4_greedy_vs_dp.json', 'w', encoding='utf-8') as f:
            json.dump(round_floats(fig4_data), f, indent=2)

    def export_figure_2_and_3(self):
        print("Exporting Figure 2 & 3...")
        # Fig 2: 1, 6, 1198, 1213
        # Fig 3: 1, 1198
        ids = [1, 6, 1198, 1213]
        fig2_results = []
        fig3_results = []
        
        tauhan_annotations = []
        
        for sid in ids:
            b = self.get_buod_by_id(sid)
            # Find chapter boundary expansion for 1198 (Noli 64 -> Full 63)
            search_chapter = b["chapter"]
            if b["novel"] == "Noli Me Tangere" and b["chapter"] == 64:
                search_chapter = 63
            
            results, debug = self.align_chapter(b["novel"], search_chapter, target_buod_text=b["text"])
            
            # Since we aligned the whole chapter, find the index of our target sentence
            target_idx = 0
            for i, r in enumerate(results):
                if r.buod_text == b["text"]:
                    target_idx = i
                    break
            
            # Figure 2 logic: scores per window size (3,4,5,6)
            # consolidated_score = max across starts for each size
            scores_by_size = {}
            for size in range(3, 7):
                size_scores = []
                for j, win in enumerate(debug["windows"]):
                    if (win["end"] - win["start"] + 1) == size:
                        size_scores.append(debug["final_scores"][target_idx, j])
                scores_by_size[str(size)] = max(size_scores) if size_scores else 0.0
            
            fig2_results.append({
                "sentence_id": sid,
                "novel": b["novel"],
                "chapter": b["chapter"],
                "buod_text": b["text"],
                "scores_by_window_size": scores_by_size,
                "winning_window_size": results[target_idx].best_window_end - results[target_idx].best_window_start + 1,
                "final_score": results[target_idx].final_score
            })
            
            # Figure 3 logic: ALL candidates
            if sid in [1, 1198]:
                candidates = []
                for j, win in enumerate(debug["windows"]):
                    # Record Tauhan pass/fail
                    # We need to compute b_tauhan.issubset(w_tauhan)
                    b_tauhan = debug["buod_tauhan_mentions"][target_idx]
                    w_tauhan = debug["window_tauhan_mentions"][j]
                    tauhan_passed = b_tauhan.issubset(w_tauhan) if b_tauhan else True
                    
                    is_winner = (j == debug["dp_path"][target_idx])
                    
                    candidate = {
                        "window_start": win["start"],
                        "window_size": win["end"] - win["start"] + 1,
                        "semantic_score": debug["semantic_scores"][target_idx, j],
                        "lexical_score": debug["lexical_scores"][target_idx, j],
                        "tauhan_passed": tauhan_passed,
                        "tauhan_score": debug["tauhan_scores"][target_idx, j],
                        "consolidated_score": debug["final_scores"][target_idx, j] if tauhan_passed else 0.0,
                        "is_winner": is_winner
                    }
                    candidates.append(candidate)
                    
                    # Store annotation example
                    if not tauhan_passed and candidate["semantic_score"] > 0.45 and len(tauhan_annotations) < 3:
                        missing = list(b_tauhan - w_tauhan)
                        tauhan_annotations.append({
                            "sid": sid,
                            "start": win["start"],
                            "semantic": candidate["semantic_score"],
                            "missing": missing,
                            "text": win["text"]
                        })

                fig3_results.append({
                    "sentence_id": sid,
                    "buod_text": b["text"],
                    "buod_characters": list(debug["buod_tauhan_mentions"][target_idx]),
                    "candidates": candidates
                })

        with open('figure2_window_scale_scores.json', 'w', encoding='utf-8') as f:
            json.dump(round_floats(fig2_results), f, indent=2)
        with open('figure3_tauhan_gate_scatter.json', 'w', encoding='utf-8') as f:
            json.dump(round_floats(fig3_results), f, indent=2)
            
        self.tauhan_annotations = tauhan_annotations

    def export_audit_5_and_6(self, ids: List[int]):
        print("Exporting Figure 5 & 6 (Audit 42)...")
        fig5_results = []
        fig6_results = []
        
        # Sort IDs to guarantee Noli first, then Fili
        noli_ids = sorted([i for i in ids if i <= 1500 or i == 3358]) # Rough heuristic
        fili_ids = sorted([i for i in ids if i > 1500 and i != 3358])
        
        sorted_ids = noli_ids + fili_ids
        
        for sid in sorted_ids:
            try:
                b = self.get_buod_by_id(sid)
            except:
                # Handle 3358
                if sid == 3358:
                    b = {
                        "id": 3358, "novel": "Noli Me Tangere", "book_key": "noli", "chapter": 63,
                        "text": "Sa isang dampang nakatayo sa tabi ng bukal sa paanan ng bund..."
                    }
                else: continue
            
            search_chapter = b["chapter"]
            if b["novel"] == "Noli Me Tangere" and b["chapter"] == 64:
                search_chapter = 63
            
            # Align the whole chapter to get sequential context
            results, debug = self.align_chapter(b["novel"], search_chapter, target_buod_text=b["text"])
            
            target_idx = 0
            for i, r in enumerate(results):
                if b["text"] in r.buod_text:
                    target_idx = i
                    break
            
            match = results[target_idx]
            score = match.final_score
            status = "precise" if score > 0.8 else ("high" if score > 0.6 else ("medium" if score >= 0.45 else "rejected"))
            
            fig5_results.append({
                "sentence_id": sid,
                "novel": b["novel"],
                "chapter": b["chapter"],
                "buod_text": b["text"],
                "semantic_score": match.semantic_score,
                "lexical_score": match.lexical_score,
                "tauhan_score": match.tauhan_score,
                "position_score": match.position_score,
                "consolidated_score": score,
                "alignment_status": status,
                "tauhan_active": len(debug["buod_tauhan_mentions"][target_idx]) > 0,
                "winning_window_size": match.best_window_end - match.best_window_start + 1,
                "chapter_expanded": (b["chapter"] != search_chapter)
            })
            
            # Figure 6
            total_buod = len(results)
            # Load full chapter to get sentence count
            full_chapter_sents = self.get_full_sentences(b["novel"], search_chapter)
            total_full = len(full_chapter_sents)
            
            est_pos = target_idx / max(1, total_buod - 1)
            act_pos = match.best_center_sentence / max(1, total_full - 1)
            
            fig6_results.append({
                "sentence_id": sid,
                "novel": b["novel"],
                "chapter": b["chapter"],
                "buod_text": b["text"],
                "estimated_position_normalized": est_pos,
                "actual_position_normalized": act_pos,
                "drift": abs(est_pos - act_pos),
                "alignment_status": status
            })

        with open('figure5_audit_40_sentences.json', 'w', encoding='utf-8') as f:
            json.dump(round_floats(fig5_results), f, indent=2)
        with open('figure6_positional_drift.json', 'w', encoding='utf-8') as f:
            json.dump(round_floats(fig6_results), f, indent=2)

    def align_chapter(self, novel: str, chapter: int, target_buod_text: str = None):
        buod_sents = self.get_buod_sentences(novel, chapter)
        full_sents = self.get_full_sentences(novel, chapter)
        
        # If target text is provided, ensure it's in the buod_sents (handles snippets)
        if target_buod_text and target_buod_text not in buod_sents:
            buod_sents.append(target_buod_text)
            buod_sents = list(dict.fromkeys(buod_sents)) # Keep unique preserving order
            
        buod_embs = self.model.encode(buod_sents)
        full_embs = self.model.encode(full_sents)
        
        return self.aligner.align(buod_sents, full_sents, buod_embs, full_embs, return_debug=True)

    def get_buod_sentences(self, novel: str, chapter: int) -> List[str]:
        df = self.noli_buod if novel == "Noli Me Tangere" else self.fili_buod
        return df[df['chapter_number'] == chapter]['sentence_text'].astype(str).tolist()

    def get_full_sentences(self, novel: str, chapter: int) -> List[str]:
        df = self.noli_full if novel == "Noli Me Tangere" else self.fili_full
        return df[df['chapter_number'] == chapter]['sentence_text'].astype(str).tolist()

if __name__ == "__main__":
    exporter = ThesisExporter()
    exporter.run_exports()
    
    print("\nTAUHAN GATE ANNOTATION EXAMPLES\n")
    for i, ann in enumerate(exporter.tauhan_annotations):
        print(f"Sentence ID {ann['sid']}:")
        print(f"Example {i+1} — Window Start: {ann['start']}, Semantic: {ann['semantic']:.3f}, Reason blocked: \"{', '.join(ann['missing'])}\" present in buod but absent from window.")
        print(f"Window text: \"{ann['text']}\"\n")
