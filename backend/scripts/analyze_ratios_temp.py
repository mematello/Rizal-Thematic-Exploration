import pandas as pd
import os
import sys

def get_stats():
    base_dir = r"c:\Users\ianku\Documents\VS Code\Rizal-Thematic-Exploration\csvFiles"
    
    books = [('Noli', 'noli_chapters.csv', 'fullversion_noli.csv'), ('Fili', 'elfili_chapters.csv', 'fullversion_elfili.csv')]
    
    for book_name, sum_file, full_file in books:
        sum_df = pd.read_csv(os.path.join(base_dir, sum_file))
        full_df = pd.read_csv(os.path.join(base_dir, full_file))
        
        sum_counts = sum_df.groupby('chapter_number').size()
        full_counts = full_df.groupby('chapter_number').size()
        
        ratios = []
        for ch in sum_counts.index:
            if ch in full_counts:
                sum_c = sum_counts[ch]
                full_c = full_counts[ch]
                ratios.append((ch, sum_c, full_c, full_c/sum_c))
        
        print(f"\nStats for {book_name}:")
        import numpy as np
        ratio_vals = [r[3] for r in ratios]
        print(f"Average Ratio (Full/Summary): {np.mean(ratio_vals):.2f}")
        print(f"Min Ratio: {np.min(ratio_vals):.2f}")
        print(f"Max Ratio: {np.max(ratio_vals):.2f}")
        print(f"Std Dev: {np.std(ratio_vals):.2f}")
        print("Sample Chapters:")
        for r in ratios[:5]:
            print(f"  Ch {r[0]}: {r[1]} buod, {r[2]} full (1:{r[3]:.1f})")

if __name__ == "__main__":
    get_stats()
