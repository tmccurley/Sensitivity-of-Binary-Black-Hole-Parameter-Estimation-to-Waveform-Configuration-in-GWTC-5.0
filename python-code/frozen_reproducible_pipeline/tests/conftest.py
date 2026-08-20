from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'03_confirmatory_analysis/analysis'))
sys.path.insert(0,str(ROOT/'04_post_confirmatory_robustness/analysis'))
