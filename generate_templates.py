#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "classify"))
from classifier import NewsLayoutClassifier

DATA = Path("/home/ubuntu/.openclaw/workspace-coding/youtube_screenshots/youtube_screenshots")
COLS = [0,1,2,3,4,5,6,7,8,10,73,74,75,76,77,78,79,80,81,82,83,97,100,120,121,122,123,124,125,126,127]
clf = NewsLayoutClassifier(active_cols=COLS)
clf.train(left_dir=DATA/"left", right_dir=DATA/"right ")
clf.save_templates(Path(__file__).parent / "templates" / "news_layout_templates.npz")
