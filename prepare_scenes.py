#!/usr/bin/env python3
"""
Splits cleaned screenplay data into individual scenes for training.

Training on whole scripts doesn't work great - they're too long and the model
learns to generate rambling text. Individual scenes are better training examples
since each one is a complete narrative unit that fits in the context window.

Usage:
    python prepare_scenes.py --input ../cleaned_data/training_data.txt
"""

import argparse
import json
import re
from pathlib import Path


def extract_scenes(text):
    """
    Pull out individual scenes from a screenplay.
    
    Looks for <SCENE>...</SCENE> tags and grabs everything until the next
    scene or end of script.
    """
    pattern = r'<SCENE>([^<]*)</SCENE>(.*?)(?=<SCENE>|<END>|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    scenes = []
    for heading, content in matches:
        heading = heading.strip()
        content = content.strip()
        if heading and content:
            scenes.append({'heading': heading, 'content': content})
    return scenes


def format_scene(scene):
    """Wrap scene with boundary markers for training."""
    return '\n'.join([
        "<SCENE_START>",
        f"<SCENE>{scene['heading']}</SCENE>",
        scene['content'],
        "<SCENE_END>"
    ])


def process(input_path, output_dir, min_len=100):
    """Extract scenes from all scripts and write training data."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {input_path}...")
    with open(input_path) as f:
        text = f.read()
    
    # scripts are separated by ===...===
    scripts = text.split('=' * 50)
    print(f"Found {len(scripts)} scripts")
    
    all_scenes = []
    n_scripts = 0
    n_short = 0
    
    for script in scripts:
        script = script.strip()
        if not script:
            continue
        
        # get title if there
        m = re.search(r'<SCRIPT>([^<]*)</SCRIPT>', script)
        title = m.group(1).strip() if m else "Unknown"
        
        scenes = extract_scenes(script)
        n_scripts += 1
        
        for i, scene in enumerate(scenes):
            if len(scene['content']) < min_len:
                n_short += 1
                continue
            
            scene['title'] = title
            scene['num'] = i + 1
            all_scenes.append(scene)
    
    print(f"Extracted {len(all_scenes)} scenes from {n_scripts} scripts")
    print(f"Skipped {n_short} short scenes")
    
    # write training file
    train_path = out / 'scene_training_data.txt'
    with open(train_path, 'w') as f:
        for scene in all_scenes:
            f.write(format_scene(scene))
            f.write('\n\n')
    
    print(f"Wrote {train_path}")
    
    # metadata
    lengths = [len(s['content']) for s in all_scenes]
    meta = {
        'n_scenes': len(all_scenes),
        'n_scripts': n_scripts,
        'avg_len': sum(lengths) / len(lengths) if lengths else 0,
        'min_len': min(lengths) if lengths else 0,
        'max_len': max(lengths) if lengths else 0,
    }
    
    with open(out / 'scene_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"\nStats:")
    print(f"  Avg scene: {meta['avg_len']:.0f} chars")
    print(f"  Shortest:  {meta['min_len']} chars")
    print(f"  Longest:   {meta['max_len']} chars")


def main():
    parser = argparse.ArgumentParser(description='Extract scenes for training')
    parser.add_argument('--input', default='./cleaned_data/training_data.txt')
    parser.add_argument('--output', default='./cleaned_data/scenes')
    parser.add_argument('--min_len', type=int, default=100, help='Min scene length in chars')
    args = parser.parse_args()
    
    process(args.input, args.output, args.min_len)


if __name__ == '__main__':
    main()