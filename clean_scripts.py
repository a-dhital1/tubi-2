#!/usr/bin/env python3
"""
Cleans raw screenplay data for LM training.

Takes annotated screenplay files (manual or BERT-labeled) and outputs clean
tokenized training data. Filters out junk, normalizes formatting, wraps
elements in special tokens that the model can learn.

Usage:
    python clean_scripts.py --input_dir ./screenplay_data/data --output_dir ./cleaned_data
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


# stuff we want to filter out - page numbers, transitions, etc
NOISE_PATTERNS = [
    r'^\s*\d+\.\s*$',
    r'^\s*CONTINUED:?\s*$',
    r'^\s*\(CONTINUED\)\s*$',
    r'^\s*MORE\s*$',
    r'^\s*\(MORE\)\s*$',
    r'^\s*FADE IN:?\s*$',
    r'^\s*FADE OUT\.?\s*$',
    r'^\s*THE END\.?\s*$',
    r'^\s*CUT TO:?\s*$',
    r'^\s*DISSOLVE TO:?\s*$',
]


class ScriptCleaner:
    """Parses and cleans screenplay annotation files."""
    
    def __init__(self, input_dir, output_dir, fmt='tokens'):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.fmt = fmt
        self.stats = defaultdict(int)
        self._noise_re = [re.compile(p, re.I) for p in NOISE_PATTERNS]
    
    def clean_text(self, text):
        """Basic text cleanup - whitespace, quotes, artifacts."""
        text = re.sub(r'[ \t]+', ' ', text).strip()
        text = re.sub(r'[|¦]', '', text)  # OCR junk
        # normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        # drop non-printable chars
        text = ''.join(c for c in text if c.isprintable() or c == '\n')
        return text
    
    def is_noise(self, text):
        """Check if line is screenplay noise (page nums, transitions, etc)."""
        text = text.strip()
        if not text:
            return True
        return any(p.match(text) for p in self._noise_re)
    
    def parse_annotation(self, path):
        """
        Parse an annotation file into structured elements.
        
        Format is like:
            dialog: Hello there!
            scene_heading: INT. OFFICE - DAY
            speaker_heading: JOHN
            text: He walks in.
        """
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        except Exception:
            self.stats['read_errors'] += 1
            return []
        
        elements = []
        label = None
        lines = []
        
        for line in content.split('\n'):
            m = re.match(r'^(dialog|scene_heading|speaker_heading|text):\s*(.*)$', line, re.I)
            if m:
                # save previous
                if label and lines:
                    combined = '\n'.join(lines)
                    if not self.is_noise(combined):
                        elements.append({'type': label, 'content': self.clean_text(combined)})
                label = m.group(1).lower()
                lines = [m.group(2)] if m.group(2).strip() else []
            elif label and line.strip():
                lines.append(line)
        
        # don't forget last one
        if label and lines:
            combined = '\n'.join(lines)
            if not self.is_noise(combined):
                elements.append({'type': label, 'content': self.clean_text(combined)})
        
        return elements
    
    def format_tokens(self, elements, title):
        """Format with special tokens for LM training."""
        out = [f"<SCRIPT>{title}</SCRIPT>", ""]
        
        for el in elements:
            t, c = el['type'], el['content']
            if not c:
                continue
            
            if t == 'scene_heading':
                out.append(f"<SCENE>{c}</SCENE>")
            elif t == 'speaker_heading':
                # strip parentheticals like (V.O.)
                speaker = re.sub(r'\s*\([^)]*\)\s*', '', c).strip()
                out.append(f"<CHARACTER>{speaker}</CHARACTER>")
            elif t == 'dialog':
                out.append(f"<DIALOG>{re.sub(r's+', ' ', c).strip()}</DIALOG>")
            elif t == 'text':
                out.append(f"<ACTION>{c}</ACTION>")
            out.append("")
        
        out.append("<END>")
        return '\n'.join(out)
    
    def format_plain(self, elements, title):
        """Simple readable format."""
        out = [f"# {title}", ""]
        for el in elements:
            t, c = el['type'], el['content']
            if not c:
                continue
            if t == 'scene_heading':
                out.append(f"\n[{c}]\n")
            elif t == 'speaker_heading':
                out.append(f"\n{c}:")
            elif t == 'dialog':
                out.append(f'"{c}"')
            elif t == 'text':
                out.append(c)
        return '\n'.join(out)
    
    def validate(self, elements):
        """Check if script has enough content to be useful."""
        dialogs = sum(1 for e in elements if e['type'] == 'dialog')
        scenes = sum(1 for e in elements if e['type'] == 'scene_heading')
        
        if dialogs < 20:
            self.stats['few_dialogs'] += 1
            return False
        if scenes < 3:
            self.stats['few_scenes'] += 1
            return False
        return True
    
    def get_title(self, filename):
        """Extract title from filename like 'Movie Name_12345_anno.txt'."""
        m = re.match(r'^(.+?)_\d+(?:_(?:manual_)?anno)?\.txt$', filename)
        if m:
            return m.group(1).replace('_', ' ')
        return filename.replace('.txt', '').replace('_', ' ')
    
    def run(self):
        """Process all files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # find annotation files
        files = []
        for subdir in ['manual_annotations/manual_annotations', 'manual_annotations',
                       'BERT_annotations/BERT_annotations', 'BERT_annotations']:
            d = self.input_dir / subdir
            if d.exists():
                found = list(d.glob('*.txt'))
                print(f"Found {len(found)} files in {subdir}")
                files.extend(found)
        
        if not files:
            print("No annotation files found!")
            return
        
        # dedupe by title (prefer manual over BERT since they come first)
        seen = set()
        unique = []
        for f in files:
            title = self.get_title(f.name)
            if title not in seen:
                seen.add(title)
                unique.append(f)
        
        print(f"Processing {len(unique)} unique scripts...")
        
        scripts = []
        for path in unique:
            self.stats['total'] += 1
            
            if path.stat().st_size < 5000:
                self.stats['too_small'] += 1
                continue
            
            elements = self.parse_annotation(path)
            if not elements:
                self.stats['empty'] += 1
                continue
            
            if not self.validate(elements):
                continue
            
            title = self.get_title(path.name)
            if self.fmt == 'tokens':
                content = self.format_tokens(elements, title)
            else:
                content = self.format_plain(elements, title)
            
            scripts.append({'title': title, 'content': content, 'n_elements': len(elements)})
            self.stats['processed'] += 1
        
        # write output
        ind_dir = self.output_dir / 'individual'
        ind_dir.mkdir(exist_ok=True)
        
        for s in scripts:
            safe = re.sub(r'[^\w\s-]', '', s['title'])[:50]
            with open(ind_dir / f"{safe}.txt", 'w') as f:
                f.write(s['content'])
        
        with open(self.output_dir / 'training_data.txt', 'w') as f:
            for s in scripts:
                f.write(s['content'])
                f.write('\n\n' + '=' * 50 + '\n\n')
        
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump({
                'total': len(scripts),
                'format': self.fmt,
                'scripts': [{'title': s['title'], 'elements': s['n_elements']} for s in scripts]
            }, f, indent=2)
        
        print(f"\nWrote {len(scripts)} scripts to {self.output_dir}")
        self._print_stats()
    
    def _print_stats(self):
        print("\n" + "=" * 40)
        print(f"Total files:    {self.stats['total']}")
        print(f"Processed:      {self.stats['processed']}")
        print(f"Too small:      {self.stats['too_small']}")
        print(f"Empty:          {self.stats['empty']}")
        print(f"Few dialogs:    {self.stats['few_dialogs']}")
        print(f"Few scenes:     {self.stats['few_scenes']}")
        print(f"Read errors:    {self.stats['read_errors']}")
        print("=" * 40)


def main():
    parser = argparse.ArgumentParser(description='Clean screenplay data for LM training')
    parser.add_argument('--input_dir', default='./screenplay_data/data')
    parser.add_argument('--output_dir', default='./cleaned_data')
    parser.add_argument('--format', choices=['tokens', 'plain'], default='tokens')
    args = parser.parse_args()
    
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Format: {args.format}")
    
    cleaner = ScriptCleaner(args.input_dir, args.output_dir, args.format)
    cleaner.run()


if __name__ == '__main__':
    main()
