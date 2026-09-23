"""Import Table 10 from the official LAS-VAD supplementary PDF.

Requires the system `pdftotext` utility. No network access or LLM calls.
The output records the source hash and separate UCF/XD descriptions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from las_data import UCF_NAMES, XD_NAMES


def parse_table(text):
    # PDF curly quotes delimit entries, including descriptions wrapped by
    # pdftotext. Splitting the two occurrences of normal preserves the small
    # dataset-specific punctuation difference in the fighting description.
    entries = re.findall(r'“([^”]+)”\s*:\s*“([^”]+)”', text)
    blocks = []
    for name, description in entries:
        if name == 'normal':
            blocks.append({})
        if not blocks:
            continue
        name = {'roadAccidents': 'road accident'}.get(name, name)
        if name in blocks[-1]:
            raise ValueError(f'Duplicate attribute category: {name}')
        blocks[-1][name] = ' '.join(description.split())
    if len(blocks) != 2 or set(blocks[0]) != set(XD_NAMES) or set(blocks[1]) != set(UCF_NAMES):
        raise ValueError('Expected Table 10: exactly seven XD and fourteen UCF categories')
    return dict(xd=blocks[0], ucf=blocks[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.pdf)
    result = subprocess.run(['pdftotext', '-layout', str(source), '-'],
                            check=True, capture_output=True, text=True)
    attributes = parse_table(result.stdout)
    attributes['_source'] = dict(title='LAS-VAD CVPR 2026 supplementary material, Table 10',
                                 sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                 url='https://openaccess.thecvf.com/content/CVPR2026/supplemental/'
                                     'Wang_Weakly_Supervised_Video_CVPR_2026_supplemental.pdf')
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(attributes, indent=2, ensure_ascii=False)+'\n')
    print(f'Imported {len(attributes["ucf"])} UCF and {len(attributes["xd"])} XD descriptions to {destination}')


if __name__ == '__main__':
    main()
