# -*- coding: utf-8 -*-
"""Xoa toan bo comment '#' trong cac code cell cua DEMO_TANG_METRIC_S.ipynb.

Dung tokenize de xac dinh COMMENT token -> khong dinh nham dau '#' trong chuoi.
Giu nguyen: markdown cell, output mau, execution_count, docstring.
"""
import io
import json
import tokenize

PATH = 'DEMO_TANG_METRIC_S.ipynb'

nb = json.load(open(PATH, encoding='utf-8'))
total_removed = 0

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    if not src.endswith('\n'):
        src += '\n'

    comments = []  # (row 1-based, col) cua tung COMMENT token
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            comments.append((tok.start[0], tok.start[1]))

    lines = src.split('\n')
    drop = set()
    for row, col in comments:
        i = row - 1
        cut = lines[i][:col].rstrip()
        if cut:
            lines[i] = cut          # comment cuoi dong -> giu phan code
        else:
            drop.add(i)             # dong chi co comment -> xoa ca dong
    total_removed += len(comments)

    lines = [ln for i, ln in enumerate(lines) if i not in drop]

    # Don dep dong trong: bo dong trong dau/cuoi cell, gom >2 dong trong lien
    # tiep ve dung 2 (chuan PEP8 giua cac ham).
    cleaned, blanks = [], 0
    for ln in lines:
        if ln.strip() == '':
            blanks += 1
            if cleaned and blanks <= 2:
                cleaned.append('')
        else:
            blanks = 0
            cleaned.append(ln)
    while cleaned and cleaned[-1] == '':
        cleaned.pop()

    new_src = '\n'.join(cleaned) + '\n'

    # Kiem chung: khong con COMMENT token, va cu phap van hop le
    for tok in tokenize.generate_tokens(io.StringIO(new_src).readline):
        assert tok.type != tokenize.COMMENT, f'cell {idx} van con comment'
    compile(new_src, f'<cell {idx}>', 'exec')

    cell['source'] = new_src.splitlines(keepends=True)

with open(PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'Da xoa {total_removed} comment; notebook van hop le (da compile tung cell).')
