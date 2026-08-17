# -*- coding: utf-8 -*-
"""Don dep sau khi xoa comment: rstrip khoang trang cuoi dong (tru cac dong nam
trong chuoi nhieu dong nhu docstring/prompt), va phuc hoi execution_count 1..11."""
import io
import json
import tokenize

PATH = 'DEMO_TANG_METRIC_S.ipynb'
nb = json.load(open(PATH, encoding='utf-8'))

ec = 0
n_fixed_total = 0
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    if not src.endswith('\n'):
        src += '\n'

    protected = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        assert tok.type != tokenize.COMMENT, f'cell {idx} van con comment'
        if tok.type == tokenize.STRING and tok.end[0] > tok.start[0]:
            for r in range(tok.start[0], tok.end[0] + 1):
                protected.add(r - 1)

    lines = src.split('\n')
    n_fixed = 0
    for i, ln in enumerate(lines):
        if i in protected:
            continue
        if ln != ln.rstrip():
            lines[i] = ln.rstrip()
            n_fixed += 1
    n_fixed_total += n_fixed

    new_src = '\n'.join(lines)
    compile(new_src, f'<cell {idx}>', 'exec')

    ec += 1
    cell['execution_count'] = ec
    cell['source'] = new_src.splitlines(keepends=True)
    if n_fixed:
        print(f'cell {idx}: rstrip {n_fixed} dong')

with open(PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
bad = [ln for c in code_cells
       for ln in ''.join(c['source']).split('\n') if ln != ln.rstrip()]
print(f'Tong: rstrip {n_fixed_total} dong | exec_count =',
      [c['execution_count'] for c in code_cells],
      '| outputs =', [len(c['outputs']) for c in code_cells])
print('Con dong thua khoang trang (ngoai chuoi nhieu dong):', len(bad))
