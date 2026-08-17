import json

nb = json.load(open('DEMO_TANG_METRIC_S.ipynb', encoding='utf-8'))
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
print('cells:', len(nb['cells']), '| code:', len(code_cells),
      '| outputs/code cell:', [len(c['outputs']) for c in code_cells],
      '| exec:', [c['execution_count'] for c in code_cells])

n_hash = sum(1 for c in code_cells for ln in c['source'] if ln.lstrip().startswith('#'))
print('dong bat dau bang #: ', n_hash)

print('\n----- CELL CODE DAU TIEN (Buoc 0) -----')
print(''.join(code_cells[0]['source']))
print('----- CELL CHOT (Buoc 9), 30 dong dau -----')
print(''.join(code_cells[9]['source'].__getitem__(slice(0, 30)) if isinstance(code_cells[9]['source'], list) else []))
