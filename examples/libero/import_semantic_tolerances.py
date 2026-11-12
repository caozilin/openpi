"""Read the editable degree workbook into the default LIBERO runtime config."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[2]
WORKBOOK = ROOT / 'outputs/01a195ea-15f0-71b1-81ca-78023e28fe20/libero40_semantic_tolerances_deg.xlsx'
LEGACY = ROOT / 'data/libero_tasktol_joint_servo/20261031_mask30_confirmed_v1/confirmed_masks.json'
OUTPUT = ROOT / 'examples/libero/configs/libero40_semantic_tolerances_deg.json'
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
SUITES = {'Spatial': 'libero_spatial', 'Object': 'libero_object', 'Goal': 'libero_goal', 'Long': 'libero_10'}


def read_cells(filename):
    with zipfile.ZipFile(filename) as archive:
        workbook = ET.fromstring(archive.read('xl/workbook.xml'))
        sheet = next(s for s in workbook.findall('s:sheets/s:sheet', NS) if s.get('name') == '语义容差编辑')
        relid = sheet.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
        rels = ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
        target = next(r.get('Target') for r in rels if r.get('Id') == relid)
        target = str(PurePosixPath('xl') / target) if not target.startswith('/') else target.lstrip('/')
        shared = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            shared = [''.join(t.text or '' for t in si.findall('.//s:t', NS)) for si in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('s:si', NS)]
        cells = {}
        for cell in ET.fromstring(archive.read(target)).findall('.//s:c', NS):
            address, kind = cell.get('r'), cell.get('t')
            value = cell.find('s:v', NS)
            if kind == 's': value = shared[int(value.text)]
            elif kind == 'inlineStr': value = ''.join(t.text or '' for t in cell.findall('.//s:t', NS))
            elif value is not None: value = value.text
            cells[address] = (value, kind, cell.find('s:f', NS) is not None)
        return cells


def convert(cells, original):
    expected_headers = [f'{p}_{a}(°)' for p in ('APP', 'EXEC') for a in ('xF', 'yF', 'zF')]
    if [cells.get(f'{c}8', (None,))[0] for c in 'EFGHIJ'] != expected_headers:
        raise ValueError('Workbook degree columns E:J must be APP/EXEC xF,yF,zF in that order')
    records = {}
    for row in sorted({int(''.join(filter(str.isdigit, address))) for address in cells}):
        if row < 9 or all(cells.get(f'{col}{row}', (None,))[0] in (None, '') for col in 'ABCDEFGHIJKL'):
            continue
        def value(col):
            raw, kind, formula = cells.get(f'{col}{row}', (None, None, False))
            if formula: raise ValueError(f'{col}{row}: formulas are not accepted as frozen settings')
            return raw, kind
        suite = SUITES.get(value('A')[0])
        if suite is None: raise ValueError(f'A{row}: unknown suite')
        def identifier(col):
            raw, kind = value(col)
            try: number = float(raw)
            except (TypeError, ValueError): raise ValueError(f'{col}{row}: invalid identifier') from None
            if not math.isfinite(number) or number != int(number): raise ValueError(f'{col}{row}: identifier must be an integer')
            return int(number)
        key = suite, identifier('B'), identifier('K')
        if key in records: raise ValueError(f'Row{row}: duplicate task/subtask')
        numbers = []
        for col in 'EFGHIJ':
            raw, kind = value(col)
            if kind not in (None, 'n'): raise ValueError(f'{col}{row}: angle must be a numeric cell')
            try: number = float(raw)
            except (TypeError, ValueError): raise ValueError(f'{col}{row}: missing or invalid angle') from None
            if not math.isfinite(number) or not 0 <= number <= 180: raise ValueError(f'{col}{row}: angle must be within0..180 degrees')
            numbers.append(int(number) if number.is_integer() else number)
        records[key] = numbers
    expected = {(t['suite'], t['task_id'], s['index']) for t in original['tasks'] for s in t['subtasks']}
    if set(records) != expected: raise ValueError('Workbook must contain exactly all40 tasks and49 unique semantic subtasks')
    tasks = deepcopy(original['tasks'])
    for task in tasks:
        for sub in task['subtasks']:
            angles = records[task['suite'], task['task_id'], sub['index']]
            sub.update(APP=angles[:3], ENG=[0, 0, 0], EXEC=angles[3:], DIS=[0, 0, 0])
    return dict(schema='libero40_semantic_tolerances_deg.1', mask_order=['xF', 'yF', 'zF'],
                angle_unit='degree', bound_semantics='symmetric_half_width', tasks=tasks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workbook', type=Path, default=WORKBOOK)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    table = convert(read_cells(args.workbook), json.loads(LEGACY.read_text()))
    table['source_workbook_sha256'] = hashlib.sha256(args.workbook.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(table, ensure_ascii=False, indent=2) + '\n')
    print(f'Imported40 tasks/49 subtasks: {args.output}')


if __name__ == '__main__': main()
