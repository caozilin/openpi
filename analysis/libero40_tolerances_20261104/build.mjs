import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const root = '/media/czl/sata/franka_my_code/openpi';
const run = path.join(root, 'data/libero_evaluation/pi05_reconstructed_10000_tol30_confirmedmask_hold20_40tasks_10seeds_20261104');
const output = path.join(root, 'outputs/01a195ea-15f0-71b1-81ca-78023e28fe20');
const support = path.join(root, 'analysis/libero40_tolerances_20261104');
const bytes = await fs.readFile(path.join(run, 'confirmed_masks.json'));
const hash = crypto.createHash('sha256').update(bytes).digest('hex');
const masks = JSON.parse(bytes);
const request = JSON.parse(await fs.readFile(path.join(run, 'evaluation_request.json'), 'utf8'));
const tasks = JSON.parse(await fs.readFile(path.join(run, 'official_tasks.json'), 'utf8'));
assert.equal(hash, request.confirmed_masks_sha256);
assert.deepEqual(masks.mask_order, ['xF', 'yF', 'zF']);
assert.equal(masks.tasks.length, 40);

const descriptions = {
  libero_spatial: ['夹起盘子和烤盅之间的黑碗，放到盘子上', '夹起烤盅旁的黑碗，放到盘子上', '夹起桌子中央的黑碗，放到盘子上', '夹起饼干盒上的黑碗，放到盘子上', '夹起木柜顶层抽屉内的黑碗，放到盘子上', '夹起烤盅上的黑碗，放到盘子上', '夹起饼干盒旁的黑碗，放到盘子上', '夹起炉灶上的黑碗，放到盘子上', '夹起盘子旁的黑碗，放到盘子上', '夹起木柜上的黑碗，放到盘子上'],
  libero_object: ['将字母汤罐放入篮子', '将奶油奶酪放入篮子', '将沙拉酱放入篮子', '将烧烤酱放入篮子', '将番茄调味酱（ketchup）放入篮子', '将番茄酱（tomato sauce）放入篮子', '将黄油放入篮子', '将牛奶放入篮子', '将巧克力布丁放入篮子', '将橙汁放入篮子'],
  libero_goal: ['打开柜子中层抽屉', '将碗放到炉灶上', '将酒瓶放到柜顶', '打开顶层抽屉并将碗放进去', '将碗放到柜顶', '将盘子推到炉灶前方', '将奶油奶酪放入碗中', '打开炉灶开关', '将碗放到盘子上', '将酒瓶放到架子上'],
  libero_10: ['将字母汤罐和番茄酱都放入篮子', '将奶油奶酪盒和黄油都放入篮子', '打开炉灶并将摩卡壶放到炉灶上', '将黑碗放入底层抽屉并关上抽屉', '白杯放左盘，黄白杯放右盘', '将书放入收纳架后格', '白杯放盘子上，巧克力布丁放盘子右侧', '将字母汤罐和奶油奶酪盒都放入篮子', '将两个摩卡壶都放到炉灶上', '将黄白杯放入微波炉并关门'],
};
const labels = {
  'black bowl': '黑碗', 'alphabet soup': '字母汤罐', 'cream cheese': '奶油奶酪', 'salad dressing': '沙拉酱', 'bbq sauce': '烧烤酱', ketchup: '番茄调味酱 ketchup', 'tomato sauce': '番茄酱 tomato sauce', butter: '黄油', milk: '牛奶', 'chocolate pudding': '巧克力布丁', 'orange juice': '橙汁', 'middle drawer': '中层抽屉', bowl: '碗', 'wine bottle': '酒瓶', 'open top drawer, then grasp and place bowl; drawer opening is APP': '开顶层抽屉→抓碗→放碗；开抽屉计APP', plate: '盘子', 'stove switch': '炉灶开关', 'cream cheese box': '奶油奶酪盒', 'moka pot': '摩卡壶', 'bottom drawer': '底层抽屉', 'white mug': '白杯', 'yellow and white mug': '黄白杯', book: '书', 'moka pot initially at image right': '初始图像右侧摩卡壶', 'moka pot initially at image left': '初始图像左侧摩卡壶', 'microwave door': '微波炉门',
};
const suiteNames = {libero_spatial: 'Spatial', libero_object: 'Object', libero_goal: 'Goal', libero_10: 'Long'};
const phases = ['APP', 'EXEC'];
const headers = ['任务集', 'task_id', '任务说明', '语义子任务', ...phases.flatMap(p => ['xF', 'yF', 'zF'].map(a => `${p}_${a}(°)`)), 'subtask_id', '官方任务描述（英文）'];
const csvHeaders = ['suite', 'task_id', 'task_zh', 'subtask_label', ...phases.flatMap(p => ['xF', 'yF', 'zF'].map(a => `${p}_${a}_deg`)), 'subtask_id', 'task_language'];
const rows = [];
const csvRows = [];
for (const task of masks.tasks) {
  const official = tasks.find(t => t.suite === task.suite && t.task_id === task.task_id);
  assert.ok(official);
  for (const subtask of task.subtasks) {
    const values = phases.flatMap(phase => [...subtask[phase]].map(bit => {
      assert.ok(bit === '0' || bit === '1');
      return bit === '1' ? 30 : 0;
    }));
    const label = labels[subtask.label];
    assert.ok(label, subtask.label);
    rows.push([suiteNames[task.suite], task.task_id, descriptions[task.suite][task.task_id], label, ...values, subtask.index, official.language]);
    csvRows.push([task.suite, task.task_id, descriptions[task.suite][task.task_id], subtask.label, ...values, subtask.index, official.language]);
  }
}
assert.equal(rows.length, 49);
assert.equal(new Set(csvRows.map(r => `${r[0]}/${r[1]}`)).size, 40);
assert.ok(rows.every(r => r.slice(4, 10).every(v => v === 0 || v === 30)));

await fs.mkdir(output, { recursive: true });
const wb = Workbook.create();
const sheet = wb.worksheets.add('语义容差编辑');
sheet.showGridLines = false;
sheet.getRange('A2').values = [['LIBERO 40任务语义姿态容差']];
sheet.getRange('A3').values = [['共40任务、49个语义子任务。黄色数值格可直接修改，单位为度。30表示对称±30°，0表示严格锁定。']];
sheet.getRange('A4').values = [['轴顺序：xF、yF、zF，均为容差坐标系的旋转轴。仅列APP（接近）和EXEC（执行/搬运），省略中间过渡阶段。']];
sheet.getRange('A5').values = [['task_id与subtask_id均从0开始。此表是编辑副本，修改不会自动写回评测配置。运行时heading退化保护仍可能收紧有效容差。']];
sheet.getRange('A6').values = [['来源：2026-11-04对应任务mask的400回合容差评测冻结配置；按原语义子任务和阶段逐位将1→30、0→0。']];
const first = 9;
const last = first + rows.length - 1;
sheet.getRange(`A8:L${last}`).values = [headers, ...rows];
sheet.getRange(`A2:L${last}`).format.font = { name: 'Arial', size: 10, color: '#203047' };
sheet.getRange(`A2:L${last}`).format.verticalAlignment = 'center';
sheet.getRange('A2').format.font = { name: 'Arial', size: 15, bold: true, color: '#173255' };
sheet.getRange('A3:A6').format.font = { name: 'Arial', size: 10, color: '#56657A' };
sheet.getRange('A3:L6').format.rowHeight = 20;
sheet.getRange('A2:L2').format.rowHeight = 28;
sheet.getRange(`A8:L${last}`).format.rowHeight = 40;
sheet.getRange('A8:L8').format = { fill: '#243C60', font: { name: 'Arial', size: 10, bold: true, color: '#FFFFFF' }, horizontalAlignment: 'center', verticalAlignment: 'center', wrapText: true, rowHeight: 34 };
sheet.getRange(`A${first}:A${last}`).format.columnWidthPx = 80;
sheet.getRange(`B${first}:B${last}`).format.columnWidthPx = 62;
sheet.getRange(`C${first}:C${last}`).format.columnWidthPx = 285;
sheet.getRange(`D${first}:D${last}`).format.columnWidthPx = 210;
sheet.getRange(`E${first}:J${last}`).format.columnWidthPx = 87;
sheet.getRange(`K${first}:K${last}`).format.columnWidthPx = 78;
sheet.getRange(`L${first}:L${last}`).format.columnWidthPx = 500;
sheet.getRange(`C${first}:D${last}`).format.wrapText = true;
sheet.getRange(`L${first}:L${last}`).format.wrapText = true;
sheet.getRange(`E${first}:J${last}`).format = { fill: '#FFF7DC', numberFormat: '0', horizontalAlignment: 'right', font: { name: 'Arial', size: 10, color: '#235E96' } };
sheet.getRange(`B${first}:B${last}`).format.numberFormat = '0';
sheet.getRange(`K${first}:K${last}`).format.numberFormat = '0';
for (const col of ['E', 'H']) sheet.getRange(`${col}8:${col}${last}`).format.borders = { left: { style: 'thin', color: '#CAD3E1' } };
for (let i = 0; i < rows.length; i++) {
  if (i === 0 || rows[i][0] !== rows[i - 1][0]) sheet.getRange(`A${first + i}:L${first + i}`).format.borders = { top: { style: 'medium', color: '#8295AF' } };
}
sheet.freezePanes.freezeRows(8);
sheet.freezePanes.freezeColumns(4);
const table = sheet.tables.add(`A8:L${last}`, true, 'SemanticToleranceDegrees');
table.style = 'TableStyleLight1';
sheet.getRange(`E${first}:J${last}`).conditionalFormats.add('cellIs', { operator: 'equal', formula: 0, format: { font: { color: '#7C8797' } } });
wb.recalculate();
console.log((await wb.inspect({kind: 'table', range: '语义容差编辑!A8:L12', include: 'values,formulas', tableMaxRows: 5, tableMaxCols: 12, maxChars: 2200})).ndjson);
console.log((await wb.inspect({kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!', options: {useRegex: true, maxResults: 20}, maxChars: 400})).ndjson);
assert.deepEqual(sheet.getRange(`E${first}:J${last}`).values, rows.map(r => r.slice(4,10)));
const xlsxPath = path.join(output, 'libero40_semantic_tolerances_deg.xlsx');
await (await SpreadsheetFile.exportXlsx(wb)).save(xlsxPath);
const escapeCsv = x => '"' + String(x).replaceAll('"', '""') + '"';
await fs.writeFile(path.join(output, 'libero40_semantic_tolerances_deg.csv'), '\uFEFF' + [csvHeaders, ...csvRows].map(r => r.map(escapeCsv).join(',')).join('\r\n') + '\r\n');
for (const [label, range] of [['spatial', 'A1:J15'], ['goal_long', 'A29:J42'], ['long', 'A43:J57']]) {
  const image = await wb.render({sheetName: sheet.name, range, scale: 1.25, format: 'png'});
  await fs.writeFile(path.join(support, `${label}_preview.png`), new Uint8Array(await image.arrayBuffer()));
}
await fs.writeFile(path.join(support, 'verification.json'), JSON.stringify({tasks:40,semantic_subtasks:49,phases,editable_numeric_cells:49*6,mask_order:masks.mask_order,unit:'degree; positive symmetric half width',source_sha256:hash,source_matches_frozen_request:true,all_values_match_bitwise_conversion:true,xlsx:xlsxPath,csv:path.join(output,'libero40_semantic_tolerances_deg.csv')},null,2));
console.log(JSON.stringify({xlsx:xlsxPath,rows:49,tasks:40}));
