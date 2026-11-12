import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {FileBlob, SpreadsheetFile} from '@oai/artifact-tool';
const root = '/media/czl/sata/franka_my_code/openpi';
const filename = root + '/outputs/01a195ea-15f0-71b1-81ca-78023e28fe20/libero40_semantic_tolerances_deg.xlsx';
const before = crypto.createHash('sha256').update(await fs.readFile(filename)).digest('hex');
const wb = await SpreadsheetFile.importXlsx(await FileBlob.load(filename));
const sheet = wb.worksheets.getItem('语义容差编辑');
const rows = sheet.getRange('A9:L57').values;
const taskKeys = new Set(), subtaskKeys = new Set(), changes = [], invalid = [], idIssues = [];
const source = JSON.parse(await fs.readFile(root+'/data/libero_evaluation/pi05_reconstructed_10000_tol30_confirmedmask_hold20_40tasks_10seeds_20261104/confirmed_masks.json','utf8'));
const suites = {Spatial:'libero_spatial',Object:'libero_object',Goal:'libero_goal',Long:'libero_10'};
const degreeCounts = {};
for (let i=0;i<rows.length;i++) {
  const row = rows[i], suite = suites[row[0]], id = row[1], subid = row[10];
  const task = source.tasks.find(t=>t.suite===suite && t.task_id===id);
  const sub = task?.subtasks.find(s=>s.index===subid);
  const key = `${suite}/${id}/${subid}`;
  if (!sub || subtaskKeys.has(key)) idIssues.push({row:i+9,key});
  taskKeys.add(`${suite}/${id}`);subtaskKeys.add(key);
  const original = sub ? ['APP','EXEC'].flatMap(p=>[...sub[p]].map(b=>Number(b)*30)) : [];
  for (let j=0;j<6;j++) {
    const value=row[4+j], cell=String.fromCharCode(69+j)+(i+9);
    if (typeof value!=='number'||!Number.isFinite(value)||value<0||value>180) invalid.push({cell,value});
    degreeCounts[value]=(degreeCounts[value]||0)+1;
    if (value!==original[j]) changes.push({cell,suite,task_id:id,subtask_id:subid,phase:j<3?'APP':'EXEC',axis:['xF','yF','zF'][j%3],old:original[j],new:value});
  }
}
const report={workbook_sha256:before,rows:rows.length,tasks:taskKeys.size,unique_subtasks:subtaskKeys.size,invalid,idIssues,degreeCounts,changed_cells:changes.length,changed_tasks:new Set(changes.map(c=>`${c.suite}/${c.task_id}`)).size,changed_subtasks:new Set(changes.map(c=>`${c.suite}/${c.task_id}/${c.subtask_id}`)).size,changes};
assert.equal(taskKeys.size,40);assert.equal(subtaskKeys.size,49);assert.equal(rows.length,49);
assert.equal(invalid.length,0);assert.equal(idIssues.length,0);
console.log(JSON.stringify({...report,changes:undefined}));
console.log((await wb.inspect({kind:'table',range:'语义容差编辑!E32:J32',include:'values,formulas',tableMaxRows:1,tableMaxCols:6,maxChars:500})).ndjson);
console.log((await wb.inspect({kind:'table',range:'语义容差编辑!E50:J51',include:'values,formulas',tableMaxRows:2,tableMaxCols:6,maxChars:700})).ndjson);
await fs.writeFile(root+'/analysis/libero40_tolerances_20261104/user_edit_review.json',JSON.stringify(report,null,2));
assert.equal(crypto.createHash('sha256').update(await fs.readFile(filename)).digest('hex'),before);
