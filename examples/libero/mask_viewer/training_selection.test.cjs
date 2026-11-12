const assert = require('node:assert/strict');
const {selected, visible} = require('./training_selection.js');
const d = {suite:'libero_spatial',task_id:0};
const snapshot = {episodes:{'libero_spatial/00/demo_2':{episode_index:2}}};
assert.equal(selected(null,d,{path:'demo_2'}),false);
assert.equal(selected(snapshot,d,{path:'demo_2'}),true);
assert.equal(selected(snapshot,d,{path:'demo_20'}),false);
assert.equal(selected(snapshot,{suite:'libero_goal',task_id:0},{path:'demo_2'}),false);
assert.equal(selected(snapshot,{suite:'libero_spatial',task_id:1},{path:'demo_2'}),false);
assert.deepEqual(visible(snapshot,d,[{path:'demo_0'},{path:'demo_2'},{path:'demo_20'}]),[{path:'demo_2'}]);
if(process.argv[2]) {
  const fs=require('node:fs'),path=require('node:path');
  const root=process.argv[2];
  const selection=JSON.parse(fs.readFileSync(path.join(root,'training_selection.json')));
  const catalog=JSON.parse(fs.readFileSync(path.join(root,'viewer_catalog.json')));
  let count=0,taskCount=0;
  for(const dataset of catalog.datasets) {
    const matching=visible(selection,dataset,dataset.episodes);count+=matching.length;
    assert(matching.length>0);taskCount++;
    for(const ep of matching)assert.equal(ep.tolerance.result.success,true);
  }
  assert.equal(count,301);assert.equal(count,selection.selected);assert.equal(taskCount,40);
  console.log('301 exact selections across40 tasks; no other demos accepted');
}
console.log('Training selection membership checks passed');
