const assert = require('node:assert/strict');
const {values}=require('./release_values.js');
const rad=d=>d*Math.PI/180;
let frame={rotation_release_rad:[rad(5),rad(-18),rad(2)],bounds_rad:Array(6).fill(rad(30)),semantic_mask:'011',effective_mask:'010',reference_held:true,rejected_rotation_release_rad:[rad(90),rad(90),rad(90)]};
let v=values(frame,true);
assert.equal(v.available,true);
assert.ok(Math.abs(v.degrees[1]+18)<1e-12);
assert.deepEqual(v.limits.map(x=>x.map(Math.round)),[[0,0],[30,30],[0,0]]);
assert.equal(v.held,true);
assert.ok(v.rejected[1]>89 && v.degrees[1]<0); // never chart failed candidate
v=values({...frame,rotation_release_rad:null},true);
assert.equal(v.available,false);
assert.deepEqual(v.degrees,[null,null,null]); // missing telemetry is not zero
v=values(frame,false);
assert.equal(v.available,false);
assert.deepEqual(v.degrees,[null,null,null]);
v=values({...frame,bounds_rad:[0,0,0,0,0,0],effective_mask:'111',held_reference_bounds_rad:[5,5,15,15,30,30].map(rad)},true);
assert.deepEqual(v.limits.map(a=>a.map(Math.round)),[[5,5],[15,15],[30,30]]); // accepted bounds during hold
v=values({...frame,reference_held:false,effective_mask:'111',bounds_rad:[5,5,15,15,30,30].map(rad)},true);
assert.deepEqual(v.limits.map(a=>a.map(Math.round)),[[5,5],[15,15],[30,30]]);
console.log('Release telemetry unit checks passed');
