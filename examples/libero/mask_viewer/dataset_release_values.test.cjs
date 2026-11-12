const assert = require('node:assert/strict');
const {series} = require('./dataset_release_values.js');
const radians = d => d * Math.PI / 180;
const accepted = {rotation_release_rad: [radians(2), radians(-12), radians(4)], bounds_rad: Array(6).fill(radians(30)), semantic_mask: '011', effective_mask: '010', optimizer: {feasible: true}, phase: 'APP'};
let values = series([accepted, {...accepted, aborted: true, rotation_release_rad: Array(3).fill(radians(90))}, {...accepted, aborted: true, rotation_release_rad: Array(3).fill(radians(80))}]);
assert.deepEqual(values[0].degrees.map(Math.round), [2, -12, 4]);
assert.deepEqual(values[1].degrees, values[0].degrees); // rejected candidate cannot become release
assert.equal(values[1].rejected, true);
assert.equal(values[2].acceptedIndex, 0); // multiple rejects retain the same accepted frame
assert.deepEqual(values[0].limits.map(x => x.map(Math.round)), [[0,0],[30,30],[0,0]]);
values = series([{...accepted, rotation_release_rad: undefined}, {...accepted, rotation_release_rad: undefined, aborted: true}]);
assert.equal(values[0].available, false);
assert.deepEqual(values[1].degrees, [null,null,null]); // missing is never forged as zero
values = series([{...accepted, aborted: true}]);
assert.equal(values[0].available, false);
assert.equal(values[0].acceptedIndex, null);
values = series([{...accepted, effective_mask: '000'}]);
assert.deepEqual(values[0].degrees.map(Math.round), [2,-12,4]); // strict axes can inherit accepted release
assert.deepEqual(values[0].limits, [[0,0],[0,0],[0,0]]);
assert.deepEqual(series([]), []);
console.log('Dataset release accepted/rejected/missing/strict-axis checks passed');
