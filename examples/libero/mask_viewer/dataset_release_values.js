/* Accepted recorded release for the dataset comparison timeline. */
(function (scope) {
  const telemetry = typeof module === 'undefined' ? scope.LiberoRelease : require('../benchmark_viewer/release_values.js');
  function series(rows) {
    let accepted = null, acceptedIndex = null;
    return rows.map((row, index) => {
      const rejected = Boolean(row.aborted || row.optimizer?.feasible === false || row.reference_held);
      const frame = rejected ? {
        ...row,
        rotation_release_rad: accepted?.rotation_release_rad ?? null,
        effective_mask: accepted?.effective_mask ?? '000',
        bounds_rad: accepted?.bounds_rad ?? null,
        reference_held: true,
        rejected_rotation_release_rad: row.rotation_release_rad,
      } : row;
      if (!rejected) { accepted = row; acceptedIndex = index; }
      return {
        ...telemetry.values(frame, true),
        rejected,
        phase: row.phase,
        semantic_subtask_index: row.semantic_subtask_index,
        heading_guarded: Boolean(row.heading_guarded),
        sourceIndex: index,
        acceptedIndex,
        boundsAvailable: Array.isArray(frame.bounds_rad) && frame.bounds_rad.length === 6 && frame.bounds_rad.every(Number.isFinite),
      };
    });
  }
  scope.DatasetRelease = {series};
  if (typeof module !== 'undefined') module.exports = scope.DatasetRelease;
})(typeof window === 'undefined' ? globalThis : window);
