/* Exact recorded telemetry only; no pose-error substitution for release. */
(function (scope) {
  function values(frame, enabled) {
    const vector = frame.rotation_release_rad;
    const available = enabled && Array.isArray(vector) && vector.length === 3 && vector.every(Number.isFinite);
    const effective = frame.effective_mask || '000';
    const semantic = frame.semantic_mask || '000';
    const acceptedBounds = frame.reference_held && Array.isArray(frame.held_reference_bounds_rad) ? frame.held_reference_bounds_rad : frame.bounds_rad;
    const bounds = Array.isArray(acceptedBounds) ? acceptedBounds : [0,0,0,0,0,0];
    return {
      available,
      degrees: available ? vector.map(v => v * 180 / Math.PI) : [null,null,null],
      limits: [0,1,2].map(i => effective[i] === '1' ? [bounds[2*i] * 180 / Math.PI, bounds[2*i+1] * 180 / Math.PI] : [0,0]),
      semantic, effective,
      held: Boolean(frame.reference_held),
      // A held reference keeps its accepted release; failed candidate values
      // are never substituted into the accepted telemetry or chart.
      rejected: Array.isArray(frame.rejected_rotation_release_rad) ? frame.rejected_rotation_release_rad.map(v=>v*180/Math.PI) : null,
    };
  }
  scope.LiberoRelease = {values};
  if (typeof module !== 'undefined') module.exports = scope.LiberoRelease;
})(typeof window === 'undefined' ? globalThis : window);
