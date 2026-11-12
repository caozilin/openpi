/* Membership comes only from the frozen training replacement request. */
(function(scope) {
  function key(dataset, episode) {
    return `${dataset.suite}/${String(dataset.task_id).padStart(2, '0')}/${episode.path}`;
  }
  function selected(snapshot, dataset, episode) {
    return Boolean(snapshot?.episodes && Object.prototype.hasOwnProperty.call(snapshot.episodes, key(dataset, episode)));
  }
  function visible(snapshot, dataset, episodes) {
    return episodes.filter(episode => selected(snapshot, dataset, episode));
  }
  scope.LiberoTrainingSelection = {key, selected, visible};
  if (typeof module !== 'undefined') module.exports = scope.LiberoTrainingSelection;
})(typeof window === 'undefined' ? globalThis : window);
