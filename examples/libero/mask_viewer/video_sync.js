"use strict";
// One master clock; a shorter rollout holds its final decoded frame.
(function(root){
  function sync(video, master, speed, force=false){
    if(!video?.getAttribute('src'))return;
    video.playbackRate=speed;
    if(video.readyState<1||!Number.isFinite(video.duration)||video.duration<=0)return;
    const at=Math.max(0,Math.min(master.currentTime,Math.max(0,video.duration-.001)));
    if(!video.seeking&&(force||Math.abs(video.currentTime-at)>.025))video.currentTime=at;
    if(master.paused||master.currentTime>=video.duration-.001)video.pause();
    else if(video.paused)video.play().catch(()=>{});
  }
  const api={sync};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
  root.LiberoVideoSync=api;
})(typeof window!=='undefined'?window:globalThis);
