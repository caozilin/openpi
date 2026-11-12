const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const {sync}=require('./video_sync.js');
function video(duration=5){return {duration,readyState:1,seeking:false,currentTime:0,paused:true,playbackRate:1,src:'test.mp4',
 getAttribute(k){return this[k];},removeAttribute(k){delete this[k];},load(){this.currentTime=0;},
 pause(){this.paused=true;},play(){this.paused=false;return Promise.resolve();},addEventListener(){}};}
const master=video(5),slave=video(2);
master.currentTime=1;master.paused=false;sync(slave,master,2,true);
assert.equal(slave.currentTime,1);assert.equal(slave.playbackRate,2);assert.equal(slave.paused,false);
master.currentTime=4;sync(slave,master,1,true);assert.equal(slave.currentTime,1.999);assert.equal(slave.paused,true);
master.currentTime=.5;master.paused=true;sync(slave,master,.5,true);assert.equal(slave.currentTime,.5);assert.equal(slave.paused,true);
slave.readyState=0;slave.duration=NaN;sync(slave,master,1,true);assert.equal(slave.currentTime,.5);
slave.removeAttribute('src');sync(slave,master,1,true);sync(null,master,1,true);

// Execute the real four-video controller with a small DOM/video double, without a server.
const elements=Object.fromEntries(['toleranceVideo','nominalWristVideo','toleranceWristVideo'].map(k=>[k,video()]));
for(const k of ['toleranceCaption','toleranceWristCaption','episodeScope','comparisonStatus','toleranceVideoState','playButton'])elements[k]={textContent:''};
elements['.comparison-toolbar']={appendChild(){}};
elements.useRecovery={checked:false};
const main=video(5),state={speed:1,frameIndex:0,frames:[{}],episode:{path:'demo_2',wrist_video_path:'nominal/wrist.mp4',
 tolerance:{video:'tolerance/main.mp4',wrist_video:'tolerance/wrist.mp4',frames:40,result:{success:false,executed_frames:40,planned_frames:100}}}};
const context={console,state,mainVideo:main,LiberoVideoSync:{sync},$:s=>elements[s.startsWith('#')?s.slice(1):s],
 document:{createElement:()=>({innerHTML:'',classList:{add(){}}})},
 configureVideos(){main.currentTime=0;},pauseVideos(){main.pause();},seekFrame(i){main.currentTime=i/20;},
 togglePlayback(){},animationLoop(){},requestAnimationFrame(){},showError(m){throw new Error(m);}};
const source=fs.readFileSync(__dirname+'/tolerance_comparison.js','utf8').split('renderSharedPrediction=function(){')[0];
vm.createContext(context);vm.runInContext(source,context);
(async()=>{
 context.configureVideos();assert.equal(elements.nominalWristVideo.src,'nominal/wrist.mp4');assert.equal(elements.toleranceWristVideo.src,'tolerance/wrist.mp4');
 await context.togglePlayback();for(const v of [main,elements.toleranceVideo,elements.nominalWristVideo,elements.toleranceWristVideo])assert.equal(v.paused,false);
 context.seekFrame(10);assert.equal(main.currentTime,.5);assert.equal(elements.nominalWristVideo.currentTime,.5);assert.equal(elements.toleranceWristVideo.currentTime,.5);
 context.pauseVideos();for(const v of [main,elements.toleranceVideo,elements.nominalWristVideo,elements.toleranceWristVideo])assert.equal(v.paused,true);
 // Switching to an unlabelled demo must unload both old tolerance camera streams.
 state.episode={path:'demo_4',wrist_video_path:'next/wrist.mp4'};context.configureVideos();
 assert.equal(elements.nominalWristVideo.src,'next/wrist.mp4');assert.equal(elements.toleranceVideo.src,undefined);assert.equal(elements.toleranceWristVideo.src,undefined);
 await context.togglePlayback();assert.equal(main.paused,false);assert.equal(elements.nominalWristVideo.paused,false);assert.equal(elements.toleranceWristVideo.paused,true);
 console.log('PASS: four-camera configure/play/pause/seek, shorter-video hold, missing tolerance, speed sync');
})().catch(e=>{console.error(e);process.exitCode=1;});
