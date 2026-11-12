from pathlib import Path
import json, subprocess
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
files=[]
for suite in ['libero_object','libero_goal','libero_10']:
 for p in sorted((ROOT/'data/libero/videos'/f'{suite}_pi05_libero_lora'/'29999').glob('*failure.mp4')):
  meta=json.loads(subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,width,height','-of','json',str(p)],capture_output=True,text=True,check=True).stdout)['streams'][0]
  count=int(meta['nb_frames'])
  if count<=1:continue
  name=p.stem.removeprefix('rollout_').removesuffix('_failure')
  indices=sorted(set(round(i*(count-1)/15) for i in range(16)))
  folder=OUT/'frames'/name;folder.mkdir(exist_ok=True)
  expr='+'.join(f'eq(n,{i})' for i in indices)
  subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(p),'-vf',f'select={expr.replace(",",chr(92)+",")}', '-vsync','0','-threads','1',str(folder/'%02d.png')],check=True)
  grid=Image.new('RGB',(4*336,4*360+55),'white');draw=ImageDraw.Draw(grid);draw.text((8,8),suite+' / '+name,fill='black')
  for j,(i,f) in enumerate(zip(indices,sorted(folder.glob('*.png')),strict=True)):
   x=j%4*336;y=j//4*360+55
   grid.paste(Image.open(f).resize((336,336)),(x,y));draw.text((x+8,y+339),f'frame {i+1}/{count}',fill='black')
  target=OUT/(name+'.jpg');grid.save(target,quality=92)
  files.append({'suite':suite,'source':str(p),'frames':count,'sheet':str(target),'source_resolution':[meta['width'],meta['height']]})
(OUT/'video_inventory.json').write_text(json.dumps(files,indent=2)+'\n')
print(json.dumps(files,indent=2))
