import argparse,io,json
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageEnhance,ImageFilter
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification

from data.imagenet_dataset import ImageNetDataset
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment24_classification_aware_residual_predictor import HiddenLinear
from scripts.experiment41_disjoint_gate_development import paired_comparison

ROOT=Path(__file__).parent.parent; OUT=ROOT/'results/sae/experiment68_unseen_online_corruptions'; BLOCK=6
CORRUPTIONS=('brightness','contrast','jpeg','pixelate','defocus','shot_noise','impulse_noise')
TABLES={'brightness':[.9,.75,.6,.45,.3],'contrast':[.85,.65,.45,.3,.2],'jpeg':[70,50,35,20,10],'pixelate':[.8,.65,.5,.35,.25],'defocus':[1,2,3,4,6],'shot_noise':[250,100,50,25,12],'impulse_noise':[.01,.025,.05,.08,.12]}

def corrupt(image,name,severity,seed):
 v=TABLES[name][severity-1]; rng=np.random.default_rng(seed)
 if name=='brightness': return ImageEnhance.Brightness(image).enhance(v)
 if name=='contrast': return ImageEnhance.Contrast(image).enhance(v)
 if name=='jpeg':
  buf=io.BytesIO(); image.save(buf,format='JPEG',quality=v); buf.seek(0); return Image.open(buf).convert('RGB')
 if name=='pixelate':
  w,h=image.size; small=image.resize((max(1,int(w*v)),max(1,int(h*v))),Image.Resampling.BOX); return small.resize((w,h),Image.Resampling.NEAREST)
 if name=='defocus': return image.filter(ImageFilter.GaussianBlur(radius=v))
 x=np.asarray(image,dtype=np.float32)/255
 if name=='shot_noise': x=rng.poisson(x*v)/v
 else:
  mask=rng.random(x.shape[:2]); x[mask<v/2]=0; x[(mask>=v/2)&(mask<v)]=1
 return Image.fromarray((np.clip(x,0,1)*255).astype(np.uint8))

class OnlineDataset(ImageNetDataset):
 def __init__(self,*a,corruption_name=None,severity=1,seed=0,**kw): super().__init__(*a,**kw); self.name=corruption_name; self.severity=severity; self.seed=seed
 def __getitem__(self,i):
  image=Image.open(self.image_paths[i]).convert('RGB')
  if self.name: image=corrupt(image,self.name,self.severity,self.seed+self.start_index+i)
  return self.processor(images=image,return_tensors='pt')['pixel_values'].squeeze(0),self.labels[i]

def load_group(root,filename,device):
 out={}
 for seed in (0,1,2):
  a=HiddenLinear().to(device); a.load_state_dict(torch.load(root/f'seed_{seed}'/filename,map_location=device,weights_only=True)); out[seed]=a.eval()
 return out

def evaluate(model,groups,loader,device,args,stat_seed):
 arrays={'baseline':[]}|{f'{g}_seed{s}':[] for g in groups for s in groups[g]}
 with torch.no_grad():
  for images,labels in tqdm(loader,desc='frozen corruption evaluation'):
   labels=labels.to(device); out=model(pixel_values=images.to(device),output_hidden_states=True); arrays['baseline'].extend((out.logits.argmax(1)==labels).cpu().tolist()); h=out.hidden_states[BLOCK]
   for group,adapters in groups.items():
    for seed,a in adapters.items():
     candidate=torch.cat([h[:,:1],h[:,1:]+a(h[:,1:])],1); logits=downstream_from_layer(model,candidate,BLOCK-1); arrays[f'{group}_seed{seed}'].extend((logits.argmax(1)==labels).cpu().tolist())
 arrays={k:np.asarray(v,bool) for k,v in arrays.items()}; result={'baseline_accuracy':float(arrays['baseline'].mean()),'methods':{}}
 for i,(name,v) in enumerate(arrays.items()):
  if name!='baseline': result['methods'][name]=paired_comparison(arrays['baseline'],v,stat_seed+i,args.bootstrap)
 return result,arrays

def main():
 p=argparse.ArgumentParser(); p.add_argument('--samples',type=int,default=3000); p.add_argument('--start-index',type=int,default=39000); p.add_argument('--severities',type=int,nargs='+',default=[1,2,3,4,5]); p.add_argument('--batch-size',type=int,default=8); p.add_argument('--workers',type=int,default=2); p.add_argument('--corruption-seed',type=int,default=2068); p.add_argument('--bootstrap',type=int,default=5000); p.add_argument('--run-name',required=True); args=p.parse_args()
 if args.start_index<39000 or args.start_index+args.samples>42000: raise ValueError('Experiment 68 locked to [39000,42000)')
 out=OUT/args.run_name; out.mkdir(parents=True,exist_ok=False); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('Device:',device)
 model=ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval(); model.requires_grad_(False)
 groups={'mixed':load_group(ROOT/'results/sae/experiment67_mixed_noise_blur_block6/full_3seed_noise_blur_identity0p2_v1','mixed_identity0p2.pt',device),'noise_only':load_group(ROOT/'results/sae/experiment50_block6_clean_preservation/full_3seed_identity_sweep_v1','identity_0p2.pt',device)}
 results={}; outcomes={}; conditions=[('clean',None,0)]+[(f'{c}_{s}',c,s) for c in CORRUPTIONS for s in args.severities]
 for ci,(key,c,s) in enumerate(conditions):
  ds=OnlineDataset(ROOT/'Dataset',args.samples,args.start_index,corruption_name=c,severity=s,seed=args.corruption_seed); loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=args.workers); results[key],arr=evaluate(model,groups,loader,device,args,args.corruption_seed+ci*20)
  for name,v in arr.items(): outcomes[f'{key}__{name}']=v
  (out/f'{key}.json').write_text(json.dumps(results[key],indent=2))
 aggregate={}
 for group in groups:
  gains=[]
  for key,c,s in conditions[1:]:
   gains.extend([results[key]['methods'][f'{group}_seed{seed}']['accuracy_difference'] for seed in groups[group]])
  aggregate[group]={'mean_gain_all_corruptions_severities':float(np.mean(gains)),'per_seed_mean_gain':[float(np.mean([results[key]['methods'][f'{group}_seed{seed}']['accuracy_difference'] for key,_,_ in conditions[1:]])) for seed in groups[group]]}
 summary={'configuration':vars(args)|{'corruptions':CORRUPTIONS,'severity_tables':TABLES,'benchmark_label':'controlled deterministic online benchmark; not official ImageNet-C','model':BASE_MODEL,'vit_frozen':True,'adapters_frozen':True,'evaluation_range':[args.start_index,args.start_index+args.samples],'training_or_tuning':False,'imageNetV2_accessed':False},'results':results,'aggregate':aggregate,'limitations':['Corruptions approximate common families but are not official ImageNet-C implementations.','This reserve benchmark must not be used to retrain or tune the frozen adapters.']}
 np.savez_compressed(out/'paired_outcomes.npz',**outcomes); (out/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(aggregate,indent=2)); print('Saved',out)
if __name__=='__main__': main()
