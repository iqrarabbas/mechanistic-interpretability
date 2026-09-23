import argparse,json
from pathlib import Path
import numpy as np,torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
from transformers import ViTForImageClassification

from data.imagenet_dataset import ImageNetDataset
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment24_classification_aware_residual_predictor import HiddenLinear
from scripts.experiment39_disjoint_adapter_training import DEFAULT_MANIFEST,locked_seed_splits
from scripts.experiment41_disjoint_gate_development import paired_comparison
from scripts.experiment50_block6_clean_preservation import adapter_loss

ROOT=Path(__file__).parent.parent; OUT=ROOT/'results/sae/experiment67_mixed_noise_blur_block6'; BLOCK=6

class Triple(Dataset):
 def __init__(self,split,seed,max_samples=None):
  n=split['end']-split['start']; n=min(n,max_samples) if max_samples else n; common=dict(dataset_dir=ROOT/'Dataset',max_samples=n,start_index=split['start'])
  self.clean=ImageNetDataset(**common); self.blur=ImageNetDataset(**common,corruption='blur',blur_severity=4); self.noise=ImageNetDataset(**common,corruption='noise',noise_severity=4,corruption_seed=seed)
 def __len__(self): return len(self.clean)
 def __getitem__(self,i):
  c,y=self.clean[i]; b,yb=self.blur[i]; n,yn=self.noise[i]
  if y!=yb or y!=yn: raise RuntimeError('label mismatch')
  return c,b,n,y

def make_loader(split,seed,args,shuffle):
 g=torch.Generator().manual_seed(seed+6700)
 return DataLoader(Triple(split,seed,args.max_samples),batch_size=args.batch_size,shuffle=shuffle,num_workers=args.workers,generator=g if shuffle else None)

def epoch(model,adapter,loader,device,args,opt=None):
 train=opt is not None; adapter.train(train); total=np.zeros(5); count=0
 for clean,blur,noise,labels in tqdm(loader,desc='mixed train' if train else 'mixed validation',leave=False):
  labels=labels.to(device)
  with torch.no_grad(): h=model(pixel_values=torch.cat([clean,blur,noise]).to(device),output_hidden_states=True).hidden_states[BLOCK]; c,b,n=h.split(len(clean))
  if train: opt.zero_grad(set_to_none=True)
  lb=adapter_loss(model,adapter,c,b,labels,args.identity_weight,args); ln=adapter_loss(model,adapter,c,n,labels,args.identity_weight,args); losses=tuple((x+y)/2 for x,y in zip(lb,ln))
  if train: losses[0].backward(); torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.0); opt.step()
  total+=np.array([float(x.detach()) for x in losses])*len(clean); count+=len(clean)
 return dict(zip(['total','residual','classification','preservation','identity'],(total/count).tolist()))

def evaluate(model,adapter,loader,device,args,seed):
 arrays={k:[] for k in ['clean','blur','noise','adapted_clean','adapted_blur','adapted_noise']}
 with torch.no_grad():
  for clean,blur,noise,labels in tqdm(loader,desc='mixed paired evaluation'):
   labels=labels.to(device); out=model(pixel_values=torch.cat([clean,blur,noise]).to(device),output_hidden_states=True); m=len(clean); logits=out.logits.split(m); hidden=out.hidden_states[BLOCK].split(m)
   for name,l in zip(['clean','blur','noise'],logits): arrays[name].extend((l.argmax(1)==labels).cpu().tolist())
   for name,h in zip(['adapted_clean','adapted_blur','adapted_noise'],hidden):
    candidate=torch.cat([h[:,:1],h[:,1:]+args.alpha*adapter(h[:,1:])],1); l=downstream_from_layer(model,candidate,BLOCK-1); arrays[name].extend((l.argmax(1)==labels).cpu().tolist())
 arrays={k:np.asarray(v,bool) for k,v in arrays.items()}
 return {c:paired_comparison(arrays[c],arrays['adapted_'+c],seed+6700+i,args.bootstrap) for i,c in enumerate(['clean','noise','blur'])},arrays

def main():
 p=argparse.ArgumentParser(); p.add_argument('--split-manifest',type=Path,default=DEFAULT_MANIFEST); p.add_argument('--split-protocol',default='proposed_protocol'); p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]); p.add_argument('--identity-weight',type=float,default=.2); p.add_argument('--classification-weight',type=float,default=.05); p.add_argument('--preservation-weight',type=float,default=.05); p.add_argument('--train-alpha',type=float,default=.5); p.add_argument('--alpha',type=float,default=1.); p.add_argument('--epochs',type=int,default=5); p.add_argument('--patience',type=int,default=2); p.add_argument('--learning-rate',type=float,default=1e-4); p.add_argument('--weight-decay',type=float,default=1e-4); p.add_argument('--smooth-l1-beta',type=float,default=1.); p.add_argument('--batch-size',type=int,default=4); p.add_argument('--workers',type=int,default=0); p.add_argument('--bootstrap',type=int,default=5000); p.add_argument('--max-samples',type=int); p.add_argument('--run-name',required=True); p.add_argument('--resume',action='store_true'); args=p.parse_args()
 out=OUT/args.run_name
 if out.exists() and not args.resume: raise FileExistsError(out)
 out.mkdir(parents=True,exist_ok=args.resume); splits=locked_seed_splits(args.split_manifest,args.split_protocol,args.seeds); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('Device:',device)
 model=ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval(); model.requires_grad_(False); training={}; validation={}; outcomes={}
 for seed in args.seeds:
  torch.manual_seed(seed+6700); np.random.seed(seed+6700); sd=out/f'seed_{seed}'; sd.mkdir(exist_ok=True); ck=sd/'mixed_identity0p2.pt'; adapter=HiddenLinear().to(device)
  if ck.exists() and args.resume: adapter.load_state_dict(torch.load(ck,map_location=device,weights_only=True)); training[str(seed)]=json.loads((sd/'training.json').read_text())
  else:
   for p0 in adapter.parameters(): torch.nn.init.zeros_(p0)
   opt=AdamW(adapter.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay); sch=CosineAnnealingLR(opt,T_max=args.epochs); best=float('inf'); stale=0; hist=[]
   tr=make_loader(splits[seed]['train'],seed,args,True); va=make_loader(splits[seed]['validation'],seed,args,False)
   for e in range(1,args.epochs+1):
    tm=epoch(model,adapter,tr,device,args,opt); vm=epoch(model,adapter,va,device,args); hist.append({'epoch':e,'train':tm,'validation':vm}); sch.step(); print(seed,e,vm['total'])
    if vm['total']<best: best=vm['total']; stale=0; torch.save(adapter.state_dict(),ck)
    else: stale+=1
    if stale>=args.patience: break
   training[str(seed)]={'best_validation_total':best,'history':hist}; (sd/'training.json').write_text(json.dumps(training[str(seed)],indent=2)); adapter.load_state_dict(torch.load(ck,map_location=device,weights_only=True))
  result,arr=evaluate(model,adapter,make_loader(splits[seed]['validation'],seed,args,False),device,args,seed); validation[str(seed)]=result
  for k,v in arr.items(): outcomes[f'seed{seed}_{k}']=v
 agg={c:{'gains_by_seed':[validation[str(s)][c]['accuracy_difference'] for s in args.seeds]} for c in ['clean','noise','blur']}
 for c in agg: agg[c]['mean_gain']=float(np.mean(agg[c]['gains_by_seed']))
 summary={'configuration':vars(args)|{'split_manifest':str(args.split_manifest.resolve()),'model':BASE_MODEL,'vit_frozen':True,'adapter_parameters':sum(p.numel() for p in HiddenLinear().parameters()),'training_corruptions':['noise4','blur4'],'imageNetV2_accessed':False,'status':'development comparison'},'splits':{str(s):splits[s] for s in args.seeds},'training':training,'validation':validation,'aggregate':agg,'limitations':['Only Noise-4 and Blur-4 are implemented online; unseen corruption families require a separate standardized benchmark.']}
 np.savez_compressed(out/'paired_validation_outcomes.npz',**outcomes); (out/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(agg,indent=2)); print('Saved',out)
if __name__=='__main__': main()
