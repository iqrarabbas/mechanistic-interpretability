import argparse, json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment12_failure_targeted_quantile_repair import PairedCorruptionDataset
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment24_classification_aware_residual_predictor import HiddenLinear
from scripts.experiment41_disjoint_gate_development import paired_comparison

ROOT=Path(__file__).parent.parent
OUT=ROOT/'results/sae/experiment66_block1_mlp_neuron_causality'
KS=(16,32,64,128,256)

def loader(corruption,start,n,seed,batch,workers):
    return DataLoader(PairedCorruptionDataset(corruption,n,start,seed),batch_size=batch,shuffle=False,num_workers=workers)

def parts(layer,x):
    a=layer.dropout(layer.attention(layer.layernorm_before(x))[0])
    y=x+a
    u=layer.mlp.activation_fn(layer.mlp.fc1(layer.layernorm_after(y)))
    return y,u

def margin(logits,labels):
    true=logits.gather(1,labels[:,None]).squeeze(1); other=logits.clone(); other.scatter_(1,labels[:,None],-torch.inf)
    return true-other.max(1).values

def discover(model,corruption,start,n,seed,args,device):
    sums={k:np.zeros(3072,np.float64) for k in ('x','x2','xy')}; sy=sy2=0.; count=0
    layer=model.vit.layers[0]
    with torch.no_grad():
        for clean,bad,labels in tqdm(loader(corruption,start,n,seed,args.batch,args.workers),desc=f'discover {corruption}'):
            labels=labels.to(device); out=model(pixel_values=torch.cat([clean,bad]).to(device),output_hidden_states=True); b=len(clean)
            _,cu=parts(layer,out.hidden_states[0][:b]); _,bu=parts(layer,out.hidden_states[0][b:])
            x=(bu-cu).mean(1).cpu().numpy(); y=(margin(out.logits[:b],labels)-margin(out.logits[b:],labels)).cpu().numpy()
            sums['x']+=x.sum(0); sums['x2']+=(x*x).sum(0); sums['xy']+=(x*y[:,None]).sum(0); sy+=y.sum(); sy2+=(y*y).sum(); count+=b
    mx=sums['x']/count; my=sy/count; vx=np.maximum(sums['x2']/count-mx*mx,1e-12); vy=max(sy2/count-my*my,1e-12)
    corr=(sums['xy']/count-mx*my)/np.sqrt(vx*vy); score=np.abs(corr)*np.abs(mx)/np.sqrt(vx)
    return {'mean_delta':mx,'correlation':corr,'score':score}

def load_adapters(root,device):
    out={}
    for seed in (0,1,2):
        a=HiddenLinear().to(device); a.load_state_dict(torch.load(root/f'seed_{seed}/identity_0p2.pt',map_location=device,weights_only=True)); out[seed]=a.eval()
    return out

def evaluate(model,adapters,features,corruption,start,n,seed,args,device):
    names=['baseline','block6']+[f'k{k}' for k in KS]+[f'k{k}_block6' for k in KS]+[f'random{k}' for k in KS]
    stores={s:{name:[] for name in names} for s in adapters}; layer=model.vit.layers[0]; rng=np.random.default_rng(args.control_seed)
    random={k:rng.choice(np.setdiff1d(np.arange(3072),features[:k]),k,replace=False) for k in KS}
    with torch.no_grad():
        for clean,bad,labels in tqdm(loader(corruption,start,n,seed,args.batch,args.workers),desc=f'eval {corruption}'):
            labels=labels.to(device); out=model(pixel_values=torch.cat([clean,bad]).to(device),output_hidden_states=True); b=len(clean)
            cy,cu=parts(layer,out.hidden_states[0][:b]); by,bu=parts(layer,out.hidden_states[0][b:]); base=(out.logits[b:].argmax(1)==labels).cpu().numpy()
            variants={'baseline':out.hidden_states[1][b:]}
            for k in KS:
                selected=features[:k]; control=random[k]
                u=bu.clone(); u[:,:,selected]=cu[:,:,selected]; variants[f'k{k}']=by+layer.dropout(layer.mlp.fc2(u))
                selected_delta=torch.nn.functional.linear(cu[:,:,selected]-bu[:,:,selected],layer.mlp.fc2.weight[:,selected])
                control_change=cu[:,:,control]-bu[:,:,control]
                control_delta=torch.nn.functional.linear(control_change,layer.mlp.fc2.weight[:,control])
                scale=(selected_delta.norm(dim=-1)/control_delta.norm(dim=-1).clamp_min(1e-8)).unsqueeze(-1)
                u=bu.clone(); u[:,:,control]=bu[:,:,control]+scale*control_change
                variants[f'random{k}']=by+layer.dropout(layer.mlp.fc2(u))
            for s,a in adapters.items():
                stores[s]['baseline'].extend(base)
                for name,h1 in variants.items():
                    h=h1
                    for block in range(1,6): h=model.vit.layers[block](h,attention_mask=None)
                    h6=torch.cat([h[:,:1],h[:,1:]+a(h[:,1:])],1)
                    if name=='baseline': stores[s]['block6'].extend((downstream_from_layer(model,h6,5).argmax(1)==labels).cpu().numpy())
                    elif name.startswith('k'):
                        stores[s][name].extend((downstream_from_layer(model,h,5).argmax(1)==labels).cpu().numpy())
                        stores[s][name+'_block6'].extend((downstream_from_layer(model,h6,5).argmax(1)==labels).cpu().numpy())
                    else: stores[s][name].extend((downstream_from_layer(model,h,5).argmax(1)==labels).cpu().numpy())
    return {s:{k:np.asarray(v,bool) for k,v in d.items()} for s,d in stores.items()}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run-name',required=True); p.add_argument('--discovery-samples',type=int,default=5000); p.add_argument('--confirmation-samples',type=int,default=2000); p.add_argument('--evaluation-samples',type=int,default=3000); p.add_argument('--batch',type=int,default=4); p.add_argument('--workers',type=int,default=0); p.add_argument('--seed',type=int,default=2066); p.add_argument('--control-seed',type=int,default=6600); p.add_argument('--bootstrap',type=int,default=5000); p.add_argument('--max-samples',type=int); args=p.parse_args()
    if args.max_samples: args.discovery_samples=args.confirmation_samples=args.evaluation_samples=args.max_samples
    ranges={'discovery':[0,args.discovery_samples],'confirmation':[11000,11000+args.confirmation_samples],'evaluation':[36000,36000+args.evaluation_samples]}
    if ranges['evaluation'][1]>50000: raise ValueError('evaluation exceeds reserve')
    out=OUT/args.run_name; out.mkdir(parents=True,exist_ok=False); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('Device:',device)
    model=ViTForImageClassification.from_pretrained(BASE_MODEL,attn_implementation='eager').to(device).eval(); model.requires_grad_(False)
    discovery={c:discover(model,c,0,args.discovery_samples,args.seed,args,device) for c in ('noise','blur')}; confirmation={c:discover(model,c,11000,args.confirmation_samples,args.seed+1,args,device) for c in ('noise','blur')}
    combined=np.sqrt(discovery['noise']['score']*discovery['blur']['score']); order=np.argsort(combined)[::-1].copy(); confirm_order=np.argsort(np.sqrt(confirmation['noise']['score']*confirmation['blur']['score']))[::-1].copy()
    adapters=load_adapters(ROOT/'results/sae/experiment50_block6_clean_preservation/full_3seed_identity_sweep_v1',device)
    evaluations={}
    for c in ('noise','blur'):
        ev=evaluate(model,adapters,order,c,36000,args.evaluation_samples,args.seed+2,args,device); evaluations[c]={}
        for s,d in ev.items():
            np.savez_compressed(out/f'{c}_seed{s}_outcomes.npz',**d); evaluations[c][str(s)]={name:paired_comparison(d['baseline'],v,args.seed+s+len(name),args.bootstrap) for name,v in d.items() if name!='baseline'}
    summary={'configuration':vars(args)|{'model':BASE_MODEL,'model_frozen':True,'block6_adapter_frozen':True,'ranges':ranges,'imageNetV2_accessed':False},'selected_features':order[:max(KS)].tolist(),'top16_confirmation_overlap':len(set(order[:16])&set(confirm_order[:16])),'evaluations':evaluations,'limitations':['Block-1 neuron restoration is paired-clean oracle analysis, not deployment.','Selection uses feature-development data; causal evaluation uses disjoint reserve data.']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps({'top16_confirmation_overlap':summary['top16_confirmation_overlap']},indent=2)); print('Saved',out)
if __name__=='__main__': main()
