import argparse
import csv
import itertools
import json
from pathlib import Path
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL, EPSILON, PairedDataset, SAE_DIR, load_sae
from scripts.experiment2_sae_strength_vs_classification import classification_margin


OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment3_causal_intervention"
EXPERIMENT2_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment2_strength_vs_classification" / "full_1000"
CANDIDATES = [6730, 3122, 22739, 6966]


def downstream_logits(model, hidden):
    assert hidden.ndim == 3 and hidden.shape[1:] == (197, 768)
    hidden = model.vit.layers[11](hidden, attention_mask=None)
    hidden = model.vit.layernorm(hidden)
    return model.classifier(hidden[:, 0])


def logits_metrics(logits, labels):
    true_logit, margin = classification_margin(logits, labels)
    return true_logit, margin, logits.argmax(dim=1)


def patch_delta(decoder_weight, latent_delta, indices):
    selected = torch.as_tensor(indices, device=latent_delta.device, dtype=torch.long)
    return F.linear(latent_delta[..., selected], decoder_weight[:, selected], bias=None)


def bootstrap_ci(values, rng, draws=2000):
    values = np.asarray(values, dtype=np.float64)
    means = np.empty(draws)
    for index in range(draws):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    return np.quantile(means, [.025, .975]).tolist()


def summarize(name, num_features, true_changes, margin_changes, recovered, random_stats=None, matched_stats=None):
    margin_changes = np.asarray(margin_changes)
    true_changes = np.asarray(true_changes)
    try:
        test = wilcoxon(margin_changes)
        pvalue = float(test.pvalue)
    except ValueError:
        pvalue = 1.0
    effect = float(np.mean(margin_changes) / max(np.std(margin_changes, ddof=1), EPSILON))
    rng = np.random.default_rng(0)
    low, high = bootstrap_ci(margin_changes, rng)
    row = {
        "intervention_name": name,
        "num_features": num_features,
        "n_samples": len(margin_changes),
        "mean_true_logit_change": float(true_changes.mean()),
        "median_true_logit_change": float(np.median(true_changes)),
        "mean_margin_change": float(margin_changes.mean()),
        "median_margin_change": float(np.median(margin_changes)),
        "margin_change_95CI_low": low,
        "margin_change_95CI_high": high,
        "percent_margin_improved": float((margin_changes > 0).mean() * 100),
        "num_predictions_recovered": int(np.sum(recovered)),
        "recovery_rate": float(np.mean(recovered)),
        "paired_pvalue": pvalue,
        "effect_size": effect,
        "random_control_mean": None,
        "random_control_std": None,
        "random_empirical_pvalue": None,
        "magnitude_matched_control_mean": None,
        "magnitude_matched_control_std": None,
        "magnitude_matched_empirical_pvalue": None,
    }
    for prefix, stats in [("random", random_stats), ("magnitude_matched", matched_stats)]:
        if stats is not None:
            distribution = np.asarray(stats)
            row[f"{prefix}_control_mean"] = float(distribution.mean())
            row[f"{prefix}_control_std"] = float(distribution.std(ddof=1))
            row[f"{prefix}_empirical_pvalue"] = float((1 + np.sum(distribution >= row["mean_margin_change"])) / (1 + len(distribution)))
    return row


def load_ranked_features():
    with (EXPERIMENT2_ROOT / "features_ranked_by_margin_correlation.csv").open() as source:
        return [int(row["feature_index"]) for row in csv.DictReader(source)]


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: SAE causal intervention")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=11000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-failures", type=int, default=0)
    parser.add_argument("--random-draws", type=int, default=100)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists(): raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PairedDataset(args.samples, args.start_index, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval(); model.requires_grad_(False)
    sae, metadata = load_sae(device)
    assert max(CANDIDATES) < sae.latent_dim
    ranked = load_ranked_features()
    decoder_weight = sae.decoder.weight

    print("Architecture: ViT-B/16, 12 pre-LN blocks, hidden=768, tokens=197")
    print("Intervention: block-10 output hidden_states[-2]; preserve CLS; replace decoded 196 patches")
    print("Downstream: block 11 -> final LayerNorm -> CLS -> classifier")
    print(f"SAE: Vanilla/ReLU, patch [B,196,768] -> latent [B,196,{sae.latent_dim}] -> decode [B,196,768]")

    reconstruction = {kind: {key: [] for key in ["mse", "cosine", "relative_error", "agreement", "margin_change", "original_correct", "recon_correct"]} for kind in ["clean", "blur"]}
    failures = []
    clean_correct = blur_correct = 0

    with torch.no_grad():
        for clean, blur, labels, image_ids, paths in tqdm(loader, desc="Baseline/reconstruction"):
            batch = clean.shape[0]; labels = labels.to(device); images = torch.cat([clean, blur]).to(device)
            outputs = model(pixel_values=images, output_hidden_states=True)
            original_logits_clean, original_logits_blur = outputs.logits.split(batch)
            site_clean, site_blur = outputs.hidden_states[-2].split(batch)
            for kind, site, original_logits in [("clean", site_clean, original_logits_clean), ("blur", site_blur, original_logits_blur)]:
                patches = site[:, 1:]; z = sae.encode(patches.flatten(0,1)).reshape(batch,196,-1); decoded = sae.decode(z)
                reconstructed_site = torch.cat([site[:, :1], decoded], dim=1)
                reconstructed_logits = downstream_logits(model, reconstructed_site)
                original_true, original_margin, original_pred = logits_metrics(original_logits, labels)
                recon_true, recon_margin, recon_pred = logits_metrics(reconstructed_logits, labels)
                reconstruction[kind]["mse"].extend((decoded-patches).pow(2).mean(dim=(1,2)).cpu().tolist())
                reconstruction[kind]["cosine"].extend(F.cosine_similarity(patches,decoded,dim=-1).mean(1).cpu().tolist())
                reconstruction[kind]["relative_error"].extend(((decoded-patches).norm(dim=-1)/patches.norm(dim=-1).clamp_min(EPSILON)).mean(1).cpu().tolist())
                reconstruction[kind]["agreement"].extend((recon_pred==original_pred).cpu().tolist())
                reconstruction[kind]["margin_change"].extend((recon_margin-original_margin).cpu().tolist())
                reconstruction[kind]["original_correct"].extend((original_pred==labels).cpu().tolist())
                reconstruction[kind]["recon_correct"].extend((recon_pred==labels).cpu().tolist())

            clean_pred = original_logits_clean.argmax(1); blur_pred = original_logits_blur.argmax(1)
            clean_correct += int((clean_pred==labels).sum()); blur_correct += int((blur_pred==labels).sum())
            clean_z = sae.encode(site_clean[:,1:].flatten(0,1)).reshape(batch,196,-1)
            blur_z = sae.encode(site_blur[:,1:].flatten(0,1)).reshape(batch,196,-1)
            clean_decoded = sae.decode(clean_z); blur_decoded = sae.decode(blur_z)
            clean_recon_logits = downstream_logits(model, torch.cat([site_clean[:,:1],clean_decoded],1))
            blur_recon_logits = downstream_logits(model, torch.cat([site_blur[:,:1],blur_decoded],1))
            for index in range(batch):
                if clean_pred[index] == labels[index] and blur_pred[index] != labels[index]:
                    failures.append(dict(
                        image_id=image_ids[index], label=labels[index:index+1].cpu(),
                        clean_pred=int(clean_pred[index]), blur_pred=int(blur_pred[index]),
                        clean_site=site_clean[index:index+1].cpu(), blur_site=site_blur[index:index+1].cpu(),
                        latent_delta=(clean_z[index:index+1]-blur_z[index:index+1]).half().cpu(),
                        candidate_clean=clean_z[index,:,CANDIDATES].mean(0).cpu(),
                        candidate_blur=blur_z[index,:,CANDIDATES].mean(0).cpu(),
                        clean_decoded=clean_decoded[index:index+1].cpu(), blur_decoded=blur_decoded[index:index+1].cpu(),
                        clean_original_logits=original_logits_clean[index:index+1].cpu(), blur_original_logits=original_logits_blur[index:index+1].cpu(),
                        clean_recon_logits=clean_recon_logits[index:index+1].cpu(), blur_recon_logits=blur_recon_logits[index:index+1].cpu(),
                    ))
    if args.samples == 1000 and (clean_correct/args.samples != .798 or blur_correct/args.samples != .633): raise AssertionError("Baseline mismatch")
    if args.max_failures: failures = failures[:args.max_failures]
    print(f"Baseline clean/blur accuracy: {clean_correct/args.samples:.2%}/{blur_correct/args.samples:.2%}; failures={len(failures)}")

    recon_summary = {}
    for kind in ["clean","blur"]:
        r=reconstruction[kind]; recon_summary[kind] = {"hidden_mse":float(np.mean(r["mse"])),"hidden_cosine":float(np.mean(r["cosine"])),"relative_error":float(np.mean(r["relative_error"])),"original_accuracy":float(np.mean(r["original_correct"])),"reconstructed_accuracy":float(np.mean(r["recon_correct"])),"top1_agreement":float(np.mean(r["agreement"])),"mean_margin_change":float(np.mean(r["margin_change"]))}
    print("SAE reconstruction sanity", json.dumps(recon_summary, indent=2))
    if min(recon_summary[k]["hidden_cosine"] for k in recon_summary) < .5: raise RuntimeError("Clearly incorrect reconstruction; stopping")

    intervention_defs = {}
    for feature in CANDIDATES: intervention_defs[f"candidate_{feature}"]=[feature]
    for size in range(1,5): intervention_defs[f"candidate_top_{size}"]=CANDIDATES[:size]
    for size in [1,2,4,8,16,32,64]: intervention_defs[f"global_top_{size}"]=ranked[:size]
    for size in [1,2,4,8,16,32]: intervention_defs[f"personal_top_{size}"]=None
    for size in [2,3]:
        for combo in itertools.combinations(CANDIDATES,size): intervention_defs["combo_"+"_".join(map(str,combo))]=list(combo)

    effects = {name:{"true":[],"margin":[],"recovered":[]} for name in intervention_defs}
    reverse_effects = {name:{"true":[],"margin":[],"wrong":[]} for name in intervention_defs}
    dose = {alpha:[] for alpha in [0,.25,.5,.75,1.]}
    image_rows=[]
    with torch.no_grad():
        for record in tqdm(failures, desc="Restoration"):
            record={key:(value.to(device) if torch.is_tensor(value) else value) for key,value in record.items()}
            label=record["label"]; delta=record["latent_delta"].float()
            base_true,base_margin,_=logits_metrics(record["blur_recon_logits"],label)
            clean_original_margin=logits_metrics(record["clean_original_logits"],label)[1].item(); blur_original_margin=logits_metrics(record["blur_original_logits"],label)[1].item()
            clean_recon_margin=logits_metrics(record["clean_recon_logits"],label)[1].item()
            row={"image_id":record["image_id"],"ground_truth":int(label),"clean_prediction":record["clean_pred"],"blur_prediction":record["blur_pred"],"clean_original_margin":clean_original_margin,"blur_original_margin":blur_original_margin,"clean_reconstructed_margin":clean_recon_margin,"blur_reconstructed_margin":base_margin.item()}
            personal_order=torch.topk(delta.abs().mean(dim=1)[0],64).indices.tolist()
            for name,indices in intervention_defs.items():
                chosen=personal_order[:int(name.rsplit("_",1)[1])] if indices is None else indices
                modified=record["blur_decoded"]+patch_delta(decoder_weight,delta,chosen)
                logits=downstream_logits(model,torch.cat([record["blur_site"][:,:1],modified],1)); true,margin,pred=logits_metrics(logits,label)
                effects[name]["true"].append((true-base_true).item()); effects[name]["margin"].append((margin-base_margin).item()); effects[name]["recovered"].append(bool(pred==label))
                if name.startswith("candidate_") and name.count("_")==1:
                    feature=chosen[0]; position=CANDIDATES.index(feature); row[f"candidate_{feature}_clean_value"]=record["candidate_clean"][position].item(); row[f"candidate_{feature}_blur_value"]=record["candidate_blur"][position].item(); row[f"candidate_{feature}_restored_margin"]=margin.item()
                if name=="candidate_top_4": row.update(all4_restored_margin=margin.item(),all4_restored_prediction=int(pred),all4_margin_recovery=(margin-base_margin).item(),all4_true_logit_recovery=(true-base_true).item())
            for alpha in dose:
                modified=record["blur_decoded"]+alpha*patch_delta(decoder_weight,delta,CANDIDATES)
                dose[alpha].append(logits_metrics(downstream_logits(model,torch.cat([record["blur_site"][:,:1],modified],1)),label)[1].item())
            image_rows.append(row)

        for record in tqdm(failures, desc="Reverse"):
            record={key:(value.to(device) if torch.is_tensor(value) else value) for key,value in record.items()}
            label=record["label"]; delta=-record["latent_delta"].float(); base_true,base_margin,_=logits_metrics(record["clean_recon_logits"],label); personal_order=torch.topk(delta.abs().mean(dim=1)[0],64).indices.tolist()
            for name,indices in intervention_defs.items():
                chosen=personal_order[:int(name.rsplit("_",1)[1])] if indices is None else indices
                modified=record["clean_decoded"]+patch_delta(decoder_weight,delta,chosen); true,margin,pred=logits_metrics(downstream_logits(model,torch.cat([record["clean_site"][:,:1],modified],1)),label)
                reverse_effects[name]["true"].append((true-base_true).item()); reverse_effects[name]["margin"].append((margin-base_margin).item()); reverse_effects[name]["wrong"].append(bool(pred!=label))

    rng=np.random.default_rng(args.seed); eligible=np.array(ranked[:1074]); random_controls={}; matched_controls={}
    for name,indices in intervention_defs.items():
        if indices is None or not (name.startswith("global_top_") or name=="candidate_top_4"): continue
        size=len(indices); random_means=[]; matched_means=[]
        for draw in range(args.random_draws):
            random_set=rng.choice(eligible,size=size,replace=False).tolist(); random_effect=[]; matched_effect=[]
            for record in failures:
                record={key:(value.to(device) if torch.is_tensor(value) else value) for key,value in record.items()}
                label=record["label"]; base_margin=logits_metrics(record["blur_recon_logits"],label)[1]; delta=record["latent_delta"].float()
                magnitudes=delta.abs().mean(dim=(0,1)); target_magnitudes=magnitudes[torch.as_tensor(indices,device=device)]; matched=[]
                for magnitude in target_magnitudes: matched.append(int(torch.argmin((magnitudes-magnitude).abs()+torch.arange(sae.latent_dim,device=device).eq(torch.as_tensor(indices,device=device)[:,None]).any(0)*1e9)))
                for chosen,store in [(random_set,random_effect),(matched,matched_effect)]:
                    modified=record["blur_decoded"]+patch_delta(decoder_weight,delta,chosen); margin=logits_metrics(downstream_logits(model,torch.cat([record["blur_site"][:,:1],modified],1)),label)[1]; store.append((margin-base_margin).item())
            random_means.append(np.mean(random_effect)); matched_means.append(np.mean(matched_effect))
        random_controls[name]=random_means; matched_controls[name]=matched_means

    summaries=[]
    for name,data in effects.items(): summaries.append(summarize(name,len(intervention_defs[name]) if intervention_defs[name] else int(name.rsplit("_",1)[1]),data["true"],data["margin"],data["recovered"],random_controls.get(name),matched_controls.get(name)))
    for name,data in reverse_effects.items(): summaries.append(summarize("reverse_"+name,len(intervention_defs[name]) if intervention_defs[name] else int(name.rsplit("_",1)[1]),data["true"],data["margin"],data["wrong"]))
    fields=list(summaries[0]);
    with (output_dir/"intervention_summary.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(summaries)
    with (output_dir/"image_level_failures.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(image_rows[0]));w.writeheader();w.writerows(image_rows)

    progressive=[next(r for r in summaries if r["intervention_name"]==f"global_top_{n}") for n in [1,2,4,8,16,32,64]]
    plt.figure();plt.plot([r["num_features"] for r in progressive],[r["mean_margin_change"] for r in progressive],marker="o");plt.xscale("log",base=2);plt.xlabel("Restored global features");plt.ylabel("Mean margin recovery");plt.tight_layout();plt.savefig(output_dir/"plot1_features_vs_margin_recovery.png",dpi=250);plt.close()
    plt.figure();plt.plot([r["num_features"] for r in progressive],[r["recovery_rate"] for r in progressive],marker="o");plt.xscale("log",base=2);plt.xlabel("Restored global features");plt.ylabel("Recovery rate");plt.tight_layout();plt.savefig(output_dir/"plot2_features_vs_recovery_rate.png",dpi=250);plt.close()
    target=next(r for r in summaries if r["intervention_name"]=="candidate_top_4");plt.figure();plt.bar(["Targeted","Random"],[target["mean_margin_change"],target["random_control_mean"]],yerr=[0,target["random_control_std"]]);plt.ylabel("Mean margin recovery");plt.tight_layout();plt.savefig(output_dir/"plot3_targeted_vs_random.png",dpi=250);plt.close()
    plt.figure();plt.bar(["Targeted","Magnitude matched"],[target["mean_margin_change"],target["magnitude_matched_control_mean"]],yerr=[0,target["magnitude_matched_control_std"]]);plt.ylabel("Mean margin recovery");plt.tight_layout();plt.savefig(output_dir/"plot4_targeted_vs_matched.png",dpi=250);plt.close()
    reverse=[next(r for r in summaries if r["intervention_name"]==f"reverse_global_top_{n}") for n in [1,2,4,8,16,32,64]];plt.figure();plt.plot([r["num_features"] for r in reverse],[r["mean_margin_change"] for r in reverse],marker="o");plt.xscale("log",base=2);plt.xlabel("Inserted Blur values");plt.ylabel("Mean clean-margin change");plt.tight_layout();plt.savefig(output_dir/"plot5_reverse_margin.png",dpi=250);plt.close()
    plt.figure();plt.plot(list(dose),[np.mean(dose[a]) for a in dose],marker="o");plt.xlabel("Restoration fraction alpha");plt.ylabel("Mean reconstructed Blur margin");plt.tight_layout();plt.savefig(output_dir/"plot6_dose_response.png",dpi=250);plt.close()
    individuals=[next(r for r in summaries if r["intervention_name"]==f"candidate_{f}") for f in CANDIDATES];plt.figure();plt.errorbar([str(f) for f in CANDIDATES],[r["mean_margin_change"] for r in individuals],yerr=[[r["mean_margin_change"]-r["margin_change_95CI_low"] for r in individuals],[r["margin_change_95CI_high"]-r["mean_margin_change"] for r in individuals]],fmt="o");plt.ylabel("Margin effect (95% CI)");plt.tight_layout();plt.savefig(output_dir/"plot7_candidate_effects.png",dpi=250);plt.close()
    best=max(summaries[:len(effects)],key=lambda r:r["mean_margin_change"]);best_values=effects[best["intervention_name"]]["margin"];base=[r["blur_reconstructed_margin"] for r in image_rows];plt.figure();plt.hist(base,bins=30,alpha=.5,label="Before");plt.hist(np.array(base)+np.array(best_values),bins=30,alpha=.5,label="After");plt.legend();plt.xlabel("Ground-truth margin");plt.tight_layout();plt.savefig(output_dir/"plot8_best_before_after.png",dpi=250);plt.close()

    direction={}
    for position,feature in enumerate(CANDIDATES):
        differences=[(r["candidate_clean"][position]-r["candidate_blur"][position]).item() for r in failures];direction[str(feature)]={"restoration_lowers_percent":float(np.mean(np.array(differences)<0)*100),"restoration_raises_percent":float(np.mean(np.array(differences)>0)*100),"mixed_or_equal_percent":float(np.mean(np.array(differences)==0)*100)}
    result={"config":vars(args)|{"model":BASE_MODEL,"sae":str(SAE_DIR.relative_to(PROJECT_ROOT)),"site":"hidden_states[-2] block-10 output; patches only"},"baseline":{"clean_accuracy":clean_correct/args.samples,"blur_accuracy":blur_correct/args.samples,"failure_samples":len(failures)},"reconstruction":recon_summary,"candidate_direction":direction,"dose_response":{str(a):float(np.mean(v)) for a,v in dose.items()},"best_intervention":best,"summaries":summaries}
    (output_dir/"summary.json").write_text(json.dumps(result,indent=2));(output_dir/"config.json").write_text(json.dumps(result["config"],indent=2))
    print("Best intervention",json.dumps(best,indent=2));print(f"Saved Experiment 3 to {output_dir}")


if __name__ == "__main__": main()
