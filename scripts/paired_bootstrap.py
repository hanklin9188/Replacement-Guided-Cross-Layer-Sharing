#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REQUIRED = {"model_id", "seed", "task", "example_id", "source_index", "gold_answer", "prediction", "correct"}
EXPECTED_MODELS = {"3b_15", "3b_20", "3b_25", "8b_15", "8b_20", "8b_25"}
EXPECTED_SEEDS = {42, 43, 44}
EXPECTED_TASKS = {"ARC-Challenge", "ARC-Easy", "hellaswag", "openbookqa", "piqa", "social_i_qa", "winogrande"}
TASK_ALIASES = {
    "ARC-Challenge": "ARC-Challenge", "ARC_C": "ARC-Challenge",
    "ARC-Easy": "ARC-Easy", "ARC_E": "ARC-Easy",
    "hellaswag": "hellaswag", "HellaSwag": "hellaswag",
    "openbookqa": "openbookqa", "OpenBookQA": "openbookqa",
    "piqa": "piqa", "PIQA": "piqa",
    "social_i_qa": "social_i_qa", "SocialIQA": "social_i_qa",
    "winogrande": "winogrande", "WinoGrande": "winogrande",
}


def boolean(value):
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}: return True
    if text in {"0", "false", "no"}: return False
    raise ValueError(f"invalid correct value: {value!r}")


def normalize_answer(task, value):
    text = str(value).strip()
    if not text.lstrip('-').isdigit():
        return text
    index = int(text)
    prefixes = {
        "ARC-Challenge": "answer", "ARC-Easy": "answer", "openbookqa": "answer",
        "social_i_qa": "answer", "hellaswag": "ending", "piqa": "solution",
        "winogrande": "option",
    }
    if index < 0:
        raise ValueError(f"negative answer index: task={task} value={value!r}")
    return f"{prefixes[task]}{index + 1}"


def load_predictions(path, expected_method):
    groups = defaultdict(dict)
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing: raise RuntimeError(f"{path}: missing columns {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            if row.get("method") and row["method"] != expected_method:
                raise RuntimeError(f"{path}:{line_number}: method={row['method']!r}, expected={expected_method!r}")
            raw_task = row["task"]
            if raw_task not in TASK_ALIASES:
                raise RuntimeError(f"{path}:{line_number}: unsupported task label {raw_task!r}")
            key = (row["model_id"], int(row["seed"]), TASK_ALIASES[raw_task])
            source_index = int(row["source_index"])
            example_id = str(row["example_id"])
            if source_index in groups[key]: raise RuntimeError(f"{path}:{line_number}: duplicate {key}/source_index={source_index}")
            gold = normalize_answer(key[2], row["gold_answer"])
            prediction = normalize_answer(key[2], row["prediction"])
            correct = boolean(row["correct"])
            if correct != (prediction == gold):
                raise RuntimeError(f"{path}:{line_number}: normalized prediction/gold disagrees with correct")
            groups[key][source_index] = {"gold":gold,"prediction":prediction,"correct":correct,"original_example_id":example_id,"canonical_example_id":f"{key[2]}:{source_index}"}
    expected_keys={(model,seed,task) for model in EXPECTED_MODELS for seed in EXPECTED_SEEDS for task in EXPECTED_TASKS}
    if set(groups)!=expected_keys:
        raise RuntimeError(f"{path}: key coverage mismatch missing={sorted(expected_keys-set(groups))[:20]} extra={sorted(set(groups)-expected_keys)[:20]}")
    return groups


def quantile(values):
    return float(np.quantile(values,0.025)),float(np.quantile(values,0.975))


def compare(ours, other, other_method, samples, rng):
    results=[];coverage=[]
    for model_id in sorted(EXPECTED_MODELS):
        for seed in sorted(EXPECTED_SEEDS):
            task_boot=[]; total_n=0
            for task in sorted(EXPECTED_TASKS):
                key=(model_id,seed,task); left=ours[key];right=other[key]
                if set(left)!=set(right):
                    raise RuntimeError(f"Ours vs {other_method} {key}: source-index mismatch ours_only={sorted(set(left)-set(right))[:10]} other_only={sorted(set(right)-set(left))[:10]}")
                ids=sorted(left)
                if any(left[x]["canonical_example_id"]!=right[x]["canonical_example_id"] for x in ids):raise RuntimeError(f"Ours vs {other_method} {key}: canonical ID mismatch")
                if any(left[x]["gold"]!=right[x]["gold"] for x in ids):raise RuntimeError(f"Ours vs {other_method} {key}: normalized gold mismatch")
                diff=np.asarray([float(left[x]["correct"])-float(right[x]["correct"]) for x in ids],dtype=np.float64)
                boot_sum=np.zeros(samples,dtype=np.float64)
                chunk=512
                for start in range(0,len(diff),chunk):
                    width=min(chunk,len(diff)-start)
                    picks=rng.integers(0,len(diff),size=(samples,width),dtype=np.int32)
                    boot_sum+=diff[picks].sum(axis=1)
                boot=boot_sum/len(diff); low,high=quantile(boot)
                results.append({"comparison":f"Ours-{other_method}","model_id":model_id,"seed":seed,"task":task,"n":len(diff),"ours_accuracy":float(np.mean([left[x]['correct'] for x in ids])),"other_accuracy":float(np.mean([right[x]['correct'] for x in ids])),"delta_accuracy":float(diff.mean()),"ci95_low":low,"ci95_high":high,"significant":bool(low>0 or high<0),"bootstrap_samples":samples})
                coverage.append({"comparison":f"Ours-{other_method}","model_id":model_id,"seed":seed,"task":task,"ours_n":len(left),"other_n":len(right),"matched_n":len(ids),"canonical_id_match":True,"source_index_match":True,"normalized_gold_match":True,"original_id_scheme_same":all(left[x]["original_example_id"]==right[x]["original_example_id"] for x in ids)})
                task_boot.append(boot);total_n+=len(diff)
            macro_boot=np.stack(task_boot,axis=1).mean(axis=1);low,high=quantile(macro_boot)
            task_results=results[-len(EXPECTED_TASKS):]
            results.append({"comparison":f"Ours-{other_method}","model_id":model_id,"seed":seed,"task":"macro","n":total_n,"ours_accuracy":float(np.mean([x['ours_accuracy'] for x in task_results])),"other_accuracy":float(np.mean([x['other_accuracy'] for x in task_results])),"delta_accuracy":float(np.mean([x['delta_accuracy'] for x in task_results])),"ci95_low":low,"ci95_high":high,"significant":bool(low>0 or high<0),"bootstrap_samples":samples})
    return results,coverage


def write(path,rows):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--ours',required=True);p.add_argument('--basis',required=True);p.add_argument('--svd',required=True);p.add_argument('--output',required=True);p.add_argument('--coverage-output',required=True);p.add_argument('--samples',type=int,default=10000);p.add_argument('--seed',type=int,default=2027);a=p.parse_args()
    ours=load_predictions(a.ours,'Ours');basis=load_predictions(a.basis,'Basis Sharing');svd=load_predictions(a.svd,'SVD-LLM');rng=np.random.default_rng(a.seed)
    r1,c1=compare(ours,basis,'Basis Sharing',a.samples,rng);r2,c2=compare(ours,svd,'SVD-LLM',a.samples,rng);write(a.output,r1+r2);write(a.coverage_output,c1+c2)
    print(json.dumps({'status':'PASS','result_rows':len(r1)+len(r2),'coverage_rows':len(c1)+len(c2),'bootstrap_samples':a.samples},indent=2))
if __name__=='__main__':main()
