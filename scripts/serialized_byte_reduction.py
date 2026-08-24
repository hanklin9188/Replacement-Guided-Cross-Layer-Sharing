#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
REQUIRED={"method","backbone","nominal_target","dense_serialized_bytes","compressed_serialized_bytes"}
EXPECTED_METHODS={"Basis Sharing","SVD-LLM"};EXPECTED_BACKBONES={"Llama-3.2-3B","Llama-3.1-8B"};EXPECTED_TARGETS={15,20,25}
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args();rows=[];seen=set()
 with Path(a.manifest).open(encoding='utf-8',newline='') as h:
  reader=csv.DictReader(h);missing=REQUIRED-set(reader.fieldnames or [])
  if missing:raise RuntimeError(f'missing columns {sorted(missing)}')
  for n,r in enumerate(reader,start=2):
   key=(r['method'],r['backbone'],int(r['nominal_target']))
   if key in seen:raise RuntimeError(f'duplicate key {key}')
   seen.add(key);dense=int(r['dense_serialized_bytes']);compressed=int(r['compressed_serialized_bytes'])
   if dense<=0 or compressed<=0:raise RuntimeError(f'nonpositive bytes at line {n}')
   rows.append({**r,'nominal_target':int(r['nominal_target']),'dense_serialized_bytes':dense,'compressed_serialized_bytes':compressed,'serialized_byte_reduction_percent':100.0*(dense-compressed)/dense})
 expected={(m,b,t) for m in EXPECTED_METHODS for b in EXPECTED_BACKBONES for t in EXPECTED_TARGETS}
 if seen!=expected:raise RuntimeError(f'coverage mismatch missing={sorted(expected-seen)} extra={sorted(seen-expected)}')
 fields=list(rows[0]);Path(a.output).parent.mkdir(parents=True,exist_ok=True)
 with Path(a.output).open('w',encoding='utf-8',newline='') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
 print(json.dumps({'status':'PASS','rows':len(rows),'output':a.output},indent=2))
if __name__=='__main__':main()
