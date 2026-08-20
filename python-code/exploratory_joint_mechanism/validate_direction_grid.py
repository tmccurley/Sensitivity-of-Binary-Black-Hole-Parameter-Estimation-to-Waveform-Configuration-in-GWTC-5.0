from __future__ import annotations
import csv
import json
import numpy as np
import joint_geometry_analysis as j


def run(stage, events, samples):
    rows=[]
    for i,event in enumerate(events,1):
        print(f"[{stage}] {i}/{len(events)} {event}",flush=True)
        xq=samples[(event,j.XPHM,'mass_ratio')]; yq=samples[(event,j.XPNR,'mass_ratio')]
        xc=samples[(event,j.XPHM,'chi_eff')]; yc=samples[(event,j.XPNR,'chi_eff')]
        qs=0.5*(j.central_width(xq)+j.central_width(yq)); cs=0.5*(j.central_width(xc)+j.central_width(yc))
        value,_,_=j.sliced_wasserstein(xq,xc,yq,yc,qs,cs)
        rows.append({'stage':stage,'event':event,'sliced_w1_180':value})
    return rows


def main():
    j.DIRECTION_COUNT=180
    ce,cs,_=j.load_confirmatory(); re,rs,_=j.load_replication()
    rows=run('confirmatory',ce,cs)+run('replication',re,rs)
    with open(j.OUTPUT/'joint_geometry_event_metrics.csv',newline='',encoding='utf-8-sig') as f:
        primary={(r['stage'],r['event']):float(r['joint_q_chi_eff_sliced_w1']) for r in csv.DictReader(f)}
    diffs=[]
    for row in rows:
        v360=primary[(row['stage'],row['event'])]; v180=row['sliced_w1_180']
        row['sliced_w1_360']=v360; row['absolute_difference']=abs(v180-v360); row['relative_difference']=abs(v180-v360)/v360
        diffs.append(row['relative_difference'])
    with open(j.OUTPUT/'joint_direction_grid_validation_180_vs_360.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary={'event_count':len(rows),'maximum_relative_difference':max(diffs),'median_relative_difference':float(np.median(diffs))}
    (j.OUTPUT/'joint_direction_grid_validation.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
