#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / 'data' / 'ours'
EXTERNAL = ROOT / 'data' / 'processed'
PACKAGE = ROOT / 'assets' / 'figures'
FIGURE_DIR = ROOT / 'paper' / 'Figure'
DATA_DIR = ROOT / 'data' / 'processed'

COLORS = {'Ours':'#0072B2','Basis Sharing':'#D55E00','SVD-LLM':'#009E73'}
BACKBONE_STYLE = {'Llama-3.2-3B':('-', 'o', '3B'),'Llama-3.1-8B':('--','s','8B')}
BACKBONE_LAYERS = {'Llama-3.2-3B':28,'Llama-3.1-8B':32}
ORDER_METHOD = ['Basis Sharing','SVD-LLM','Ours']
ORDER_BACKBONE = ['Llama-3.2-3B','Llama-3.1-8B']
TARGETS = [15,20,25]

mpl.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
    'font.size':8.0,'axes.titlesize':8.5,'axes.labelsize':8.0,
    'xtick.labelsize':7.2,'ytick.labelsize':7.2,'legend.fontsize':6.5,
    'axes.linewidth':0.75,'lines.linewidth':1.55,'lines.markersize':4.2,
    'xtick.major.width':0.65,'ytick.major.width':0.65,
    'xtick.direction':'out','ytick.direction':'out',
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.color':'#D9D9D9','grid.linewidth':0.45,'grid.alpha':0.75,
    'figure.facecolor':'white','axes.facecolor':'white',
    'savefig.facecolor':'white','savefig.bbox':'tight','savefig.pad_inches':0.06,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
})


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer=csv.DictWriter(handle, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def c_star(row):
    """Return simultaneous-deployment distortion normalized per logical layer."""
    return float(row['C_joint']) / BACKBONE_LAYERS[row['backbone']]


def annotate_joint_point(ax, row, x, y):
    """Place operating-point labels without collisions across the two backbones."""
    key=(row['backbone'],int(row['nominal_target']))
    offsets={
        ('Llama-3.2-3B',15):(3,-8),
        ('Llama-3.2-3B',20):(3,-8),
        ('Llama-3.2-3B',25):(3,3),
        ('Llama-3.1-8B',15):(-3,5),
        ('Llama-3.1-8B',20):(-3,3),
        ('Llama-3.1-8B',25):(-3,3),
    }
    offset=offsets[key]
    ax.annotate(
        f"{row['nominal_target']}%",(x,y),xytext=offset,
        textcoords='offset points',fontsize=6.2,
        ha='right' if offset[0] < 0 else 'left',
        va='bottom' if offset[1] >= 0 else 'top',
    )


def save_figure(fig, name, copy_to_paper=False):
    PACKAGE.mkdir(parents=True, exist_ok=True);FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf=PACKAGE/f'{name}.pdf';png=PACKAGE/f'{name}.png';svg=PACKAGE/f'{name}.svg'
    fig.savefig(pdf);fig.savefig(png,dpi=600);fig.savefig(svg)
    if copy_to_paper: shutil.copy2(pdf, FIGURE_DIR/f'{name}.pdf')
    plt.close(fig)


def quality_rows():
    frozen = {
      ('Llama-3.2-3B','Basis Sharing','Pure'):([.3320,.3320,.3322],[0,0,0]),
      ('Llama-3.2-3B','Basis Sharing','CE'):([.7104,.4891,.3373],[.0012,.1384,.0048]),
      ('Llama-3.2-3B','Basis Sharing','CE+KD'):([.7128,.6815,.6506],[.0052,.0054,.0020]),
      ('Llama-3.2-3B','SVD-LLM','Pure'):([.3320,.3320,.3322],[0,0,0]),
      ('Llama-3.2-3B','SVD-LLM','CE'):([.5736,.3349,.3362],[.2032,.0022,.0037]),
      ('Llama-3.2-3B','SVD-LLM','CE+KD'):([.7078,.6732,.6358],[.0013,.0028,.0073]),
      ('Llama-3.1-8B','Basis Sharing','Pure'):([.3327,.3320,.3320],[0,0,0]),
      ('Llama-3.1-8B','Basis Sharing','CE'):([.7605,.7281,.3316],[.0039,.0075,.0004]),
      ('Llama-3.1-8B','Basis Sharing','CE+KD'):([.7643,.7389,.6996],[.0047,.0054,.0064]),
      ('Llama-3.1-8B','SVD-LLM','Pure'):([.3321,.3320,.3321],[0,0,0]),
      ('Llama-3.1-8B','SVD-LLM','CE'):([.7443,.5877,.3332],[.0019,.2174,.0026]),
      ('Llama-3.1-8B','SVD-LLM','CE+KD'):([.7531,.7301,.6964],[.0089,.0074,.0034]),
    }
    rows=[]
    for (backbone,method,regime),(means,sds) in frozen.items():
        for target,mean,sd in zip(TARGETS,means,sds):
            rows.append({'backbone':backbone,'method':method,'target':target,'x_compression':target,'regime':regime,'mean_macro':mean,'sd_macro':sd,'source':'frozen main.tex observed table'})
    ours=read_csv(ANALYSIS/'main_summary.csv')
    for row in ours:
        rows.append({'backbone':row['backbone'],'method':'Ours','target':int(row['nominal_target']),'x_compression':float(row['actual_compression']),'regime':row['regime'],'mean_macro':float(row['mean_macro']),'sd_macro':0.0 if row['sd_macro']=='NA' else float(row['sd_macro']),'source':'validated ours_main_summary.csv'})
    rows.sort(key=lambda r:(ORDER_BACKBONE.index(r['backbone']),ORDER_METHOD.index(r['method']),r['regime'],r['target']))
    return rows


def plot_quality(rows, regime, name, ylim):
    fig,ax=plt.subplots(figsize=(3.05,2.22))
    for method in ORDER_METHOD:
        for backbone in ORDER_BACKBONE:
            selected=sorted([r for r in rows if r['method']==method and r['backbone']==backbone and r['regime']==regime],key=lambda r:r['target'])
            x=np.asarray([r['x_compression'] for r in selected]);y=np.asarray([r['mean_macro'] for r in selected]);sd=np.asarray([r['sd_macro'] for r in selected])
            line,marker,_=BACKBONE_STYLE[backbone]
            ax.errorbar(x,y,yerr=sd if regime!='Pure' else None,color=COLORS[method],linestyle=line,marker=marker,capsize=2.0,capthick=.7,elinewidth=.75,markeredgecolor='white',markeredgewidth=.45,zorder=3)
    ax.set_title(regime.replace('CE+KD','CE + KD'),pad=4,fontweight='semibold')
    ax.set_xlabel('Whole-model reduction (%)');ax.set_ylabel('7-task macro accuracy')
    ax.set_xticks([15,20,25]);ax.set_xlim(12.7,27.0);ax.set_ylim(*ylim)
    method_handles=[Line2D([0],[0],color=COLORS[m],lw=2,label=m) for m in ORDER_METHOD]
    backbone_handles=[Line2D([0],[0],color='#555',ls=BACKBONE_STYLE[b][0],marker=BACKBONE_STYLE[b][1],lw=1.2,label=BACKBONE_STYLE[b][2]) for b in ORDER_BACKBONE]
    if regime=='Pure':
        leg1=ax.legend(handles=method_handles,loc='upper right',frameon=False,handlelength=1.55,borderpad=0.1,fontsize=5.7)
        ax.add_artist(leg1)
        ax.legend(handles=backbone_handles,loc='lower left',frameon=False,ncol=2,handlelength=1.45,borderpad=0.1,fontsize=5.7)
    fig.tight_layout(pad=.45)
    save_figure(fig,name,copy_to_paper=True)


def plot_recovery():
    rows=read_csv(ANALYSIS/'recovery_analysis.csv')
    fig,axes=plt.subplots(2,2,figsize=(7.0,4.15),sharex='col',gridspec_kw={'height_ratios':[2.1,1]})
    for col,backbone in enumerate(ORDER_BACKBONE):
        selected=sorted([r for r in rows if r['backbone']==backbone],key=lambda r:int(r['nominal_target']))
        x=np.asarray([float(r['actual_compression']) for r in selected])
        ce=np.asarray([float(r['CE_mean']) for r in selected]);cesd=np.asarray([float(r['CE_sd']) for r in selected])
        kd=np.asarray([float(r['CEKD_mean']) for r in selected]);kdsd=np.asarray([float(r['CEKD_sd']) for r in selected])
        pure=np.asarray([float(r['Pure_macro']) for r in selected]);teacher=float(selected[0]['dense_teacher_macro'])
        ax=axes[0,col]
        ax.errorbar(x,ce,yerr=cesd,color='#E69F00',marker='o',capsize=2.5,label='CE')
        ax.errorbar(x,kd,yerr=kdsd,color=COLORS['Ours'],marker='s',capsize=2.5,label='CE + KD')
        ax.plot(x,pure,color='#777',marker='^',ls=':',label='Pure')
        ax.axhline(teacher,color='#222',lw=1.0,ls='--',label='Dense teacher')
        ax.set_title(backbone,fontweight='semibold');ax.set_ylim(.28,.90)
        if col==0:ax.set_ylabel('Macro accuracy')
        ax.legend(frameon=False,ncol=2,loc='lower left')
        ax=axes[1,col]
        ax.plot(x,cesd,color='#E69F00',marker='o',label='CE SD')
        ax.plot(x,kdsd,color=COLORS['Ours'],marker='s',label='CE+KD SD')
        ax.set_yscale('log');ax.set_ylim(5e-4,3e-1);ax.set_xlabel('Actual reduction (%)')
        if col==0:ax.set_ylabel('Seed SD (log)')
        ax.legend(frameon=False,loc='upper left')
    fig.tight_layout(pad=.55,h_pad=.55,w_pad=.8)
    save_figure(fig,'recovery_stability')


def plot_asymmetry():
    rows=read_csv(ANALYSIS/'pair_analysis.csv')
    fig,ax=plt.subplots(figsize=(2.25,2.1))
    palette={'Llama-3.2-3B':'#0072B2','Llama-3.1-8B':'#CC79A7'}
    allv=[]
    for backbone in ORDER_BACKBONE:
        sel=[r for r in rows if r['backbone']==backbone]
        x=np.asarray([float(r['C_i_to_j']) for r in sel]);y=np.asarray([float(r['C_j_to_i']) for r in sel]);allv.extend(x);allv.extend(y)
        ax.scatter(x,y,s=7,alpha=.30,color=palette[backbone],edgecolors='none',rasterized=True,label=BACKBONE_STYLE[backbone][2])
    low=max(.05,min(allv)*.8);high=max(allv)*1.15
    ax.plot([low,high],[low,high],ls='--',lw=.9,color='#333',label='identity')
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(low,high);ax.set_ylim(low,high)
    ax.set_xlabel(r'$C_{i\rightarrow j}$');ax.set_ylabel(r'$C_{j\rightarrow i}$')
    ax.legend(frameon=False,loc='lower right',handletextpad=.35)
    fig.tight_layout(pad=.35)
    save_figure(fig,'diag_asymmetry',copy_to_paper=True)


def plot_envelope():
    rows=read_csv(ANALYSIS/'group_analysis.csv')
    fig,ax=plt.subplots(figsize=(2.25,2.1));palette={'Llama-3.2-3B':'#0072B2','Llama-3.1-8B':'#CC79A7'}
    for backbone in ORDER_BACKBONE:
        sel=[r for r in rows if r['backbone']==backbone]
        x=np.asarray([float(r['Delta']) for r in sel]);y=np.asarray([float(r['delta']) for r in sel]);sizes=np.asarray([float(r['group_size']) for r in sel])*8
        ax.scatter(x,y,s=sizes,color=palette[backbone],alpha=.78,edgecolor='white',linewidth=.5,label=BACKBONE_STYLE[backbone][2],zorder=3)
    lim=(0.18,1.04);ax.plot(lim,lim,ls='--',lw=.9,color='#333');ax.set_xlim(lim);ax.set_ylim(lim)
    ax.set_xlabel('Worst pairwise replacement cost\n' + r'$C_{\max}(\mathcal{G})$')
    ax.set_ylabel('Best representative cost\n' + r'$C_{\mathrm{rep}}(\mathcal{G})$')
    ax.legend(frameon=False,loc='upper left')
    fig.tight_layout(pad=.35)
    save_figure(fig,'diag_envelope',copy_to_paper=True)


def plot_joint():
    rows=read_csv(ANALYSIS/'joint_analysis.csv')
    fig,ax=plt.subplots(figsize=(2.65,2.05));palette={'Llama-3.2-3B':'#0072B2','Llama-3.1-8B':'#CC79A7'}
    for backbone in ORDER_BACKBONE:
        sel=sorted([r for r in rows if r['backbone']==backbone],key=lambda r:int(r['nominal_target']))
        x=np.asarray([float(r['Delta_max']) for r in sel]);y=np.asarray([c_star(r) for r in sel]);sizes=260*np.asarray([float(r['Pure_drop']) for r in sel])+18
        ax.plot(x,y,color=palette[backbone],lw=1.0,alpha=.65)
        ax.scatter(x,y,s=sizes,color=palette[backbone],edgecolor='white',linewidth=.6,label=BACKBONE_STYLE[backbone][2],zorder=3)
        for r,xx,yy in zip(sel,x,y):annotate_joint_point(ax,r,xx,yy)
    ax.plot([.44,1.0],[.44,1.0],ls='--',lw=.9,color='#333',zorder=1)
    ax.set_xlim(.44,1.0);ax.set_ylim(.50,1.40)
    ax.set_xlabel(r'$\max_k C_{\max}(\mathcal{G}_k)$');ax.set_ylabel(r'Per-layer joint cost $C^\star$')
    ax.legend(frameon=False,loc='upper left');fig.tight_layout(pad=.4)
    save_figure(fig,'diag_joint',copy_to_paper=True)


def plot_structural_validation_combined():
    """Render the three structural diagnostics with identical panel geometry."""
    palette={'Llama-3.2-3B':'#0066CC','Llama-3.1-8B':'#CC0000'}
    fig,axes=plt.subplots(
        1,3,
        figsize=(7.15,2.35),
        gridspec_kw={'width_ratios':[1,1,1]},
    )

    # (a) Directed donor--target replacement costs.
    rows=read_csv(ANALYSIS/'pair_analysis.csv');ax=axes[0];allv=[]
    for backbone in ORDER_BACKBONE:
        selected=[row for row in rows if row['backbone']==backbone]
        x=np.asarray([float(row['C_i_to_j']) for row in selected])
        y=np.asarray([float(row['C_j_to_i']) for row in selected])
        allv.extend(x);allv.extend(y)
        ax.scatter(x,y,s=7,alpha=.40,color=palette[backbone],edgecolors='none',rasterized=True,label=BACKBONE_STYLE[backbone][2])
    low=max(.05,min(allv)*.8);high=max(allv)*1.15
    ax.plot([low,high],[low,high],ls='--',lw=.9,color='#333',label='Identity')
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(low,high);ax.set_ylim(low,high)
    ax.set_xlabel(r'$C_{i\rightarrow j}$');ax.set_ylabel(r'$C_{j\rightarrow i}$')
    ax.set_title('(a)',fontweight='semibold',pad=6)
    ax.legend(frameon=False,loc='lower right',handletextpad=.35,fontsize=5.8)

    # (b) Worst pairwise replacement cost and best representative cost.
    rows=read_csv(ANALYSIS/'group_analysis.csv');ax=axes[1]
    for backbone in ORDER_BACKBONE:
        selected=[row for row in rows if row['backbone']==backbone]
        x=np.asarray([float(row['Delta']) for row in selected])
        y=np.asarray([float(row['delta']) for row in selected])
        sizes=np.asarray([float(row['group_size']) for row in selected])*8
        ax.scatter(x,y,s=sizes,color=palette[backbone],alpha=1.0,edgecolor='white',linewidth=.5,label=BACKBONE_STYLE[backbone][2],zorder=3)
    lim=(.18,1.04);ax.plot(lim,lim,ls='--',lw=.9,color='#333');ax.set_xlim(lim);ax.set_ylim(lim)
    ax.set_xlabel('Worst pairwise replacement cost\n' + r'$C_{\max}(\mathcal{G})$')
    ax.set_ylabel('Best representative cost\n' + r'$C_{\mathrm{rep}}(\mathcal{G})$')
    ax.set_title('(b)',fontweight='semibold',pad=6)
    ax.legend(frameon=False,loc='upper left',fontsize=5.8)

    # (c) Maximum worst pairwise replacement cost versus per-layer joint cost.
    rows=read_csv(ANALYSIS/'joint_analysis.csv');ax=axes[2]
    for backbone in ORDER_BACKBONE:
        selected=sorted([row for row in rows if row['backbone']==backbone],key=lambda row:int(row['nominal_target']))
        x=np.asarray([float(row['Delta_max']) for row in selected])
        y=np.asarray([c_star(row) for row in selected])
        sizes=260*np.asarray([float(row['Pure_drop']) for row in selected])+18
        ax.plot(x,y,color=palette[backbone],lw=1.0,alpha=1.0)
        ax.scatter(x,y,s=sizes,color=palette[backbone],alpha=1.0,edgecolor='white',linewidth=.6,label=BACKBONE_STYLE[backbone][2],zorder=3)
        for row,xx,yy in zip(selected,x,y):annotate_joint_point(ax,row,xx,yy)
    ax.plot([.44,1.0],[.44,1.0],ls='--',lw=.9,color='#333',zorder=1)
    ax.set_xlim(.44,1.0);ax.set_ylim(.50,1.40)
    ax.set_xlabel(r'$\max_k C_{\max}(\mathcal{G}_k)$');ax.set_ylabel(r'Per-layer joint cost $C^\star$')
    ax.set_title('(c)',fontweight='semibold',pad=6)
    ax.legend(frameon=False,loc='upper left',fontsize=5.8)

    # Fixed margins preserve identical axes widths/heights across all panels.
    fig.subplots_adjust(left=.070,right=.992,bottom=.225,top=.825,wspace=.49)
    save_figure(fig,'diag_structural_validation',copy_to_paper=True)


def plot_ablation():
    rows=read_csv(ANALYSIS/'structural_ablation.csv')
    order=['full_method','no_directional_cost','no_envelope_mean','single_link','symmetric_representative']
    labels=['Full','No directional','Two-way mean*','Single-link*','Symmetric donor']
    lookup={r['variant']:r for r in rows};x=np.arange(len(order));width=.34
    fig,axes=plt.subplots(1,2,figsize=(7.0,2.55),gridspec_kw={'width_ratios':[1.45,1]})
    pure=[float(lookup[v]['Pure_macro']) for v in order];kd=[float(lookup[v]['CEKD_macro']) for v in order]
    axes[0].bar(x-width/2,pure,width,color='#999999',label='Pure')
    axes[0].bar(x+width/2,kd,width,color=COLORS['Ours'],label='CE + KD')
    axes[0].set_ylim(.30,.88);axes[0].set_ylabel('Macro accuracy');axes[0].set_xticks(x,labels,rotation=18,ha='right');axes[0].legend(frameon=False,ncol=2)
    cstar=[c_star(lookup[v]) for v in order]
    bars=axes[1].bar(x,cstar,color=['#0072B2','#D55E00','#B3B3B3','#B3B3B3','#CC79A7'])
    for idx in (2,3):bars[idx].set_hatch('///')
    axes[1].set_ylabel(r'Per-layer joint cost $C^\star$');axes[1].set_xticks(x,labels,rotation=18,ha='right')
    axes[1].text(.98,.96,'* structurally identical to Full',transform=axes[1].transAxes,ha='right',va='top',fontsize=6.5,color='#555')
    fig.tight_layout(pad=.55,w_pad=1.0)
    save_figure(fig,'structural_ablation')


def plot_bootstrap():
    rows=[r for r in read_csv(EXTERNAL/'paired_bootstrap_results.csv') if r['task']=='macro']
    opponents=['Ours-Basis Sharing','Ours-SVD-LLM'];ops=['3b_15','3b_20','3b_25','8b_15','8b_20','8b_25'];labels=['3B–15%','3B–20%','3B–25%','8B–15%','8B–20%','8B–25%']
    seed_colors={42:'#56B4E9',43:'#0072B2',44:'#003B5C'};offset={42:-.18,43:0,44:.18}
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.45),sharex=True,sharey=True)
    for ax,comparison in zip(axes,opponents):
        lookup={(r['model_id'],int(r['seed'])):r for r in rows if r['comparison']==comparison}
        for yi,op in enumerate(ops):
            for seed in (42,43,44):
                r=lookup[(op,seed)];mean=100*float(r['delta_accuracy']);low=100*float(r['ci95_low']);high=100*float(r['ci95_high'])
                ax.errorbar(mean,yi+offset[seed],xerr=[[mean-low],[high-mean]],fmt='o',ms=4,color=seed_colors[seed],ecolor=seed_colors[seed],elinewidth=1.05,capsize=2.0,markeredgecolor='white',markeredgewidth=.35)
        ax.axvline(0,color='#333',ls='--',lw=.9);ax.set_yticks(np.arange(len(ops)),labels);ax.invert_yaxis();ax.set_xlim(0,22)
        ax.set_xlabel('Ours advantage (percentage points)');ax.set_title(comparison.replace('Ours-','vs. '),fontweight='semibold')
    handles=[Line2D([0],[0],marker='o',color=seed_colors[s],lw=0,label=f'seed {s}') for s in (42,43,44)]
    axes[0].legend(handles=handles,frameon=False,loc='lower right')
    fig.tight_layout(pad=.6,w_pad=.8)
    save_figure(fig,'paired_bootstrap_macro')
    plot_rows=[]
    for r in rows:plot_rows.append({**r,'delta_pp':100*float(r['delta_accuracy']),'ci95_low_pp':100*float(r['ci95_low']),'ci95_high_pp':100*float(r['ci95_high'])})
    write_csv(DATA_DIR/'paired_bootstrap_macro.csv',plot_rows)


def plot_bytes():
    rows=[]
    for r in read_csv(ANALYSIS/'serialized_bytes.csv'):
        rows.append({'method':'Ours','model_id':r['model_id'],'backbone':r['backbone'],'target':int(r['nominal_target']),'reduction':float(r['serialized_byte_reduction_percent']),'dense_bytes':int(r['dense_serialized_weight_bytes']),'compressed_bytes':int(r['compressed_standalone_weight_bytes'])})
    for r in read_csv(EXTERNAL/'serialized_byte_reduction_combined.csv'):
        if r['method'] == 'Ours':
            continue
        rows.append({'method':r['method'],'model_id':r['model_id'],'backbone':r['backbone'],'target':int(r['nominal_target']),'reduction':float(r['serialized_byte_reduction_percent']),'dense_bytes':int(r['dense_serialized_bytes']),'compressed_bytes':int(r['compressed_serialized_bytes'])})
    fig,axes=plt.subplots(1,2,figsize=(7.0,2.65),sharex=True,sharey=True)
    for ax,backbone in zip(axes,ORDER_BACKBONE):
        ax.plot([14,26],[14,26],color='#777',ls=':',lw=1,label='ideal target')
        for method in ORDER_METHOD:
            sel=sorted([r for r in rows if r['backbone']==backbone and r['method']==method],key=lambda r:r['target'])
            ax.plot([r['target'] for r in sel],[r['reduction'] for r in sel],color=COLORS[method],marker={'Ours':'o','Basis Sharing':'s','SVD-LLM':'^'}[method],label=method)
        ax.set_title(backbone,fontweight='semibold');ax.set_xlabel('Nominal target (%)');ax.set_xticks(TARGETS);ax.set_xlim(13.5,26.5);ax.set_ylim(13,27)
    axes[0].set_ylabel('Serialized weight reduction (%)');axes[0].legend(frameon=False,loc='upper left')
    fig.tight_layout(pad=.55,w_pad=.8);save_figure(fig,'serialized_byte_reduction');write_csv(DATA_DIR/'serialized_byte_reduction_combined.csv',rows)


def main():
    PACKAGE.mkdir(parents=True,exist_ok=True);DATA_DIR.mkdir(parents=True,exist_ok=True)
    qrows=quality_rows();write_csv(DATA_DIR/'quality_frontiers.csv',qrows)
    plot_quality(qrows,'Pure','quality_pure',(.315,.385));plot_quality(qrows,'CE','quality_ce',(.28,.90));plot_quality(qrows,'CE+KD','quality_kd',(.60,.89))
    plot_recovery();plot_asymmetry();plot_envelope();plot_joint();plot_structural_validation_combined();plot_ablation();plot_bootstrap();plot_bytes()
    captions=r'''# Suggested captions

## Quality frontiers
Seven-task macro accuracy versus matched whole-model reduction. Points show means over seeds 42/43/44 for CE and CE+KD; error bars are sample SD. Pure is evaluated without recovery. Solid/circle and dashed/square curves denote 3B and 8B, respectively.

## Recovery stability
Recovery behavior of Ours. CE+KD stabilizes the high-variance 8B operating points while retaining 98.9--99.2% of the dense teacher; the lower panels report seed SD on a logarithmic scale.

## Directional asymmetry
Held-out directed replacement costs. Departures from the identity line demonstrate that donor--target replacement is asymmetric; axes use logarithmic scales and diagonal self-replacements are excluded.

## Envelope tightness
Groupwise pairwise envelope versus the best outward donor. Marker area is proportional to group size; every observed group satisfies delta(G) <= Delta(G).

## Pairwise-to-joint cost
Maximum pairwise replacement cost versus the simultaneous full-partition cost per logical layer, $C^\star$. The dashed identity line gives the equal-cost reference; labels give nominal reduction and marker area encodes Pure macro degradation.

## Structural ablation
Structural ablation at the 3B 20% operating point. Hatched variants collapse to the Full structure under the frozen K/pin/regime constraints and therefore reuse its observed result; the remaining variants are independently evaluated.

## Paired bootstrap
Task-stratified paired bootstrap differences under the synchronized per-example evaluator. Error bars are 95% intervals from 10,000 resamples; every operating-point/seed macro interval is strictly above zero.

## Serialized reduction
Weight-only serialized reduction under a common dense-versus-standalone definition. All methods closely realize their nominal budgets.
'''
    (PACKAGE/'CAPTIONS.md').write_text(captions,encoding='utf-8')
    (PACKAGE/'README.md').write_text('''# ICASSP paper figures

All figures are generated from validated observed CSV data. PDF files are vector outputs; PNG files are 600 dpi; SVG files retain editable text. The six manuscript-facing PDFs are copied into `icassp2027/Figure/`.

Important provenance: quality frontiers use the frozen H200 main-table aggregates. Paired-bootstrap figures use synchronized current per-example evaluation; maximum Ours rerun drift from the frozen aggregate is 0.341 percentage point.
''',encoding='utf-8')
    expected=['quality_pure','quality_ce','quality_kd','recovery_stability','diag_asymmetry','diag_envelope','diag_joint','diag_structural_validation','structural_ablation','paired_bootstrap_macro','serialized_byte_reduction']
    for name in expected:
        for suffix in ('pdf','png','svg'):
            path=PACKAGE/f'{name}.{suffix}'
            if not path.is_file() or path.stat().st_size<=0:raise RuntimeError(f'missing figure {path}')
    print({'status':'PASS','figure_count':len(expected),'package':str(PACKAGE)})


if __name__=='__main__':main()
