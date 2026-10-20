import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

U="/Users/ian/Desktop/VSCode/QuasicrystalCoding/output/writing_selection/20261010_104702_904056/03_numbers/"
B="/Users/ian/Desktop/Peak_Identification/"
OUT="/Users/ian/Desktop/Parity_paper_figures_and_sources/figures/"
RESP=U+"response_summaries_all_grids.csv"; RQ2=U+"rq2_main_table.csv"
PAR=U+"parity_finest_with_refinement.csv"; PALL=U+"parity_all_saved_grids.csv"
MODES=["potential_only", "substitution", "width_only","phason"]
LAB={"potential_only":"Potential-only", "substitution":"Substitution", "width_only":"Width-only", "phason":"Phason"}
COL={"potential_only":"#bb00ff", "substitution":"#0000ed", "width_only":"#00ed00", "phason":"#ed0000"}
MK={"potential_only":"o", "substitution":"s", "width_only":"^", "phason":"D"}
INK="#0b0b0b"; INK2="#383838"; GRID="#cfceca"
plt.rcParams.update({
                    "font.family":"DejaVu Sans",
                    "font.size":10,
                    "axes.edgecolor":INK2,
                    "axes.labelcolor":INK,
                    "xtick.color":INK2,
                    "ytick.color":INK2,
                    "axes.spines.top":False,
                    "axes.spines.right":False,
                    "axes.grid":True,
                    "grid.color":GRID,
                    "grid.linewidth":0.6,
                    "legend.frameon":False,
                    "lines.linewidth":2,
                    "figure.dpi":200,
                    })
GEN=list(range(10,21)); NL=[55,89,144,233,377,610,987,1597,2584,4181,6765]

def curve(n,mode):
    tag="bond_4" if mode=="phason" else "site_5"
    f=f"n{n}_e31985_{mode}/n{n}_{mode}_{tag}_parity_curves.csv"
    return pd.read_csv(B+f), f
def genaxis(ax):
    ax.set_xticks(GEN); ax.set_xlabel("Fibonacci generation n")
def save(fig,name): fig.savefig(OUT+name,bbox_inches="tight",facecolor="white"); plt.close(fig)

# For RQ1
r=pd.read_csv(RESP)
pv=r[r.window=="full"].pivot_table(index=["mode","generation"],columns="n_energy",values="mean_abs")
dref=np.maximum((pv[15993]-pv[7997]).abs(),(pv[31985]-pv[15993]).abs())
fig,ax=plt.subplots(figsize=(6.4,4.0))
for m in MODES:
    y=pv.loc[m][31985]; e=dref.loc[m]
    ax.errorbar(GEN,y,yerr=e,color=COL[m],marker=MK[m],ms=5,capsize=2,label=LAB[m])
    ax.text(20.25,y.iloc[-1],LAB[m],va="center",fontsize=9,color=INK)
genaxis(ax); ax.set_xlim(9.6,22.3); ax.set_ylim(0,0.07)
ax.set_ylabel("Mean |ΔT| over all (p, E) cells, 0.1-10 eV")
ax.legend(loc="lower left",ncol=2,fontsize=8.5)
save(fig,"rq1_mean_abs_response.png")

# For RQ2
q=pd.read_csv(RQ2); q=q[q.window=="full"].sort_values("generation")
fig,ax=plt.subplots(figsize=(6.4,3.8))
ax.plot(q.generation,q.mean_abs_I_over_sub_response,color="#0000ed",marker="o",ms=5,label="mean|I| / mean|ΔT_sub|")
ax.plot(q.generation,q.mean_abs_I_over_sum_component_responses,color="#ed0000",marker="s",ms=5,label="mean|I| / (mean|ΔT_pot| + mean|ΔT_width|)")
ax.axhline(1,color=INK2,lw=1,ls="--"); ax.text(10,1.03,"remainder as large as the substitution response",fontsize=8,color=INK2)
genaxis(ax); ax.set_ylim(0,1.5); ax.set_ylabel("Ratio (dimensionless)")
ax.legend(loc="lower left",fontsize=8.5)
save(fig,"rq2_remainder_ratio.png")

# For parity differences (main finding)
fig,axs=plt.subplots(2,1,figsize=(7.2,4.8),sharex=True,sharey=True)
for ax,n in zip(axs,[10,20]):
    c,f=curve(n,"phason"); s=c[c.energy_ev>=5]
    ax.plot(s.energy_ev,s.D_residual*100,color="#0000ed",lw=0.7)
    i=np.abs(c.D_residual.values).argmax()
    ax.plot(c.energy_ev[i],c.D_residual[i]*100,"o",ms=6,mfc="none",mec="#ed0000",mew=1.5)
    ax.annotate(f"S_max = {abs(c.D_residual[i])*100:.1f} pp at {c.energy_ev[i]:.3f} eV",(c.energy_ev[i],c.D_residual[i]*100),
        xytext=(14, 4 if c.D_residual[i]<0 else -4),textcoords="offset points",fontsize=8.5,color=INK,va="center")
    Sint=np.trapezoid(np.abs(c.D_residual),c.energy_ev)
    ax.text(0.99,0.95,f"n = {n} (N = {NL[n-10]})   S_int = {Sint:.4f} eV",transform=ax.transAxes,ha="right",va="top",fontsize=9,color=INK)
    ax.set_ylabel("R_G(E) (pp)"); ax.axhline(0,color=INK2,lw=0.6)
axs[1].set_xlabel("Energy E (eV)"); axs[0].set_xlim(5,10); axs[0].set_ylim(-32,32)
save(fig,"phason_residual_curves.png")

# The one after that
p=pd.read_csv(PAR); pf=p[(p.window=="full")&(p["mode"]=="phason")].set_index("generation").sort_index()
a=pd.read_csv(PALL)
smax={}; smax_g={}
for n in GEN:
    for g in [7997,15993,31985]:
        tag="bond_4"
        c=pd.read_csv(B+f"n{n}_e{g}_phason/n{n}_phason_{tag}_parity_curves.csv")
        smax_g[(n,g)]=np.abs(c.D_residual).max()
sm=pd.Series({n:smax_g[(n,31985)] for n in GEN})
sm_ref=pd.Series({n:max(abs(smax_g[(n,15993)]-smax_g[(n,7997)]),abs(smax_g[(n,31985)]-smax_g[(n,15993)])) for n in GEN})
si=pf.residual_absolute; si_ref=pd.Series({n:max(abs(v1-v0),abs(v2-v1)) for n,(v0,v1,v2) in
     a[(a.window=="full")&(a["mode"]=="phason")&a.n_energy.isin([7997,15993,31985])].pivot_table(index="generation",columns="n_energy",values="residual_absolute").iterrows()})
rm=pf.rms; rm_ref=pf.rms_max_observed_change
fig,ax=plt.subplots(figsize=(6.6,4.3))
series=[("S_max = max |R_G(E)| (largest sampled residual)",sm,sm_ref,"#ed0000","D"),
        ("D_RMS (root-mean-square of observed contrast)",rm,rm_ref,"#bb00ff","s"),
        ("S_int = ∫|R_G(E)| dE (integrated residual)",si,si_ref,"#0000ed","o")]
for lab,y,e,cc,mk in series:
    base=y.loc[10]; ax.errorbar(GEN,100*y/base,yerr=100*e/base,color=cc,marker=mk,ms=5,capsize=2,label=lab)
ax.set_yscale("log"); ax.set_ylim(1,200); genaxis(ax)
ax.set_yticks([1,2,5,10,20,50,100]); ax.set_yticklabels(["1","2","5","10","20","50","100"])
ax.set_ylabel("Index (generation 10 = 100), log scale")
ax.text(20.2,100*sm[20]/sm[10],f"{100*sm[20]/sm[10]:.0f}",va="center",fontsize=9,color=INK)
ax.text(20.2,100*rm[20]/rm[10],f"{100*rm[20]/rm[10]:.0f}",va="center",fontsize=9,color=INK)
ax.text(20.2,100*si[20]/si[10],f"{100*si[20]/si[10]:.1f}",va="center",fontsize=9,color=INK)
ax.set_xlim(9.6,21); ax.legend(loc="lower left",fontsize=8.3)
save(fig,"phason_indexed_summaries.png")

# Same section
d=pd.read_csv("/Users/ian/Desktop/VSCode/QuasicrystalCoding/output/writing_selection/20261010_104702_904056/03_numbers/rq3_derived_metrics_e31985.csv")
fig,ax=plt.subplots(figsize=(6.4,3.2))
ys=np.arange(len(MODES))[::-1]
for yy,m in zip(ys,MODES):
    s=d[d["mode"]==m].set_index("generation")
    a1=100*(1-s.S_int[20]/s.S_int[10]); a2=100*(1-s.S_max[20]/s.S_max[10])
    ax.plot([a2,a1],[yy,yy],color=GRID,lw=3,zorder=1)
    ax.plot(a1,yy,"o",color="#0000ed",ms=8,zorder=2,label="S_int (integrated)" if m==MODES[0] else None)
    ax.plot(a2,yy,"D",color="#ed0000",ms=7,zorder=2,label="S_max (largest sampled)" if m==MODES[0] else None)
    ax.text(a1+1.5,yy,f"{a1:.0f}%",va="center",fontsize=8.5,color=INK); ax.text(a2-1.5,yy,f"{a2:.0f}%",va="center",ha="right",fontsize=8.5,color=INK)
ax.set_yticks(ys); ax.set_yticklabels([LAB[m] for m in MODES]); ax.set_xlim(0,105); ax.grid(axis="y",visible=False)
ax.set_xlabel("Decline from generation 10 to generation 20 (%)"); ax.legend(loc="lower center",bbox_to_anchor=(0.5,1.0),ncol=2,fontsize=8.5)
save(fig,"decline_by_mode.png")

# Again, same section
fig,ax=plt.subplots(figsize=(6.4,3.9))
for m in MODES:
    s=d[d["mode"]==m].set_index("generation")
    ax.plot(GEN,100*s.frac_E_absR_gt_0p05,color=COL[m],marker=MK[m],ms=5,label=LAB[m])
ax.set_yscale("log"); genaxis(ax); ax.set_ylabel("Energy samples with |R_G(E)| > 5 pp (%)\n(of 31,985. log scale)")
ax.axhline(100/31985,color=INK2,lw=1,ls="--"); ax.text(10,100/31985*1.25,"one sample",fontsize=8,color=INK2)
ax.legend(loc="upper right",fontsize=8.5)
save(fig,"energy_concentration.png")

# Where largest residuals occur
E0=5.773061530765383
vals=[]
for n in GEN:
    c,_=curve(n,"phason"); i=np.argmin(np.abs(c.energy_ev.values-E0)); vals.append((c.energy_ev[i],c.D_residual[i]))
fig,ax=plt.subplots(figsize=(6.4,3.6))
for n,(e,v),N in zip(GEN,vals,NL):
    ev=(N%2==0)
    ax.bar(n,abs(v)*100,width=0.6,color="#d3d3d3" if ev else "#ffae00",edgecolor="white")
    ax.text(n,abs(v)*100+0.6,f"{abs(v)*100:.1f}",ha="center",fontsize=7.5,color=INK)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#ffae00",label="odd chain length N"),Patch(color="#d3d3d3",label="even chain length N (n = 12, 15, 18)")],loc="upper left",fontsize=8.5,ncol=2)
genaxis(ax); ax.set_ylabel("|R_G| at E = 5.7731 eV (pp)"); ax.set_ylim(0,30); ax.grid(axis="x",visible=False)
save(fig,"fig8_phason_5p773_by_generation.png")
pd.DataFrame({"generation":GEN,"N":NL,"energy_ev":[e for e,_ in vals],"D_residual":[v for _,v in vals]}).to_csv(OUT+"fig8_data.csv",index=False)
print("done", sm.round(4).to_dict())