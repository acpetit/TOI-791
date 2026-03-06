import warnings
from jnkepler.jaxttv import utils
from jnkepler.jaxttv import JaxTTV

import matplotlib.pyplot as plt
import numpy as np


col_b='#EE4266'
col_c='#0EAD69'

def plot_modelcustom(self, tcmodellist, tcmodelsamples=None,oldmodel=None,extrapoints=None, tcobslist=None, errorobslist=None, t0_lin=None, p_lin=None,
                tcmodelunclist=None, tmargin=None, save=None, marker=None, ylims=None, ylims_residual=None,
                unit=1440., ylabel='TTV (min)', xlabel='Transit time (BJD - 2458000)',bjdoffset=2458000,planetcolors=[col_b,col_c],labels=["TOI-791 b","TOI-791 c"]):
    """plot transit time model

        Args:
            tcmodellist: list of the arrays of model transit times for each planet
            tcobslist: list of the arrays of observed transit times for each planet
            errorobslist: list of the arrays of observed transit time errors for each planet
            t0_lin, p_lin: linear ephemeris used to show TTVs (n_planet,)
            tcmodelunclist: model uncertainty (same format as tcmodellist)
            tmargin: margin in x axis (float or tuple)
            save: if not None, plot is saved as "save_planet#.png"
            marker: marker for model
            unit: TTV unit (defaults to minutes)
            ylabel, xlabel: axis labels in the plots
            ylims, ylims_residual: y ranges in the plots

    """
    if tcobslist is None:
        tcobslist = self.tcobs
        xmin,xmax = np.min(np.concat(tcobslist)),np.max(np.concat(tcobslist))

    if errorobslist is None:
        if self.errorobs is not None:
            errorobslist = self.errorobs
        else:
            errorobslist = [np.zeros_like(_t) for _t in tcobslist]

    if (t0_lin is None) or (p_lin is None):
        t0_lin, p_lin = self.linear_ephemeris()
        warnings.warn(
            "using t0 and P from a linear fit to the observed transit times.")

    figs = []


    for j, (tcmodel, tcobs, errorobs, t0, p,pcol,plabel) in enumerate(zip(tcmodellist, tcobslist, errorobslist, t0_lin, p_lin,planetcolors,labels)):
        tcmodel, tcobs, errorobs = np.array(
            tcmodel), np.array(tcobs), np.array(errorobs)

        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6, 6),
                                        sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        if tmargin is not None:
            try:
                tmlow,tmmax = tmargin
            except TypeError:
                tmlow = tmargin
                tmmax = tmargin
            plt.xlim(xmin-tmlow-bjdoffset, xmax+tmmax-bjdoffset)
        ax.set_ylabel(ylabel)
        ax2.set_xlabel(xlabel)

        ax.axhline(0,lw=1,c="lightgray",zorder=0)

        tnumobs = np.round((tcobs - t0)/p).astype(int)
        tnummodel = np.round((tcmodel - t0)/p).astype(int)
        ax.errorbar(tcobs-bjdoffset, (tcobs-t0-tnumobs*p)*unit, yerr=errorobs*unit, zorder=100,
                    fmt='o', mfc='white', color='dimgray', label='Observations', lw=1, markersize=7)
        idxm = tcmodel > 0
        tlin = t0 + tnummodel * p
        ax.plot(tcmodel[idxm]-bjdoffset, (tcmodel-tlin)[idxm]*unit, '-', marker=marker, lw=2, mfc='white', color=pcol,
                zorder=1, label='Model', alpha=0.9)
        if tcmodelunclist is not None:
            munc = tcmodelunclist[j]
            ax.fill_between(tcmodel[idxm]-bjdoffset, (tcmodel-munc-tlin)[idxm]*unit,
                            (tcmodel+munc-tlin)[idxm]*unit,
                            lw=1, color=pcol, zorder=99, alpha=0.2)
            
        if tcmodelsamples is not None:
            tnummodel_2 = np.round((tcmodelsamples[0][j] - t0)/p).astype(int)
            tlin_2 = t0 + tnummodel_2 * p
            for tmod in tcmodelsamples:
                ax.plot(tmod[j]-bjdoffset, (tmod[j]-tlin_2)*unit, '-',  lw=1, color=pcol,
                    zorder=1, alpha=0.2)
                ax2.plot(tcmodel-bjdoffset, (tmod[j]-tcmodel)*unit,lw=1, color=pcol,
                    zorder=1, alpha=0.2)
                
        if oldmodel is not None:
            ax.plot(oldmodel[j][idxm]-bjdoffset, (oldmodel[j]-tlin)[idxm]*unit, '-', marker=marker, lw=1.8, mfc='white', color="darkgrey",
                zorder=1, label='Old Model', alpha=0.9)
            ax2.plot(tcmodel-bjdoffset, (oldmodel[j]-tcmodel)*unit,lw=1.8, color="darkgrey",
                    zorder=1, alpha=0.9)
            
        if extrapoints is not None:
            if len(extrapoints[j]):
                ax.errorbar(extrapoints[j][0]-bjdoffset, (extrapoints[j][0]-t0-extrapoints[j][1]*p)*unit, yerr=extrapoints[j][2]*unit, zorder=1000,
                    fmt='o', mfc='black', color='dimgray', label='Observations', lw=1, markersize=7)
                ax2.errorbar(extrapoints[j][0]-bjdoffset, (extrapoints[j][0]-tcmodel[extrapoints[j][1]])*unit, yerr=extrapoints[j][2]*unit, zorder=1000,
                    fmt='o', mfc='black', color='dimgray', label='Observations', lw=1, markersize=7)

        
        # ax.set_title("planet %d" % (j+1))
        if ylims is not None and len(ylims) == len(t0_lin):
            ax.set_ylim(ylims[j])

        idxm = utils.findidx_map(tcmodel, tcobs)
        ax2.errorbar(tcobs-bjdoffset, (tcobs-tcmodel[idxm])*unit, yerr=errorobs*unit, zorder=1000,
                        fmt='o', mfc='white', color='dimgray', label='data', lw=1, markersize=7)
        ax2.axhline(y=0, color=pcol, alpha=1,lw=2)
        ax2.set_ylabel("Residual (min)")
        if ylims_residual is not None and len(ylims_residual) == len(t0_lin):
            ax2.set_ylim(ylims_residual[j])

        #Adding planet label
        ax.text(.05,.9,plabel,
        horizontalalignment='left',color=pcol,
        transform=ax.transAxes)

        # change legend order
        handles, labels = ax.get_legend_handles_labels()
        order = [1, 0]
        if oldmodel: order = [2,0,1]
        ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
                    loc='best',)

        fig.tight_layout(pad=0.05)

        if save is not None:
            plt.savefig(save+"_planet%d.pdf" %
                        (j+1), dpi=200, bbox_inches="tight")
        figs.append(fig)
    return figs
