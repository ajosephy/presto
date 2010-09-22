#!/usr/bin/env python

"""A python version of PRESTO's show_pfd.
    Patrick Lazarus, Sept. 1st, 2010
"""
import sys
import copy

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.cm
from mpl_toolkits import axes_grid1
import psr_utils
import prepfold
import presto

# Define "anti-rainbow" colormap.
# (See line 951 in $PRESTO/src/prepfold_plot.c)
cdict = {'red': [(0.0, 1.0, 1.0),
                 (0.035, 1.0, 1.0),
                 (0.045, 0.947, 0.947),
                 (0.225, 0.0, 0.0),
                 (0.4, 0.0, 0.0),
                 (0.41, 0.0, 0.0),
                 (0.6, 0.0, 0.0),
                 (0.775, 1.0, 1.0),
                 (0.985, 1.0, 1.0),
                 (1.0, 1.0, 1.0)],
         'green': [(0.0, 1.0, 1.0),
                 (0.035, 0.844, 0.844),
                 (0.045, 0.8, 0.8),
                 (0.225, 0.0, 0.0),
                 (0.4, 0.946, 0.946),
                 (0.41, 1.0, 1.0),
                 (0.6, 1.0, 1.0),
                 (0.775, 1.0, 1.0),
                 (0.985, 0.0, 0.0),
                 (1.0, 0.0, 0.0)], 
         'blue': [(0.0, 1.0, 1.0),
                 (0.035, 1.0, 1.0),
                 (0.045, 1.0, 1.0),
                 (0.225, 1.0, 1.0),
                 (0.4, 1.0, 1.0),
                 (0.41, 0.95, 0.95),
                 (0.6, 0.0, 0.0),
                 (0.775, 0.0, 0.0),
                 (0.985, 0.0, 0.0),
                 (1.0, 0.0, 0.0)]}
matplotlib.cm.anti_rainbow = matplotlib.colors.LinearSegmentedColormap('anti_rainbow', cdict, plt.rcParams['image.lut'])


def set_defaults():
    # Set defaults for plot
    plt.rc(('xtick.major', 'ytick.major'), size=6)
    plt.rc(('xtick.minor', 'ytick.minor'), size=3)
    plt.rc('axes', labelsize='small')
    plt.rc(('xtick', 'ytick'), labelsize='x-small')


class PrepfoldPlot:
    """Prepfold plot object.

        Recreates the standard prepfold plot using
        matplotlib and adds interactive functionality.
    """
    def __init__(self, pfdfn):
        """PrefoldPlot constructor.

            Input:
                pfdfn: Name of .pfd file.
        """
        # Open .pfd file
        self.pfdfn = pfdfn
        self.origpfd = prepfold.pfd(pfdfn)
        self.pfd = copy.deepcopy(self.origpfd)

        # Prep pfd
        self.dm = self.pfd.bestdm
        self.pfd.dedisperse(doppler=True)
        self.pfd.adjust_period()
        self.subints_to_kill = []
        self.subbands_to_kill = []
        
        # Create matplotlib figure
        self.fig = plt.figure(figsize=(11,8.5), facecolor='w')
        # Create axes for plotting
        self.sumprof_ax = plt.axes((0.06, 0.68, 0.21, 0.26))
        self.timephs_ax = plt.axes((0.06, 0.09, 0.21, 0.59), \
                                                sharex=self.sumprof_ax)
        self.freqphs_ax = axes_grid1.host_axes((0.44, 0.3, 0.22, 0.38), \
                                                sharex=self.sumprof_ax)
        self.dmchi2_ax = plt.axes((0.44, 0.09, 0.22, 0.13))
        self.timechi2_ax = plt.axes((0.27, 0.09, 0.11, 0.59))
        self.ppdot_ax = plt.axes((0.74, 0.09, 0.2, 0.2))
        self.pchi2_ax = plt.axes((0.74, 0.41, 0.2, 0.1))
        self.pdchi2_ax = plt.axes((0.74, 0.58, 0.2, 0.1))

        # Plot
        self.plot()
        # Add text
        self.addtext()

        # Register callbacks
        self.fig.canvas.mpl_connect('key_press_event', self.keypress)
        self.fig.canvas.mpl_connect('button_press_event', self.mousepress)

    def plot(self):
        """Plot all elements.
        """
        self.plot_prof(self.sumprof_ax)
        self.plot_timephs(self.timephs_ax)
        self.plot_freqphs(self.freqphs_ax)
        self.plot_dmchi2(self.dmchi2_ax)
        self.plot_timechi2(self.timechi2_ax)
        self.plot_ppdot(self.ppdot_ax)
        self.plot_pchi2(self.pchi2_ax)
        self.plot_pdchi2(self.pdchi2_ax)

    def addtext(self):
        """Add texture to the figure
            (i.e. the Data Info area).
        """
        DEFSIZE = 'small'
        plt.figtext(0.06, 0.01, self.pfd.filenm, size=DEFSIZE)
        plt.figtext(0.29, 0.94, "Candidate: %s" % self.pfd.candnm, size=DEFSIZE)
        plt.figtext(0.29, 0.92, "Telescope: %s" % self.pfd.telescope, size=DEFSIZE)
        if self.pfd.tepoch != 0.0:
            tepoch_str = r"Epoch$_{\mathsf{topo}}$ = %-.11f" % self.pfd.tepoch
        else:
            tepoch_str = r"Epoch$_{\mathsf{topo}}$ = N/A"
        plt.figtext(0.29, 0.90, tepoch_str, size=DEFSIZE)
        if self.pfd.bepoch != 0.0:
            bepoch_str = r"Epoch$_{\mathsf{bary}}$ = %-.11f" % self.pfd.bepoch
        else:
            tepoch_str = r"Epoch$_{\mathsf{bary}}$ = N/A"
        plt.figtext(0.29, 0.88, bepoch_str, size=DEFSIZE)
        plt.figtext(0.29, 0.86, r"T$_{\mathsf{sample}}$", size=DEFSIZE)
        plt.figtext(0.39, 0.86, "= %.5g" % self.pfd.dt, size=DEFSIZE)
        plt.figtext(0.29, 0.84, "Data Folded", size=DEFSIZE)
        plt.figtext(0.39, 0.84, "= %d" % self.pfd.Nfolded, size=DEFSIZE)
        # sum all subbands and average over all subints.
        data_avg, data_var = self.pfd.stats.sum(axis=1).mean(axis=0)[1:3]
        plt.figtext(0.29, 0.82, "Data Avg", size=DEFSIZE)
        plt.figtext(0.39, 0.82, "= %.4g" % data_avg, size=DEFSIZE)
        plt.figtext(0.29, 0.80, r"Data StdDev", size=DEFSIZE)
        plt.figtext(0.39, 0.80, "= %.4g" % np.sqrt(data_var), size=DEFSIZE)
        plt.figtext(0.29, 0.78, "Profile Bins", size=DEFSIZE)
        plt.figtext(0.39, 0.78, "= %d" % self.pfd.proflen, size=DEFSIZE)
        plt.figtext(0.29, 0.76, "Profile Avg", size=DEFSIZE)
        plt.figtext(0.39, 0.76, "= %.4g" % self.pfd.avgprof, size=DEFSIZE)
        plt.figtext(0.29, 0.74, r"Profile StdDev", size=DEFSIZE)
        plt.figtext(0.39, 0.74, "= %.4g" % np.sqrt(self.pfd.varprof), size=DEFSIZE)
        plt.figtext(0.65, 0.96, "Search Information", size=DEFSIZE)
        plt.figtext(0.53, 0.94, r"RA$_{\mathsf{J2000}}$ = %s" % self.pfd.rastr, \
                        size=DEFSIZE)
        plt.figtext(0.72, 0.94, r"Dec.$_{\mathsf{J2000}}$ = %s" % self.pfd.decstr, \
                        size=DEFSIZE)
        plt.figtext(0.58, 0.92, "Folding Parameters", size=DEFSIZE)

        # Use combine_profs to find red chi2 because I'm too lazy
        # to find out why I'm calculating it wrong.
        f = presto.foldstats()
        outprof = np.empty(self.pfd.proflen)
        delays = np.zeros(self.pfd.npart)
        presto.combine_profs(self.pfd.profs.sum(axis=1).ravel(),\
                             self.pfd.stats.sum(axis=1).ravel(), \
                             self.pfd.npart, self.pfd.proflen, \
                             delays, outprof, f)
        df = self.pfd.proflen-1
        prob = sp.stats.chi2.sf(f.redchi*df, df)
        sig = -sp.stats.norm.ppf(prob)
        if prob == 0.0:
            plt.figtext(0.53, 0.90, r"Reduced $\chi^2$ = %.3f   P(Noise) ~ 0" % \
                        f.redchi, size=DEFSIZE)
        else:
            plt.figtext(0.53, 0.90, r"Reduced $\chi^2$ = %.3f   P(Noise) < %.3g  " \
                        r"($\approx$ %.1f $\sigma$)" % (f.redchi, prob, sig), size=DEFSIZE)


    # Define callback functions
    def keypress(self, event):
        if event.key in ['q', 'Q']:
            plt.close()
        elif event.key in [' ']:
            print "Replotting..."
            self.plot()
            print "Done."
        elif event.key in ['k', 'K']:
            if self.subints_to_kill:
                print "Killing %d subints:" % len(self.subints_to_kill)
                print "\t%s" % self.subints_to_kill
                self.pfd.kill_intervals(self.subints_to_kill)
                self.subints_to_kill = []
            else:
                print "No subints to kill..."
            if self.subbands_to_kill:
                print "Killing %d subbands:" % len(self.subbands_to_kill)
                print "\t%s" % self.subbands_to_kill
                self.pfd.kill_subbands(self.subbands_to_kill)
                self.subbands_to_kill = []
            else:
                print "No subbands to kill..."


    def mousepress(self, event):
#        if (event.inaxes == sumprof_ax) and (event.button==1):
#            plt.axes(sumprof_ax)
#            cid_mousemove = fig.canvas.mpl_connect('motion_notify_event', \
#                                                            mousemove)
#            cid_mouserelease = fig.canvas.mpl_connect('button_release_event', \
#                                                            mouserelease)
#            xlow = np.fmod(event.xdata, 1) # Ensure between 0-1
#            
#            vspan1 = plt.axvspan(event.xdata, event.xdata+1/pfd.proflen, \
#                                    fc='b', lw=0, alpha=0.15)
#            vspan1.xstart = event.xdata # store starting x
#            vspan2 = plt.axvspan(1+event.xdata, 1+event.xdata+1/pfd.proflen, \
#                                    fc='b', lw=0, alpha=0.15)
#            vspan2.xstart = event.xdata # store starting x
#            plt.draw()
        if (event.inaxes == self.timephs_ax):
            # Check to see if there is already a patch
            patch_found = None
            redraw = False
            subint = np.floor(event.ydata/self.pfd.T*self.pfd.npart)
            for patch in self.timephs_ax.patches:
                if patch.contains(event)[0]:
                    patch_found = patch
            # Update the plot
            if event.button==1 and patch_found is not None:
                # Set subint to not be killed
                patch_found.remove()
                self.subints_to_kill.remove(subint)
                redraw = True
            elif event.button==3:
                # Set subint to be killed
                if subint not in self.subints_to_kill and \
                        subint < self.pfd.npart and \
                        subint >= 0:
                    self.subints_to_kill.append(subint)
                    plt.axes(self.timephs_ax)
                    plt.axis('tight')
                    plt.axhspan(subint*self.pfd.T/self.pfd.npart, \
                                (subint+1)*self.pfd.T/self.pfd.npart, \
                                fc='r', lw=0, alpha=0.15, zorder=1)
                    redraw = True
            if redraw:
                plt.draw()
        elif (event.inaxes == self.freqphs_ax) and (event.button==3):
            # Check to see if there is already a patch
            patch_found = None
            redraw = False
            subband = np.floor(event.ydata)
            for patch in self.freqphs_ax.patches:
                if patch.contains(event)[0]:
                    patch_found = patch
            # Update the plot
            if event.button==1 and patch_found is not None:
                patch_found.remove()
                self.subbands_to_kill.remove(subband)
                redaw = True
            elif event.button==3:
                if subband not in self.subbands_to_kill and \
                        subband < self.pfd.nsub and \
                        subband >= 0:
                    self.subbands_to_kill.append(subband)
                    plt.axes(self.freqphs_ax)
                    plt.axis('tight')
                    plt.axhspan(subband, (subband+1), \
                                fc='r', lw=0, alpha=0.15, zorder=1)
                    redraw = True
            plt.draw()
#    
#    def mousemove(self, event):
#        if event.inaxes==sumprof_ax:
#            verts = lambda x1, x2: np.array([[x1, 0.], [x1, 1.], [x2, 1.], [x2, 0.], [x1, 0.]])
#            print sumprof_ax.patches
#            plt.draw()

    def plot_prof(self, sumprof_ax=None):
        """Plot summed profile.
        """
        if sumprof_ax is None:
            # Create new axes
            sumprof_ax = plt.axes()
        else:
            # Set given axes as current
            plt.axes(sumprof_ax)
        plt.cla()
        phase = np.linspace(0,2, self.pfd.proflen*2, endpoint=False)
        prof = np.tile(self.pfd.sumprof, (2,))
        # prof = np.concatenate([self.pfd.sumprof]*2, axis=0)
        plt.plot(phase, prof, 'k-')
        plt.axhline(self.pfd.avgprof, c='k', ls='dotted')
        for loc, spine in sumprof_ax.spines.iteritems():
            if loc != 'bottom':
                spine.set_color('none')
        bspine_ax = sumprof_ax.spines['bottom'].axis
        bspine_ax.set_ticks(np.linspace(0,2,20, endpoint=False), minor=True)
        bspine_ax.set_ticks(np.linspace(0,2,4, endpoint=False), minor=False)
        plt.ticklabel_format(style='plain', useOffset=False, axis='both')
        plt.tick_params(which='both', top='off', right='off', left='off', \
                        labeltop='off', labelright='off', labelleft='off', \
                        labelbottom='off')
        if False:
            divider = axes_grid1.make_axes_locatable(sumprof_ax)
            prof_yerr = divider.append_axes("left", size="-10%", pad="10%", sharey=sumprof_ax)
            plt.cla()
            # Add vertical error bar
            plt.axis('off')
            plt.errorbar(0, self.pfd.avgprof, yerr=np.sqrt(self.pfd.varprof), \
                            ecolor='k', marker='x', mec='k')
            plt.axhline(self.pfd.avgprof, c='k', ls='dotted')
            plt.ticklabel_format(style='plain', useOffset=False, axis='both')

        # Add horizontal error bar (radio data only)
        if self.pfd.lofreq > 0.0 and self.pfd.chan_wid > 0.0 and False:
            bw = self.pfd.numchan*self.pfd.chan_wid
            centre_freq = self.pfd.lofreq+bw/2.0
            tdm = psr_utils.dm_smear(dm, bw, centre_freq)
            ttot = np.sqrt(tdm**2 + pfd.dt**2 + (self.pfd.chan_wid*1e6)**-2)
            prof_xerr = divider.append_axes("top", size="10%", pad=0, sharex=sumprof_ax)
            plt.axis('off')
            plt.errorbar(1, 0, xerr=ttot*0.5, ecolor='k', marker='none', mec='k')
            plt.ticklabel_format(style='plain', useOffset=False, axis='both')
        plt.draw()

    def plot_timephs(self, timephs_ax=None):
        """Time vs. Phase plot.
        """
        if timephs_ax is None:
            # Create new axes
            timephs_ax = plt.axes()
        else:
            # Set given axes as current
            plt.axes(timephs_ax)
        plt.cla()
        timephs = np.tile(self.pfd.profs.sum(axis=1).squeeze(), (2,))
        # timephs = np.concatenate([timephs]*2, axis=1)
        scaled_timephs = scale2d(timephs, indep=False)
        plt.imshow(scaled_timephs, interpolation='nearest', aspect='auto', \
                    origin='lower', cmap=matplotlib.cm.gist_yarg, \
                    extent=(0,2,0,self.pfd.T))
        plt.ylabel("Time (s)")
        plt.xlabel("Phase")
        plt.yticks(rotation=90)
        timephs_ax.xaxis.set_ticks(np.linspace(0,2,20, endpoint=False), minor=True)
        timephs_ax.xaxis.set_ticks(np.linspace(0,2,4, endpoint=False), minor=False)
        plt.draw()

    def plot_freqphs(self, freqphs_ax=None):
        """Freq vs. Phase plot.
        """
        if freqphs_ax is None:
            # Create new axes
            freqphs_ax = axes_grid1.host_axes()
        else:
            # Set given axes as current
            plt.axes(freqphs_ax)
        plt.cla()
        freqphs = np.tile(self.pfd.profs.sum(axis=0).squeeze(), (2,))
        # freqphs = np.concatenate([freqphs]*2, axis=1)
        #freqphs =  np.tile(freqphs, (1,2))
        scaled_freqphs = scale2d(freqphs, indep=False)
        plt.imshow(scaled_freqphs, interpolation='nearest', aspect='auto', \
                    origin='lower', cmap=matplotlib.cm.gist_yarg, \
                    extent=(0,2,0,self.pfd.nsub))
        plt.ylabel("Sub-band")
        plt.xlabel("Phase")
        plt.yticks(rotation=90)
        freqphs_ax.xaxis.set_ticks(np.linspace(0,2,20, endpoint=False), minor=True)
        freqphs_ax.xaxis.set_ticks(np.linspace(0,2,4, endpoint=False), minor=False)
        trans = matplotlib.transforms.Affine2D().scale(1.0, \
                        self.pfd.subdeltafreq).translate(0,self.pfd.lofreq)
        para = freqphs_ax.twin(trans.inverted())
        para.set_viewlim_mode("transform")
        para.set_ylabel("Freq (MHz)")
        plt.setp(para.xaxis.get_ticklabels(), visible=False)
        plt.setp(para.yaxis.get_ticklabels(), rotation=90)
        plt.draw()

    def plot_dmchi2(self, dmchi2_ax=None):
        """Chi2 vs. DM plot.
        """
        if dmchi2_ax is None:
            # Create new axes
            dmchi2_ax = plt.axes()
        else:
            # Set given axes as current
            plt.axes(dmchi2_ax)
        plt.cla()
        chi2s = np.zeros_like(self.pfd.dms)
        for ii, dm in enumerate(self.pfd.dms):
            self.pfd.dedisperse(dm, doppler=True)
            chi2s[ii] = self.pfd.calc_redchi2()
        plt.plot(self.pfd.dms, chi2s, 'k-')
        plt.axis([np.min(self.pfd.dms), np.max(self.pfd.dms), 0, 1.1*np.max(chi2s)])
        plt.xlabel(r"DM (cm$^{-3}$ pc)")
        plt.ylabel(r"Reduced $\chi^2$")
        plt.setp(dmchi2_ax.yaxis.get_ticklabels(), rotation=90)
        self.pfd.dedisperse(doppler=True)
        plt.draw()

    def plot_timechi2(self, timechi2_ax=None):
        """Chi2 vs. Time plot.
        """
        if timechi2_ax is None:
            # Create new axes
            timechi2_ax = plt.axes()
        else:
            # Set given axes as current
            plt.axes(timechi2_ax)
        plt.cla()
        chi2s = np.ones(self.pfd.npart+1)
        timephs = self.pfd.time_vs_phase()
        profs = np.cumsum(timephs, axis=0)
        prof_avg = 0
        prof_var = 0
        for ii, prof in enumerate(profs):
            prof_avg += np.sum(self.pfd.stats[ii][:,4])
            prof_var += np.sum(self.pfd.stats[ii][:,5])
            chi2s[ii+1] = self.pfd.calc_redchi2(prof, prof_avg, prof_var)
        plt.plot(chi2s, np.linspace(0,1,self.pfd.npart+1), 'k-')
        plt.axis([1.1*np.max(chi2s), 0, 0, 1])
        timechi2_ax.yaxis.set_ticks_position('right')
        timechi2_ax.yaxis.set_label_position('right')
        plt.yticks(rotation=90)
        plt.xlabel(r"Reduced $\chi^2$")
        plt.ylabel("Fraction of Observation")
        plt.draw()

    def plot_ppdot(self, ppdot_ax=None):
        """Chi2 vs. (P, Pdot) Plot.
        """
        if ppdot_ax is None:
            # Create new axes
            ppdot_ax = plt.axes()
        else:
            # Set given axes as current
            plt.axes(ppdot_ax)
        plt.cla()

        # Prepare delays
        parttimes = self.pfd.start_secs.astype('float32').astype('float64')
        curr_f1, curr_f2 = psr_utils.p_to_f(self.pfd.curr_p1, self.pfd.curr_p2)
        freqs = 1.0/self.pfd.periods
        tmp, freqderivs = psr_utils.p_to_f(1.0/self.pfd.fold_p1, self.pfd.pdots)
        df = freqs - self.pfd.fold_p1
        dfd = freqderivs - self.pfd.fold_p2
        f_delays = np.outer(df, parttimes)
        fd_delays = np.outer(dfd, (parttimes)**2 / 2.0)

        chi2s = np.empty(self.pfd.pdots.size * self.pfd.periods.size)
        outprof = np.empty(self.pfd.proflen)
        outstats = presto.foldstats()
        instats = self.pfd.stats.sum(axis=1).ravel()
        inprofs = self.pfd.profs.sum(axis=1).ravel()
        phasedelays = fd_delays.repeat(self.pfd.periods.size, axis=0) + \
                         np.tile(f_delays, (self.pfd.pdots.size,1))
        bindelays = phasedelays*self.pfd.proflen - self.pfd.pdelays_bins
       
        for ii, delay in enumerate(bindelays):
            presto.combine_profs(inprofs, instats, self.pfd.npart, \
                                     self.pfd.proflen, delay, outprof, outstats)
            chi2s[ii] = outstats.redchi
        chi2s.shape = (self.pfd.pdots.size, self.pfd.periods.size)
        
        plt.imshow(chi2s, interpolation='nearest', aspect='auto', \
                    origin='lower', cmap=matplotlib.cm.anti_rainbow)
        plt.draw()

    def plot_pchi2(self, pchi2_ax=None):
        """Chi2 vs. P plot.
        """
        if pchi2_ax is None:
            # Create new axes
            pchi2_ax = plt.axes()
        else:
            # Set given axes as current
            plt.axes(pchi2_ax)
        plt.cla()
        
        # Prepare delays
        parttimes = self.pfd.start_secs.astype('float32').astype('float64')
        curr_f2 = -self.pfd.curr_p2/(self.pfd.periods**2.0)
        freqs = 1.0/self.pfd.periods
        df = freqs - self.pfd.fold_p1
        dfd = curr_f2 - self.pfd.fold_p2
        f_delays = np.outer(df, parttimes)
        fd_delays = np.outer(dfd, (parttimes)**2 / 2.0)
        
        chi2s = np.empty(self.pfd.periods.size)
        outprof = np.empty(self.pfd.proflen)
        outstats = presto.foldstats()
        instats = self.pfd.stats.sum(axis=1).ravel()
        inprofs = self.pfd.profs.sum(axis=1).ravel()
        phasedelays = f_delays + fd_delays
        bindelays = phasedelays*self.pfd.proflen - self.pfd.pdelays_bins

        print "Period vs. Chi2" # DEBUG
        for ii, delay in enumerate(bindelays):
            presto.combine_profs(inprofs, instats, self.pfd.npart, \
                                     self.pfd.proflen, delay, outprof, outstats)
            chi2s[ii] = outstats.redchi
            print "%d - %.12f" % (ii, chi2s[ii]) # DEBUG
        pfold = psr_utils.p_to_f(self.pfd.fold_p1, self.pfd.fold_p2, \
                                    self.pfd.fold_p3)[0]
        periods = (self.pfd.periods-pfold)*1000 # in ms
        plt.plot(periods, chi2s, 'k-')
        plt.axis([np.max(periods), np.min(periods), 0, np.max(chi2s)*1.1])
        plt.ticklabel_format(style='sci', scilimits=(4,4), useOffset=False)
        plt.draw()

    def plot_pdchi2(self, pdchi2_ax=None):
        """Chi2 vs. Pd plot.
        """
        if pdchi2_ax is None:
            # Create new axes
            pdchi2_ax = plt.axes()
        else:
            # Set given axes as current
            plt.axes(pdchi2_ax)
        plt.cla()
        
        # Prepare delays
        parttimes = self.pfd.start_secs.astype('float32').astype('float64')
        curr_f1 = 1.0/self.pfd.curr_p1
        freqderivs = -self.pfd.pdots/(self.pfd.curr_p1**2.0)
        df = curr_f1 - self.pfd.fold_p1
        dfd = freqderivs - self.pfd.fold_p2
        f_delays = np.outer(df, parttimes)
        fd_delays = np.outer(dfd, (parttimes)**2 / 2.0)
        
        chi2s = np.empty(self.pfd.pdots.size)
        outprof = np.empty(self.pfd.proflen)
        outstats = presto.foldstats()
        instats = self.pfd.stats.sum(axis=1).ravel()
        inprofs = self.pfd.profs.sum(axis=1).ravel()
        phasedelays = f_delays + fd_delays
        bindelays = phasedelays*self.pfd.proflen - self.pfd.pdelays_bins

        print "P-dot vs. Chi2" # DEBUG
        for ii, delay in enumerate(bindelays):
            presto.combine_profs(inprofs, instats, self.pfd.npart, \
                                     self.pfd.proflen, delay, outprof, outstats)
            chi2s[ii] = outstats.redchi
            print "%s - %.12f" % (ii, chi2s[ii]) # DEBUG
        pdfold = psr_utils.p_to_f(self.pfd.fold_p1, self.pfd.fold_p2, \
                                    self.pfd.fold_p3)[1]
        pdots = self.pfd.pdots-pdfold
        plt.plot(pdots, chi2s, 'k-')
        plt.axis([np.max(pdots), np.min(pdots), 0, np.max(chi2s)*1.1])
        plt.draw()


def scale2d(array2d, indep=False):
    """Scale a 2D array for plotting.
        Subtract min from each row.
        Divide by global max (if indep==False)
        or divide by max of each row (if indep==True)
    """
    # Min of each row
    min = np.outer(array2d.min(axis=1), np.ones(array2d.shape[1]))
    if indep==False:
        # Global maximum
        max = array2d.max()
    else:
        # maximum for each row
        max = np.outer(arra2d.max(axis=1), np.ones(array2d.shape[1]))
    return (array2d - min)/max


if __name__=='__main__':
    # Adjust some matplotlib configurations
    set_defaults()
    pfdplot = PrepfoldPlot(sys.argv[1])
    plt.show()
