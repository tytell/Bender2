import h5py
import numpy as np
from scipy import signal, fftpack, integrate
import matplotlib.pyplot as plt

class BendingData(object):
    def __init__(self, filename, datasets=('Fx', 'Fx', 'Fz', 'Tx', 'Ty', 'Tz', 'Lact', 'Ract')):
        self.filename = filename

        with h5py.File(filename, 'r') as h5file:
            self.sampfreq = h5file['RawInput'].attrs['SampleFrequency']

            stim = {n: v for n, v in h5file['NominalStimulus'].attrs.items()}
            self.stim = stim

            data = [np.array(h5file['Calibrated'][dataset]) for dataset in datasets]
            self.data = np.array(data)
            self.datanames = datasets
            self.angle = np.array(h5file['RawInput']['Encoder'])

        self.t = np.arange(len(self.angle)) / self.sampfreq

    def get_data(self, dataname):
        return [self.data[i, :] for i, n in enumerate(self.datanames) if n == dataname][0]

    def EI(self, din, doutvert, dclamp):
        EI1 = self.get_data('Tx') * din / doutvert * dclamp / 1000.0
        EI2 = self.get_data('Fy') * din / 1000.0 * dclamp / 1000.0

        return EI1, EI2


class ProcessPassiveBending(object):
    def __init__(self, filename, din, doutvert, dclamp, width, height, isfreqsweep=False):
        self.filename = filename

        self.din = din
        self.doutvert = doutvert
        self.dclamp = dclamp

        self.width = width
        self.height = height
        self.I = self.second_moment()

        self.isfreqsweep = isfreqsweep
        self.phase = None

        self.load_data()

    def second_moment(self):
        I = np.pi / 4 * (float(self.width) / 2 / 1000) ** 3 * (float(self.height) / 2 / 1000)
        return I

    def load_data(self):
        with h5py.File(self.filename, 'r') as h5file:
            self.sampfreq = h5file['RawInput'].attrs['SampleFrequency']

            stim = {n: v for n, v in h5file['NominalStimulus'].attrs.items()}
            self.stim = stim

            self.Fy = np.array(h5file['Calibrated']['Fy'])
            self.Tx = np.array(h5file['Calibrated']['Tx'])
            self.angle = np.array(h5file['RawInput']['Encoder'])

            if len(h5file['Output']['Lactcmd']) > 2:
                self.Lact = np.array(h5file['Output']['Lactcmd'])
                self.Ract = np.array(h5file['Output']['Ractcmd'])

        self.t = np.arange(len(self.angle)) / self.sampfreq

    def filter_data(self, cutoff=None, b=None, a=None):
        if cutoff:
            b, a = signal.butter(5, cutoff / (self.sampfreq / 2))

        self.Fys = signal.filtfilt(b, a, self.Fy)
        self.Txs = signal.filtfilt(b, a, self.Tx)

    def generate_activation(self, actphase, startcycle, actduty=0.3, actpulsefreq=75.0):
        if self.isfreqsweep:
            raise NotImplementedError("Activation phase for frequency sweeps isn't implemented")

        actburstdur = actduty / self.stim['Frequency']
        actburstdur = np.floor(actburstdur * actpulsefreq * 2) / (actpulsefreq * 2)
        actduty = actburstdur * self.stim['Frequency']

        self.Lact = []
        self.Ract = []
        self.isLact = np.zeros_like(self.t)
        self.isRact = np.zeros_like(self.t)

        for c in range(int(startcycle), int(self.stim['Cycles'])):
            tstart = ((c + 0.25 + actphase) / self.stim['Frequency']) + self.stim['WaitPre']
            tend = tstart + actburstdur
            self.Lact.append([tstart, tend])
            self.isLact[np.logical_and(self.t >= tstart, self.t < tend)] = 1

            tstart -= 0.5/self.stim['Frequency']
            tend -= 0.5/self.stim['Frequency']

            self.Ract.append([tstart, tend])
            self.isRact[np.logical_and(self.t >= tstart, self.t < tend)] = 1

        self.Lact = np.array(self.Lact)
        self.Ract = np.array(self.Ract)

    def get_moduli(self):
        good = np.logical_and(self.t > self.stim['WaitPre'] + 1.0 / self.stim['Frequency'],
                              self.t < self.t[-1] - self.stim['WaitPost'])

        self.good = good
        self.angle_fft = fftpack.fft(self.angle[good])
        self.Fy_fft = fftpack.fft(self.Fys[good])
        self.Tx_fft = fftpack.fft(self.Txs[good])

        self.f = fftpack.fftfreq(np.sum(good), 1.0 / self.sampfreq)

        idx = np.argmin(np.abs(self.stim['Frequency'] - self.f))
        self.basefreq_idx = idx

        mod1 = self.Tx_fft[idx] / self.angle_fft[idx]
        mod2 = self.Fy_fft[idx] / self.angle_fft[idx]

        self.EITx = mod1 * self.din / self.doutvert * (self.dclamp / 1000.0)
        self.EIFy = mod2 * self.din / 1000.0 * self.dclamp / 1000.0

    def plot_vs_time(self, ax):
        ax[0].plot(self.t, self.Tx, 'k')
        ax[0].plot(self.t, self.Txs, 'b', linewidth=3)
        ax[0].set_ylabel('Tx')

        if len(ax) > 1:
            ax[1].plot(self.t, self.Fy, 'k')
            ax[1].plot(self.t, self.Fys, 'g', linewidth=3)
            ax[1].set_ylabel('Fy')

    def plot_power(self, ax=None, showsmooth=True, excludeprepost=True):
        if ax is None:
            fig, ax = plt.subplots(2,1, sharex=True, sharey=True)

        # check for a list
        try:
            ax[0]
        except TypeError:
            ax = [ax]

        if excludeprepost:
            good = np.logical_and(self.t > self.stim['WaitPre'],
                                  self.t < self.t[-1] - self.stim['WaitPost'])
        else:
            good = np.isfinite(self.t)

        f, spec1 = signal.periodogram(self.Tx[good], fs=self.sampfreq)
        spec = [spec1]
        label = ['Tx']
        if showsmooth:
            f, spec2 = signal.periodogram(self.Txs[good], fs=self.sampfreq)
            spec.append(spec2)
            label.append('Txs')

            col = ['k','r']
        else:
            col = ['k']

        spec = [spec]
        label = [label]

        if len(ax) == 3:
            f, spec1 = signal.periodogram(self.Fy[good], fs=self.sampfreq)
            if not showsmooth:
                spec.append([spec1])
                label.append(['Fy'])
            else:
                f, spec2 = signal.periodogram(self.Fys[good], fs=self.sampfreq)
                spec.append([spec1, spec2])
                label.append(['Fy', 'Fys'])

        if len(ax) >= 2:
            f, spec1 = signal.periodogram(self.angle[good], fs=self.sampfreq)
            spec.append([spec1])
            label.append(['angle'])

        for ax1, spec1, label1 in zip(ax,spec,label):
            for spec11, label11, col1 in zip(spec1, label1, col):
                ax1.loglog(f,spec11, col1, label=label11)
            ax1.set_ylabel(label1[0])

        ax[-1].set_xlabel('Frequency (Hz')

    def plot_vs_phase(self, ax):
        if self.isfreqsweep and not self.phase:
            raise NotImplementedError('no phase for frequency sweeps yet')

        phase = (self.t - self.stim['WaitPre']) * self.stim['Frequency']

        ax[0].plot(phase, self.Tx, 'k')
        ax[0].plot(phase, self.Txs, 'b', linewidth=3)
        ax[0].set_ylabel('Tx')

        if len(ax) > 1:
            ax[1].plot(phase, self.Fy, 'k')
            ax[1].plot(phase, self.Fys, 'g', linewidth=3)
            ax[1].set_ylabel('Fy')

    def plot_vs_angle(self, ax, showcycles=None, addarrows=(1.15, 1.65)):
        if showcycles is not None:
            phase = (self.t - self.stim['WaitPre']) * self.stim['Frequency']
            good = np.logical_and(phase >= showcycles[0], phase < showcycles[1])
            arrowadd = showcycles[0]
        else:
            good = self.good
            arrowadd = 0.0

        ax[0].plot(self.angle[good], self.Txs[good])

        idx = [int(np.round(((a+arrowadd) / self.stim['Frequency'] + self.stim['WaitPre']) * self.sampfreq))
               for a in addarrows]
        for i1 in idx:
            ax[0].annotate('', xy=(self.angle[i1], self.Txs[i1]),
                           xytext=(self.angle[i1 - 20], self.Txs[i1 - 20]),
                           arrowprops=dict(arrowstyle='fancy', facecolor='black', mutation_scale=36))

        if len(ax) > 1:
            ax[1].plot(self.angle[good], self.Fys[good])
            for i1 in idx:
                ax[1].annotate('', xy=(self.angle[i1], self.Fys[i1]),
                               xytext=(self.angle[i1 - 10], self.Fys[i1 - 10]),
                               arrowprops=dict(facecolor='black', shrink=0.05))

    def plot_work_loop(self, ax, actstartcycle, leftcolor='darkred', rightcolor='darkblue',
                       actthickness=3, color='k', linestyle='-',
                       passivecolor='darkgreen', passivestyle='--',
                       normalize=True, yscale=1000.0, cycles=None, showarrows=True):
        phase = (self.t - self.stim['WaitPre']) * self.stim['Frequency']
        ispass = phase < actstartcycle
        isact = phase >= actstartcycle

        if cycles is not None:
            incycles = np.array([np.floor(p) in cycles for p in phase])

            ispass = np.logical_and(ispass, incycles)
            isact = np.logical_and(isact, incycles)
        else:
            incycles = np.full_like(phase, True, dtype=bool)

        if normalize:
            y = self.Txs * self.din/self.doutvert
            x = self.angle # * np.pi/180.0 / (self.dclamp / 1000.0)
        else:
            y = self.Txs
            x = self.angle

        if normalize:
            ishold = np.logical_or(self.t < self.stim['WaitPre'],
                                   self.t > self.t[-1] - self.stim['WaitPost']/2)
            y0 = np.mean(y[ishold])

            y -= y0

        ax.plot(x[isact], y[isact]*yscale, linestyle, color=color)
        for onoff in self.Lact:
            ison = np.logical_and(np.logical_and(self.t >= onoff[0], self.t < onoff[1]), incycles)
            h = ax.plot(x[ison], y[ison]*yscale, linestyle, linewidth=actthickness, color=leftcolor)
        h[0].set_label('left')

        for onoff in self.Ract:
            ison = np.logical_and(np.logical_and(self.t >= onoff[0], self.t < onoff[1]), incycles)
            h = ax.plot(x[ison], y[ison]*yscale, linestyle, linewidth=actthickness, color=rightcolor)
        h[0].set_label('right')

        if showarrows:
            if cycles is None:
                c = -1
            else:
                Lactphase = (self.Lact - self.stim['WaitPre']) * self.stim['Frequency']
                Ractphase = (self.Lact - self.stim['WaitPre']) * self.stim['Frequency']

                Lactincycles = np.array([np.floor(l) in cycles for l in Lactphase.flat]).reshape(Lactphase.shape)
                Ractincycles = np.array([np.floor(r) in cycles for r in Ractphase.flat]).reshape(Ractphase.shape)

                isactcyc = np.logical_and(np.all(Lactincycles, axis=1),
                                          np.all(Ractincycles, axis=1))

                c = np.argwhere(isactcyc)
                c = c[-1]

            arrowtimes = [np.mean(self.Lact[c, :]), np.mean(self.Ract[c, :])]
            col = [leftcolor, rightcolor]

            for t1, col1 in zip(arrowtimes, col):
                idx = int(t1 * self.sampfreq)

                ax.annotate('', xy=(x[idx + 10], y[idx + 10]*yscale),
                            xytext=(x[idx], y[idx]*yscale),
                            arrowprops=dict(arrowstyle='fancy', facecolor=col1, mutation_scale=48))

        ax.plot(x[ispass], y[ispass]*yscale, passivestyle, label='passive', color=passivecolor)
        # ax.set_xlabel('Angle (deg)')
        # ax.set_ylabel('Torque (N m)')

    def get_work(self, actstartcycle, actphase):
        phase = (self.t - self.stim['WaitPre']) * self.stim['Frequency']
        ispass = phase < actstartcycle
        isact = phase >= actstartcycle

        phase = phase - actphase

        y = self.Txs * self.din / self.doutvert
        x = self.angle * np.pi/180.0

        work = []
        isactcycle = []
        for c in range(0, int(self.stim['Cycles'])-1):
            iscycle = np.logical_and(phase >= c, phase <= c+1)
            if any(np.logical_and(iscycle, ispass)) and any(np.logical_and(iscycle, isact)):
                print("Cycle {} contains active and passive periods. Skipping".format(c))
                work.append(np.nan)
                isactcycle.append(False)
            elif any(iscycle):
                w1 = integrate.trapz(y[iscycle], x=x[iscycle])
                work.append(w1)
                isactcycle.append(any(np.logical_and(iscycle, isact)))

        return np.array(work), np.array(isactcycle)

    def plot_stiffness(self, ax, yscale=1e-3):
        y = self.Txs * self.din / self.doutvert
        x = self.angle * np.pi/180.0 / (self.dclamp / 1000.0)

        dt = 1/self.sampfreq

        dy = np.gradient(y, dt)
        dx = np.gradient(x, dt)

        EI = dy / dx
        I = self.second_moment()

        E = EI / I

        ax.plot(self.t, E*yscale)

        return EI, E

    def get_stiffness(self, actstartcycle, avgdur=0.4):
        phase = (self.t - self.stim['WaitPre']) * self.stim['Frequency']
        ispass = phase < actstartcycle
        isact = phase >= actstartcycle

        y = self.Txs * self.din / self.doutvert
        x = self.angle * np.pi/180.0 / (self.dclamp / 1000.0)

        dt = 1/self.sampfreq

        dy = np.gradient(y, dt)
        dx = np.gradient(x, dt)

        I = self.second_moment()
        Eall = (dy / dx) / I

        E = []
        isactcycle = []
        phasecycle = []
        for c in np.arange(0.3, int(self.stim['Cycles'])-1, 0.5):
            iscycle = np.logical_and(phase >= c, phase <= c+avgdur)
            phasecycle.append([c, c+avgdur])
            if any(np.logical_and(iscycle, ispass)) and any(np.logical_and(iscycle, isact)):
                print("Cycle {} contains active and passive periods. Skipping".format(c))
                E.append(np.nan)
                isactcycle.append(False)
            else:
                E1 = Eall[iscycle]
                E1[np.isinf(E1)] = np.nan
                E.append(np.nanmedian(E1))
                isactcycle.append(any(np.logical_and(iscycle, isact)))

        E = np.array(E)
        isactcycle = np.array(isactcycle)
        phasecycle = np.array(phasecycle)
        return E, isactcycle, phasecycle

    def plot_with_moduli(self, ax):
        idx = self.basefreq_idx
        mod1 = self.Fy_fft[idx] / self.angle_fft[idx]
        mod2 = self.Tx_fft[idx] / self.angle_fft[idx]

        good = self.good
        ax[0].plot(self.t[good], self.angle[good])
        ax[0].plot(self.t[good], self.stim['Amplitude'] *
                   np.sin(2 * np.pi * self.stim['Frequency'] * (self.t[good] - self.stim['WaitPre'])), 'k--')

        delta1 = np.arccos(np.real(mod1) / np.abs(mod1))
        delta2 = np.arccos(np.real(mod2) / np.abs(mod2))

        ax[1].plot(self.t[good], self.Fys[good])
        mag = np.max(self.Fys[good] - np.mean(self.Fys[good]))
        ax[1].plot(self.t[good], np.mean(self.Fys[good]) + mag *
                   np.sin(2 * np.pi * self.stim['Frequency'] * (self.t[good] - self.stim['WaitPre']) + delta1), 'k--')

        ax[2].plot(self.t[good], self.Txs[good])
        mag = np.max(self.Txs[good] - np.mean(self.Txs[good]))
        ax[2].plot(self.t[good], np.mean(self.Txs[good]) + mag *
                   np.sin(2 * np.pi * self.stim['Frequency'] * (self.t[good] - self.stim['WaitPre']) + delta2), 'k--')