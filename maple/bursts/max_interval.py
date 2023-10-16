import numpy as np

from maple.bursts.detector import Detector as DetectorBase


# =============================================================================
class Pars:
    """Parameters (ms)
    """

    def __init__(self,
                 max_isi_first: float,
                 max_isi_last: float,
                 min_ibi: float,
                 min_dur: float,
                 min_nspikes: int):
        self.max_isi_first = max_isi_first  # max ISI at burst start (ms)
        self.max_isi_last = max_isi_last    # max ISI at burst end (ms)
        self.max_isi_intnl = max(max_isi_first, max_isi_last)
        self.min_ibi = min_ibi              # min inter-burst interval (ms)
        self.min_dur = min_dur              # min burst duration (ms)
        self.min_nspikes = min_nspikes      # min nunber of spikes in burst
        self.min_nisis = min_nspikes - 1

    def is_beg_isi_ok(self, isi: float):

        return isi < self.max_isi_first

    def is_end_isi_ok(self, isi: float):

        return isi < self.max_isi_last

    def is_int_isi_ok(self, isi: float):

        return np.alltrue(isi < self.max_isi_intnl)

    def is_burst(self, isis: list):
        return self.is_beg_isi_ok(isis[0]) and \
               self.is_end_isi_ok(isis[-1]) and \
               self.is_int_isi_ok(isis[1:-2])

    def __str__(self):

        return f"isiB_{self.max_isi_first}_isiE_{self.max_isi_last}_" \
               f"ibi_{self.min_ibi}_dur_{self.min_dur}_nsp_{self.min_nspikes}"


# =============================================================================
default_pars = Pars(
    max_isi_first=30.,   # max ISI at burst start (ms)
    max_isi_last=30.,    # max ISI at burst end (ms)
    min_ibi=100.,        # min inter-burst interval (ms)
    min_dur=20.,         # min burst duration (ms)
    min_nspikes=3       # min nunber of spikes in burst
)


# =============================================================================
class Detector(DetectorBase):

    def __init__(self,
                 filename: str,
                 pars: Pars):

        super().__init__(filename)
        self.pars = pars

    def signature(self):

        return f"maxint_{str(self.pars)}"

    def find_bursts(self,
                    spike_times: dict,
                    isis: dict):

        """ For single spike train, finds the burst using max interval method.
        # params currently in par
        ##

        # TODO: all our burst analysis routines should use the same
        # value to indiciate "no bursts" found.
        # no.bursts = NA;                  #value to return if no bursts found.
        no_bursts = matrix(nrow=0,ncol=1)  #emtpy value nrow()=length() = 0.
        """

        indx = {k: np.full_like(t, np.nan, dtype=np.int)
                for k, t in spike_times.items()}

        for ti, si in isis.items():
            ns = si.size + 1   # number of spikes
            ind = indx[ti]
            bi = 0       # burst index
            ii = 0       # starting spike index
            while ii < ns:
                # burst length (spikes):
                bl = self.pars.min_nspikes \
                    if ii <= ns - self.pars.min_nspikes \
                    else 1
                while bl <= ns - ii:
                    if ii <= ns - self.pars.min_nspikes:
                        isburst = self.pars.is_burst(si[ii:ii+bl-1])
                    else:
                        isburst = False
                    if isburst:
                        bl += 1
                    if not isburst or bl + ii > ns:
                        if bl > self.pars.min_nspikes:
                            bl -= 1
                        elif bl > 1:
                            bl = 1
                        ind[ii:ii+bl] = bi
                        ii += bl
                        bi += 1
                        break

        return indx
