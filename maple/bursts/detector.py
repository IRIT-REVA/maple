import abc
from typing import Optional

from maple.basic_analysis import ExtendedResults

from maple.bursts.container import Bursts


class Detector(abc.ABC):

    # Create a temp array for the storage of the bursts. Assume that
    # it will not be longer than nspikes/2 since we need at least two
    # spikes to be in a burst.

    def __init__(self,
                 file_name: str):

        self.file_name = file_name

    def apply(self,
              template_ids: Optional[list] = None):
        extres = ExtendedResults(self.file_name,
                                 fields=[ExtendedResults.Fields.spike_times,
                                         ExtendedResults.Fields.isis],
                                 templ_ids=template_ids)

        b = Bursts(self)
        b.ids = self.find_bursts(extres.spike_times, extres.isis)

        return b

    @abc.abstractmethod
    def signature(self):
        pass

    @abc.abstractmethod
    def find_bursts(self,
                    spike_times: dict,
                    spike_intervals: dict):

        """ For single spike train, finds the burst using max interval method.
        # params currently in par
        ##

        # TODO: all our burst analysis routines should use the same
        # value to indiciate "no bursts" found.
        # no.bursts = NA;                  #value to return if no bursts found.
        no_bursts = matrix(nrow=0,ncol=1)  #emtpy value nrow()=length() = 0.
        """
        pass
