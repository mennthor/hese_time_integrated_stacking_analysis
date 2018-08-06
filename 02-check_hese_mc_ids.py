# coding:utf-8

"""
Apply HESE filter to sim files and store runIDs, eventIDs and MC primary
energies for all events surviving the filter.
The IDs are checked against the final level MCs to sort out any HESE like events
for sensitivity calulations.
"""

from __future__ import division, print_function

import json
import argparse

from I3Tray import *
from icecube import icetray, dataclasses, dataio
from icecube import VHESelfVeto, DomTools, weighting


class collector(icetray.I3ConditionalModule):
    """
    Collect run ID, event ID and MC primary energy for events surviving the
    HESE filter.
    """

    def __init__(self, ctx):
        icetray.I3ConditionalModule.__init__(self, ctx)
        self.AddParameter("outfilename", "outfilename", "")

    def Configure(self):
        # HESE total charge cut: wiki.icecube.wisc.edu/index.php/HESE-7year#Summary_of_Cuts_and_Parameters
        self.minimum_charge = 6000.

        self.outfile = self.GetParameter("outfilename")
        self.run_id = []
        self.event_id = []
        # Just save some extra stuff
        self.qtot = []
        self.energy = []

    def Physics(self, frame):
        # If HESE veto is passed, VHESelfVeto variable is False.
        # Also VHESelfVeto doesn't always write the key, which means the event
        # is vetoed.
        try:
            hese_veto = frame["VHESelfVeto"].value
        except KeyError:
            hese_veto = True
        if not hese_veto:
            # Not HESE vetoed: Candidate event
            qtot = frame["HESE_CausalQTot"].value
            if qtot >= self.minimum_charge:
                # Also over QTot cut: Winner
                evt_header = frame["I3EventHeader"]
                prim = frame["MCPrimary"]

                self.run_id.append(evt_header.run_id)
                self.event_id.append(evt_header.event_id)
                self.energy.append(prim.energy)
                self.qtot.append(qtot)

                self.PushFrame(frame)

    def Finish(self):
        out_dict = {"energy": self.energy, "run_id": self.run_id,
                    "event_id": self.event_id, "qtot": self.qtot}
        with open(self.outfile, "w") as outf:
            json.dump(out_dict, fp=outf, indent=2)
        print("Wrote output file to:\n  ", self.outfile)


def main(in_files, out_file, gcd_file):
    files = []
    files.append(gcd_file)
    if not isinstance(in_files, list):
        in_files = [in_files]
    files.extend(in_files)

    tray = I3Tray()

    # Read files
    tray.AddModule("I3Reader", "reader", Filenamelist=files)

    # Create correct MCPrimary
    tray.AddModule(weighting.get_weighted_primary, "weighted_primary",
                   If=lambda frame: not frame.Has("MCPrimary"))

    def rename(frame):
        """ Rename for older files not having 'InIcePulses' key """
        if not frame.Has('InIcePulses'):
            frame['InIcePulses'] = frame['OfflinePulses']
        return True

    # Rename frames if InIcePulses
    tray.AddModule(rename, 'rename')

    ##########################################################################
    # Following code from hesefilter.py
    # Prepare Pulses
    tray.AddModule("I3LCPulseCleaning", "cleaning_HLC",
                   OutputHLC="InIcePulsesHLC",
                   OutputSLC="",
                   Input="InIcePulses",
                   If=lambda frame: not frame.Has("InIcePulsesHLC"))

    # Apply HESE filter
    tray.AddModule("VHESelfVeto",
                   "selfveto",
                   Pulses="InIcePulsesHLC")

    # Add CausalQTot frame
    tray.AddModule('HomogenizedQTot', 'qtot_causal',
                   Pulses="InIcePulses",
                   Output='HESE_CausalQTot',
                   VertexTime='VHESelfVetoVertexTime')
    ##########################################################################

    # Save IDs if event is HESE-like
    tray.AddModule(collector, "collector", outfilename=out_file)

    tray.AddModule("TrashCan", "NacHsart")
    tray.Execute()
    tray.Finish()


###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    # Parse options and call `main`
    parser = argparse.ArgumentParser(description="Check HESE filter")
    parser.add_argument("--infiles", type=str)
    parser.add_argument("--gcdfile", type=str)
    parser.add_argument("--outfile", type=str)

    args = parser.parse_args()

    in_files = args.infiles.split(",")
    gcd_file = args.gcdfile
    out_file = args.outfile

    main(in_files, out_file, gcd_file)
