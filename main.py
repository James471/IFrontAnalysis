import os
import sys
import numpy as np
import yt

from .StromgrenSphere import StromgrenSphere

def get_photon_err(snapshot, snapshot0, Q):
    ds = snapshot.ds
    ad = snapshot.ad
    ad0 = snapshot0.ad
    volume = (ad['dx'][0]**3).value
    n_photon_added = Q * ds.current_time.value / 8.0
    n_photon_inflight = (ad['n_photon'].sum() - ad0['n_photon'].sum()) * volume 
    n_electron_added = (ad['n_e'].sum() - ad0['n_e'].sum()) * volume
    photon_absorbed = n_electron_added
    photon_err = (n_photon_added - (n_photon_inflight + photon_absorbed)) / n_photon_added
    return np.abs(photon_err)

cwd = os.getcwd()
Q    = float(sys.argv[1])
output_file = os.path.join(sys.argv[2])
start_pattern = sys.argv[3]

sphere = StromgrenSphere(cwd, start_pattern=start_pattern)
first_snapshot = sphere.snapshot_list[0]
last_snapshot = sphere.snapshot_list[-1]

photon_err = get_photon_err(last_snapshot, first_snapshot, Q)

with open(output_file, 'w') as f:
    f.write(f"{photon_err}\n")