#!/usr/bin/env python3

from os.path import join

import hong2p import util, db


def test_load_experiment():
    r1 = join(util.raw_data_root(), '2019-11-21/4/tif_stacks/fn_0001.tif')
    r2 = join(util.raw_data_root(), '2020-03-09/2/fn_002/tif_stacks/fn_002.tif')
    c1 = join(util.analysis_output_root(),
        '2019-11-21/4/tif_stacks/fn_0001_nr.tif'
    )
    tiffs = [c1, r1, r2]
    for t in tiffs: #[2:]:
        print(t)
        data = db.load_recording(t)
        #import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    test_load_experiment()

