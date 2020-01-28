#!/usr/bin/env python3

"""
"""

import time

import numpy as np
import pandas as pd
from scipy.optimize import brute

import chemutils as cu
import olfsysm as osm
from drosolf.orns import orns

import hong2p.util as u


# TODO what should this return?
def fit_model(frac_responder_df):
    # With add_spontaneous=False, it should just return the deltas...
    orn_deltas = orns(add_spontaneous=False, drop_sfr=False).T
    sfr = orn_deltas['spontaneous firing rate']
    orn_deltas = orn_deltas.iloc[:, :-1]

    tmp_mp = osm.ModelParams()
    osm.load_hc_data(tmp_mp, 'hc_data.csv')
    skip_idx = None
    for i, osm_one_orn_deltas in enumerate(tmp_mp.orn.data.delta):
        my_one_orn_deltas = orn_deltas.iloc[i]
        if not np.array_equal(osm_one_orn_deltas, my_one_orn_deltas):
            assert np.array_equal(orn_deltas.iloc[i + 1], osm_one_orn_deltas)
            skip_idx = i
            break
    assert skip_idx is not None

    # TODO TODO print which receptor(?) is being removed

    shared_idx = np.setdiff1d(np.arange(len(orn_deltas)), [skip_idx])
    sfr = sfr[shared_idx]
    orn_deltas = orn_deltas.iloc[shared_idx]
    assert np.array_equal(sfr, tmp_mp.orn.data.spont[:, 0])
    assert np.array_equal(orn_deltas, tmp_mp.orn.data.delta)
    kc_cxn_distrib = tmp_mp.kc.cxn_distrib.copy()
    del tmp_mp

    # TODO TODO i guess acetoin isn't in hallem?? maybe i'm misunderstanding
    # some other naming convention? try stripping inchi?
    hc_inchi = cu.convert(orn_deltas.columns, from_type='name')
    assert len(hc_inchi) == len(set(hc_inchi))

    mean_frac_responding = frac_responder_df.groupby(['odor_set','name1']
        ).frac_responding.mean()

    new_odor_series = []
    target_response_fracs = []
    # TODO maybe just loop over abbreviated things?
    for oset, odors in u.odor_set2order.items():
        # Skipping this for now so as not to have to worry about acetoin.
        if oset == 'flyfood':
            continue

        # So we know we can later use these prefixes to pull out just the data
        # for each odor set.
        assert not orn_deltas.columns.map(lambda c: c.startswith(oset)).any()

        for o in odors:
            abbrev = cu.odor2abbrev(o)

            # Natural stuff will be excluded by breaking out of order early.
            if o in u.solvents:
                continue

            target_response_fracs.append(mean_frac_responding.loc[oset, abbrev])

            # Anything after mix is either a mix or real as I have it now.
            #if cu.odor_is_mix(o):
            if abbrev == 'mix':
                # These will be filled in (potentially followed by another
                # optimization step) AFTER the single components are optimized
                # to produce model KC activations are similar levels to the
                # data.
                new_ser = pd.Series(
                    index=sfr.index,
                    data=np.full(sfr.shape, np.nan),
                    name=f'{oset} mix'
                )
                new_odor_series.append(new_ser)
                break

            inchi = cu.convert(o, from_type='name')
            assert inchi in hc_inchi

            new_ser = orn_deltas.T[hc_inchi == inchi].iloc[0].copy()
            #new_ser.name = f'{oset} {new_ser.name}'
            new_ser.name = f'{oset} {abbrev}'
            # Because we will use this later to identify the mix columns.
            assert not new_ser.name.endswith('mix')
            new_odor_series.append(new_ser)

    del mean_frac_responding
    target_response_fracs = np.array(target_response_fracs)

    len_before = len(orn_deltas)
    assert orn_deltas.shape[1] == 110
    orn_deltas = pd.concat([orn_deltas] + new_odor_series, axis=1)
    assert len(orn_deltas) == len_before

    # This will give the the "1" at the end of its shape that Matt's code
    # may be expecting.
    sfr = sfr.to_frame()

    mp = osm.ModelParams()

    # Set memory-related params
    # The data matrices can be edited later, but can't change size
    # (nglom x 1 matrix of spontaneous firing rates)
    mp.orn.data.spont = sfr
    # (nglom x nodor matrix of odor-evoked rate deltas)
    mp.orn.data.delta = orn_deltas
    mp.kc.cxn_distrib = kc_cxn_distrib

    # spike timings are thrown out by default
    mp.kc.save_spike_recordings = True

    # Allocate memory
    rv = osm.RunVars(mp)

    # TODO or should i just disable tuning?
    # Tune thresholds/the APL to the Hallem odors only
    mp.kc.tune_from = range(110)
    # non-zero for consistent PN-KC connectivity generation
    mp.kc.seed = 12345

    # TODO TODO should i set sim_only to exclude new stuff here?
    # (particularly if i'm gonna have null stuff for mixture response)
    # (any reason not to?)
    mp.sim_only = range(110)

    before = time.time()
    print('running initial sims.', end='', flush=True)
    # Run a 1st simulation to tune KC thresholds and APL weights
    osm.run_ORN_LN_sims(mp, rv)
    print('.', end='', flush=True)
    osm.run_PN_sims(mp, rv)
    print('.', end='', flush=True)
    osm.run_KC_sims(mp, rv, True) # True -> (re)gen connectivity, retune stuff
    print(' done ({:.1f}s)'.format(time.time() - before), flush=True)

    # TODO TODO TODO do synthetic mixture studies at ORNs using any data i can
    # find on the fruits they use (could maybe use our own peach / mango /
    # banana data)

    # TODO what does "*** Not yet returned!" mean when i try rv.kc.responses???
    # (seems to only come up when i try to access the value in ipdb. Matt have
    # any idea why this would be?)

    # TODO TODO TODO check whether any of my odors are among the set of odors
    # with measurements at multiple concentrations (could maybe use that data to
    # inform modifications to ORN representation)

    # After this point the responses to the Hallem set won't change
    assert orn_deltas.shape[1] > 110, 'no new odors to sim'

    sfr = sfr.iloc[:, 0]
    # TODO could also try using one max firing rate across all
    # (maybe they do all have a similar max firing rate, and some odors just
    # didn't have particularly activating odors found for them?)
    orn_maxes = orn_deltas.max(axis=1) + sfr

    def constrain_hallem_deltas(odor_deltas):
        odor_orn_rates = sfr + odor_deltas
        odor_orn_rates[odor_orn_rates < 0] = 0
        over_max = odor_orn_rates > orn_maxes
        odor_orn_rates[over_max] = orn_maxes[over_max]
        odor_deltas = odor_orn_rates - sfr
        return odor_deltas

    def scale_hallem_odor_deltas(odor_deltas, scale):
        # TODO does matt's code behave appropriately if deltas would send 
        # spont negative (should either err or be treated same as if sum
        # were a 0 firing rate)?
        # TODO maybe subtract lowest value in delta before multiplying
        # by scale?
        return constrain_hallem_deltas(odor_deltas * scale)

    def one_odor_model_response_fraction(scale, oi):
        deltas = orn_deltas.copy()

        # for a pandas dataframe of the same dimensions
        curr = deltas.iloc[:, oi]
        deltas.iloc[:, oi] = scale_hallem_odor_deltas(curr, scale)
        mp.orn.data.delta = deltas

        verbose = False
        if verbose:
            print(f'running sims for scale={scale[0]:.3f}', flush=True)

        mp.sim_only = [oi]
        osm.run_ORN_LN_sims(mp, rv)
        osm.run_PN_sims(mp, rv)
        osm.run_KC_sims(mp, rv, False)
        r = np.mean(rv.kc.responses[:, oi], axis=0)

        if verbose:
            print(f'response rate: {r:.3f}', flush=True)

        return r

    def err_fn(rt, r):
        #return (rt - r)**2
        return abs(rt - r)

    scales = []
    for oi, rt in zip(range(110, orn_deltas.shape[1]), target_response_fracs):
        def one_odor_sparsity_err(scale):
            r = one_odor_model_response_fraction(scale, oi)
            return err_fn(rt, r)

        name = orn_deltas.iloc[:, oi].name
        if name.endswith('mix'):
            parts = name.split()
            assert len(parts) == 2
            oset = parts[0]

            # TODO TODO TODO maybe use various kinds of derivative free
            # optimization methods to fit a few different mixture models here?
            # TODO TODO may want to fit mixture model parameters using more
            # information than just the sparsity of the mixture response though
            # (so maybe factor out that part of fitting, and just return
            # component scales / scaled data here?)

            # TODO when is the appropriate time to constrain this?
            # normalize before doing so? some other scaling fn?
            # TODO also try w/ max here rather than sum -> some scaling
            # (that would avoid need for normalizing...)
            component_cols = orn_deltas.columns.map(
                lambda c: c.startswith(oset) and not c.endswith('mix')
            )
            #model_orn_mix = orn_deltas.loc[:, component_cols].sum(axis=1)
            model_orn_mix = orn_deltas.loc[:, component_cols].mean(axis=1)

            orn_deltas.iloc[:, oi] = model_orn_mix

            # TODO asserts to ensure this is last thing seen from this odorset

        print(name)
        #print(f'oi={oi}')
        #print(orn_deltas.iloc[:, oi])
        print(f'rt={rt:.3f}')

        # This was necessary to find a suitable scale for this particular odor.
        if oi == 111:
            rmin = 2
            rmax = 4
        else:
            rmin = 0.1
            rmax = 2.5
        # TODO err / warn / modify bounds if opt scale was equal to one of the
        # bounds

        # TODO any similar fn that can be configured to return as soon as
        # error is sufficiently low?
        # TODO something that makes grid increasingly more fine around minima
        # from previous searches?
        ret, fval, _, _ = brute(one_odor_sparsity_err, ((rmin, rmax),),
            finish=None, full_output=True, Ns=50
        )
        print(f'(brute) best scale: {ret:.3f}')
        print(f'(brute) best error: {fval:.3f}')
        # would at least need to increase Ns to do better than those for all
        # odors (if even possible, given realities of model)
        assert fval <= 0.02

        # TODO why doesn't minimize work? it doesn't seem to update parameter
        # at all... (i mean these gradient methods will not work w/o some
        # kind of modification for my problem w/o meaningful gradients.
        # though see: https://arxiv.org/abs/1706.04698 )
        # tried x0 an array and w/ values of x0=0.5,1,2
        # of all the methods available, only Nelder-Mead, Powell, and COBYLA
        # work, with the first seeming to perhaps converge the fastest
        # (but all of these fail in some of the cases. kiwi 3-methylbutanol, for
        # whatever reason)

        # TODO could try using line_search here
        # TODO (linear?) constraint to keep scale positive?
        # TODO [maybe implement max activation constraint this way too?]
        '''
        x0 = 1.0
        mret = minimize(one_odor_sparsity_err, x0, method='COBYLA')
        assert mret.success
        # since at least 'Powell' method will return something with an empty
        # shape
        # (cannot be indexed w/ x[0]))
        mret = mret.x.reshape((1,))[0]
        print(f'(nm) best scale: {mret:.3f}')
        rel_err = abs(mret - ret) / max(abs(mret), abs(ret))
        print(f'relative error between two scales: {rel_err:.4f}')
        #assert np.isclose(mret, ret, rtol=0.05)
        '''
        print('')

        # TODO TODO look at which / how many of ORN responses are
        # at min / max?
        orn_deltas.iloc[:, oi] = \
            scale_hallem_odor_deltas(orn_deltas.iloc[:, oi], ret)

        scales.append(ret)

    # TODO TODO run sims w/ all of above (w/ diff weight draws?)
    # TODO make sense to run w/ fit scales w/ diff weight draws? or always want
    # to fit scales for a given set of weight draws?
    # (if former, may just want to return fit scales / scaled data, and have
    # another fn to generate multiple versions of simulated data w/ those fits)

    # TODO TODO TODO convert output to something suitable for use as my real
    # data
    responses = rv.kc.responses

    '''
    spike_recordings = rv.kc.spike_recordings
    spike_counts = rv.kc.spike_counts
    # couldn't get shape of this cause it is apparently a list
    print(len(spike_recordings))
    print(spike_counts.shape)

    assert (spike_recordings[0].shape[1] ==
        int(np.round((mp.time_end - mp.time_pre_start) / mp.time_dt)))

    stim_start_idx = int(np.round(
        (mp.time_stim_start - mp.time_pre_start) / mp.time_dt 
    ))
    '''
    # TODO are there ever spikes before mp.time_start?
    # if not, just throw out that data before converting to df, right?
    # TODO ever multiple spikes fired?

    # TODO TODO if i'm just going to binarize anyway, should i just use 
    # rv.kc.respones rather than spike_recordings?

    # TODO TODO TODO maybe first check what total output of
    # linear-sum-at-periphery is in model kcs?
    # TODO TODO TODO maybe then try scaling sum to equal real mix scale in KCs?
    # TODO TODO TODO try each of linear models at periphery? (but how to fit if
    # minimize not working...)

    # TODO TODO TODO somehow test goodness of linearity scaling fit for across
    # concentration data? or come up with a better method of scaling knowing
    # that? possible? may very well not be...

    odor_sets = []
    name1s = []
    for c in orn_deltas.iloc[:, 110:].columns:
        parts = c.split()
        odor_sets.append(parts[0])
        name1s.append(' '.join(parts[1:]))

    col_index = pd.MultiIndex.from_frame(pd.DataFrame({
        'odor_set': odor_sets,
        'name1': name1s
    }))
    model_df = pd.DataFrame(responses[:, 110:], columns=col_index)
    model_df.index.name = 'cell'

    # From how the docs describe unstack, it's not obvious it would do what I
    # want, but it seems to...
    model_ser = model_df.unstack()
    model_ser.name = 'responded'
    model_df = model_ser.reset_index()
    # TODO maybe also return scales / scaled data for use in other fitting steps
    # for mixture responses?
    return model_df


