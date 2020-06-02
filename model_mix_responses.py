#!/usr/bin/env python3

"""
"""

import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import brute

import chemutils as cu
import olfsysm as osm
from drosolf.orns import orns

import hong2p.util as u


# TODO TODO refactor so there is another fn to retrieve the model outputs on
# just the unmodified hallem inputs?
# TODO maybe allow configuring max vs sum/mean for mixing rule
# (and diff normalization options? not sure how to specify those though...)
def fit_model(frac_responder_df, require_all_components=True, fit_mix=True):
    # With add_sfr=False, it should just return the deltas...
    orn_deltas = orns(add_sfr=False, drop_sfr=False).T
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

    # TODO what is this shared_idx thing again?
    shared_idx = np.setdiff1d(np.arange(len(orn_deltas)), [skip_idx])
    sfr = sfr[shared_idx]
    orn_deltas = orn_deltas.iloc[shared_idx]
    assert np.array_equal(sfr, tmp_mp.orn.data.spont[:, 0])
    assert np.array_equal(orn_deltas, tmp_mp.orn.data.delta)
    kc_cxn_distrib = tmp_mp.kc.cxn_distrib.copy()
    del tmp_mp

    assert orn_deltas.columns.name == 'odor'
    assert orn_deltas.index.name == 'receptor'
    n_orn_types = len(orn_deltas.index)

    # TODO TODO i guess acetoin isn't in hallem?? maybe i'm misunderstanding
    # some other naming convention? try stripping inchi?
    hc_inchi = cu.convert(orn_deltas.columns, from_type='name')
    assert len(hc_inchi) == len(set(hc_inchi))

    mean_frac_responding = frac_responder_df.groupby(['odor_set','name1']
        ).frac_responding.mean()

    # Will set values of this to False, as appropriate, within the loop.
    # TODO delete this if not used
    oset2all_components_in_hallem = {o: True for o in u.odor_set2order.keys()}

    new_odor_series = []
    target_response_fracs = []
    odor_metadata_list = []
    oi = 110 - 1
    # TODO maybe just loop over abbreviated things?
    for oset, odors in u.odor_set2order.items():

        # TODO TODO make a kwarg that toggles whether mixtures w/ any missing
        # components should be simulated (and maybe another flag for whether
        # mixture responses should be created for them?)

        # So we know we can later use these prefixes to pull out just the data
        # for each odor set.
        assert not orn_deltas.columns.map(lambda c: c.startswith(oset)).any()

        for o in odors:
            abbrev = cu.odor2abbrev(o)

            # Natural stuff will be excluded by breaking out of order early.
            if o in u.solvents:
                continue
            oi += 1

            target_response_frac = mean_frac_responding.loc[oset, abbrev]
            target_response_fracs.append(target_response_frac)

            # TODO TODO TODO probably switch new_ser stuff to a multiindex, w/
            # (oset, abbrev) as the two levels (would probably need to fill nan
            # for the hallem oset)

            # Anything after mix is either a mix or real as I have it now.
            #if cu.odor_is_mix(o):
            if abbrev == 'mix':
                inchi = None
                # These will be filled in (potentially followed by another
                # optimization step) AFTER the single components are optimized
                # to produce model KC activations are similar levels to the
                # data.
                new_ser = pd.Series(
                    index=sfr.index,
                    data=np.full(sfr.shape, np.nan)
                )
                # TODO TODO TODO refactor so we don't depend on all "real" stuff
                # being after the first "mix" in an odor set!!!!
            else:
                inchi = cu.convert(o, from_type='name')
                if inchi in hc_inchi:
                    new_ser = orn_deltas.T[hc_inchi == inchi].iloc[0].copy()
                    assert new_ser.shape == (n_orn_types,)
                else:
                    oset2all_components_in_hallem[oset] = False
                    warnings.warn(f'{inchi} (name={o}) not in hc_inchi. setting'
                        ' ORN deltas to NaN for this odor!'
                    )
                    nan_deltas = np.zeros(n_orn_types) * np.nan
                    new_ser = pd.Series(index=orn_deltas.index, data=nan_deltas)
                    del nan_deltas

                    # TODO TODO TODO does matt's model (as i would hope) just
                    # straight ignore all nan, or does it propagate it?
                    # TODO TODO if the model doesn't ignore nan, maybe handle by
                    # setting the model to not similuate fully nan indices?

            new_ser.name = f'{oset} {abbrev}'
            new_odor_series.append(new_ser)

            odor_metadata_list.append({
                'odor_set': oset,
                'abbrev': abbrev,
                'oi': oi,
                'name': o,
                'inchi': inchi,
                'target_response_frac': target_response_frac
            })

            # TODO see note in end of abbrev == 'mix' block about refactoring
            # this. this is just to skip the "real" stuff after mixes.
            if abbrev == 'mix':
                break

    # TODO TODO is there any sensible way to fit mixture responses (as a
    # weighted sum of component ORN responses) IF we are missing some of the
    # component responses (from hallem, like in the oset=='flyfood' case)??

    # TODO delete this + code that generates it, if i don't end up using
    odor_metadata = pd.DataFrame(odor_metadata_list)
    odor_metadata['fit_hallem_orns_scale']= \
        np.zeros(len(odor_metadata)) * np.nan

    oi_ser = odor_metadata.oi
    # Showing there are no gaps.
    assert oi_ser.nunique() == (oi_ser.max() - oi_ser.min() + 1)
    del oi_ser

    odor_metadata.set_index('oi', inplace=True)

    del target_response_frac, mean_frac_responding, oi
    target_response_fracs = np.array(target_response_fracs)

    len_before = len(orn_deltas)
    assert orn_deltas.shape[1] == 110
    orn_deltas = pd.concat([orn_deltas] + new_odor_series, axis=1)
    assert len(orn_deltas) == len_before
    del len_before

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

    # TODO TODO TODO get an understanding of how matt's tuning process works

    # TODO or should i just disable tuning?
    # TODO TODO for using neuprint data, probably!

    # Note Matt's 2020-02-02 commit (Which I think he didn't push until
    # ~2020-05-30, but honestly I'm not sure. Github says it was updated more
    # recently than last commit.)
    # Tune thresholds/the APL to the Hallem odors only
    mp.kc.tune_from = range(110)
    # non-zero for consistent PN-KC connectivity generation
    mp.kc.seed = 12345

    # TODO TODO should i set sim_only to exclude new stuff here?
    # (particularly if i'm gonna have null stuff for mixture response)
    # (any reason not to?)
    mp.sim_only = range(110)

    import ipdb; ipdb.set_trace()

    before = time.time()
    print('running initial sims.', end='', flush=True)
    # Run a 1st simulation to tune KC thresholds and APL weights
    osm.run_ORN_LN_sims(mp, rv)
    print('.', end='', flush=True)
    osm.run_PN_sims(mp, rv)
    print('.', end='', flush=True)
    osm.run_KC_sims(mp, rv, True) # True -> (re)gen connectivity, retune stuff
    print(' done ({:.1f}s)'.format(time.time() - before), flush=True)

    before_fitting = time.time()

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

    import ipdb; ipdb.set_trace()
    # TODO maybe also return maxes, to inspect the effect of that constraint out
    # of this fn?

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

    # TODO be consistent about using either the term "response_fraction" or
    # "sparsity"?
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

    # TODO TODO maybe move body of this loop into loop above (maybe moving
    # a bit of the between code above the loop above?) any real reason to 
    # make a list of those target_response_fracs first??

    # TODO improve this message?
    print('rescaling Hallem ORN deltas to match observed KC sparsity (with '
        'concentrations potentially deviating from those in Hallem)...'
    )
    default_rmin = 0.1
    default_rmax = 2.5
    # TODO TODO probably change the type of this to a pandas series?

    # Since we want to set the mix from the *scaled* components, we need to make
    # sure that after we see a mix (for a given odor set), we never see anything
    # *else* from that odor set. This will help with that.
    odor_sets_with_seen_mixes = set()

    # TODO TODO i'm not indexing target_response_fracs incorrectly or something
    # one time, am i? that part of reason for scale mismatch?
    # TODO TODO TODO related to above, print sparsity achieved during fit in
    # here, to see it's actually somewhat observable at the output of this fn,
    # in case i don't pass something correctly
    for oi, rt in zip(range(110, orn_deltas.shape[1]), target_response_fracs):
        def one_odor_sparsity_err(scale):
            r = one_odor_model_response_fraction(scale, oi)
            return err_fn(rt, r)

        # TODO could replace most of this stuff w/ odor_metadata
        name = orn_deltas.iloc[:, oi].name
        parts = name.split()
        assert len(parts) == 2
        oset = parts[0]
        assert oset not in odor_sets_with_seen_mixes, 'out of order'

        if name.endswith('mix'):
            odor_sets_with_seen_mixes.add(oset)

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
            # At least in the deterministic case, the overall scale shouldn't
            # affect the correlation between this synthetic mix and the
            # components...

            # The data we are pulling from `orn_deltas` here ***was*** scaled
            # (at the bottom of this loop, when `orn_deltas` is assigned into)
            all_component_data = orn_deltas.loc[:, component_cols]

            if require_all_components:
                if all_component_data.isnull().any().any():
                    continue
                model_orn_mix = all_component_data.mean(axis=1)
            else:
                # TODO + check that denominator doesn't include nan
                # TODO TODO find some replacement fn. this doesn't work.
                #model_orn_mix = all_component_data.nanmean(axis=1)
                import ipdb; ipdb.set_trace()

            # TODO maybe pick a scale (relative to sum / max of component
            # scales) that best matches the scale ratio / difference of the real

            # TODO TODO maybe just as a sanity check, copy orn_deltas before
            # any modifications (up top) and then define model_orn_mix
            # in terms of unweighted component sum, and check that it looks
            # different what is calculated here (where scales actually do factor
            # in, via the modifications to orn_deltas at end of this loop)

            orn_deltas.iloc[:, oi] = model_orn_mix
            if not fit_mix:
                continue
        else:
            one_component_data = orn_deltas.iloc[:, oi]
            if one_component_data.isnull().any():
                continue
        print(name)

        # TODO print range brute search is over, and what resolution.
        # (for each odor if i continue special casing that 111 odor)

        # TODO TODO TODO see comment below part of kc_mix_analysis that calls
        # this code about apparent mismatch between the rank ordering of
        # response strengths of various odors in the model outputs vs input
        # data. is this special casing of 111, whatever that is, part of the
        # reason????
        # TODO str about what 111 is. maybe make some lookup fn to translate
        # from odor names to hallem indices (or whatever this actually is, maybe
        # there is some offset?), and use that?
        # This was necessary to find a suitable scale for this particular odor.
        if oi == 111:
            rmin = 2
            rmax = 4
        else:
            rmin = default_rmin
            rmax = default_rmax
        # TODO err / warn / modify bounds if opt scale was equal to one of the
        # bounds

        # TODO TODO save rmin, rmax, and resolution to odor_metadata for each
        # odor

        # TODO any similar fn that can be configured to return as soon as
        # error is sufficiently low?
        # TODO something that makes grid increasingly more fine around minima
        # from previous searches?
        # TODO TODO probably rename ret and fval so it's more obvious to me what
        # they are... seems like ret is supposed to be the scale... is fval the
        # achieved sparsity?
        # oh, i guess fval is the error? then can i pass extra return stuff, or
        # should i just recompute the sparsity out here, if i want to check it?
        ret, fval, _, _ = brute(one_odor_sparsity_err, ((rmin, rmax),),
            finish=None, full_output=True, Ns=50
        )
        print(f'(brute) best scale: {ret:.3f}')
        print(f'(brute) best error: {fval:.3f}')
        print(f'target sparsity: {rt:.3f}')

        # TODO don't recompute this if i can just return it as something extra
        # from brute maybe?
        r = one_odor_model_response_fraction(ret, oi)
        print(f'achieved sparsity: {r:.3f}')
        #

        # TODO TODO TODO maybe visualize the effects of my scaling +
        # constraining and those effects on model outputs, to sanity check?

        # would at least need to increase Ns to do better than those for all
        # odors (if even possible, given realities of model)
        assert fval <= 0.02, f'fval: {fval}'
        # TODO TODO maybe assert that ret (scale?) is in the range the fitting
        # was limited to, as another check it's the right return value from
        # brute?

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

        # TODO TODO TODO check that modifying the orn_deltas in place like this
        # is actually affecting the ultimate "responses" output, and in a way
        # that makes sense
        # (maybe also provide option to return unscaled version, and use that
        # as one flawed test that the scaling is behaving in a reasonable way?)
        # (test scaling to essentially the floor too. might want to add noise to
        # also make correlations vanish at floor / behave like real data
        # correlations w/o responses, if necessary)

        # TODO TODO maybe double check that the RHS orn_deltas has actually
        # been modified to the new scale + constraints
        # TODO TODO look at which / how many of ORN responses are
        # at min / max?
        orn_deltas.iloc[:, oi] = \
            scale_hallem_odor_deltas(orn_deltas.iloc[:, oi], ret)

        # TODO maybe still have this be defined in the `not fit_mix` case
        odor_metadata.loc[oi, 'fit_hallem_orns_scale'] = ret

        # TODO maybe also add a field the odor_metadata for the fit error for
        # each odor / mix?

    print(f' done ({time.time() - before_fitting:.1f}s)', flush=True)
    print(f'Total time: {time.time() - before:.1f}s', flush=True)

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


    # TODO TODO TODO options to add noise to simulations!
    # (e.g. normal noise at orns propagated up through to other stuff)

    # TODO TODO TODO can i pull out model pn responses, for stuff like
    # correlation matrices? how? (put in returned data!)

    odor_sets = []
    name1s = []
    # TODO don't hardcode the 110 everywhere. use constant at least.
    for c in orn_deltas.iloc[:, 110:].columns:
        parts = c.split()
        odor_sets.append(parts[0])
        name1s.append(' '.join(parts[1:]))

    col_index = pd.MultiIndex.from_frame(pd.DataFrame({
        'odor_set': odor_sets,
        'name1': name1s
    }))
    # TODO don't hardcode the 110 everywhere. use constant at least.
    model_df = pd.DataFrame(responses[:, 110:], columns=col_index)
    model_df.index.name = 'cell'

    # TODO double check this part
    # From how the docs describe unstack, it's not obvious it would do what I
    # want, but it seems to...
    model_ser = model_df.unstack()
    model_ser.name = 'responded'
    model_df = model_ser.reset_index()

    # TODO maybe cast model_ser 0.0 / 1.0 float to bool?

    orn_deltas = orn_deltas.iloc[:, 110:].T
    assert len(sfr) == orn_deltas.shape[1]
    orn_abs_rates = orn_deltas + sfr

    assert len(orn_deltas) == len(orn_abs_rates)
    assert len(odor_metadata) == len(orn_abs_rates)

    # TODO maybe rename abbrev to 'name1' for consistency w/ stuff
    # in kc_mix_analysis.py ...
    odor_metadata.set_index(['odor_set', 'abbrev'], inplace=True)

    # TODO TODO TODO either have orn_deltas be multiindexed the whole time,
    # or zip its orig index here w/ the new index we are gonna give it,
    # and check if you join the str parts of each rows multiindex together,
    # you get the index of that orn_deltas row!

    orn_deltas.index = odor_metadata.index
    orn_abs_rates.index = odor_metadata.index

    # TODO TODO TODO null the parts of returned model data that are not null
    # when they are supposed to be!!!! (like acetoin or (when
    # require_all_components == True) the flyfood mix)

    # TODO TODO similar plots (of orn firing rates) BUT divided by the max for
    # each orn type? and maybe a more conservative upper bound on max firing
    # rate? or a constant / normal distribution of max firing rates? or what
    # distribution would make sense? have people measured this type of stuff
    # anywhere? maybe in vision?

    orn_deltas.index.names = ['odor_set', 'name1']
    orn_abs_rates.index.names = ['odor_set', 'name1']

    model_df.set_index(['odor_set', 'name1'], inplace=True)
    for row in fit_scales.itertuples():
        if pd.isnull(row.fit_hallem_orns_scale):
            model_df.loc[row.Index, 'responded'] = np.nan
    model_df.reset_index(inplace=True)
    # TODO maybe convert 'responded' col to boolean before returning?

    # TODO TODO TODO also return PN data.
    # TODO TODO TODO probaly just return all data i can get out of matt's model
    # (just check it's working to serialize it and stuff...)
    # (maybe copy or something here to be safe)
    import ipdb; ipdb.set_trace()

    return {
        'model_df': model_df,
        'fit_scales': odor_metadata,
        'orn_deltas': orn_deltas,
        'orn_abs_rates': orn_abs_rates,
        'orn_maxes': orn_maxes
    }


