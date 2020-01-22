#!/usr/bin/env python3

from os.path import join, split, abspath, normpath, dirname, getmtime
import glob
from datetime import date, datetime
from collections import OrderedDict
from pprint import pprint
import sys
import argparse
import pickle

from jinja2.loaders import FileSystemLoader
from latex import build_pdf, escape, LatexBuildError
from latex.jinja2 import make_env
import matplotlib.pyplot as plt

import hong2p.util as u


def clean_generated_latex(latex_str):
    """Returns str w/ any consecutive empty lines removed.
    """
    lines = latex_str.splitlines()
    cleaned_lines = []
    for first, second in zip(lines, lines[1:]):
        if not (first == '' and second == ''):
            cleaned_lines.append(first)
    cleaned_lines.append(lines[-1])
    return '\n'.join(cleaned_lines)


pdfdir = abspath(normpath(join('mix_figs', 'pdf')))
matched_file2longest_glob_str = dict()
def glob_plots(glob_str, filenames_to_use):
    """Returns list of filenames of PDF plots to use (no directories).

    If second argument is a list, only paths also in this list are returned.
    """
    matching_files = [split(p)[1] for p in glob.glob(join(pdfdir, glob_str))]
    if filenames_to_use is not None:
        matching_files = [f for f in matching_files if f in filenames_to_use]

    for f in matching_files:
        if f in matched_file2longest_glob_str:
            last_glob_str = matched_file2longest_glob_str[f]

            # Can't disambiguate in this case.
            assert len(glob_str) != len(last_glob_str)

            if len(glob_str) > len(last_glob_str):
                matched_file2longest_glob_str[f] = glob_str
        else:
            matched_file2longest_glob_str[f] = glob_str

    return matching_files


def plot_files_in_order(glob_str, filenames_to_use):
    """Returns filenames in order for plotting.

    Reverse chronological w/ panels in [kiwi, control, fly food] order.
    None in list where fly was missing a panel.
    """
    files = glob_plots(glob_str, filenames_to_use)
    if len(files) == 0:
        return []

    reverse_chronological = True
    # TODO refactor s.t. this indicates order of all panels, not just where
    # control should be (since supporting >=2 panels now)
    control_last = True
    # Since we change sort direction in reverse_chronological case:
    if reverse_chronological:
        control_last = not control_last

    exclude_str = glob_str.replace('*','')
    def file_keys(fname):
        assert fname[-4] == '.', 'following slicing assumes 3 char extension'
        parts = fname[:-4].replace(exclude_str, '').split('_')
        date_part = None
        fly_part = None
        panel_part = None
        for p in parts:
            # TODO refactor to not need a separate hardcoded branch per panel
            if p == 'kiwi':
                if panel_part is not None:
                    pass
                    # This fails w/ discrim_kiwi_kiwi_<date>_<fly>_<thorimageid>
                    # output.
                    # TODO fix
                    '''
                    raise ValueError(
                        f'duplicate panel parts in filename {fname}')
                    '''

                if control_last:
                    panel_part = 1
                else:
                    panel_part = 2
                continue

            elif p == 'control':
                if panel_part is not None:
                    raise ValueError(
                        f'duplicate panel parts in filename {fname}')

                if control_last:
                    panel_part = 2
                else:
                    panel_part = 1
                continue

            elif p == 'flyfood':
                if panel_part is not None:
                    raise ValueError(
                        f'duplicate panel parts in filename {fname}')

                panel_part = 0
                continue

            try:
                fly_part_already_found = fly_part is not None
                curr_fly_part = int(p)

                # Leading zeros might mean this part was actually one of the 
                # "_NNN" format ThorImage IDs. Either way, fly_num should not
                # be formatted into filename with any leading zeros.
                if p[0] == '0':
                    # We can continue here, because were this part a date,
                    # it would have failed int parsing above anyway.
                    continue
                else:
                    if fly_part_already_found:
                        # Can't be a ValueError b/c using that as indication
                        # part couldn't be converted to int...
                        raise RuntimeError(
                            f'duplicate fly_num in filename {fname}')
                    fly_part = curr_fly_part
                    continue
            except ValueError:
                pass

            try:
                date_part_already_found = date_part is not None
                date_part = datetime.strptime(p, u.date_fmt_str)
                if date_part_already_found:
                    # Can't be a ValueError b/c using that as indication
                    # part couldn't be converted to a date...
                    raise RuntimeError( f'duplicate date in filename {fname}')
                continue
            except ValueError:
                pass

        if date_part is None:
            raise ValueError(f'file {fname} had no date part')
        if fly_part is None:
            raise ValueError(f'file {fname} had no fly_num part')
        if panel_part is None:
            raise ValueError(f'file {fname} had no panel part')

        return (date_part, fly_part, panel_part)

    # Checking that we have all pairs for all flies.
    # Important since plot layout assumes everything can map to rows of 2
    # elements nicely.
    fname2keys = {f: file_keys(f) for f in files}
    date_fly2seen_panels = dict()
    for f, (date, fly, panel) in fname2keys.items():
        df = (date, fly)
        if df not in date_fly2seen_panels:
            date_fly2seen_panels[df] = {panel}
        else:
            date_fly2seen_panels[df].add(panel)

    # TODO delete
    '''
    p0 = list(date_fly2seen_panels.values())[0]
    for date_fly, seen in date_fly2seen_panels.items():
        if len(seen) != 2 or seen != p0:
            raise ValueError(f'fly {date_fly} did not have all expected panels')
    '''

    files = sorted(files, key=lambda f: fname2keys[f])
    files_with_fillers = []
    last_date_fly = None
    max_panel = max([x[-1] for x in fname2keys.values()])
    panel = None
    for f in files:
        keys = fname2keys[f]
        date_fly = keys[:2]
        if date_fly != last_date_fly:
            last_date_fly = date_fly
            next_panel = 0
            if panel is not None:
                while panel < max_panel:
                    files_with_fillers.append(None)
                    panel += 1

        panel = keys[-1]
        while next_panel < panel:
            files_with_fillers.append(None)
            next_panel += 1
        next_panel = panel + 1
        files_with_fillers.append(f)

    while panel < max_panel:
        files_with_fillers.append(None)
        panel += 1

    assert (len(files_with_fillers) ==
        (max_panel + 1) * len({x[:2] for x in fname2keys.values()})
    )

    if reverse_chronological:
        files_with_fillers = files_with_fillers[::-1]

    return files_with_fillers


def agg_within_sections(sec_names_and_files):
    """
    Returns new list of sections and files, with no duplicate section names,
    and file lists behind originally-duplicate section names concatenated
    together.
    """
    name2files = OrderedDict()
    for name, files in sec_names_and_files:
        if name not in name2files:
            name2files[name] = files
        else:
            name2files[name] += files

    return list(name2files.items())


def compile_tex_to_pdf(latex_str, pdf_fname, verbose, gen_latex_msg=None):
    current_dir = abspath(dirname(__file__))
    try:
        # Current dir needs to be passed so that 'template.tex', and any 
        # other dependencies in current directory, can be accessed in the
        # temporary build dir created by the latex package.
        # The empty string as the last element retains any default search paths
        # TeX builder would have.
        pdf = build_pdf(latex_str, texinputs=[current_dir, ''])
        print('Writing output to {}'.format(pdf_fname))
        pdf.save_to(pdf_fname)

    except LatexBuildError as err:
        if gen_latex_msg is not None and not verbose:
            print(gen_latex_msg)
        raise


def main(*args, **kwargs):
    debug = False
    verbose = False
    write_latex_for_testing = False
    only_print_latex = False
    write_latex_like_pdf = False
    date_in_pdf_name = True
    last_inputs_pickle_fname = 'last_inputs_genpdfreport.p'
    if debug:
        verbose = True
        write_latex_for_testing = True

    # TODO some more idiomatic / cleaner way to pass cli args to main?

    if 'rerun_last_inputs' in kwargs:
        rerun_last_inputs = kwargs.pop('rerun_last_inputs')
    else:
        rerun_last_inputs = False

    if rerun_last_inputs:
        with open(last_inputs_pickle_fname, 'rb') as f:
            data = pickle.load(f)
        args = data['args']
        kwargs = data['kwargs']

    elif debug:
        with open(last_inputs_pickle_fname, 'wb') as f:
            pickle.dump({'args': args, 'kwargs': kwargs}, f)

    test_tex_fname = 'test.tex'
    if 'only_recompile_test_tex' in kwargs:
        only_recompile_test_tex = kwargs.pop('only_recompile_test_tex')
    else:
        only_recompile_test_tex = False

    if only_recompile_test_tex:
        print(f'Reading test TeX from {test_tex_fname}')
        with open(test_tex_fname, 'r') as f:
            latex_str = f.read()
        pdf_fname = 'test.pdf'
        # TODO maybe remove test.pdf before starting
        compile_tex_to_pdf(latex_str, pdf_fname, verbose)
        sys.exit()

    if 'filenames' in kwargs:
        filenames_to_use = kwargs.pop('filenames')
        if filenames_to_use is not None:
            # Any matching files will have to also be present in here.
            filenames_to_use = {split(p)[1] for p in filenames_to_use}
            # TODO maybe print which files are excluded as they are? (if
            # verbose?)
    else:
        # This means to use all matching files.
        filenames_to_use = None

    env = make_env(loader=FileSystemLoader('.'))
    template = env.get_template('template.tex')

    all_pdfdir_files = set(glob_plots('*', filenames_to_use))

    # TODO test that auto finding + ordering missing stuff is still working when
    # these are not passed in kwargs
    # These will all be stacked top to bottom.
    if 'section_names_and_globstrs' in kwargs:
        section_names_and_globstrs = kwargs.pop('section_names_and_globstrs')
    else:
        section_names_and_globstrs = []

    # These will be lined up in two columns, with one odor_set in left column
    # and the other in the right column.
    if 'paired_section_names_and_globstrs' in kwargs:
        paired_section_names_and_globstrs = \
            kwargs.pop('paired_section_names_and_globstrs')
    else:
        paired_section_names_and_globstrs = []

    sections = [(n, glob_plots(gs, filenames_to_use)) for n, gs
        in section_names_and_globstrs
    ]

    matched_file_set_before = set()
    for _, fs in sections:
        for f in fs:
            matched_file_set_before.add(f)

    # Only the most specific glob str should point to a file.
    sections = [(n, [f for f in fs if matched_file2longest_glob_str[f] == gs])
        for (n, fs), (_, gs) in zip(sections, section_names_and_globstrs)
    ]

    matched_file_set_after = set()
    for _, fs in sections:
        for f in fs:
            matched_file_set_after.add(f)
    # To make sure the above filtering is not removing all references
    # to any plots.
    assert matched_file_set_after == matched_file_set_before
    del matched_file_set_after, matched_file_set_before

    seen_plots = set()
    for _, fs in sections:
        for f in fs:
            assert f not in seen_plots, f'{f} was referred to multiple times!'
            seen_plots.add(f)
    del seen_plots

    # TODO assert no glob strs in this line point to same file / also
    # assign files only to longest matching glob str here
    # (this can cause plots to be included twice, in unpredictable order)
    paired_sections = [(n, plot_files_in_order(gs, filenames_to_use)) for n, gs
        in paired_section_names_and_globstrs
    ]
    unclaimed_pdfdir_files = (all_pdfdir_files -
        (set([f for s in sections for f in s[1]]) | 
         set([f for s in paired_sections for f in s[1]]))
    )
    def globstr(fname):
        assert fname[-4] == '.'
        parts = fname[:-4].split('_')

        # To exclude something I know won't generate sensible globstrs.
        if parts[-1] in ('sorted', 'seg', 'discrim'):
            return None

        prefix_parts = []
        for i, p in enumerate(parts):
            last_i = i
            if p in ('kiwi', 'control'):
                break
            try:
                datetime.strptime(p, u.date_fmt_str)
                break
            except ValueError:
                prefix_parts.append(p)

        if last_i == 0:
            gs = '*'
        else:
            gs = ''

        # TODO test last_i dependent stuff in boundary cases:
        # 1) no '_' to split on
        # 2) lone date/odor_set (at start, in middle, at end)
        # 3) no date/odor_set
        # 4) date/odor_set followed by thorimage_id

        intermediate_asterisk = False
        suffix_parts = []
        for j, p in enumerate(parts[::-1]):
            i = len(parts) - j - 1
            if i <= last_i:
                if i < len(parts) - 1:
                    intermediate_asterisk = True
                break

            # Not just checking for a number because some odor abbreviations
            # have numbers in them.
            # Just crudely matching the common-end of my thorimage_id
            # conventions..
            if len(p) >= 3 and p[-3] == '0':
                try:
                    int(p[-2:])
                    # Did seem to be a thorimage_id suffix.
                    break
                except ValueError:
                    pass

            suffix_parts.append(p)

        suffix_parts = suffix_parts[::-1]

        gs_parts = prefix_parts
        if intermediate_asterisk:
            gs_parts.append('*')
        gs_parts += suffix_parts
        gs += '_'.join(gs_parts) + '*'

        return gs

    unclaimed_files2globstrs = {f: globstr(f) for f in unclaimed_pdfdir_files}
    unclaimed_files2globstrs = {f: gs for f, gs in
        unclaimed_files2globstrs.items() if gs is not None
    }
    unclaimed_file2mtime = {f: getmtime(join(pdfdir, f)) for f in
        unclaimed_pdfdir_files
    }
    # This will make it so files within globstrs are sorted by with recent ones
    # first.
    unclaimed_pdfdir_files = sorted(unclaimed_pdfdir_files,
        key=unclaimed_file2mtime.get
    )

    unclaimed_files2globstrs = dict()
    globstrs2unclaimed_files = dict()
    gs2max_mtime = dict()
    for f in unclaimed_pdfdir_files:
        gs = globstr(f)
        if gs is None:
            continue
        unclaimed_files2globstrs[f] = gs
        mtime = unclaimed_file2mtime[f]
        if gs not in gs2max_mtime:
            gs2max_mtime[gs] = mtime
            globstrs2unclaimed_files[gs] = [f]
        else:
            gs2max_mtime[gs] = max(gs2max_mtime[gs], mtime)
            globstrs2unclaimed_files[gs].append(f)
        
    # TODO worth providing some manual checkpoint for adding these? always
    # prompt as to whether any should be rejected, then list numbers or
    # something?
    # Will still need to manually filter these. Sometimes something we don't
    # want to match on will show up in generated glob strings.
    globstrs_to_exclude = {
        'odorandfit*',
        'oorder_corr_mean*',
        'pca*',
        'pca_unstandardized*',
        'porder_corr_max*',
        'porder_corr_mean*',
        'skree*',
        'skree_unstandardized*'
    }
    unclaimed_file_globstr_set = {g for g in unclaimed_files2globstrs.values()}
    unclaimed_file_globstr_set -= globstrs_to_exclude
    if len(unclaimed_file_globstr_set) > 0:
        globstrs2unclaimed_files = {g: fs for g, fs in
            globstrs2unclaimed_files.items() if g in unclaimed_file_globstr_set
        }
        print('Also including unclaimed plots matching these glob strings:')
        pprint(globstrs2unclaimed_files)
        print('')

        # TODO TODO maybe (at least for some kind of table of contents?)
        # parse the figure titles for a plot type description
        # (how hard is it w/ matplotlib PDF output? i assume svg would have been
        # easier, but then that was a pain to use in LaTeX)

        new_sections = [('', globstrs2unclaimed_files[gs]) for gs in
            sorted(unclaimed_file_globstr_set, key=gs2max_mtime.get)
        ]
        sections += new_sections

    # TODO may want to change this / how sections are rendered so sections
    # are not just figure titles, but can be something that contain multiple
    # differently-titled figures, that both go in the same named section
    # (that can hopefully be referred to in some kind of table of contents)
    sections = agg_within_sections(sections)
    paired_sections = agg_within_sections(paired_sections)

    # TODO if i'm gonna do this, maybe warn about which desired sections
    # were missing plots (or do that regardless...)?
    sections = [(name, plots) for name, plots in sections if len(plots) > 0]
    paired_sections = [(name, plots) for name, plots in paired_sections
        if len(plots) > 0
    ]
    # TODO TODO delete hack. requires figuring out how to get rows with three
    # elements to layout in single rows in the pdf (even down to 0.05\textwidth,
    # it made one row per plot)
    # this isn't even working... even with 2 per row now, stuff only ever seems
    # to be placed by itself on a row...
    new_paired_sections = []
    for name, plots in paired_sections:
        assert len(plots) % 3 == 0
        new_plots = []
        for i in range(len(plots) // 3):
            assert len(plots[3*i:3*i + 3]) == 3
            new_plots.extend(plots[3*i:3*i + 3] + [None])
        assert len(new_plots) % 4 == 0
        new_paired_sections.append((name, new_plots))
    #
    if verbose:
        print('Section names and input files:')
        pprint(sections)
        print('Paired section names and input files:')
        pprint(paired_sections)
        print('')

    # TODO even if not using sections in latex, maybe include quick list of
    # types of figures to expect at top. maybe even bulleted.

    latex_str = template.render(pdfdir=pdfdir, sections=sections,
        paired_sections=paired_sections, filename_captions=debug, **kwargs
    )
    latex_str = clean_generated_latex(latex_str)

    if write_latex_for_testing:
        print('Writing LaTeX to {} (for testing)'.format(test_tex_fname))
        with open(test_tex_fname, 'w') as f:
            f.write(latex_str)

    gen_latex_msg = 'Rendered TeX:\n{}\n'.format(latex_str)
    if only_print_latex or verbose:
        print(gen_latex_msg)
    if only_print_latex:
        sys.exit()

    # TODO maybe make this share less of a prefix w/ kc_mix_analysis.py
    # could be annoying
    # the fact that the date prefix avoided that was kinda nice i guess...
    pdf_fname = 'kc_mix_analysis.pdf'
    if date_in_pdf_name:
        pdf_fname = date.today().strftime(u.date_fmt_str) + f'_{pdf_fname}'

    if write_latex_like_pdf:
        tex_fname = pdf_fname[:-len('.pdf')] + '.tex'
        print('Writing LaTeX to {}'.format(tex_fname))
        with open(tex_fname, 'w') as f:
            f.write(latex_str)

    # TODO only do if None in file lists / delete?
    fig, ax = plt.subplots()
    ax.axis('off')
    # For debugging layout (so this placeholder is visible).
    # this is visible in plt.show(), but not in saved figure...
    fig.savefig('empty_placeholder.pdf')#, facecolor='red')

    compile_tex_to_pdf(latex_str, pdf_fname, verbose, gen_latex_msg)

    return pdf_fname


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--only-recompile', default=False,
        action='store_true'
    )
    parser.add_argument('-l', '--rerun-last-inputs', default=False,
        action='store_true'
    )
    args = parser.parse_args()
    # TODO just pass dict of all args to kwargs?
    kwargs = dict(
        only_recompile_test_tex=args.only_recompile,
        rerun_last_inputs=args.rerun_last_inputs
    )
    main(**kwargs)

