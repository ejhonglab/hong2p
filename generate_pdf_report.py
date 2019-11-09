#!/usr/bin/env python3

from os.path import join, split, abspath, normpath, dirname, getmtime
import glob
from datetime import date, datetime
from pprint import pprint

from jinja2.loaders import FileSystemLoader
from latex import build_pdf, escape, LatexBuildError
from latex.jinja2 import make_env

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
def glob_plots(glob_str, filenames_to_use):
    matching_files = [split(p)[1] for p in glob.glob(join(pdfdir, glob_str))]
    if filenames_to_use is None:
        return matching_files
    else:
        return [f for f in matching_files if f in filenames_to_use]


def plot_files_in_order(glob_str, filenames_to_use):
    """Returns filenames in order for plotting.

    Reverse chronological w/ control following kiwi.
    """
    files = glob_plots(glob_str, filenames_to_use)
    if len(files) == 0:
        return []

    reverse_chronological = True
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
            if p == 'kiwi':
                if panel_part is not None:
                    raise ValueError(
                        f'duplicate panel parts in filename {fname}')

                if control_last:
                    panel_part = 0
                else:
                    panel_part = 1
                continue

            elif p == 'control':
                if panel_part is not None:
                    raise ValueError(
                        f'duplicate panel parts in filename {fname}')

                if control_last:
                    panel_part = 1
                else:
                    panel_part = 0
                continue

            try:
                fly_part_already_found = fly_part is not None
                curr_fly_part = int(p)

                # TODO probably also want to fail in >1 consectutive zero case.
                # may not be important to support fly_num == 0 case either.

                # Leading zeros might mean this part was actually one of the 
                # "_NNN" format ThorImage IDs. Either way, fly_num should not
                # be formatted into filename with any leading zeros.
                if curr_fly_part != 0 and p[0] == '0':
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

    p0 = list(date_fly2seen_panels.values())[0]
    for df, seen in date_fly2seen_panels.items():
        if len(seen) != 2 or seen != p0:
            raise ValueError(f'fly {df} did not have all expected panels')


    files = sorted(files, key=lambda f: fname2keys[f])
    if reverse_chronological:
        files = files[::-1]
    return files


def main(*args, **kwargs):
    # TODO pop these out of kwargs if i'm gonna call this from outside,
    # while still keeping this separate module. pass rest of kwargs to rendering
    # fn?
    verbose = False
    only_print_latex = False
    write_latex_for_testing = False
    date_in_pdf_name = True

    if 'filenames' in kwargs:
        filenames_to_use = kwargs.pop('filenames')
        # Any matching files will have to also be present in here.
        filenames_to_use = {split(p)[1] for p in filenames_to_use}
        # TODO maybe print which files are excluded as they are? (if verbose?)
    else:
        # This means to use all matching files.
        filenames_to_use = None

    env = make_env(loader=FileSystemLoader('.'))
    template = env.get_template('template.tex')

    # TODO TODO TODO maybe just lump everything in plot dir w/ unrecognized
    # prefix into this section, just w/o secion name?
    # would prob make it easier to add new plot types, w/o some other
    # refactoring
    # TODO maybe order these previously-unclaimed sections by (latest?) mtime
    # of files matched by the glob?
    # TODO how to either show "Figure N" (without colon) or no figure label
    # (while still incrementing figure num)? (for this stuff)

    # TODO TODO TODO maybe allow the second element of each tuple being
    # an iterable of glob strs, which defines order w/in a section
    # (so i can put shuffle control after cell tuning breadth, for example)
    # (or ctrl after kiwi stuff, if can't get single facetgrids to do what
    # i want)
    all_pdfdir_files = set(glob_plots('*', filenames_to_use))

    # These will all be stacked top to bottom (?)
    section_names_and_globstrs = [
        ('Cell tuning breadth', 'n_ro_hist*'),
        ('Mean fraction responding', 'mean_frac_responding*'),
        ('Kiwi panel odor correlations', 'kiwi_corr*'),
        ('Control panel odor correlations', 'ctrl_corr*'),
        ('Cell linearity distributions', 'cell_linearity_dists*')
    ]
    sections = [(n, glob_plots(gs, filenames_to_use)) for n, gs
        in section_names_and_globstrs
    ]
    # These will be lined up in two columns, with one odor_set in left column
    # and the other in the right column.
    paired_section_names_and_globstrs = [
        ('Threshold sensitivity', 'threshold_sensitivity*'),
        ('Response rates', 'resp_frac_*'),
        ('Response reliability', 'reliable_of_resp_*'),
        ('Normalized mix responder tuning', 'ratio_mix_rel_to_others_*'),
        ('Mix responder tuning', 'mix_rel_to_others_*'),
        ('Trial-max response correlations', 'oorder_corr_max*'),
    ]
    # TODO option to use passed in filenames rather than stuff from glob
    # (to only include plots generated in one analysis run, for example)
    # TODO maybe find intersection of globstrs w/ those filenames, and fail
    # if any filenames are passed in w/ unrecognized glob strs
    # might make more sense to just include all passed in actually...
    # but should still group by non-id part of filename (prefix always?)
    # (maybe order by suffix if there are other suffix keys besides
    # date / fly_num / panel??)
    # (put all those in their own section, like at top, for across fly
    # stuff?)
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

    # TODO maybe exclude stuff older than a day by default? or at least
    # warn (i mean we might not want to regenerate, but should be aware it's
    # old maybe)?
    # Could also do something like this to exclude stuff that is sufficiently
    # older than other plots, such that it was likely to have been generated in
    # a different run (and thus likely diff. code / data / parameters).
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

        # TODO TODO maybe prompt to have this script edit itself to add the
        # discovered plot type (empty title + the globstr)?
        # or specify plot names + globstrs in some external config file anyway,
        # and then edit that?

    # TODO if i'm gonna do this, maybe warn about which desired sections
    # were missing plots (or do that regardless...)?
    sections = [(name, plots) for name, plots in sections if len(plots) > 0]
    paired_sections = [(name, plots) for name, plots in paired_sections
        if len(plots) > 0
    ]
    if verbose:
        print('Section names and input files:')
        pprint(sections)
        print('Paired section names and input files:')
        pprint(paired_sections)
        print('')

    # TODO even if not using sections in latex, maybe include quick list of
    # types of figures to expect at top. maybe even bulleted.

    latex_str = template.render(pdfdir=pdfdir, sections=sections,
        paired_sections=paired_sections, filename_captions=False, **kwargs
    )
    latex_str = clean_generated_latex(latex_str)

    if write_latex_for_testing:
        tex_fname = 'test.tex'
        print('Writing LaTeX to {} (for testing)'.format(tex_fname))
        with open(tex_fname, 'w') as f:
            f.write(latex_str)

    gen_latex_msg = 'Rendered TeX:\n{}\n'.format(latex_str)
    if only_print_latex or verbose:
        print(gen_latex_msg)
    if only_print_latex:
        import sys; sys.exit()

    current_dir = abspath(dirname(__file__))

    # TODO maybe make this share less of a prefix w/ kc_mix_analysis.py
    # could be annoying
    # the fact that the date prefix avoided that was kinda nice i guess...
    pdf_fname = 'kc_mix_analysis.pdf'
    if date_in_pdf_name:
        pdf_fname = date.today().strftime(u.date_fmt_str) + f'_{pdf_fname}'

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
        if not verbose:
            print(gen_latex_msg)
        raise

    return pdf_fname


if __name__ == '__main__':
    main()

