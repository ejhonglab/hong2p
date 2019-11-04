#!/usr/bin/env python3

from os.path import join, split, abspath, normpath, dirname
import glob
from datetime import date
from pprint import pprint

from jinja2.loaders import FileSystemLoader
from latex import build_pdf, escape, LatexBuildError
from latex.jinja2 import make_env

import hong2p.util as u


def clean_generated_latex(latex_str):
    lines = latex_str.splitlines()
    cleaned_lines = []
    for first, second in zip(lines, lines[1:]):
        if not (first == '' and second == ''):
            cleaned_lines.append(first)
    cleaned_lines.append(lines[-1])
    return '\n'.join(cleaned_lines)


def main():
    verbose = False
    only_print_latex = False
    write_latex_for_testing = True

    env = make_env(loader=FileSystemLoader('.'))
    template = env.get_template('template.tex')
    section_names_and_globstrs = [
        ('Threshold sensitivity', 'threshold_sensitivity*'),
        ('Response rates', 'resp_frac*')
        #('Correlations', ('c1.svg', 'c2.svg'))
    ]
    pdfdir = abspath(normpath(join('mix_figs', 'pdf')))
    sections = [
        (n, [split(p)[1] for p in glob.glob(join(pdfdir, gs))])
        for n, gs in section_names_and_globstrs
    ]
    # TODO if i'm gonna do this, maybe warn about which desired sections
    # were missing plots (or do that regardless...)
    sections = [(name, plots) for name, plots in sections if len(plots) > 0]
    if verbose:
        print('Section names and input files:')
        pprint(sections)
        print('')

    latex_str = template.render(pdfdir=pdfdir, sections=sections,
        filename_captions=False)
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
    try:
        pdf_fname = (date.today().strftime(u.date_fmt_str) +
            '_kc_mix_analysis.pdf')

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


if __name__ == '__main__':
    main()

