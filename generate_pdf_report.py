#!/usr/bin/env python3

from os.path import join, split, abspath, normpath, dirname
import glob
from datetime import date
from pprint import pprint

from jinja2.loaders import FileSystemLoader
from latex import build_pdf, escape, LatexBuildError
from latex.jinja2 import make_env

import hong2p.util as u


def main():
    verbose = False
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

    template_str = template.render(pdfdir=pdfdir, sections=sections,
        filename_captions=False)
    template_msg = 'Rendered TeX:\n{}\n'.format(template_str)
    if verbose:
        print(template_msg)

    current_dir = abspath(dirname(__file__))
    try:
        pdf_fname = (date.today().strftime(u.date_fmt_str) +
            '_kc_mix_analysis.pdf')

        pdf = build_pdf(template_str, texinputs=[current_dir, ''])
        print('Writing output to {}'.format(pdf_fname))
        pdf.save_to(pdf_fname)

    except LatexBuildError as err:
        if not verbose:
            print(template_msg)
        raise


if __name__ == '__main__':
    main()

