# TODO TODO figure out how to get dependencies for this (template.tex, figureSeries.sty)
# distributed with hong2p via setuptools (and try not to need either in this directory
# with the python modules)
# TODO add conda deps for system latex requirements?

# TODO convert to pathlib
from pathlib import Path
from os.path import abspath, dirname
from typing import Optional

from jinja2.loaders import PackageLoader, FileSystemLoader
from latex import build_pdf, escape, LatexBuildError
from latex.jinja2 import make_env


def get_latex_dir() -> Path:
    # TODO change to have latex dir under module dir if PackageLoader doesn't work w/
    # latex dir where it currently is (at root of repo)
    return Path(__file__).resolve().parent.parent / 'latex'


def clean_generated_latex(latex_str: str) -> str:
    """Returns str w/ any consecutive empty lines removed.
    """
    lines = latex_str.splitlines()
    cleaned_lines = []
    for first, second in zip(lines, lines[1:]):
        if not (first == '' and second == ''):
            cleaned_lines.append(first)
    cleaned_lines.append(lines[-1])
    return '\n'.join(cleaned_lines)


def compile_tex_to_pdf(latex_str, pdf_fname, failed_latex_msg: Optional[str] = None,
    verbose: bool = False) -> None:

    latex_dir = get_latex_dir()
    try:
        # TODO update comment. don't actually need template.tex right? since processed
        # contents of that should have since been put in latex_str?
        #
        # latex_dir needs to be passed so that 'template.tex', and any other
        # dependencies in current directory, can be accessed in the temporary build dir
        # created by the latex package.
        #
        # The empty string as the last element retains any default search paths TeX
        # builder would have.
        pdf = build_pdf(latex_str, texinputs=[str(latex_dir), ''])

        if verbose:
            print(f'Writing output to {pdf_fname}')

        pdf.save_to(pdf_fname)

    except LatexBuildError as err:
        # TODO shouldn't i print err too?
        if failed_latex_msg is not None:
            print(failed_latex_msg)
        raise


# TODO maybe delete fig_dir (and compute here from fig paths in section_names2figs?)
# TODO doc
def make_pdf(pdf_path, fig_dir, section_names2figs, header: Optional[str] = None,
    template_path='template.tex', write_latex_like_pdf: bool = False,
    write_latex_for_testing: bool = False, test_tex_path='test.tex',
    only_print_latex: bool = False, verbose: bool = False
    ) -> None:

    fig_dir = Path(fig_dir).resolve()

    # TODO test this works w/ hong2p installed in a more proper way (like directly from
    # github, but non-editable)
    try:
        env = make_env(loader=PackageLoader('hong2p', 'latex'))

    # ValueError: The 'hong2p' package was not installed in a way that PackageLoader
    # understands
    # (when installed in editable mode, which is typically how I use it)
    except ValueError:
        latex_dir = get_latex_dir()
        # TODO can FileSystemLoader take Path input?
        env = make_env(loader=FileSystemLoader(latex_dir))

    template = env.get_template('template.tex')

    # TODO try to just use a dict if jinja can support that
    section_names_and_figs = [(name, figs) for name, figs in section_names2figs.items()]

    # Converting any inputs with figure paths specified as Path objects to str
    #
    # TODO might need to convert to paths relative to fig_dir? currently only testing w/
    # relative path inputs...
    # could i make all figure paths absolute and not specify fig dir?
    section_names_and_figs = [
        (name, [str(f) for f in figs]) for name, figs in section_names_and_figs
    ]

    section_names_and_figs = [(name, figs) for name, figs in section_names_and_figs
        if len(figs) > 0
    ]

    # TODO if header is None, will jinja + latex do the right thing and not show a
    # header (i.e. does it end up evaluating falsey?) may need to just not pass in
    # kwargs if not?
    latex_str = template.render(fig_dir=fig_dir, sections=section_names_and_figs,
        #pagebreak_after=pagebreak_after, filename_captions=filename_captions,
        header=header
    )
    latex_str = clean_generated_latex(latex_str)

    if write_latex_for_testing:
        if verbose:
            print(f'Writing LaTeX to {test_tex_path} (for testing)')

        with open(test_tex_path, 'w') as f:
            f.write(latex_str)

    failed_latex_msg = f'Rendered TeX:\n{latex_str}\n'
    if only_print_latex or verbose:
        print(failed_latex_msg)

    if only_print_latex:
        return

    # TODO delete?
    if write_latex_like_pdf:
        tex_fname = str(pdf_path)[:-len('.pdf')] + '.tex'

        if verbose:
            print(f'Writing LaTeX to {tex_fname}')

        with open(tex_fname, 'w') as f:
            f.write(latex_str)

    compile_tex_to_pdf(latex_str, pdf_path, failed_latex_msg=failed_latex_msg,
        verbose=verbose
    )

