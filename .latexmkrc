# Shared build settings (also used by CI).
# - pdflatex, aux files in build/
# - repo root on TEXINPUTS so notes.sty is found when a chapter is compiled
#   from inside its own folder (latexmk -cd, VS Code LaTeX Workshop, ...)
use Cwd 'abs_path';
my $root = abs_path('.');
$ENV{'TEXINPUTS'} = "$root:" . ($ENV{'TEXINPUTS'} // '');
$pdf_mode = 1;
$out_dir = "$root/build";
$pdflatex = 'pdflatex -interaction=nonstopmode -halt-on-error -file-line-error %O %S';
