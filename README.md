# Introduction to Robotics: extended notes

My own textbook for the Graduate Introduction to Robotics course, built from first principles one idea at a time.

**Latest PDF** (rebuilt automatically on every push to `main`):
https://github.com/mrizo-maruf/robotics-notes/releases/download/latest/robotics-notes.pdf

## How a new section gets written

1. Send Claude the slide images for one idea (in a session that has this repo attached), and say which chapter they belong to if it isn't obvious.
2. Claude writes the section (derivations, figures, verified worked examples), builds it, looks at the pages and pushes to `main`. `CLAUDE.md` holds the standing instructions.
3. CI rebuilds the PDF at the link above.

You can edit any `.tex` file yourself; Claude pulls before working and keeps your changes.

## Building locally (optional)

Needs TeX Live (`latexmk`, `pdflatex`) and, for figures, Python with `requirements.txt`.

```bash
make                         # whole book      -> build/notes.pdf
make ch CH=02-rotations      # one chapter     -> build/ch-02-rotations.pdf
make figs CH=02-rotations    # regenerate figures
make check CH=02-rotations   # re-run the numeric examples
make new CH=03-kinematics TITLE="Forward Kinematics" KEY=fk
```

VS Code: the LaTeX Workshop extension works if you build `notes.tex` from the repo root. Overleaf can also import the repo, but GitHub sync there is a paid feature.

## Structure

The `.tex` is split so every piece is small and edited on its own:

```
chapters/NN-name/                one chapter (NN = chapter number)
  chapter.tex                    intro + list of sections
  sections/NN-idea/section.tex   one idea: heading + list of subsections
  sections/NN-idea/MM-sub.tex    one sub-idea (~10-60 lines)
```

Chapter 1 is frames and planar rigid-body motion; chapter 2 is 3-D rotations and rigid-body motion.

See the "Layout" section of `CLAUDE.md`. Slides are not stored in the repo.
