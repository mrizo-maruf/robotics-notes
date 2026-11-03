# Robotics notes: instructions for Claude

This repo is Mukhammadrizo's personal textbook for a semester-long **Graduate Introduction to Robotics** course. It is built one idea at a time from slide images he sends in the chat. He is a robotics/CV researcher (SLAM, VIO, ROS2, Isaac Sim, 3-D perception), so write at graduate level: full derivations and no hand-holding on basics he clearly knows. Never skip a step in an argument, though.

What he asked for, in his words: *extended notes on each idea; derive things, explain why and how it works; approach from first principles and the essence of the idea; visualizations that show how and why; applied, practical, non-trivial examples and how to use the concepts in them.* Every section must deliver all of that.

## The session workflow

1. **Sync.** `git pull`. Read this file, `frontmatter/notation.tex`, and the target chapter's `chapter.tex` plus the section before the new one, so notation and story continue seamlessly.
2. **Read the slides** he sent in the chat. Slides are never stored in the repo (`slides/` is git-ignored).
3. **Place the idea.** The `.tex` is split finely so that work, edits and diffs happen one piece at a time. Never put a whole section in one file:
   - one chapter = one folder `chapters/NN-name/`, NN = chapter number in book order (`make new CH=NN-name TITLE="..." KEY=short`). Refer to chapters only by chapter, never by lecture number;
   - one idea (a group of slides) = one **section folder** `sections/NN-slug/`, containing `section.tex` (heading, a short motivating paragraph, and the `\input` list) and is `\input` in `chapter.tex`;
   - one sub-idea = one **subsection file** `sections/NN-slug/MM-slug.tex` (~10-60 lines), `\input` in that `section.tex`.

   When asked to change something, edit only the files involved. If placement is unclear, ask in one short question before writing.
4. **Write** the section (anatomy below).
5. **Compute every number** in a worked example in `chapters/<ch>/code/examples.py`, using `code/rigid.py` (extend it when a new primitive is needed). Never hand-compute a number that appears in the text. Cross-check against an independent route (finite differences, `scipy`, a second formula).
6. **Draw figures** in `chapters/<ch>/code/figs.py` (style from `code/style.py`), output PDF to `chapters/<ch>/figs/`. Render each figure to PNG and look at it before using it; fix clipping and overlapping labels.
7. **Build and inspect.** Run `make ch CH=<ch>`. It must have 0 errors and 0 overfull boxes. References into *other* chapters show as ?? in a standalone chapter build; that is expected. Render the new pages to PNG and look at every one. Then run `make` (the whole book): it must have 0 undefined references and nothing else may break.
8. **Commit and push** to `main` with a message like `ch02: add DH parameters section`. CI rebuilds the PDF at the fixed link (README).
9. **Report** in 3-5 lines: what was added (section names, pages), anything on the slides that was ambiguous or looked wrong, and what you went beyond the slides with.

He may edit any file himself. Treat his edits as authoritative: never revert or "clean up" his changes. If one conflicts with notation or correctness, say so and ask.

## Anatomy of a section

Not every element is needed in every section. Most sections need most of them.

1. **The question.** One short paragraph: what problem this idea solves, and why the naive approach fails.
2. **Key idea box**: the essence in 2-4 sentences. Someone who reads only these boxes should still get the chapter.
3. **Derivations** in `deriv` boxes, from first principles: start from definitions or physics, and show every non-obvious step. Explain *why* each step is taken, not only what it is. Prefer a geometric argument plus an algebraic one when both exist.
4. **A figure that shows the mechanism**, not decoration: what moves, what stays fixed, what breaks. The caption explains what to look at.
5. **Worked example(s)** in `example` boxes: applied and non-trivial (real robots, sensors, SLAM/VIO, manipulators, perception pipelines), with concrete numbers from `code/`, a sanity check, and the lesson. At least one per section.
6. **Pitfalls**: what goes wrong in real code or practice, and how to detect it.
7. **Connections**: one or two sentences linking back to earlier chapters with `\S\ref{...}` (and forward, if the slides foreshadow something).

Close each chapter with a summary section: a comparison table if relevant, a formula sheet, a self-assessment checklist, practice problems with answers, and "the essence in N lines". Chapter 2 (`chapters/02-rotations`) is the reference for tone, depth and layout.

## Writing rules

- **Prose first.** Lists only for genuinely parallel items. British spelling (parameterise, optimisation), as in chapter 2.
- **Honest.** If a slide's claim is wrong, imprecise, or uses a different convention, write the correct version and add a `noteb` box mapping the slide's notation or statement to ours. Mention it in the report. If something is uncertain (e.g. a library's current convention), say so or check it, never guess.
- **Beyond the slides.** Sparse slides should be filled in from standard references (Lynch & Park, Siciliano et al., Barfoot, Murray-Li-Sastry, Corke, Thrun et al., Hartley & Zisserman), so each section stands alone.
- **Notation.** Use the book's notation (`frontmatter/notation.tex`), even when the slides differ. New notation goes into `notation.tex` first. New macros used in more than one chapter go in `notes.sty`. Chapter-only macros go at the top of that chapter's `chapter.tex`.
- **Boxes.** Use only `keyidea`, `deriv`, `example`, `pitfall`, `noteb`.
- **Labels.** Always `sec:<KEY>:<slug>`, `eq:<KEY>:<slug>`, `fig:<KEY>:<slug>`, where KEY is the chapter key (chapter 1 is `pl`, chapter 2 is `rot`). **Never hard-code a section, equation or figure number**: always `\ref`/`\eqref`/`\figref`.
- **Font.** 10 pt Times (`mathptmx`), set in `notes.sty`. Do not change layout per chapter.

## Figures

- matplotlib with `from style import *`. Full width = `FULL_W` (6.6 in), half width = `HALF_W`. Axis colours: x red, y green, z blue (`CX, CY, CZ`).
- 3-D axes: `set_box_aspect(..., zoom=1.2-1.5)` and `set_axis_off()`. Check that nothing is clipped.
- Diagrams that are pure boxes and arrows go in TikZ directly in the `.tex`.
- Commit both the script and the generated PDF. CI does not run Python.

## Layout

```
notes.tex            book root: front matter + \subfile per chapter
notes.sty            fonts, layout, boxes, global macros
frontmatter/notation.tex
chapters/NN-name/    one chapter; NN = chapter number
  chapter.tex        compiles alone; intro + \input of each section.tex
  sections/NN-slug/  one idea: section.tex + MM-slug.tex per subsection
  figs/  code/       generated figure PDFs; figs.py + examples.py
chapters/01-planar-motion/  chapter 1: frames, planar poses, pose graphs, twists
chapters/_template/  copied by `make new`
code/rigid.py        shared, tested numerics (SO3/SE3/quaternions/...)
code/style.py        shared figure style
.github/workflows/build.yml   CI: build + publish PDF to release "latest"
```
