# Usage
#   make                      build the whole book          -> build/notes.pdf
#   make ch CH=02-rotations   build one chapter (fast)      -> build/ch-02-rotations.pdf
#   make figs CH=02-rotations regenerate that chapter's figures (python)
#   make check CH=02-rotations re-run the chapter's numeric examples
#   make new CH=03-kinematics TITLE="Forward Kinematics" KEY=fk   start a chapter
#   make clean
CH ?=
.PHONY: book ch figs check new clean
book:
	latexmk -pdf notes.tex
ch:
	@test -n "$(CH)" || (echo "usage: make ch CH=02-rotations" && exit 1)
	latexmk -cd -pdf -jobname=ch-$(CH) chapters/$(CH)/chapter.tex
figs:
	python3 chapters/$(CH)/code/figs.py
check:
	python3 chapters/$(CH)/code/examples.py
new:
	@test -n "$(CH)" -a -n "$(TITLE)" -a -n "$(KEY)" || (echo 'usage: make new CH=02-name TITLE="Title" KEY=short' && exit 1)
	@test ! -e chapters/$(CH) || (echo "chapters/$(CH) exists" && exit 1)
	cp -r chapters/_template chapters/$(CH)
	find chapters/$(CH) -name '*.tex' -exec sed -i 's|@CH@|$(CH)|g; s|@TITLE@|$(TITLE)|g; s|@KEY@|$(KEY)|g' {} +
	sed -i 's|^% ---- new chapters are inserted above this line|\\subfile{chapters/$(CH)/chapter}\n&|' notes.tex
	@echo "created chapters/$(CH) and added it to notes.tex"
clean:
	latexmk -C notes.tex; rm -rf build
