PYTHON ?= python3

.PHONY: check paper audit-vlm
check:
	$(PYTHON) -B scripts/check_paper_repository.py

audit-vlm:
	$(PYTHON) -B scripts/audit_vlm_reconciliation.py --archive-root . --out build/vlm.json

paper:
	mkdir -p manuscript/build
	xelatex -interaction=nonstopmode -halt-on-error -output-directory=manuscript/build manuscript/TRIX_REVISION.tex
	xelatex -interaction=nonstopmode -halt-on-error -output-directory=manuscript/build manuscript/TRIX_REVISION.tex
