PYTHON ?= python3

.PHONY: check paper response audit-vlm overleaf
check:
	$(PYTHON) -B scripts/check_paper_repository.py
	$(PYTHON) -B scripts/build_reviewer_response_map.py --check

audit-vlm:
	$(PYTHON) -B scripts/audit_vlm_reconciliation.py --archive-root . --out build/vlm.json

overleaf:
	$(PYTHON) -B scripts/build_overleaf_bundle.py

paper:
	mkdir -p manuscript/build
	xelatex -interaction=nonstopmode -halt-on-error -output-directory=manuscript/build manuscript/TRIX_REVISION.tex
	xelatex -interaction=nonstopmode -halt-on-error -output-directory=manuscript/build manuscript/TRIX_REVISION.tex

response: paper
	$(PYTHON) -B scripts/build_reviewer_response_map.py
	xelatex -interaction=nonstopmode -halt-on-error -output-directory=manuscript/build manuscript/TRIX_RESPONSE.tex
	xelatex -interaction=nonstopmode -halt-on-error -output-directory=manuscript/build manuscript/TRIX_RESPONSE.tex
