# Before publishing this repository

1. **Author approval:** confirm the final repository name and any required co-author/institutional approval. The authors have chosen **not to grant a code licence**. Do not add a `LICENSE` file without a later explicit author decision; public visibility alone does not grant general reuse rights.
2. **SEER agreement:** the authorised user should check the exact agreement accepted for this data release and the approved scope of ML analysis. This repository contains code and aggregate results only; do not add SEER case listings, individual records, predictions, SHAP values, fitted models, or model parameters.
3. **Repository boundary:** upload only the contents of this folder. Never add the restricted run directories, the surrounding `_codex_workspace`, or a project-root archive. Run future analyses outside the repository; `.gitignore` is a secondary safeguard, not a data-disclosure guarantee.
4. **Final content check:** inspect `git status` and the staged diff before the first push. Check every newly added file for identifiers, local absolute paths, credentials, restricted logs, and small-cell results.
5. **Versioned citation:** after approval, create a fixed GitHub release/tag (and, if desired, a DOI archive), then put its immutable link in the manuscript's Code availability statement. Do not cite an unversioned working branch as if its contents cannot change.
6. **Manuscript alignment:** confirm that the final manuscript, main tables, supplementary tables, figures, and legends use the same local-rerun result set as this code package. The code repository itself does not perform that editorial cross-check.

Suggested Code availability wording, to complete **after** publication of the repository:

> Analysis code and software-environment specifications are available at [versioned repository URL]. Individual-level SEER data, patient-level predictions, and fitted models are not redistributed. Researchers may request access to the relevant SEER data through the SEER Program, subject to its data-use agreement.

Do not insert a placeholder URL into the submitted manuscript.
