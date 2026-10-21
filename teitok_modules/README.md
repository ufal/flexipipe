# flexipipe's TEITOK pages

The pages that run flexipipe from TEITOK (`index.php?action=flexipipe`). TEITOK finds an action's
page in the project's `Sources/`, then in the shared project's, then in its own, so these files
work from the shared project without touching TEITOK itself.

## Install

```
sudo perl teitok_modules/install-teitok-modules.pl [--shared DIR] [--user USER] [--dry-run]
```

`*.php` go to `SHARED/Sources/`, `*.js` / `*.css` to `SHARED/Scripts/`, `*.html` to
`SHARED/Pages/`. `SHARED` is the TEITOK shared project (default: `TT_SHARED` from TEITOK's
`.htaccess`). Re-running upgrades in place; a file changed locally since the last install is kept as
`FILE.local-YYYYmmdd-HHMMSS`; `SHARED/Resources/flexipipe-teitok-modules.json` records what was
installed (version, commit, checksums). flexipipe itself runs in TEITOK's Python environment
(`defaults/base/venv`, else `SHARED/Resources/venv`); the set-up page installs and upgrades it there.

## Files

| file | what |
|---|---|
| `flexipipe.php` | the router (`action=flexipipe`) |
| `flexipipe_setup.php` | the corpus defaults (language, model, models per language, options, sentence layout, the backend for steps the model does not do — named entities, normalization, pronunciation), backends, models, profiles; install / upgrade |
| `flexipipe_train.php` | the training page (`act=training`): a running training followed live, the corpus's models (`Resources/flexipipe-models/`, any trainable backend) with held-out scores, what the corpus can teach, Train / Retrain, use / tag with / remove a model |
| `flexipipe_jobs.php` | background jobs (training, long runs): started detached, one per kind at a time, progress from flexipipe's `step i/n` lines, the log tail, stop; `act=jobs`, `act=job` |
| `flexipipe_document.php` | one document: what it contains, what a run would do and who does each step, run, sentence preview |
| `flexipipe_settings.php` | flexipipe's part of `Resources/settings.xml`, read and written the way flexipipe reads it — including `<layers><item key attribute>`: where a layer is written when the user decides (else flexipipe takes the name the corpus uses: `teitok_names.py`; set-up page, "Where flexipipe writes") |
| `flexipipe_cmd.php` | running flexipipe: document paths (checked), manifests, escaped arguments, cached `info` calls |
| `flexipipe-setup.js`, `flexipipe.css` | the defaults form, the styles |
| `flexipipe_legacy.php` | the earlier module, for the pages not rebuilt yet (trying models on a text, gold comparison, training, processing a document) |

Plan for processing single documents (one click, or step by step): `dev/TEITOK-DOCUMENT-PIPELINE.md`.
