# Before the final submission

**Deadline: Tuesday 17 November 2026, 23:59 CET.** The model (v2) is frozen. Everything left is
checking and submitting. Tick each box and push, so everyone can see what's done.

Don't change the model or tune anything on the provided match. Hand labels, tuning on the public
window, checking output against the video, and pseudo-labels on this match are all banned. The
organisers also said that anything not explicitly allowed is not permitted.

## 1. Get the latest code

All the final work is on `main`.

```bash
git fetch
git checkout main
git pull
uv sync
uv run pytest        # expect: 344 passed
```

- [x] Tests pass on a second machine. Name: Walter (344 passed, Windows)

## 2. Regenerate the file and check it's identical

You need the match bundle in `data/` (or `MTAH_DATA_DIR` pointing at it).

```bash
uv run python -m src.submit
uv run python evaluation/validate_submission.py evaluation/results/827.csv
```

Then check the SHA-256:
- Windows PowerShell: `Get-FileHash evaluation/results/827.csv`
- macOS/Linux: `shasum -a 256 evaluation/results/827.csv`

It must be (PowerShell prints it in capitals, which is the same hash):

```
311f12563a420d044646fed0d1592f43718d170d6a9fb52dacecd4b34fbe2656
```

- [x] The validator passes
- [x] The hash matches on a second machine. Name: Walter (Windows)

If the hash differs, stop and tell the team. Don't submit a file nobody can reproduce.

Never paste the CSV, scores, label counts or anything under `data/logs` into ChatGPT, Claude or any
other hosted AI. The terms forbid it.

## 3. Tidy the repo

- [x] Merge `stage-8-finalist` into `main`, so the judges see the final code first
- [x] Fix the test count in `827.md` ("340 tests" should be "342 tests", in the Reproduction section)
- [ ] Delete this file, or leave it in. Either is fine.

## 4. Make the repo public

The rules require a public repository URL. Ours is private.

- [ ] Check that `data/` is still gitignored and nothing from the match bundle is committed.
  The only files committed under `data/` and `evaluation/results/` come from the organisers'
  initial commit, which is already public.
- [ ] Read `progress/2026-11-01-walter.md` and `progress/2026-11-05-alex.md` for anything about
  results on the provided match. Everything in the repo becomes public; remove anything like that
  first. Name: ______
- [ ] GitHub, then Settings, then Change visibility, then Public
- [ ] Open the URL in a private window to check that it loads

## 5. Technical document

- [ ] Export `827.md` to `827.pdf` (the form accepts `.md` too, but it must be named `827`)
- [ ] Check that the tables and code blocks survived the export

## 6. Last checks

- [ ] Re-read the organisers' GitHub issues
  (github.com/metrica-sports/more-than-a-hack-2027/issues) and the email inbox for new rulings
- [ ] Decide who submits: ______ (they need a Google account)

## 7. Submit

Use the Google Form from the organisers' email. Only one response per team counts, and the last
one before the deadline is scored.

| Field | What to put |
|---|---|
| Team Name | `827` |
| Possession file | `827.csv`, the one with the hash above |
| Public code repository URL | https://github.com/WalterHsieh0713/more-than-a-hack-2027 |
| Short technical document | `827.pdf` |

- [ ] Submitted. By ______, on ______ at ______
- [ ] Screenshot of the confirmation posted in the team chat
