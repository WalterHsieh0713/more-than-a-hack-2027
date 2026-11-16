# Walter: how to finish and submit

**Deadline: Tuesday 17 November 2026, 23:59 CET.** Please submit as soon as you can rather than on
the day. The form keeps the last response before the deadline, so submitting early costs nothing.

Everything is built and checked. What's left is making the repo public and filling in the form.
It should take about 15 minutes. Don't change any code or rerun any training.

## Already done

- The final code is on `main`. All 344 tests pass on macOS and on your Windows machine.
- `827.csv` regenerates to the same file on both machines. Its SHA-256 is:

  ```
  311f12563a420d044646fed0d1592f43718d170d6a9fb52dacecd4b34fbe2656
  ```

- No match data is committed, and the notes in `progress/` hold nothing from the match.
- The technical document is `827.md`, in the top folder of the repo.

## Step 1. Get the latest `main`

```powershell
git checkout main
git pull
```

## Step 2. Check the file you're about to upload

You already have `evaluation/results/827.csv` from your hash check. Confirm it's still the right
one:

```powershell
Get-FileHash evaluation/results/827.csv
```

PowerShell prints the hash in capitals. It must be the hash above. If it isn't, regenerate the
file with `uv run python -m src.submit` and check again. If it still differs, stop and message
Alex.

- [ ] The hash matches

## Step 3. Make the repo public

The rules require a public repository, and only you can change this.

1. Open https://github.com/WalterHsieh0713/more-than-a-hack-2027/settings
2. Scroll to the bottom, to "Danger Zone"
3. Click "Change visibility", choose "Change to public", and confirm
4. Open https://github.com/WalterHsieh0713/more-than-a-hack-2027 in a private browser window, not
   logged in, and check that it loads

- [ ] The repo is public and loads in a private window

## Step 4. Check for late rulings

Look through the organisers' emails for anything new since 6 November. If something changes what
we're allowed to submit, stop and message Alex.

- [ ] Nothing new, or the team has discussed it

## Step 5. Submit the form

Open the Google Form linked in the organisers' email. You need to be signed in to a Google account.

| Field | What to put |
|---|---|
| Team name | `827` |
| Possession file | `evaluation/results/827.csv`, the one you checked in step 2 |
| Public code repository URL | `https://github.com/WalterHsieh0713/more-than-a-hack-2027` |
| Technical document | `827.md`, from the top folder of the repo |

The form accepts Markdown, so there's no need to export a PDF. Don't rename either file.

- [ ] Submitted, on ______ at ______

## Step 6. Confirm

1. Take a screenshot of the confirmation page and post it in the team chat.
2. Tick the boxes in this file, commit and push, so everyone can see it's done.

- [ ] Screenshot posted

## After submitting

Don't push anything else to the repo before the results are announced, apart from ticking these
boxes. The judges will rerun the code, and it has to produce the file we submitted.

## Things not to do

- Don't paste the CSV, scores, label counts or anything under `data/logs` into ChatGPT, Claude or
  any other hosted AI. The competition terms forbid it.
- Don't compare the output against the video, and don't tune anything on the provided match.

## If something goes wrong

- **The repo can't be made public:** message Alex. Alex can push the same code to a public repo
  under their own account, and you submit that URL instead.
- **The form rejects the file:** run
  `uv run python evaluation/validate_submission.py evaluation/results/827.csv` and send Alex the
  one-line result.
- **You can't submit in time:** tell Alex by Monday evening, and Alex will submit with the same
  three items.
