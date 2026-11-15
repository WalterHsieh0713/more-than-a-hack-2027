# More Than A Hack 2027 — working notes for Claude

Task: label every tactical-video frame (0..199318) as H (Barcelona), A (Athletic), D (dead ball) or X (abstain).
Ranked on spellF1 (possession segments matched by temporal IoU at 0.3-0.7); macroF1 is the tie-break.
Deadline Nov 17 2026, 23:59 CET. Read CHALLENGE.md, DATA.md, EVALUATION.md, RULES.md, SUBMISSION.md first.

Hard rules (disqualifying if broken):
- No hand labels or hand corrections on this match, for any purpose. Labels must come from code.
- The 5-minute public GT (frames 11185-18684) is for sanity checks only: never train or tune on it.
- No human-produced data about this match (event data, Opta/StatsBomb, etc).
- The repo must regenerate the exact submission file deterministically (fix seeds, pin deps).
- Never commit anything under data/ (gitignored; keep it that way).
- The organisers' email to all participants overrides RULES.md: training, retraining and tuning use external data only. No self-training or pseudo-labels on this match, whatever RULES.md says.
- Organisers' reply, 2026-11-06: anything not explicitly allowed is not permitted. Never check output against the video. Check the licence of every dataset before using it.
- Hosted LLMs must never see match data or results. Run BAR-ATH checks through `python -m src.blind`, which prints only PASS or FAIL. Never open data/logs.
- Keep evaluation/gt, evaluation/results and data/ out of every search, including git grep: a match prints their rows.

Key facts:
- ATD frame_id = video_frame + 1. Coords normalised 0-1; pitch 105.3 x 68.0 m.
- Ball missing on ~41% of graded frames. 46 track slots, no identities.
- Graded: 11185-90651 and 110808-190197. 110808-113741 is all D (2nd half kicks off at 113742).
- GT is derived mechanically from Metrica's corrected tracking: D where it marks the ball dead, else the owning team.
- Smart Tagging times are seconds from video start: frame = round(s*25). Throw-ins have Team = N/A.
- Baseline to beat: baseline_smooth50, spellF1 0.2433 on the full match.

Setup: `uv sync`, then `uv run pytest`. Python and packages are pinned in uv.lock.

Working style: concise explanations, plain prose. Build one piece at a time and check in after each milestone.
