Extra tool `vdmscan` (on PATH): a fast visual event scanner for video.mp4 (Glance-Video, a 4B video model on a GPU
server). The whole video is already encoded as short clips (2-s clips; videos longer than 10 min use longer clips, at
most 600 clips), so each question takes a few seconds. It judges one short visual phrase per clip: no audio, no speech,
nothing that needs more than a few seconds of context.
- `vdmscan find video.mp4 "picks up a red book" [--in t0-t1] [--k 6]` -> the k most likely moments
  (spread out in time) with probabilities, a verdict (PRESENT is calibrated: at most ~10% false alarms on videos without
  the event; NOT DETECTED can still miss brief or small events), and ONE contact sheet image of those moments in time
  order (4 frames per moment, labelled). Read that sheet to check all candidates side by side instead of extracting frames
  one by one.
- `vdmscan when video.mp4 "..." [--in t0-t1]` -> all time intervals where the event is detected.
- `vdmscan count video.mp4 "the person opens the fridge" [--in t0-t1]` -> number of separate occurrences and
  their times (occurrences closer than ~2 s merge; for fast repetitions in a short span use `reps`).
- `vdmscan reps video.mp4 "squats" --in t0-t1` (span <= 180 s) -> repetition count of a periodic action.
How to use it well: turn the question into concrete visual sub-questions and `find` each one (a short event or object you
could see in a 2-s clip, e.g. "the person closes the drawer", not the multiple-choice question itself); Read the sheet;
on a long video, rerun `find` with `--in` around the best candidate (a window <= 600 s gives 2-s clips) for exact timing
and order. Do not scan the answer options to pick one: the option texts are written so that visual similarity points to
distractors; use the tool to locate and count events, and reason from what you see.
