Extra tool `vdm1` (on PATH): a fast clip-level visual checker (Glance-4B) for video.mp4. It cuts the video into 2-second
clips (4 frames each, 1 s apart) and answers one yes/no question per clip, so it can see short actions and changes
(not audio, not anything longer than 2 s). Probabilities are not calibrated on video: treat them as rankings and look
at the frames around important hits yourself. Output: JSON plus a '# ...' summary line.
- `vdm1 count video.mp4 "Does a person places a cup on the table in this clip?" [--in t0-t1]` -> number of separate
  high-probability segments, the count under 9 threshold/gap settings, and the event times. Phrase the question as an
  action within one clip. Repeated actions closer than 2 s merge into one; check the times.
- `vdm1 find video.mp4 "Does a car stops at the crossing?" [--in t0-t1] [--top 5]` -> the most likely time intervals.
- `vdm1 verify video.mp4 --at t0-t1 "Is the door opened in this clip?"` -> mean/max p_yes and per-clip p_yes.
- `vdm1 choose video.mp4 --question "..." --options "A. ..." "B. ..."` -> each option's mean probability over 8 clips
  (a weak hint only; it ignores audio and long-range context).
Cost: about 1 clip per second of video; a 10-minute video takes roughly N seconds on the GPU (see README); long spans are
subsampled to at most 900 clips (the output says so) — narrow with --in for precise counts.
