Extra tool `vdm` (on PATH): a visual detector (Glance-4B) that scores sampled frames of video.mp4 one at a time:
single frames only (no motion, no audio, no temporal context), ~15 frames/s, JSON output plus a '# ...' summary line.
Probabilities are calibrated on natural images, not video frames: treat them as ranking scores and view key frames.
- `vdm find video.mp4 "a red mug on the table" [--fps 1] [--in t0-t1] [--top 5]` -> P(t) per frame, top intervals
  with peak probability and a box at the best frame. Use it to locate where/when something is visible.
- `vdm count video.mp4 "Is a person places a cup on the table?" [--fps 2] [--in t0-t1] [--mode yesno]` ->
  separated above-threshold segments, a count distribution over 9 threshold/gap settings, event timestamps.
  Object visibility is not an action: use a yes/no action question with --mode yesno, then check events visually.
- `vdm verify video.mp4 --at t0-t1 "Is the door open?"` -> p_yes per frame, mean, max.
- `vdm locate frame.jpg "the man in the blue jacket"` -> P(present) and box (x1,y1,x2,y2 px of that image).
- `vdm motion video.mp4 [--in t0-t1]` -> camera turn events (left/right, degrees) and forward motion from optical
  flow (no model; rough).
At most 1500 frames per call (fps is lowered for long spans and the output says so): scan long videos coarsely, then
re-scan a window with --in and a higher --fps. Cost: 10 min at 2 fps ~2-3 min; a 2-hour film at the cap ~5-10 min.
