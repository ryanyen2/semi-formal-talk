# Semi-formal programming (talk, 2026-11-14)

The slides of a talk, as one web page, with a live calendar inside it. The calendar reads each typed
event (day, start, end) with a small model that runs in the browser and learns from corrections:
move an event to where it belongs and the model's weights change so the sentence is read that way.

Keys: → / Space / PageDown next, ← / PageUp back. In the calendar's sentence the arrows move the caret
once there is text; Esc leaves the sentence. The address keeps the beat (`#13` is the calendar).

```bash
npm install
npm run dev
```

The stock prices are NVIDIA's daily closes from Yahoo Finance. The meme on beat 7 is by HADISY AI ACADEMY; beat 11 quotes the New York Times (Chinese edition) of 2026-10-30. Beat 14 is a recording of the calendar, for when the live one fails.
The code is copied from a research repository by `talk/export.sh`.
