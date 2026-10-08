# Live Deployment Checklist

## GitHub
1. Create a public GitHub repository, e.g. `AI-Music-Generator`.
2. Upload the project files.
3. Make sure `models/lstm_music_generator.pt` is committed (it is intentionally small).

## Streamlit Community Cloud
1. Open Streamlit Community Cloud.
2. Create **New app**.
3. Select the GitHub repository and branch.
4. Main file: `app/app.py`
5. Deploy.

## After deployment
Test:
- Generate Music
- Audio playback
- MIDI download
- WAV download
- Different styles
- Temperature/Top-K
- Mobile layout

## Important
Do not put private or copyrighted MIDI collections in a public repository. Keep the deployed checkpoint and dataset licensing information documented.
