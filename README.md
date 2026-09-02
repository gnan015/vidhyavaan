# Exotel inbound audio service

Production-minded FastAPI service for audio calls routed to an Exotel virtual number. It supports either realtime audio sent by a VoiceBot/Audio Stream applet or a Passthru/Record Applet callback with a completed recording.

## Run locally

Requires Python 3.11+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Check `http://127.0.0.1:8000/health`.

## Expose with ngrok

Install and authenticate ngrok, then in another terminal run:

```powershell
ngrok http 8000
```

Use the generated HTTPS forwarding address in the Exotel App Bazaar flow:

- Passthru / Record applet callback: `https://YOUR-NGROK-DOMAIN/api/exotel/callback?token=YOUR_SECRET`
- VoiceBot / Audio Stream applet: `wss://YOUR-NGROK-DOMAIN/ws/exotel-stream?token=YOUR_SECRET`

Set `EXOTEL_WEBHOOK_TOKEN` in `.env` to the same secret. Configure `RECORDING_ALLOWED_HOSTS` in production with Exotel's actual recording host(s), for example `my-recording-host.example`. Leave it blank only if recording URLs may originate from different known HTTPS hosts.

## Endpoints and call flow

`GET` and `POST /api/exotel/callback` accept form-urlencoded Exotel fields: `CallSid` (required), `From`, `To`, `RecordingUrl`, `Digits`, and `Duration`. A valid `RecordingUrl` is queued for bounded background download into `data/recordings`. The response is Exotel-compatible XML containing `Hangup`; change `exotel_hangup_xml()` in `app/routes/exotel.py` to return the exact `Gather`, `Dial`, or continuation XML required by your configured call flow.

`WS /ws/exotel-stream` accepts Exotel JSON handshake/media packets. It persists base64 `media.payload` frames to `recordings/call_{call_sid}.wav`, using `start.media_format` to determine the sample rate and bit depth; Exotel's absent metadata defaults to 8 kHz, 16-bit mono raw PCM. `audio/x-mulaw` frames are converted to 16-bit PCM before they are stored. Replace `process_audio_frame()` in `app/services/audio.py` with your STT provider adapter; keep it non-blocking or enqueue the frame.

## Security and operations

Use a strong `EXOTEL_WEBHOOK_TOKEN`; clients provide it through `X-Exotel-Token` or query string. `EXOTEL_SIGNATURE_SECRET` enables the included HMAC-SHA256 placeholder over `METHOD + newline + path + newline + raw body`; confirm Exotel's documented signing scheme for the product/account and adjust `app/services/security.py` accordingly. Use a durable object store and job queue instead of local files/background tasks for multi-instance production deployments. JSON logs include call metadata but never audio content.
