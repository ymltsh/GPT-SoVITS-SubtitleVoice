import os
import subprocess

from ..ffmpeg_utils import get_ffmpeg_path


def export_gpt_sovits(project_dir: str, clips: list, output_dir: str,
                      episode_videos: dict[str, str] = None) -> dict:
    """Build a GPT-SoVITS dataset by re-extracting audio from original video.

    Each clip is trimmed with ffmpeg using its effective_start / effective_end,
    so Clip Boundary Trim takes effect immediately without re-running analysis.

    AI cache (cache/wav/) is deliberately left untouched.
    """
    os.makedirs(output_dir, exist_ok=True)
    wav_dir = os.path.join(output_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    train_list_path = os.path.join(output_dir, "train.list")
    results = []
    errors = []

    ffmpeg = get_ffmpeg_path() or "ffmpeg"
    episode_videos = episode_videos or {}

    with open(train_list_path, "w", encoding="utf-8") as list_file:
        for clip in clips:
            clip_id = str(clip.id).zfill(4)
            wav_filename = f"{clip_id}.wav"
            wav_path = os.path.join(wav_dir, wav_filename)

            video_path = episode_videos.get(clip.episode, "")
            if not video_path or not os.path.isfile(video_path):
                errors.append({
                    "id": clip.id, "text": clip.text,
                    "error": f"视频文件不存在: episode={clip.episode}"
                })
                continue

            eff_start = clip.effective_start
            eff_duration = clip.effective_duration

            cmd = [
                ffmpeg, "-y",
                "-ss", str(eff_start),
                "-t", str(eff_duration),
                "-i", video_path,
                "-ac", "1",
                "-ar", "48000",
                "-c:a", "pcm_s16le",
                wav_path,
            ]

            try:
                r = subprocess.run(
                    cmd, capture_output=True, text=True,
                    encoding="utf-8", errors="replace", timeout=60,
                )
                if r.returncode != 0:
                    errors.append({
                        "id": clip.id, "text": clip.text,
                        "error": r.stderr[:200],
                    })
                    continue
            except Exception as exc:
                errors.append({
                    "id": clip.id, "text": clip.text,
                    "error": str(exc),
                })
                continue

            list_file.write(f"{wav_filename}|{clip.text}\n")
            results.append({
                "id": clip.id, "file": wav_filename, "text": clip.text,
            })

    return {
        "output_dir": output_dir,
        "wav_count": len(results),
        "error_count": len(errors),
        "train_list": train_list_path,
        "results": results,
        "errors": errors,
    }
