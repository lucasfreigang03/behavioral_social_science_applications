import json
import logging
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from config import DURATION_SEC

TIMEOUT_SEC = 300
PROBE_TIMEOUT_SEC = 60
MAX_ATTEMPTS = 3
RETRY_WAIT_SEC = 5
TRANSIENT_ERRORS = ("timeout", "timed out", "connection reset", "broken pipe",
                    "server returned 5", "temporary failure in name resolution")

class EpisodeDownloader:
    """Downloads the middle DURATION_SEC seconds of each episode as 16 kHz mono wav file with ffmpeg.
    Next to each wav file, a JSON file stores the episode duration and where the excerpt starts."""
    def __init__(self, out_dir: Path, max_workers: int = 6):
        """Create the output folder. max_workers is the number of parallel downloads."""
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers

    def download(self, df: pd.DataFrame) -> pd.DataFrame:
        """Download all episodes in df (columns episode_hash, episode_audio_url, episode_length) in parallel.
        Returns one row per episode with episode_hash, success, path, audio_duration_sec, duration_source,
        excerpt_start_sec and error."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            results = list(ex.map(self.__download_one, df["episode_hash"], df["episode_audio_url"], df["episode_length"]))
        return pd.DataFrame(results, columns=["episode_hash", "success", "path", "audio_duration_sec",
                                              "duration_source", "excerpt_start_sec", "error"])

    def __download_one(self, episode_hash: str, url: str, feed_length):
        """Download the middle of one episode, or reuse the files if they already exist.
        The duration comes from ffprobe (the audio file itself) or, if that fails, from the feed (itunes_duration).
        Transient errors (e.g. timeouts) are retried up to MAX_ATTEMPTS times with a growing wait."""
        out = self.out_dir / f"{episode_hash}.wav"
        meta_path = self.out_dir / f"{episode_hash}.json"
        if out.exists() and meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            return (episode_hash, True, str(out), meta["audio_duration_sec"], meta["duration_source"],
                    meta["excerpt_start_sec"], None)

        duration = self.__probe_duration(url)
        source = "ffprobe"
        if duration is None:
            duration = feed_length
            source = "feed"
            logging.warning(f"ffprobe failed for {episode_hash}, using the duration from the feed")
        if pd.isna(duration):
            return episode_hash, False, None, None, None, None, "duration unknown"
        start = round(max(0.0, float(duration) / 2 - DURATION_SEC / 2), 1) # midpoint - 5 min

        err = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            err = self.__run_ffmpeg(url, out, start)
            if err is None:
                meta = {"audio_duration_sec": float(duration), "duration_source": source, "excerpt_start_sec": start}
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f)
                logging.info(f"Download OK: {episode_hash}")
                return episode_hash, True, str(out), float(duration), source, start, None
            if attempt == MAX_ATTEMPTS or not self.__is_transient(err):
                break
            logging.warning(f"RETRY {episode_hash} ({attempt}/{MAX_ATTEMPTS}): {err}")
            time.sleep(RETRY_WAIT_SEC * attempt)

        logging.error(f"FAIL {episode_hash}: {err}")
        return episode_hash, False, None, float(duration), source, start, err

    def __probe_duration(self, url: str) -> float | None:
        """Return the duration of the audio file in seconds as reported by ffprobe, or None if it cannot be read."""
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", url]
        try:
            result = subprocess.run(cmd, check=True, timeout=PROBE_TIMEOUT_SEC, capture_output=True, text=True)
            duration = float(result.stdout.strip())
            return duration if duration > 0 else None
        except Exception:
            return None

    def __run_ffmpeg(self, url: str, out: Path, start: float) -> str | None:
        """Run ffmpeg once. Writes to a .part file and renames it on success, so a finished file is always complete.
        Returns None on success, otherwise the error text."""
        tmp = out.with_suffix(".part.wav")
        cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error",
               "-ss", str(start), "-t", str(DURATION_SEC), "-i", url,
               "-ar", "16000", "-ac", "1", str(tmp)]
        try:
            subprocess.run(cmd, check=True, timeout=TIMEOUT_SEC, capture_output=True, text=True)
            tmp.rename(out)
            return None

        except subprocess.TimeoutExpired:
            err = "timeout"

        except subprocess.CalledProcessError as e:
            err = (e.stderr or "").strip()[-300:] or f"ffmpeg exit {e.returncode}"

        except FileNotFoundError as e:
            err = e.filename

        except Exception as e:
            err = repr(e)

        tmp.unlink(missing_ok=True)
        return err

    @staticmethod
    def __is_transient(err: str) -> bool:
        """Return True if the error text looks temporary, so that a retry makes sense."""
        err = err.lower()
        return any(t in err for t in TRANSIENT_ERRORS)
