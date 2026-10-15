"""Generate a small, reproducible multimodal corpus for benchmarks and retrieval evals.

All content is fictional. Output (default: samples/corpus/):
  - 4 multi-page PDFs           (fpdf2)
  - 2 synthetic images          (Pillow)
  - 2 speech recordings         (macOS `say` + ffmpeg)
  - 1 narrated slide video      (Pillow frames + `say` narration + ffmpeg)
  - optional benchmark files   (--bench: 100-page PDF, 10-min speech, 5-min 720p video)
  - optional ~1 GB video        (--large, ffmpeg noise source) for upload memory tests

Usage:
    python scripts/make_sample_corpus.py [--out samples/corpus] [--large]
"""

import argparse
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

from fpdf import FPDF
from PIL import Image, ImageDraw

PDFS = {
    "northwind_handbook.pdf": [
        "Northwind Robotics Employee Handbook. Section 1: Working arrangements. "
        "Employees may work remotely up to three days per week, subject to manager approval. "
        "Core collaboration hours are 10:00 to 15:00 Central Time, and recurring meetings "
        "should be scheduled inside that window whenever possible.",
        "Section 2: Paid time off. Full-time employees accrue 1.75 days of paid time off per month. "
        "Balances are capped at 30 days. Any unused paid time off above the cap is forfeited "
        "on January 31 of the following year. Part-time staff accrue time off pro rata.",
        "Section 3: Expenses. Travel expenses must be submitted within 45 days through the Ledgerly "
        "portal, with itemized receipts attached. Meals are reimbursed up to 75 dollars per day "
        "while travelling. Alcohol is never reimbursable.",
        "Section 4: Device security. All laptops must use full-disk encryption and a screen lock "
        "of five minutes or less. Lost or stolen devices must be reported to the security desk at "
        "extension 4410 within two hours of discovery.",
    ],
    "atlas_launch_plan.pdf": [
        "Project Atlas Launch Plan. Atlas is an autonomous warehouse picking robot designed for "
        "mid-sized distribution centers. After two pilot phases, Atlas is scheduled to reach "
        "general availability on March 9, 2027.",
        "Pricing. Atlas Standard is priced at 38,500 dollars per unit. Atlas Pro is priced at "
        "52,000 dollars per unit and includes the vision upgrade and priority support. Volume "
        "discounts start at ten units.",
        "Risks. The main supply chain risk is the lidar sensor, which is sourced from a single "
        "vendor, Kestrel Optics, with a lead time of 14 weeks. A second supplier is being "
        "qualified for the second half of 2027.",
        "Pilot program. The pilot customers are Harbor Freight Logistics in Memphis and Blue Ridge "
        "Grocers in Asheville. Each pilot runs for twelve weeks with weekly performance reviews.",
    ],
    "retention_policy.pdf": [
        "Data Retention Policy. Customer invoices are retained for seven years to meet tax "
        "obligations. Support chat transcripts are deleted automatically after 18 months.",
        "Backups. Production backups are encrypted with AES-256 and stored in two regions, "
        "us-east-2 and eu-west-1. Restore tests are performed every quarter and the results are "
        "recorded in the compliance tracker.",
        "Deletion requests. Requests for deletion under GDPR must be completed within 30 days of "
        "verification. The data protection officer responsible for these requests is Miriam Okafor.",
    ],
    "incident_postmortem.pdf": [
        "Incident Postmortem: Checkout Outage. On August 14, 2026 the checkout service was "
        "unavailable for 47 minutes, affecting roughly 12 percent of daily orders.",
        "Root cause. An expired TLS certificate on the internal payments proxy caused all payment "
        "authorization calls to fail. The automated renewal job had been failing silently since July "
        "because of a revoked service account token.",
        "Action items. Add certificate expiry alerts that fire 21 days before expiry; the owner is "
        "Dev Patel and the due date is October 1. Add alerting on renewal job failures.",
    ],
}

AUDIO = {
    "weekly_standup.mp3": [
        "Good morning everyone, this is the weekly engineering standup.",
        "The mobile team shipped version four point two on Tuesday, and crash rates are down.",
        "The database migration to Postgres sixteen is scheduled for November third.",
        "Priya will own the on call rotation next week.",
        "The design review for the onboarding flow has moved to Thursday at two p m.",
        "That is everything for today, thanks all.",
    ],
    "customer_call.wav": [
        "This is a summary of the account call with Lakeside Clinics.",
        "The customer reported that exports to CSV time out above fifty thousand rows.",
        "They asked for single sign on with Okta before their next rollout.",
        "Their renewal is due on December fifteenth, and the contract value is one hundred twenty thousand dollars.",
        "We agreed to send a written plan by Friday.",
    ],
}

VIDEO_NARRATION = [
    "Welcome to the Atlas product demo.",
    "This chart shows picking throughput growing every quarter.",
    "On a full charge, the Atlas battery lasts eleven hours, and charging takes ninety minutes.",
    "The safety scanner stops the robot within thirty centimeters of a person.",
    "Thanks for watching.",
]


def make_pdf(path: Path, pages: list[str]) -> None:
    pdf = FPDF()
    pdf.set_font("Helvetica", size=12)
    for text in pages:
        pdf.add_page()
        pdf.multi_cell(0, 7, text)
    pdf.output(str(path))


def bar_chart(size=(640, 480)) -> Image.Image:
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    w, h = size
    d.line([(60, h - 60), (w - 40, h - 60)], fill="black", width=3)
    d.line([(60, 40), (60, h - 60)], fill="black", width=3)
    for i, value in enumerate([120, 200, 260, 340]):
        x0 = 100 + i * 120
        d.rectangle([x0, h - 60 - value, x0 + 70, h - 60], fill=(40, 90, 200))
    return img


def sunset(size=(640, 480)) -> Image.Image:
    w, h = size
    img = Image.new("RGB", size)
    d = ImageDraw.Draw(img)
    for y in range(h):
        t = y / h
        d.line([(0, y), (w, y)], fill=(255, int(140 - 80 * t), int(60 + 40 * t)))
    d.ellipse([w // 2 - 70, h // 2 - 20, w // 2 + 70, h // 2 + 120], fill=(255, 220, 80))
    d.rectangle([0, int(h * 0.75), w, h], fill=(20, 40, 90))
    return img


def green_field(size=(640, 480)) -> Image.Image:
    w, h = size
    img = Image.new("RGB", size, (135, 200, 245))
    d = ImageDraw.Draw(img)
    d.rectangle([0, h // 2, w, h], fill=(60, 160, 60))
    d.ellipse([w - 160, 40, w - 60, 140], fill=(255, 230, 0))
    return img


def say_to_wav(text: str, out_wav: Path, workdir: Path) -> None:
    aiff = workdir / (out_wav.stem + ".aiff")
    subprocess.run(["say", "-o", str(aiff), text], check=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff), "-ar", "16000", "-ac", "1", str(out_wav)],
        check=True,
    )


def speech_file(sentences: list[str], out: Path, workdir: Path) -> None:
    """Render each sentence separately and join them with short pauses."""
    parts = []
    for i, sentence in enumerate(sentences):
        wav = workdir / f"{out.stem}_{i}.wav"
        say_to_wav(sentence, wav, workdir)
        parts.append(wav)
    listing = workdir / f"{out.stem}_list.txt"
    silence = workdir / "silence.wav"
    if not silence.exists():
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=16000:cl=mono",
                "-t",
                "0.8",
                str(silence),
            ],
            check=True,
        )
    listing.write_text("".join(f"file '{p}'\nfile '{silence}'\n" for p in parts))
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing), str(out)],
        check=True,
    )


def media_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(out.stdout.strip())


def make_video(out: Path, workdir: Path) -> None:
    narration = workdir / "narration.wav"
    speech_file(VIDEO_NARRATION, narration, workdir)
    total = media_duration(narration)
    scenes = [green_field(), bar_chart(), sunset()]
    listing = workdir / "scenes.txt"
    lines = []
    for i, img in enumerate(scenes):
        png = workdir / f"scene_{i}.png"
        img.save(png)
        lines.append(f"file '{png}'\nduration {total / len(scenes):.2f}\n")
    lines.append(f"file '{workdir / f'scene_{len(scenes) - 1}.png'}'\n")
    listing.write_text("".join(lines))
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-i",
            str(narration),
            "-vf",
            "fps=25,format=yuv420p",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(out),
        ],
        check=True,
    )


def make_large_video(out: Path, seconds: int = 180) -> None:
    """~1 GB incompressible 1080p video (noise caps the encoder at the target bitrate)."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=1920x1080:rate=30:duration={seconds}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-vf",
            "noise=alls=60:allf=t",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-b:v",
            "45M",
            "-minrate",
            "45M",
            "-maxrate",
            "45M",
            "-bufsize",
            "90M",
            "-c:a",
            "aac",
            str(out),
        ],
        check=True,
    )


def filler_sentences(n: int, seed: int) -> list[str]:
    """Deterministic business-prose filler for size benchmarks."""
    rng = random.Random(seed)
    subjects = [
        "The operations team",
        "Our finance group",
        "The platform squad",
        "Regional managers",
        "The support desk",
        "Warehouse staff",
        "The security council",
        "Product marketing",
    ]
    verbs = ["reviewed", "approved", "postponed", "documented", "escalated", "measured", "simplified"]
    objects = [
        "the quarterly forecast",
        "the onboarding checklist",
        "vendor contracts",
        "incident runbooks",
        "shipping latency",
        "the hiring plan",
        "capacity targets",
        "the training budget",
    ]
    tails = [
        "after the steering meeting.",
        "ahead of the audit.",
        "for the northern region.",
        "with input from customers.",
        "before the holiday freeze.",
        "using last year's baseline.",
    ]
    return [f"{rng.choice(subjects)} {rng.choice(verbs)} {rng.choice(objects)} {rng.choice(tails)}" for _ in range(n)]


def make_bench_files(out: Path, workdir: Path) -> None:
    """Larger files so ingestion timings are dominated by processing, not task dispatch."""
    pages = [" ".join(filler_sentences(30, seed=p)) for p in range(100)]
    make_pdf(out / "bench_report_100p.pdf", pages)

    lecture = workdir / "lecture.aiff"
    subprocess.run(["say", "-o", str(lecture), " ".join(filler_sentences(260, seed=7))], check=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(lecture),
            "-t",
            "600",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(out / "bench_lecture_10min.mp3"),
        ],
        check=True,
    )
    seconds = 300
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=1280x720:rate=25:duration={seconds}",
            "-i",
            str(out / "bench_lecture_10min.mp3"),
            "-t",
            str(seconds),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(out / "bench_video_5min.mp4"),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="samples/corpus")
    parser.add_argument("--large", action="store_true", help="also generate a ~1 GB video")
    parser.add_argument(
        "--bench", action="store_true", help="also generate samples/bench/ (100-page PDF, 10-min audio, 5-min video)"
    )
    args = parser.parse_args()

    for tool in ("say", "ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise SystemExit(f"'{tool}' is required (macOS `say`, ffmpeg).")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, pages in PDFS.items():
        make_pdf(out / name, pages)
    bar_chart().save(out / "bar_chart.png")
    sunset().save(out / "sunset.jpg", quality=92)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name, sentences in AUDIO.items():
            speech_file(sentences, out / name, work)
        make_video(out / "atlas_demo.mp4", work)
        if args.bench:
            bench_dir = out.parent / "bench"
            bench_dir.mkdir(exist_ok=True)
            make_bench_files(bench_dir, work)

    if args.large:
        large_dir = out.parent / "large"
        large_dir.mkdir(exist_ok=True)
        make_large_video(large_dir / "large_upload_test.mp4")
        print(f"large file: {large_dir / 'large_upload_test.mp4'}")

    for f in sorted(out.iterdir()):
        print(f"{f.stat().st_size / 1024:>10.1f} KB  {f}")


if __name__ == "__main__":
    main()
