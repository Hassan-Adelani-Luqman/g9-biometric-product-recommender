"""Multimodal biometric product recommender: command-line app.

Owner: Peter Michael Angelo Rucakibungo

Chains the three trained models in the order of the brief's flow diagram:

    [1/3] Face image  -> identify_face()         no face / not an enrolled member -> ACCESS DENIED
    [2/3] Member      -> customer_id             (config.MEMBER_TO_CUSTOMER)
          customer_id -> predict_for_customer()  result is computed but HELD, not shown
    [3/3] Voice clip  -> verify_voice(expected_member=face member)
                                                 voice not recognised             -> ACCESS DENIED
                                                 voice is a different member      -> ACCESS DENIED (identity mismatch)
    APPROVED -> display the predicted product and the top 3 probabilities

The cross-modal check in step 3 is the core of the multimodal logic: the voice
must belong to the same person the face step recognised, so a stolen photo
plus someone else's voice is not enough.

Run from the project root:
    python src/app.py --face <image> --voice <clip.wav>   # one transaction
    python src/app.py                                     # interactive: asks for each file in turn
    python src/app.py --demo                              # runs the 6 required scenarios

Exit codes: 0 = access granted (demo: every scenario behaved as expected),
1 = access denied (demo: a scenario did not), 2 = missing/invalid file or model.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402  (light: paths and constants only)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3"}
WIDTH = 34  # column where stage results start


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _supports(text: str) -> bool:
    try:
        text.encode(sys.stdout.encoding or "ascii")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


OK, NO = ("✅", "❌") if _supports("✅❌…") else ("[OK]", "[X]")
DOTS = "…" if OK == "✅" else "..."


def _stage(num: int, title: str) -> None:
    print(f"[{num}/3] {title}{DOTS}".ljust(WIDTH), end="", flush=True)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
class InputError(Exception):
    """A missing or unreadable input file, or a missing model. Exit code 2."""


@dataclass
class Components:
    ip: object   # src.image_pipeline
    tf: object   # src.train_face_model
    tv: object   # src.train_voice_model
    tp: object   # src.train_product_model


_COMPONENTS: Components | None = None


def load_components(verbose: bool = True) -> Components:
    """Import the pipelines and load all models once (later calls reuse them)."""
    global _COMPONENTS
    if _COMPONENTS is not None:
        return _COMPONENTS
    t0 = time.perf_counter()
    if verbose:
        print(f"Loading models{DOTS}", end=" ", flush=True)

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    import cv2
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # hide OpenCV backend notices

    from src import image_pipeline as ip
    from src import train_face_model as tf
    from src import train_product_model as tp
    from src import train_voice_model as tv

    hints = {config.FACE_MODEL_PATH: "python src/train_face_model.py",
             config.VOICE_MODEL_PATH: "python src/train_voice_model.py",
             config.PRODUCT_MODEL_PATH: "python src/train_product_model.py",
             config.MERGED_CSV: "python src/data_merge.py"}
    missing = [f"{_rel(p)} (create it with: {cmd})" for p, cmd in hints.items() if not p.exists()]
    if missing:
        if verbose:
            print()
        raise InputError("missing model or data file:\n  " + "\n  ".join(missing))

    if verbose and not all((ip.PRETRAINED_DIR / name).exists() for name, _, _ in ip.PRETRAINED.values()):
        print("\nFirst run: downloading the pretrained face models (about 39 MB, once).")
    ip.ensure_models()                       # downloads the face detector/embedder on first run
    ip._detector(), ip._recognizer()         # noqa: B018  (build the ONNX networks now, not mid-transaction)
    tf.load_model(), tv.load_model(), tp.load_model()
    import numpy as np                       # warm up the audio feature code (first call is slow)
    tv.ap.extract_features(np.random.default_rng(0).normal(0, 0.1, tv.ap.SR).astype(np.float32))
    _COMPONENTS = Components(ip=ip, tf=tf, tv=tv, tp=tp)
    if verbose:
        print(f"done ({time.perf_counter() - t0:.1f} s)")
    return _COMPONENTS


def check_file(path: str | Path, kind: str) -> Path:
    """Return a usable path, or raise InputError with a message a user can act on."""
    exts = IMAGE_EXTS if kind == "face" else AUDIO_EXTS
    p = Path(str(path).strip().strip("'\"")).expanduser()
    if not str(path).strip():
        raise InputError(f"no {kind} file given")
    if not p.exists():
        raise InputError(f"{kind} file not found: {p}")
    if not p.is_file():
        raise InputError(f"{kind} path is a folder, not a file: {p}")
    if p.suffix.lower() not in exts:
        raise InputError(f"{kind} file must be one of {', '.join(sorted(exts))} (got '{p.suffix or 'no extension'}'): {p}")
    return p


# ---------------------------------------------------------------------------
# One transaction
# ---------------------------------------------------------------------------
@dataclass
class Outcome:
    approved: bool
    stage: str                      # "face", "voice" (where it was denied) or "approved"
    reason: str
    member: str | None = None
    customer_id: int | None = None
    product: str | None = None
    top3: list[tuple[str, float]] = field(default_factory=list)
    face: dict | None = None
    voice: dict | None = None
    seconds: dict[str, float] = field(default_factory=dict)


def run_transaction(face: str | Path, voice: str | Path | Callable[[], Path], *,
                    face_label: str | None = None, voice_label: str | None = None) -> Outcome:
    """Run the full flow on one face image and one voice clip and print each stage.

    ``voice`` may be a callable that returns a path: interactive mode uses it
    to ask for the clip only after the face has been accepted.
    """
    c = load_components()
    face = check_file(face, "face")
    if not callable(voice):
        voice = check_file(voice, "voice")

    print(f"  Face : {face_label or _rel(face)}")
    if not callable(voice):
        print(f"  Voice: {voice_label or _rel(voice)}")

    # [1/3] Facial recognition --------------------------------------------
    _stage(1, "Facial recognition")
    t0 = time.perf_counter()
    try:
        f = c.tf.identify_face(face)
    except Exception as e:  # unreadable / corrupt image
        print(f"{NO} could not read the image")
        raise InputError(f"could not read face image {face.name} ({type(e).__name__}: "
                         f"is it a real image file?)") from e
    t_face = time.perf_counter() - t0
    if not f["accepted"]:
        print(f"{NO} {f['reason']}  ({t_face:.2f} s)")
        _verdict(False, "face step", "Voice check not run; no product shown.")
        return Outcome(False, "face", f["reason"], face=f, seconds={"face": t_face})
    member = f["member"]
    print(f"{OK} {member} (similarity {f['similarity']:.2f} >= {f['threshold']:.2f})  ({t_face:.2f} s)")

    # [2/3] Product recommendation (held) ---------------------------------
    _stage(2, "Product recommendation")
    t0 = time.perf_counter()
    cid = config.MEMBER_TO_CUSTOMER[member]
    product, probs = c.tp.predict_for_customer(cid)
    t_prod = time.perf_counter() - t0
    print(f"computed for customer {cid}, held until the voice check passes  ({t_prod:.2f} s)")

    # [3/3] Voiceprint verification ---------------------------------------
    if callable(voice):
        voice = voice()
    _stage(3, "Voiceprint verification")
    t0 = time.perf_counter()
    try:
        v = c.tv.verify_voice(voice, expected_member=member)
    except Exception as e:  # unreadable / corrupt / empty audio
        print(f"{NO} could not read the audio")
        raise InputError(f"could not read voice clip {Path(voice).name} ({type(e).__name__}: "
                         f"is it a real audio file?)") from e
    t_voice = time.perf_counter() - t0
    seconds = {"face": t_face, "product": t_prod, "voice": t_voice}
    if not v["approved"]:
        print(f"{NO} {v['reason']}  ({t_voice:.2f} s)")
        _verdict(False, "voice step", "The held prediction is discarded; no product shown.")
        return Outcome(False, "voice", v["reason"], member=member, customer_id=cid,
                       face=f, voice=v, seconds=seconds)
    print(f"{OK} {member} (confidence {v['confidence']:.2f} >= {v['threshold']:.2f})  ({t_voice:.2f} s)")

    # Display --------------------------------------------------------------
    top3 = list(probs.items())[:3]
    _verdict(True, None, f"Recommended product for {config.MEMBER_NAMES[member]} (customer {cid}): {product}")
    for i, (cat, p) in enumerate(top3, 1):
        print(f"     {i}. {cat:<12}{p:6.1%}")
    print(f"     Total time {sum(seconds.values()):.2f} s")
    return Outcome(True, "approved", "access granted", member=member, customer_id=cid,
                   product=product, top3=top3, face=f, voice=v, seconds=seconds)


def _verdict(approved: bool, where: str | None, detail: str) -> None:
    if approved:
        print(f"{OK} ACCESS GRANTED. {detail}")
    else:
        print(f"{NO} ACCESS DENIED at the {where}. {detail}")


# ---------------------------------------------------------------------------
# Demo: the 6 required scenarios
# ---------------------------------------------------------------------------
RAW_AUDIO = config.AUDIO_PIPELINE_DIR / "data" / "audio" / "raw"


@dataclass
class Scenario:
    title: str
    face: Path
    voice: Path
    expect: str                    # "approved", "face" or "voice"
    expect_text: str
    short: str                     # label for the summary table
    reason_contains: str = ""      # extra check on the denial reason
    face_label: str | None = None
    voice_label: str | None = None


def demo_scenarios(workdir: Path) -> list[Scenario]:
    """Scenarios 1-6. Scenario 6's augmented inputs are generated into ``workdir``."""
    img = lambda m, e: config.IMAGES_DIR / m / f"{m}_{e}_01.jpg"   # noqa: E731
    clip = lambda m, p: RAW_AUDIO / f"{m}_{p}_1.wav"                # noqa: E731
    face6, voice6 = make_augmented_inputs(img("peter", "smiling"), clip("peter", "confirm_transaction"), workdir)
    return [
        Scenario("Authorised member (Hassan)",
                 img("hassan", "neutral"), clip("hassan", "yes_approve"),
                 "approved", "product displayed", "Hassan: own face + own voice"),
        Scenario("Emma with a different expression and phrase",
                 img("emma", "smiling"), clip("emma", "confirm_transaction"),
                 "approved", "product displayed", "Emma: other expression + phrase"),
        Scenario("Unauthorised face (non-member)",
                 config.UNAUTHORIZED_FACES_DIR / "neutral.png", clip("hassan", "yes_approve"),
                 "face", "denied at face step", "Non-member face"),
        Scenario("Parfait's face + unauthorised voice",
                 img("parfait", "neutral"), config.UNAUTHORIZED_VOICES_DIR / "approve.wav",
                 "voice", "denied at voice step", "Parfait's face + non-member voice"),
        Scenario("Parfait's face + Peter's voice",
                 img("parfait", "surprised"), clip("peter", "yes_approve"),
                 "voice", "denied: identity mismatch", "Parfait's face + Peter's voice",
                 reason_contains="identity mismatch"),
        Scenario("Robustness: augmented face and voice of Peter",
                 face6, voice6, "approved", "product displayed", "Peter: augmented face + voice",
                 face_label="data/images/peter/peter_smiling_01.jpg, rotated -15 degrees (generated)",
                 voice_label="Audio_features/.../peter_confirm_transaction_1.wav + background noise at 20 dB SNR (generated)"),
    ]


def make_augmented_inputs(face_src: Path, voice_src: Path, workdir: Path) -> tuple[Path, Path]:
    """Rotate the photo and add fresh background noise to the clip, using the
    same augmentation code as the image and audio pipelines. The noise uses a
    different random seed from the one used to build the training features."""
    import cv2
    import numpy as np
    import soundfile as sf
    c = load_components(verbose=False)
    ap = c.tv.ap
    face_out = workdir / "peter_smiling_01__rotate_m15.png"
    cv2.imwrite(str(face_out), c.ip.AUGMENTATIONS["rotate_m15"](c.ip.load_image(face_src)))
    voice_out = workdir / "peter_confirm_transaction_1__noise20db.wav"
    noisy = ap.add_noise(ap.load_clip(voice_src), snr_db=20, rng=np.random.default_rng(7))
    sf.write(voice_out, noisy, ap.SR, subtype="FLOAT")
    return face_out, voice_out


def outcome_text(o: Outcome) -> str:
    if o.approved:
        return f"product displayed ({o.product})"
    if o.stage == "voice" and "identity mismatch" in o.reason:
        return "denied at voice (identity mismatch)"
    return f"denied at {o.stage} step"


def run_demo() -> list[dict]:
    """Run every scenario, print each transaction and a summary table.

    Returns one dict per scenario (used by notebook 05).
    """
    load_components()
    results = []
    with tempfile.TemporaryDirectory(prefix="g9_demo_") as tmp:
        scenarios = demo_scenarios(Path(tmp))
        for i, s in enumerate(scenarios, 1):
            print(f"\n{'=' * 78}\nScenario {i}: {s.title}\nExpected: {s.expect_text}\n{'-' * 78}")
            o = run_transaction(s.face, s.voice, face_label=s.face_label, voice_label=s.voice_label)
            ok = o.stage == s.expect and s.reason_contains in o.reason
            results.append({"#": i, "scenario": s.title, "short": s.short, "expected": s.expect_text,
                            "result": outcome_text(o), "as_expected": ok, "reason": o.reason,
                            "seconds": round(sum(o.seconds.values()), 2), "outcome": o})

    print(f"\n{'=' * 78}\nSummary\n{'-' * 78}")
    print(f"   {'Scenario':<37}{'Expected':<28}{'Result'}")
    for r in results:
        print(f"{r['#']}. {r['short']:<37}{r['expected']:<28}{OK if r['as_expected'] else NO} {r['result']}")
    n_ok = sum(r["as_expected"] for r in results)
    print(f"{'-' * 78}\n{n_ok}/{len(results)} scenarios behaved as expected.")
    return results


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------
class Quit(Exception):
    pass


def ask_file(prompt: str, kind: str) -> Path:
    """Ask until the user gives a usable file, or 'q' to quit."""
    while True:
        try:
            answer = input(prompt)
        except EOFError:
            print()
            raise Quit from None
        if answer.strip().lower() in {"q", "quit", "exit"}:
            raise Quit
        try:
            return check_file(answer, kind)
        except InputError as e:
            print(f"  {NO} {e}. Try again, or type q to quit.")


def interactive() -> int:
    print("Multimodal Biometric Product Recommender: interactive mode")
    print("Enter file paths (you can drag a file into the terminal). Type q to quit.\n")
    try:
        load_components()
        while True:
            face = ask_file("Face image path: ", "face")
            try:
                run_transaction(face, lambda: ask_file(
                    'Voice clip path (say "Yes, approve" or "Confirm transaction"): ', "voice"))
            except InputError as e:
                print(f"{NO} {e}")
            again = input("\nAnother transaction? [y/N]: ").strip().lower()
            if again not in {"y", "yes"}:
                return 0
            print()
    except (Quit, EOFError, KeyboardInterrupt):
        print("\nBye.")
        return 0


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Face -> product prediction (held) -> voice -> display, as in the project brief.",
        epilog="With no arguments, the app asks for each file in turn.")
    parser.add_argument("--face", help="face image (.jpg, .png, ...)")
    parser.add_argument("--voice", help="voice clip (.wav, ...) saying 'Yes, approve' or 'Confirm transaction'")
    parser.add_argument("--demo", action="store_true", help="run the 6 required demo scenarios")
    args = parser.parse_args(argv)

    if args.demo and (args.face or args.voice):
        parser.error("--demo cannot be combined with --face/--voice")
    if bool(args.face) != bool(args.voice):
        parser.error("give both --face and --voice (or neither, for interactive mode)")

    try:
        if args.demo:
            results = run_demo()
            return 0 if all(r["as_expected"] for r in results) else 1
        if args.face:
            # check both files before starting, so a typo is not found halfway through
            face, voice = check_file(args.face, "face"), check_file(args.voice, "voice")
            print("Multimodal Biometric Product Recommender")
            load_components()
            return 0 if run_transaction(face, voice).approved else 1
        return interactive()
    except InputError as e:
        sys.stdout.flush()  # keep the stage lines above the error message
        print(f"{NO} Error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
