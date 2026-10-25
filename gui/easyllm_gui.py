"""EasyLLM: a chat GUI for running 4-bit / 8-bit / fp16 Hugging Face models (up to ~13B) on a small GPU.

    python gui/easyllm_gui.py                 local, opens http://127.0.0.1:7860
    python gui/easyllm_gui.py --demo          fake model, no GPU needed
    python gui/easyllm_gui.py --tunnel ...    Colab: public Cloudflare quick tunnel protected by a key
"""
import argparse
import hmac
import json
import os
import platform
import secrets
import threading
import time
import warnings
import webbrowser
from importlib import metadata

from flask import Flask, Response, g, jsonify, request, send_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WEB = os.path.join(HERE, "web")
DEMO = os.environ.get("EASYLLM_DEMO") == "1"
DEMO_SLOW = float(os.environ.get("EASYLLM_DEMO_SLOW", "1"))  # demo only: stretch fake load times
ACCESS_KEY = os.environ.get("EASYLLM_KEY", "")
GIB = 1024 ** 3
try:
    VERSION = open(os.path.join(ROOT, "VERSION"), encoding="utf-8").read().strip()
except OSError:
    VERSION = "dev"

# --------------------------------------------------------------------------- log
# One buffer for everything: app events, Python prints/warnings, library output (HF download bars,
# bitsandbytes, CUDA...) and, via ingest(), the cloudflared tunnel.
import codecs
import logging
import re
import sys

_LOG, _LOG_LOCK, _SEQ = [], threading.Lock(), [0]
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _classify(text, default="info"):
    low = text.lower()
    if "traceback" in low or "error" in low or "exception" in low or "failed" in low:
        return "error"
    if "warn" in low:
        return "warn"
    return default


def _upsert(key, text, lvl):
    """Add a log line, or rewrite the one with this key (progress bars redraw the same line)."""
    with _LOG_LOCK:
        _SEQ[0] += 1
        if key is not None:
            for e in reversed(_LOG[-60:]):
                if e["k"] == key:
                    e.update(i=_SEQ[0], m=text, l=lvl)
                    return key
        key = _SEQ[0] if key is None else key
        _LOG.append({"i": _SEQ[0], "k": key, "t": time.strftime("%H:%M:%S"), "l": lvl, "m": text})
        del _LOG[:-1500]
        return key


def log(msg, lvl="info"):
    _upsert(None, str(msg), lvl)


class _LineBuffer:
    """Turns raw text chunks into log lines. \n ends a line, \r redraws the current one (tqdm)."""

    def __init__(self, default="info", prefix=""):
        self.default, self.prefix = default, prefix
        self.cur, self.key, self.lock = "", None, threading.Lock()

    def feed(self, text):
        with self.lock:
            for tok in re.split(r"(\r\n|\n|\r)", text):
                if tok in ("\n", "\r\n"):
                    self.cur, self.key = "", None
                elif tok == "\r":
                    self.cur = ""  # keep self.key: the next text replaces the same line
                elif tok:
                    self.cur += tok
                    clean = _ANSI.sub("", self.cur).rstrip()
                    if clean:
                        lvl = _classify(clean, self.default)
                        redraw = self.key is not None
                        self.key = _upsert(self.key, self.prefix + clean, "prog" if redraw and lvl == self.default else lvl)


class _Tee:
    """Wraps sys.stdout / sys.stderr so Python-level writes are logged and still shown."""

    def __init__(self, orig, lb):
        self._o, self._lb = orig, lb

    def write(self, s):
        try:
            self._lb.feed(s)
        except Exception:  # noqa: BLE001
            pass
        return self._o.write(s)

    def __getattr__(self, name):
        return getattr(self._o, name)


def _capture_fd(fd, lb):
    """Catch C-level output (CUDA, bitsandbytes, subprocesses) written straight to fd 1/2, and pass it on."""
    try:
        saved = os.dup(fd)
        r, w = os.pipe()
        os.dup2(w, fd)
        os.close(w)
    except OSError:
        return

    def pump():
        dec = codecs.getincrementaldecoder("utf-8")("replace")
        while True:
            try:
                data = os.read(r, 4096)
            except OSError:
                break
            if not data:
                break
            try:
                lb.feed(dec.decode(data))
            except Exception:  # noqa: BLE001
                pass
            try:
                os.write(saved, data)
            except OSError:
                pass

    threading.Thread(target=pump, daemon=True).start()


def capture_output():
    """Send everything this process prints to the log. Safe to call more than once."""
    if getattr(sys, "_easyllm_capture", False):
        return
    sys._easyllm_capture = True
    out, err = _LineBuffer("info"), _LineBuffer("info")
    # In Jupyter/Colab, print() goes to a kernel object, not to fd 1, so wrap it as well.
    # In a plain terminal print() ends up on fd 1/2, which the fd capture below already sees.
    if type(sys.stdout).__module__.startswith("ipykernel") or type(sys.stdout).__module__.startswith("colab"):
        sys.stdout, sys.stderr = _Tee(sys.stdout, out), _Tee(sys.stderr, err)
    else:
        try:
            sys.stdout.reconfigure(line_buffering=True)  # keep prints in order with stderr
        except Exception:  # noqa: BLE001
            pass
    _capture_fd(1, out)
    _capture_fd(2, err)
    # keep the dev server's per-request lines (GET /api/state ...) out of the log
    logging.getLogger("werkzeug").setLevel(logging.ERROR)


def ingest(text, source=""):
    """Add text from another source (e.g. the cloudflared process) to the log."""
    lb = _LineBuffer("info", prefix=f"[{source}] " if source else "")
    lb.feed(text if text.endswith("\n") else text + "\n")


def friendly(e):
    s = str(e)
    n = type(e).__name__
    if "out of memory" in s.lower() or n == "OutOfMemoryError":
        return "Out of GPU memory. Use 4-bit, pick a smaller model, or lower max new tokens."
    if "401" in s or "gated" in s.lower() or "GatedRepoError" in n:
        return "This model is gated. Accept its license on Hugging Face and add an HF token (Colab secret HF_TOKEN or the Advanced field)."
    if "404" in s or "RepositoryNotFound" in n:
        return "Model repo not found. Check the repo id."
    return s or n


def _ver(pkg):
    try:
        return metadata.version(pkg)
    except Exception:
        return "-"


# ------------------------------------------------------------------------ engine
def _dtype_kw(torch):
    """transformers >= 4.56 renamed torch_dtype to dtype; pass whichever this version understands."""
    try:
        import transformers
        from packaging.version import Version

        key = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    except Exception:  # noqa: BLE001
        key = "torch_dtype"
    return {key: torch.float16}


class Engine:
    def __init__(self):
        self.status = "idle"  # idle | loading | ready | generating | error
        self.phase = ""
        self.error = ""
        self.model_id = ""
        self.quant = ""
        self.model = None
        self.tok = None
        self.load_t0 = 0.0
        self.load_secs = 0.0
        self.stop_flag = threading.Event()
        self.was_stopped = False
        self.lock = threading.Lock()
        self.gen = {"prompt": 0, "tokens": 0, "tps": 0.0, "elapsed": 0.0, "max": 0}
        self._demo_vram = 0.35

    # ---- hardware
    def gpu(self):
        if DEMO:
            return {"name": "Tesla T4 (demo)", "total": 14.6, "used": self._demo_vram}
        try:
            import torch

            if not torch.cuda.is_available():
                return {"name": "No GPU", "total": 0.0, "used": 0.0}
            free, total = torch.cuda.mem_get_info()
            return {
                "name": torch.cuda.get_device_properties(0).name,
                "total": total / GIB,
                "used": (total - free) / GIB,
            }
        except Exception:
            return {"name": "Unknown", "total": 0.0, "used": 0.0}

    def system(self):
        return {
            "python": platform.python_version(),
            "torch": "demo" if DEMO else _ver("torch"),
            "transformers": "demo" if DEMO else _ver("transformers"),
            "bitsandbytes": "demo" if DEMO else _ver("bitsandbytes"),
        }

    # ---- load / unload
    def load(self, model_id, quant, token):
        with self.lock:
            if self.status in ("loading", "generating"):
                raise RuntimeError("Busy. Wait for the current job to finish.")
            self.status, self.error, self.phase = "loading", "", "Starting"
            self.model_id, self.quant = model_id, quant
            self.load_t0 = time.time()
        threading.Thread(target=self._load, args=(model_id, quant, token), daemon=True).start()

    def _free(self):
        self.model = None
        self.tok = None
        if DEMO:
            self._demo_vram = 0.35
            return
        import gc

        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    def _load(self, model_id, quant, token):
        try:
            log(f"Loading {model_id} ({quant})")
            self.phase = "Freeing memory"
            self._free()
            if DEMO:
                self.phase = "Loading tokenizer"
                time.sleep(0.5 * DEMO_SLOW)
                self.phase = "Downloading weights"
                print("Some weights of the model were not initialized from the checkpoint (demo)")
                for pct in range(0, 101, 20):
                    sys.stderr.write(f"\rDownloading model.safetensors: {pct:3d}%|{'#' * (pct // 10):<10}| {pct * 0.8:.1f}M/80.0M")
                    time.sleep(0.25 * DEMO_SLOW)
                sys.stderr.write("\n")
                warnings.warn("`torch_dtype` is deprecated! Use `dtype` instead! (demo)")
                self.phase = "Moving to GPU"
                os.write(2, b"CUDA: device 0 initialised (demo, written at C level)\n")
                time.sleep(0.6 * DEMO_SLOW)
                self._demo_vram = {"fp16": 6.2, "8bit": 8.1, "4bit": 5.6}.get(quant, 5.6)
                self.model, self.tok = object(), object()
            else:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

                token = token or os.environ.get("HF_TOKEN") or None
                self.phase = "Loading tokenizer"
                tok = AutoTokenizer.from_pretrained(model_id, token=token)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                kw = dict(device_map="cuda", token=token, **_dtype_kw(torch))  # T4: fp16, no bf16
                if quant == "4bit":
                    kw["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                elif quant == "8bit":
                    kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                self.phase = "Downloading and loading weights"
                model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
                model.eval()
                self.tok, self.model = tok, model
            self.load_secs = time.time() - self.load_t0
            self.phase = ""
            self.status = "ready"
            log(f"Model ready in {self.load_secs:.0f}s, VRAM {self.gpu()['used']:.1f} GiB", "ok")
        except Exception as e:  # noqa: BLE001
            self._free()
            self.status, self.error, self.phase = "error", friendly(e), ""
            log(f"Load failed: {self.error}", "error")

    def unload(self):
        with self.lock:
            if self.status in ("loading", "generating"):
                raise RuntimeError("Busy. Stop generation first.")
            self._free()
            self.status, self.error, self.phase, self.model_id = "idle", "", "", ""
        log("Model unloaded")

    # ---- generation
    def stream(self, messages, p):
        with self.lock:
            if self.status != "ready":
                raise RuntimeError("No model loaded.")
            self.status = "generating"
        self.stop_flag.clear()
        self.was_stopped = False
        try:
            yield from (self._demo_stream if DEMO else self._real_stream)(messages, p)
            self.was_stopped = self.stop_flag.is_set()  # user pressed Stop (read before cleanup sets the flag)
        finally:
            self.stop_flag.set()  # also stops the worker if the client disconnected
            if self.status == "generating":
                self.status = "ready"

    def _demo_stream(self, messages, p):
        last = messages[-1]["content"].lower()
        if "thread" in last or "process" in last:
            text = (
                "A **process** is an independent program with its own memory space; a **thread** is a unit of "
                "execution that lives *inside* a process and shares its memory.\n\n"
                "- Processes are isolated, so a crash in one rarely affects others.\n"
                "- Threads are cheaper to create and switch, but need locks to share data safely.\n"
                "- Use processes for CPU-bound parallelism, threads for I/O-bound work.\n\n"
                "```python\nimport threading\n\ndef work(n):\n    print('thread', n)\n\n"
                "for i in range(3):\n    threading.Thread(target=work, args=(i,)).start()\n```\n"
            )
        else:
            text = "This is a **demo** reply. Load a real model in Colab to get real answers.\n"
        words = text.split(" ")
        self.gen = {"prompt": 48, "tokens": 0, "tps": 0.0, "elapsed": 0.0, "max": int(p.get("max_new_tokens", 512))}
        t0 = time.time()
        for i, w in enumerate(words):
            if self.stop_flag.is_set():
                return
            time.sleep(0.04)
            self.gen["tokens"] += 2
            self.gen["elapsed"] = time.time() - t0
            self.gen["tps"] = self.gen["tokens"] / max(self.gen["elapsed"], 1e-3)
            yield w + ("" if i == len(words) - 1 else " ")

    def _inputs(self, messages):
        tok = self.tok
        try:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:  # template rejects the system role: fold it into the first user turn
            msgs = [dict(m) for m in messages]
            if msgs and msgs[0]["role"] == "system":
                sys_txt = msgs.pop(0)["content"]
                if msgs:
                    msgs[0]["content"] = sys_txt + "\n\n" + msgs[0]["content"]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")

    def _real_stream(self, messages, p):
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
        import torch

        flag = self.stop_flag

        class Stop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kw):
                return torch.full((input_ids.shape[0],), flag.is_set(), dtype=torch.bool, device=input_ids.device)

        class Counting(TextIteratorStreamer):
            n = 0
            first = True

            def put(self, value):
                if self.first:
                    self.first = False
                else:
                    self.n += int(value.numel())
                super().put(value)

        inputs = self._inputs(messages)
        n_prompt = int(inputs["input_ids"].shape[1])
        ctx = getattr(self.model.config, "max_position_embeddings", 4096) or 4096
        if n_prompt >= ctx - 16:
            raise RuntimeError(f"Conversation is too long for this model ({n_prompt} tokens). Clear the chat.")
        max_new = max(16, min(int(p.get("max_new_tokens", 512)), ctx - n_prompt))
        streamer = Counting(self.tok, skip_prompt=True, skip_special_tokens=True)
        kw = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new,
            repetition_penalty=float(p.get("repetition_penalty", 1.05)),
            pad_token_id=self.tok.pad_token_id,
            stopping_criteria=StoppingCriteriaList([Stop()]),
        )
        temp = float(p.get("temperature", 0.7))
        if temp > 0:
            kw.update(do_sample=True, temperature=temp, top_p=float(p.get("top_p", 0.9)))
        else:
            kw.update(do_sample=False)

        err = []

        def run():
            try:
                with torch.inference_mode():
                    self.model.generate(**kw)
            except Exception as e:  # noqa: BLE001
                err.append(e)
                streamer.end()

        self.gen = {"prompt": n_prompt, "tokens": 0, "tps": 0.0, "elapsed": 0.0, "max": max_new}
        t0 = time.time()
        th = threading.Thread(target=run, daemon=True)
        th.start()
        for chunk in streamer:
            self.gen["tokens"] = streamer.n
            self.gen["elapsed"] = time.time() - t0
            self.gen["tps"] = streamer.n / max(self.gen["elapsed"], 1e-3)
            yield chunk
        th.join()
        self.gen["tokens"] = streamer.n
        self.gen["elapsed"] = time.time() - t0
        self.gen["tps"] = streamer.n / max(self.gen["elapsed"], 1e-3)
        if err:
            raise err[0]


eng = Engine()
app = Flask(__name__)


# ----------------------------------------------------------------------- access
@app.before_request
def guard():
    if not ACCESS_KEY:
        return None
    ck = request.cookies.get("easyllm_key", "")
    if hmac.compare_digest(ck, ACCESS_KEY):
        return None
    qk = request.args.get("key", "")
    if qk and hmac.compare_digest(qk, ACCESS_KEY):
        g.set_cookie = True
        return None
    return Response("Forbidden. Open the full link printed by the notebook (it ends with ?key=...).", 403)


@app.after_request
def after(resp):
    if g.get("set_cookie"):
        secure = request.headers.get("X-Forwarded-Proto", "") == "https"
        resp.set_cookie("easyllm_key", ACCESS_KEY, httponly=True, samesite="Lax", secure=secure, max_age=86400)
    resp.headers["Cache-Control"] = "no-store"
    return resp


# -------------------------------------------------------------------------- API
@app.get("/")
def index():
    return send_file(os.path.join(WEB, "index.html"))


@app.get("/api/state")
def state():
    gpu = eng.gpu()
    return jsonify(
        status=eng.status,
        phase=eng.phase,
        error=eng.error,
        model=eng.model_id,
        quant=eng.quant,
        load_secs=(time.time() - eng.load_t0) if eng.status == "loading" else eng.load_secs,
        gpu=gpu,
        gen=eng.gen,
        demo=DEMO,
        version=VERSION,
        system=eng.system(),
        log_seq=_SEQ[0],
    )


@app.get("/api/log")
def get_log():
    since = int(request.args.get("since", 0))
    with _LOG_LOCK:
        return jsonify(lines=[x for x in _LOG if x["i"] > since], seq=_SEQ[0])


@app.post("/api/load")
def load():
    d = request.get_json(force=True)
    model = (d.get("model") or "").strip()
    quant = d.get("quant", "4bit")
    if not model or quant not in ("fp16", "8bit", "4bit"):
        return jsonify(error="Pick a model and precision."), 400
    try:
        eng.load(model, quant, (d.get("token") or "").strip())
    except RuntimeError as e:
        return jsonify(error=str(e)), 409
    return jsonify(ok=True)


@app.post("/api/unload")
def unload():
    try:
        eng.unload()
    except RuntimeError as e:
        return jsonify(error=str(e)), 409
    return jsonify(ok=True)


@app.post("/api/stop")
def stop():
    eng.stop_flag.set()
    return jsonify(ok=True)


@app.post("/api/chat")
def chat():
    d = request.get_json(force=True)
    msgs = d.get("messages") or []
    if eng.status != "ready":
        return jsonify(error="No model loaded." if eng.status != "generating" else "Already generating."), 409
    if not msgs or msgs[-1].get("role") != "user":
        return jsonify(error="Last message must be from the user."), 400
    params = d.get("params") or {}

    def gen():
        try:
            for piece in eng.stream(msgs, params):
                yield json.dumps({"t": piece}) + "\n"
            yield json.dumps({"done": True, "stopped": eng.was_stopped, "stats": eng.gen}) + "\n"
        except Exception as e:  # noqa: BLE001
            log(f"Generation failed: {friendly(e)}", "error")
            yield json.dumps({"error": friendly(e)}) + "\n"

    return Response(gen(), mimetype="application/x-ndjson", headers={"X-Accel-Buffering": "no"})


# --------------------------------------------------------------------------- main
def main():
    global DEMO, DEMO_SLOW, ACCESS_KEY
    ap = argparse.ArgumentParser(description="EasyLLM: chat GUI for LLMs on a small GPU")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    ap.add_argument("--tunnel", action="store_true", help="expose the GUI through a Cloudflare quick tunnel (Colab)")
    ap.add_argument("--url-file", help="write the public link here once the tunnel is up")
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, ".cache"), help="where the cloudflared binary is cached")
    ap.add_argument("--key", default="", help="access key required in the link / cookie (generated when --tunnel is used)")
    ap.add_argument("--demo", action="store_true", help="simulate a model, no GPU needed")
    ap.add_argument("--demo-slow", type=float, default=0.0, help="demo only: stretch the fake load time")
    args = ap.parse_args()

    DEMO = DEMO or args.demo
    if args.demo_slow:
        DEMO_SLOW = args.demo_slow
    if args.tunnel and not args.key:
        args.key = secrets.token_urlsafe(16)
    ACCESS_KEY = args.key or ACCESS_KEY

    capture_output()
    log(f"EasyLLM v{VERSION}" + (" (demo mode)" if DEMO else ""))

    from werkzeug.serving import make_server

    srv = make_server(args.host, args.port, app, threaded=True)
    print(f"EasyLLM is running at http://{args.host}:{args.port}/", flush=True)

    if args.tunnel:
        sys.path.insert(0, HERE)
        import tunnel

        def on_url(url):
            log(f"Public link ready: {url}", "ok")  # the access key is never logged
            if args.url_file:
                with open(args.url_file, "w", encoding="utf-8") as f:
                    f.write(url + "/" + (f"?key={ACCESS_KEY}" if ACCESS_KEY else ""))

        tunnel.start(args.port, args.cache_dir, on_url,
                     on_error=lambda m: log(m, "error"),
                     on_line=lambda ln: ingest(ln, "cloudflared"))
    elif not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(f"http://{args.host}:{args.port}/")).start()

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
