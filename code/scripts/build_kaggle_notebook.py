import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
cells = []
def md(s):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': s.strip() + '\n'})
def code(s):
    cells.append({'cell_type': 'code', 'metadata': {}, 'source': s.strip() + '\n', 'outputs': [], 'execution_count': None})

md('''# LifeMate — một notebook chạy trọn pipeline trên Kaggle

**Cách chạy:** Add Input bộ dữ liệu đã giải nén, bật **Internet**, chọn một GPU rồi Run All. Nếu chọn T4 x2, notebook vẫn chỉ dùng GPU được chỉ định ở `GPU_INDEX`.
Không cần clone GitHub. Mặc định notebook fine-tune LLM trước trên TRAIN rồi dùng adapter vừa train. Notebook chứa toàn bộ mã nguồn: audio tiếng Việt → faster-whisper → Qwen → validate/call trên `ToolEnv` thật → submission + logs.

Bộ dữ liệu cần có `<split>.jsonl`, `<split>_tools_catalog.json`, audio và **zaic_tools đúng phiên bản của split**. Notebook chỉ đọc manifest/catalog/audio; trạng thái ứng dụng do `ToolEnv` quản lý.

Chỉnh cấu hình bên dưới. `LIMIT=0` chạy **toàn bộ split**. Model tải từ Hugging Face trước khi suy luận; có thể dùng model gắn sẵn ở `/kaggle/input`. Model ASR mặc định là `Systran/faster-whisper-medium`, dùng trực tiếp định dạng CTranslate2.

ASR chạy toàn bộ split (bao gồm audio lịch sử), lưu transcript rồi thoát tiến trình và giải phóng GPU. LLM sau đó đọc cache; không nạp Whisper cùng lúc. Mặc định không giới hạn thời gian mỗi mẫu (`float("inf")`); vẫn giữ giới hạn tool calls và token/context ≤16.384.

Kết quả lưu trong `/kaggle/working/lifemate_runs/<RUN_NAME>/`: `transcripts.jsonl`, `transcripts.manifest.json`, `asr.log`, `submission.jsonl`, `diagnostics.jsonl`, `inference.log`, `setup.log`, `models.log`, `validation.json`, `run_config.json`, `environment.txt`, và `lifemate_artifacts.zip`. Transcript/logs có nội dung người dùng; kiểm tra trước khi chia sẻ. Đây không phải chế độ bảo đảm giới hạn 30 giây/mẫu của cuộc thi.''')

code('''from pathlib import Path
import os, sys, json, subprocess, datetime, shutil, zipfile, hashlib

# ===== CẤU HÌNH — thông thường chỉ cần chỉnh DATA_ROOT =====
DATA_ROOT = None  # None: tự tìm trong /kaggle/input; hoặc "/kaggle/input/<dataset>/train+public_test"
SPLIT = "public_test"  # train / public_test / private_test
LIMIT = 0             # 0 = toàn bộ split; đặt 5 để smoke test
REQUIRE_GPU = True    # False nếu muốn chạy CPU
GPU_INDEX = 0         # GPU cho các bước setup/train
RUN_ASR = True        # False: chỉ dùng TRANSCRIPT_CACHE đã hoàn tất, không chạy lại ASR
TRANSCRIPT_CACHE = None  # None: OUT/transcripts.jsonl; đặt đường dẫn cache để tái sử dụng
REFRESH_TRANSCRIPTS = False  # True: thay cache hoàn tất nếu đổi ASR/dữ liệu/cấu hình

# Fine-tune trước rồi tự nạp adapter cho inference.
TRAIN_LLM = True
TRAIN_BACKEND = "unsloth"  # backend mặc định; cần CUDA
TRAIN_USE_ASR = False  # bật nếu train/history thiếu transcript
TRAIN_MISSING_TOOL_POLICY = "error"  # schema thiếu sẽ được lấy qua action inspect; không mặc định bỏ mẫu
TRAIN_DATA_ROOT = None  # None: tự tìm train/train.jsonl trong /kaggle/input
LOCAL_ADAPTER = None   # TRAIN_LLM=False + đường dẫn adapter để dùng bản đã train
TRAIN_EPOCHS = 3
TRAIN_MAX_STEPS = -1   # -1: theo epochs; đặt 2 cho kiểm tra ngắn, không phải model đủ chất lượng
TRAIN_MAX_LENGTH = 8192
TRAIN_GRAD_ACCUMULATION = 8
TRAIN_VALIDATION_FRACTION = 0.1

ASR_REPO = "Systran/faster-whisper-medium"
ASR_FORMAT = "ctranslate2"  # Systran/faster-whisper-medium đã ở định dạng CTranslate2
LLM_REPO = "Qwen/Qwen3-4B-Instruct-2507"
ASR_REVISION = None  # None: resolve main thành SHA và ghi lại trong model_manifest.json
LLM_REVISION = None
LOCAL_ASR = None     # thư mục CTranslate2 có model.bin; bỏ qua tải/chuyển ASR khi đã có
LOCAL_LLM = None     # thư mục Transformers có config.json + tokenizer + trọng số
HF_SECRET_NAME = None  # tùy chọn: tên Kaggle Secret chứa HF token (public models không cần)

SAMPLE_SECONDS = float("inf")  # không giới hạn thời gian mỗi mẫu
MAX_INPUT_TOKENS = 6000
MAX_NEW_TOKENS = 192
MAX_CALLS = 8
MAX_INVALID = 3
MAX_TOOL_DETAILS = 35
ASR_BEAM = 1
SEED = 42
FAIL_FAST = False      # False: lưu lỗi từng mẫu và tiếp tục, báo lỗi ở bước kiểm chứng
USE_TRAIN_TRANSCRIPT = False  # chỉ được bật cho split train
RUN_TRAIN_SCORE = True        # proxy F1, chỉ chạy khi SPLIT == "train"
RUN_NAME = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")

APP = Path("/kaggle/working/lifemate_app")
MODEL_ROOT = Path("/kaggle/working/lifemate_models")
OUT = Path("/kaggle/working/lifemate_runs") / RUN_NAME
for folder in (APP / "lifemate", APP / "tests", MODEL_ROOT, OUT):
    folder.mkdir(parents=True, exist_ok=True)

def run_logged(command, log_path, *, env=None, cwd=None):
    # Không log env/token hoặc dùng shell=True. Giữ đúng exit status của subprocess.
    with Path(log_path).open("a", encoding="utf-8", buffering=1) as log:
        with subprocess.Popen([str(x) for x in command], cwd=cwd, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, errors="replace", bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            status = process.wait()
    if status:
        raise RuntimeError(f"Command failed (exit {status}); xem {log_path}")

assert SPLIT in {"train", "public_test", "private_test"}
assert isinstance(LIMIT, int) and LIMIT >= 0
assert not USE_TRAIN_TRANSCRIPT or SPLIT == "train"
assert ASR_FORMAT in {"ctranslate2", "transformers"}
assert isinstance(GPU_INDEX, int) and GPU_INDEX >= 0
assert MAX_INPUT_TOKENS + MAX_NEW_TOKENS <= 16384
print("Artifacts:", OUT)
''')

md('''## 1. Dependency và runtime

Giữ nguyên PyTorch/CUDA do Kaggle cung cấp, không cài đè torch. Các thư viện ASR/LLM được cài trong tiến trình riêng trước khi dùng. CTranslate2 cần CUDA 12 + cuDNN 9 khi chạy GPU; bước suy luận thực tế sẽ kiểm tra việc nạp model. Nếu GPU memory không đủ, giảm `MAX_INPUT_TOKENS`, chọn model nhỏ hơn hoặc đổi GPU.''')
# Embed the shared manifest so the standalone notebook uses the same constraints.
packages = [
    line.strip() for line in (ROOT / "code" / "requirements.txt").read_text().splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]
code('''packages = ''' + repr(packages) + '''
# Kiểm tra torch trước: tránh pip tự kéo một bản CUDA khác với Kaggle.
run_logged([sys.executable, "-c", "import torch; print('Kaggle torch:', torch.__version__)"], OUT / "setup.log")
# Constrain the preinstalled CUDA stack so dependency resolution cannot replace it.
import importlib.metadata
stack = []
for distribution in ("torch", "torchvision", "triton"):
    try:
        stack.append(f"{distribution}=={importlib.metadata.version(distribution)}")
    except importlib.metadata.PackageNotFoundError:
        pass
constraint = OUT / "kaggle_cuda_constraints.txt"
constraint.write_text("\\n".join(stack) + "\\n", encoding="utf-8")
run_logged([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--constraint", constraint,
            "--upgrade-strategy", "only-if-needed", *packages], OUT / "setup.log")
BASE_ENV = os.environ.copy()
BASE_ENV.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
                TOKENIZERS_PARALLELISM="false", CUBLAS_WORKSPACE_CONFIG=":4096:8")
BASE_ENV["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)
# Expose pip-provided NVIDIA libraries to the child process's dynamic loader.
import site
lib_dirs = []
for site_dir in site.getsitepackages():
    lib_dirs.extend(str(p) for p in Path(site_dir).glob("nvidia/*/lib") if p.is_dir())
BASE_ENV["LD_LIBRARY_PATH"] = os.pathsep.join(lib_dirs + [BASE_ENV.get("LD_LIBRARY_PATH", "")])
hardware_script = """
import json, torch, ctranslate2
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from faster_whisper import WhisperModel
gpu = torch.cuda.is_available()
print(json.dumps(dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=gpu,
    gpu_name=torch.cuda.get_device_name(0) if gpu else None,
    bf16=torch.cuda.is_bf16_supported() if gpu else False,
    ct2=ctranslate2.__version__)))
"""
hardware = subprocess.run([sys.executable, "-c", hardware_script], env=BASE_ENV,
                          capture_output=True, text=True)
(OUT / "hardware.log").write_text(hardware.stdout + hardware.stderr, encoding="utf-8")
if hardware.returncode:
    raise RuntimeError("Import dependency thất bại; xem hardware.log:\\n" + hardware.stderr[-4000:])
HW = json.loads(hardware.stdout.strip().splitlines()[-1])
print(HW)
if REQUIRE_GPU and not HW["gpu"]:
    raise RuntimeError("Bật GPU trong Kaggle Settings rồi Run All, hoặc đặt REQUIRE_GPU=False.")
DEVICE = "cuda" if HW["gpu"] else "cpu"
LLM_DTYPE = ("bfloat16" if HW["bf16"] else "float16") if HW["gpu"] else "float32"
ASR_COMPUTE = "float16" if HW["gpu"] else "int8"
run_logged([sys.executable, "-m", "pip", "freeze"], OUT / "environment.txt")
''')

md('''## 2. Tìm và kiểm tra dữ liệu

Nếu có nhiều bản của cùng split, đặt `DATA_ROOT` rõ ràng. `DATA_ROOT` có thể là thư mục cha hoặc chính thư mục split. Không cần chép dữ liệu từ vùng input chỉ đọc. Chỉ đọc manifest và kiểm tra đường dẫn audio; không duyệt nội dung các thư mục `*_environment`.''')
code('''def discover_split(root, split):
    candidates = []
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not (
            d.endswith("_environment") or d in {"audio", "__MACOSX", ".git"}
        ))
        if f"{split}.jsonl" in files:
            candidates.append(Path(current))
            dirs[:] = []
    return sorted(set(p.resolve() for p in candidates))

if DATA_ROOT:
    location = Path(DATA_ROOT).expanduser().resolve()
    if (location / SPLIT / f"{SPLIT}.jsonl").is_file():
        SPLIT_DIR = location / SPLIT
    elif (location / f"{SPLIT}.jsonl").is_file():
        SPLIT_DIR = location
    else:
        raise FileNotFoundError(f"Không tìm thấy {SPLIT}.jsonl trong {location}")
else:
    candidates = discover_split("/kaggle/input", SPLIT)
    if len(candidates) != 1:
        raise ValueError(f"Tìm thấy {len(candidates)} split: {candidates}. Đặt DATA_ROOT rõ ràng.")
    SPLIT_DIR = candidates[0]
CATALOG_PATH = SPLIT_DIR / f"{SPLIT}_tools_catalog.json"
if not CATALOG_PATH.is_file():
    raise FileNotFoundError(CATALOG_PATH)
if not ((SPLIT_DIR / "zaic_tools.py").is_file() or (SPLIT_DIR / "zaic_tools").is_dir()):
    raise FileNotFoundError(f"Cần zaic_tools đi kèm split tại {SPLIT_DIR}, không dùng package khác trên pip.")
ITEMS = []
with (SPLIT_DIR / f"{SPLIT}.jsonl").open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            ITEMS.append(json.loads(line))
if LIMIT:
    ITEMS = ITEMS[:LIMIT]
EXPECTED_IDS = [item["id"] for item in ITEMS]
if not ITEMS or len(set(EXPECTED_IDS)) != len(EXPECTED_IDS):
    raise ValueError("Manifest rỗng hoặc có id trùng trong phần cần chạy.")
for item in ITEMS:
    audio = (SPLIT_DIR / item["audio"]).resolve()
    if not audio.is_relative_to(SPLIT_DIR.resolve()) or not audio.is_file():
        raise FileNotFoundError(f"Audio không hợp lệ cho {item['id']}: {audio}")
print(f"Split: {SPLIT_DIR}; samples cần chạy: {len(ITEMS)}")
''')

md('''## 3. Toàn bộ mã pipeline nhúng trong notebook

Các cell dưới ghi mã vào `/kaggle/working/lifemate_app` để chạy trong subprocess, tách runtime tải model (online) khỏi suy luận (local-only). Logic baseline được giữ nguyên; không cần repository bên ngoài. Có cả mã fine-tune, bộ unit test và scorer train.''')
for rel in ['lifemate/__init__.py', 'lifemate/catalog.py', 'lifemate/models.py', 'lifemate/agent.py', 'lifemate/transcripts.py', 'lifemate/decoding.py',
            'lifemate/gpu.py', 'main.py', 'score_train.py', 'lifemate/sft.py', 'prepare_sft.py', 'train_lora.py',
            'merge_lora.py', 'finetune.py', 'tests/test_baseline.py', 'tests/test_sft.py',
            'tests/test_finetune_integration.py', 'tests/test_audio.py', 'tests/test_catalog_inspection.py',
            'tests/test_staged_pipeline.py', 'tests/test_decoding.py']:
    md(f'### `{rel}`')
    code(f'%%writefile /kaggle/working/lifemate_app/{rel}\n' + (ROOT/'code'/rel).read_text())

code('''run_logged([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
           OUT / "unit_tests.log", cwd=APP, env=BASE_ENV)
toolenv_check_script = """
import sys
from pathlib import Path
from main import resolve_data_layout
split_dir, env_root, holder = resolve_data_layout(Path(sys.argv[1]), sys.argv[2])
try:
    sys.path.insert(0, str(split_dir))
    from zaic_tools import ToolEnv
    env = ToolEnv(sys.argv[2], sys.argv[3], root=str(env_root))
    required = ('validate', 'call', 'get_tool_calls', 'get_state_hash')
    missing = [name for name in required if not callable(getattr(env, name, None))]
    if missing:
        raise RuntimeError('ToolEnv instance is missing callable API methods: ' + ', '.join(missing))
    print('ToolEnv instance public API OK')
finally:
    if holder is not None:
        holder.cleanup()
"""
run_logged([sys.executable, "-c", toolenv_check_script, SPLIT_DIR, SPLIT, EXPECTED_IDS[0]],
           OUT / "setup.log", cwd=APP, env=BASE_ENV)
''')

md('''## 4. Tải Hugging Face model và chuyển ASR nếu cần

Hai model mặc định là public, không yêu cầu token. Với model cần xác thực, thêm Kaggle Secret rồi đặt `HF_SECRET_NAME`; token chỉ truyền qua environment, không ghi vào cấu hình/logs. SHA thực tế được lưu để tái lập lần chạy. `LOCAL_ASR` luôn phải trỏ tới model CTranslate2 đã chuyển đổi. Không tải model trong bước inference.''')
code('''download_settings = dict(asr_repo=ASR_REPO, llm_repo=LLM_REPO, asr_format=ASR_FORMAT,
    asr_revision=ASR_REVISION, llm_revision=LLM_REVISION,
    local_asr=LOCAL_ASR, local_llm=LOCAL_LLM, model_root=str(MODEL_ROOT), out=str(OUT))
(OUT / "download_settings.json").write_text(json.dumps(download_settings, indent=2), encoding="utf-8")
DOWNLOAD_ENV = BASE_ENV.copy()
DOWNLOAD_ENV.update(HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0", HF_DATASETS_OFFLINE="0")
DOWNLOAD_ENV["HF_HUB_DISABLE_XET"] = "1"  # dùng HTTP download chuẩn của Hub
if HF_SECRET_NAME:
    from kaggle_secrets import UserSecretsClient
    DOWNLOAD_ENV["HF_TOKEN"] = UserSecretsClient().get_secret(HF_SECRET_NAME)

download_script = r\x27\x27\x27
import json, sys, os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
cfg = json.loads(Path(sys.argv[1]).read_text())
root = Path(cfg["model_root"])
api = HfApi()
manifest = {}
for part in ("asr", "llm"):
    local = cfg["local_" + part]
    if local:
        target = Path(local).expanduser().resolve()
        if not target.is_dir():
            raise FileNotFoundError(target)
        manifest[part] = {"path": str(target), "source": "local"}
        continue
    repo = cfg[part + "_repo"]
    requested = cfg[part + "_revision"] or "main"
    sha = api.model_info(repo, revision=requested).sha
    target = root / part / sha
    print(f"Downloading {part}: {repo}@{sha}", flush=True)
    snapshot_download(repo_id=repo, revision=sha, local_dir=str(target), max_workers=4)
    if part == "asr" and cfg["asr_format"] == "transformers":
        from ctranslate2.converters import TransformersConverter
        converted = root / "asr_ct2" / sha
        if not (converted / ".conversion_complete").is_file():
            # Rebuild only notebook-generated incomplete conversion, never user LOCAL_ASR.
            import shutil
            if converted.exists():
                shutil.rmtree(converted)
            converted.parent.mkdir(parents=True, exist_ok=True)
            converter = TransformersConverter(str(target))
            converter.convert(str(converted), quantization="float16")
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(str(target), use_fast=True, local_files_only=True).save_pretrained(str(converted))
            if (target / "preprocessor_config.json").is_file():
                shutil.copy2(target / "preprocessor_config.json", converted / "preprocessor_config.json")
            if not (converted / "tokenizer.json").is_file():
                raise FileNotFoundError("ASR conversion needs a local fast tokenizer.json")
            (converted / ".conversion_complete").touch()
        target = converted
    manifest[part] = {"path": str(target), "repo": repo, "revision": sha, "source": "huggingface"}
asr = Path(manifest["asr"]["path"])
llm = Path(manifest["llm"]["path"])
for file in (asr / "model.bin", asr / "config.json", llm / "config.json"):
    if not file.is_file():
        raise FileNotFoundError(f"Missing model file: {file}")
if not any(llm.glob("*.safetensors")):
    raise FileNotFoundError("LLM cần trọng số safetensors trong thư mục model.")
# Verify tokenizer loads locally now, before the long inference step.
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained(str(llm), local_files_only=True, trust_remote_code=False)
Path(cfg["out"], "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("Both model directories are staged.")
\x27\x27\x27
download_path = APP / "download_kaggle_models.py"
download_path.write_text(download_script, encoding="utf-8")
try:
    run_logged([sys.executable, download_path, OUT / "download_settings.json"],
               OUT / "models.log", env=DOWNLOAD_ENV, cwd=APP)
finally:
    # Không giữ token trong dictionary dùng cho suy luận hoặc file artifacts.
    DOWNLOAD_ENV.pop("HF_TOKEN", None)
MODEL_MANIFEST = json.loads((OUT / "model_manifest.json").read_text())
ASR_PATH = Path(MODEL_MANIFEST["asr"]["path"])
LLM_PATH = Path(MODEL_MANIFEST["llm"]["path"])
print("ASR:", ASR_PATH, "\\nLLM:", LLM_PATH)
''')

md('''## 5. ASR trước — lưu toàn bộ transcript, giải phóng GPU

Tiến trình ASR chỉ nạp Whisper. Kết thúc thành công mới có cache với manifest `complete` để LLM dùng. Cache lưu cả transcript hiện tại và toàn bộ lịch sử có audio/text; agent chỉ lấy phần lịch sử cần thiết vào prompt. Nếu lỗi ASR, cell dừng, giữ `.partial.jsonl` và báo lỗi; chạy lại cùng cấu hình sẽ tiếp tục từ mẫu chưa xong. Không biến lỗi ASR thành dự đoán rỗng.

Để chạy LLM lại mà không chạy ASR, giữ đường dẫn `TRANSCRIPT_CACHE` và manifest cạnh nó, đặt `RUN_ASR=False`. Nếu đổi ASR để ablation, dùng cache khác hoặc `REFRESH_TRANSCRIPTS=True`. Mỗi cache ghi model/settings/seed và dấu vết dữ liệu để tránh dùng nhầm transcript.''')
code('''TRANSCRIPTS = Path(TRANSCRIPT_CACHE) if TRANSCRIPT_CACHE else OUT / "transcripts.jsonl"
TRANSCRIPT_MANIFEST = TRANSCRIPTS.with_suffix(".manifest.json")
ASR_ENV = BASE_ENV.copy()
ASR_ENV["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)
ASR_ENV.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
if RUN_ASR:
    asr_command = [sys.executable, "-u", APP / "main.py", "--stage", "asr",
        "--data-root", SPLIT_DIR, "--split", SPLIT, "--asr-model", ASR_PATH,
        "--transcript-cache", TRANSCRIPTS, "--device", DEVICE,
        "--asr-compute-type", ASR_COMPUTE, "--asr-beam", str(ASR_BEAM),
        "--limit", str(LIMIT), "--seed", str(SEED)]
    if USE_TRAIN_TRANSCRIPT:
        asr_command.append("--use-train-transcript")
    if REFRESH_TRANSCRIPTS:
        asr_command.append("--refresh-transcripts")
    run_logged(asr_command, OUT / "asr.log", cwd=APP, env=ASR_ENV)
elif not TRANSCRIPTS.is_file() or not TRANSCRIPT_MANIFEST.is_file():
    raise FileNotFoundError("RUN_ASR=False cần cache JSONL và manifest đã hoàn tất.")
print("Transcripts:", TRANSCRIPTS)
print("ASR process has exited; LLM will read and verify cache before loading weights.")
''')

md('''## 6. Fine-tune tool calling trên TRAIN (LoRA/QLoRA)

Mặc định `TRAIN_LLM=True`: huấn luyện LoRA/QLoRA, rồi tự chuyển sang **base model + adapter vừa train** cho task chính. Nhãn phải là `label.tool_calls` theo thứ tự thực thi. Replay dùng `ToolEnv` thật của train; kết quả tool trước được đưa vào prompt bước sau. Chỉ tính loss trên assistant JSON, split theo sample ID trước khi tạo các bước, không đọc nhãn test.

Danh mục `tool_index` giữ toàn bộ tool được phép. Nếu gold tool chưa có schema chi tiết, dữ liệu SFT thêm action nội bộ `inspect` để yêu cầu schema trước khi gọi tool. Inference hỗ trợ cùng action; schema đã yêu cầu được giữ nguyên khi rút gọn prompt. Mặc định `TRAIN_MISSING_TOOL_POLICY="error"`, không bỏ mẫu do giới hạn top tool. `inspect` không phải tool call của BTC và không xuất vào submission.

Muốn dùng adapter đã có: đặt `TRAIN_LLM=False`, `LOCAL_ADAPTER` tới thư mục adapter. Muốn dùng model đã merge: đặt `TRAIN_LLM=False`, `LOCAL_ADAPTER=None`, `LOCAL_LLM` tới model merged. Nếu thiếu train data khi bật training, cell sẽ dừng rõ ràng thay vì dùng model gốc âm thầm.

Backend huấn luyện mặc định là **Unsloth FastLanguageModel** với QLoRA 4-bit, LoRA dropout=0 và gradient checkpointing `unsloth`. Import Unsloth trước Transformers/PEFT trong subprocess riêng. T4 dùng FP16; GPU hỗ trợ thì BF16. Sequence dài có thể hết VRAM: điều chỉnh cấu hình cho GPU thực tế. JSON target quá dài sẽ bị báo lỗi, không bị cắt âm thầm. `eval_loss` là loss trên validation theo teacher forcing, **không phải điểm F1 chính thức**. Checkpoint/adapter nằm dưới `/kaggle/working/lifemate_finetunes`; không nhét model nhiều GB vào ZIP logs. Nếu mất session, gắn lại base weights + adapter hoặc dùng checkpoint để resume bằng `train_lora.py`.''')
code('''ADAPTER_PATH = Path(LOCAL_ADAPTER) if LOCAL_ADAPTER else None
if TRAIN_LLM:
    if TRAIN_DATA_ROOT:
        training_data = Path(TRAIN_DATA_ROOT)
    else:
        training_candidates = discover_split("/kaggle/input", "train")
        if len(training_candidates) != 1:
            raise ValueError(f"Cần đúng một tập train: {training_candidates}. Đặt TRAIN_DATA_ROOT rõ ràng.")
        training_data = training_candidates[0]
    FT_RUN = Path("/kaggle/working/lifemate_finetunes") / datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    FT_ENV = BASE_ENV.copy()
    FT_ENV.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    train_command = [sys.executable, "-u", APP / "finetune.py",
        "--data-root", training_data, "--base-model", LLM_PATH, "--output-dir", FT_RUN,
        "--backend", TRAIN_BACKEND,
        "--validation-fraction", str(TRAIN_VALIDATION_FRACTION),
        "--max-tool-details", str(MAX_TOOL_DETAILS), "--seed", str(SEED),
        "--missing-tool-policy", TRAIN_MISSING_TOOL_POLICY,
        "--epochs", str(TRAIN_EPOCHS), "--max-steps", str(TRAIN_MAX_STEPS),
        "--max-length", str(TRAIN_MAX_LENGTH),
        "--gradient-accumulation", str(TRAIN_GRAD_ACCUMULATION),
        "--quantization", "4bit" if HW["gpu"] else "none"]
    if TRAIN_USE_ASR:
        train_command.extend(["--asr-model", ASR_PATH, "--asr-device", DEVICE,
                              "--asr-compute-type", ASR_COMPUTE])
    try:
        run_logged(train_command, OUT / "training_pipeline.log", cwd=APP, env=FT_ENV)
    finally:
        if FT_RUN.is_dir():
            for filename in ("prepare_sft.log", "train_lora.log", "base_model.json",
                             "training/training_config.json", "training/eval_results.json",
                             "training/train_results.json", "training/trainer_state.json", "sft/manifest.json",
                             "sft/train_skipped.jsonl", "sft/validation_skipped.jsonl"):
                path = FT_RUN / filename
                if path.is_file():
                    shutil.copy2(path, OUT / ("training_" + filename.replace("/", "_")))
    ADAPTER_PATH = FT_RUN / "training" / "adapter"
if ADAPTER_PATH is not None and not (ADAPTER_PATH / "adapter_config.json").is_file():
    raise FileNotFoundError(f"Không có adapter hợp lệ: {ADAPTER_PATH}")
print("Inference LLM:", LLM_PATH, "adapter:", ADAPTER_PATH)
''')

md('''## 7. LLM từ transcript cache → kiểm chứng → lưu kết quả và logs

Inference chỉ nạp LLM trên một GPU (`GPU_INDEX`, `--llm-device-map single`), không nạp ASR, và mặc định không giới hạn thời gian từng mẫu. Greedy decoding dừng ngay khi sinh xong JSON quyết định hoàn chỉnh (`call`, `inspect` hoặc `done`); vẫn giữ `MAX_NEW_TOKENS` làm giới hạn an toàn. JSON lồng nhau, dấu ngoặc trong chuỗi và khối thinking không gây dừng sớm. Một GPU vẫn cần đủ VRAM cho trọng số và context. stdout/stderr được ghi vào `inference.log`; `diagnostics.jsonl` có `sample_seconds` (LLM), `asr_seconds`, `total_sample_seconds` (tổng hai giai đoạn). Notebook kiểm tra đủ ID và schema output; không tự suy diễn tính đúng của state hash. Chạy lại cell dùng cache cũ và tạo thư mục kết quả mới.''')
code('''# Chạy lại cell này không ghi đè kết quả cũ.
if (OUT / "inference.log").exists():
    previous_out = OUT
    OUT = OUT.parent / datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    OUT.mkdir(parents=True)
    for name in ("setup.log", "models.log", "hardware.log", "unit_tests.log", "environment.txt", "model_manifest.json", "asr.log"):
        if (previous_out / name).is_file():
            shutil.copy2(previous_out / name, OUT / name)
    for path in previous_out.glob("training_*"):
        if path.is_file():
            shutil.copy2(path, OUT / path.name)
INFER_ENV = BASE_ENV.copy()
INFER_ENV["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)
INFER_ENV["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
INFER_ENV.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
INFER_ENV.pop("HF_TOKEN", None)
SUBMISSION = OUT / "submission.jsonl"
DIAGNOSTICS = OUT / "diagnostics.jsonl"
command = [sys.executable, "-u", APP / "main.py",
    "--stage", "llm", "--transcript-cache", TRANSCRIPTS,
    "--llm-device-map", "single",
    "--data-root", SPLIT_DIR, "--split", SPLIT,
    "--asr-model", ASR_PATH, "--llm-model", LLM_PATH,
    "--output", SUBMISSION, "--log-path", DIAGNOSTICS,
    "--device", DEVICE, "--asr-compute-type", ASR_COMPUTE, "--llm-dtype", LLM_DTYPE,
    "--limit", str(LIMIT), "--sample-seconds", str(SAMPLE_SECONDS),
    "--max-input-tokens", str(MAX_INPUT_TOKENS), "--max-new-tokens", str(MAX_NEW_TOKENS),
    "--max-calls", str(MAX_CALLS), "--max-invalid", str(MAX_INVALID),
    "--max-tool-details", str(MAX_TOOL_DETAILS), "--asr-beam", str(ASR_BEAM), "--seed", str(SEED)]
if ADAPTER_PATH is not None:
    command.extend(["--llm-adapter", ADAPTER_PATH])
if FAIL_FAST:
    command.append("--fail-fast")
if USE_TRAIN_TRANSCRIPT:
    command.append("--use-train-transcript")
for source, name in ((TRANSCRIPTS, "transcripts.jsonl"), (TRANSCRIPT_MANIFEST, "transcripts.manifest.json")):
    if source.resolve() != (OUT / name).resolve():
        shutil.copy2(source, OUT / name)
run_config = dict(split=SPLIT, data_root=str(SPLIT_DIR), sample_count=len(EXPECTED_IDS),
                  unlimited_sample_time=SAMPLE_SECONDS == float("inf"), transcript_cache=str(TRANSCRIPTS),
                  asr_from_cache=json.loads(TRANSCRIPT_MANIFEST.read_text(encoding="utf-8"))["asr"],
                  inference_gpu_ids=[GPU_INDEX], llm_device_map="single", stop_on_decision_json=True,
                  limit=LIMIT, hardware=HW, command=[str(x) for x in command], models=MODEL_MANIFEST)
(OUT / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
validation = {"status": "running", "expected_rows": len(EXPECTED_IDS)}
try:
    run_logged(command, OUT / "inference.log", cwd=APP, env=INFER_ENV)
    def read_jsonl(path):
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    rows = read_jsonl(SUBMISSION)
    diagnostics = read_jsonl(DIAGNOSTICS)
    if [r.get("id") for r in rows] != EXPECTED_IDS:
        raise ValueError("Submission thiếu, thừa hoặc sai thứ tự ID.")
    if [r.get("id") for r in diagnostics] != EXPECTED_IDS:
        raise ValueError("Diagnostics không đủ ID.")
    for row in rows:
        if set(row) != {"id", "tool_calls", "state_hash"}:
            raise ValueError(f"Sai schema output: {row.get('id')}")
        if not isinstance(row["tool_calls"], list) or not isinstance(row["state_hash"], str) or not row["state_hash"]:
            raise ValueError(f"Invalid calls/hash: {row['id']}")
    errors = [r["id"] for r in diagnostics if "error" in r]
    times = [r["sample_seconds"] for r in diagnostics]
    total_times = [r["total_sample_seconds"] for r in diagnostics]
    validation.update(status="completed" if not errors else "completed_with_sample_errors",
        rows=len(rows), error_count=len(errors), error_ids=errors,
        mean_seconds=sum(times) / len(times), max_seconds=max(times),
        mean_asr_seconds=sum(r["asr_seconds"] for r in diagnostics) / len(diagnostics),
        mean_total_seconds=sum(total_times) / len(total_times),
        samples_over_30s=sum(t > 30 for t in total_times),
        submission_sha256=hashlib.sha256(SUBMISSION.read_bytes()).hexdigest())
    if errors:
        raise RuntimeError(f"Đã xuất đủ mẫu nhưng {len(errors)} mẫu lỗi; xem diagnostics và inference.log.")
    if SPLIT == "train" and RUN_TRAIN_SCORE:
        run_logged([sys.executable, APP / "score_train.py", "--train-jsonl", SPLIT_DIR / "train.jsonl",
                    "--catalog", CATALOG_PATH, "--submission", SUBMISSION],
                   OUT / "train_score.log", cwd=APP, env=INFER_ENV)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
except BaseException as exc:
    if validation["status"] != "completed_with_sample_errors":
        validation["status"] = "failed"
    validation["failure"] = str(exc)
    raise
finally:
    (OUT / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    archive = OUT / "lifemate_artifacts.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(OUT.iterdir()):
            if path.is_file() and path != archive:
                bundle.write(path, arcname=path.name)
    print("\\nKết quả và logs:", OUT)
    print("ZIP:", archive)
''')
code('''from IPython.display import display, FileLink
for name in ("transcripts.jsonl", "transcripts.manifest.json", "asr.log", "submission.jsonl", "diagnostics.jsonl", "inference.log", "validation.json", "lifemate_artifacts.zip"):
    path = OUT / name
    if path.is_file():
        display(FileLink(str(path)))
print("Có thể tải toàn bộ thư mục run từ tab Output của Kaggle sau khi Save Version / Run All.")
''')

notebook = dict(nbformat=4, nbformat_minor=5, cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.11.0'},
    'kaggle': {'accelerator': 'gpu', 'isInternetEnabled': True, 'language': 'python', 'sourceType': 'notebook'},
})
for i, cell in enumerate(cells):
    cell['id'] = f'lifemate-{i:03d}'
target = ROOT / 'notebooks' / 'LifeMate_Kaggle.ipynb'
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print(target, len(cells), 'cells')
