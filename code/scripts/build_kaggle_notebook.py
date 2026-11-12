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
Không cần clone GitHub. Mặc định notebook fine-tune LLM trước trên TRAIN rồi dùng adapter vừa train. Notebook chứa toàn bộ mã nguồn: audio tiếng Việt → GIPformer ONNX → Qwen → validate/call trên `ToolEnv` thật → submission + logs.

Bộ dữ liệu cần có `<split>.jsonl`, `<split>_tools_catalog.json`, audio và **zaic_tools đúng phiên bản của split**. Notebook chỉ đọc manifest/catalog/audio; trạng thái ứng dụng do `ToolEnv` quản lý.

Chỉnh cấu hình bên dưới. `LIMIT=0` chạy **toàn bộ split**. Model tải từ Hugging Face trước khi suy luận; có thể dùng model gắn sẵn ở `/kaggle/input`. Model ASR mặc định là `g-group-ai-lab/gipformer1.5-68M-rnnt`, dùng trực tiếp định dạng ONNX.

Qwen3.5 dùng function calling native từ chat template của checkpoint: `tools`, assistant `tool_calls` và tool responses; SFT/inference dùng chung template, không train JSON action call làm output model. Mọi tool được phép và mọi ràng buộc schema được giữ khi finetune/inference; chỉ bỏ metadata dư và câu mô tả integer lặp. Lịch sử và kết quả tool được giữ đầy đủ; không tự rút gọn khi prompt dài. Vượt giới hạn token sẽ báo lỗi rõ ràng.

Feedback parse/precheck/validate được giữ khi LLM đang sửa lỗi. Sau khi validate thành công và call được thực thi, feedback từ chối cũ được bỏ khỏi prompt tiếp theo; mọi lời gọi đã thực thi và kết quả vẫn được giữ, kể cả call trả lỗi. Trace debug vẫn lưu đầy đủ lỗi trước đó.

ASR chạy toàn bộ split (bao gồm audio lịch sử), lưu transcript rồi thoát tiến trình và giải phóng runtime CPU. LLM sau đó đọc cache; không nạp GIPformer cùng lúc. Mặc định không giới hạn thời gian mỗi mẫu (`float("inf")`); vẫn giữ giới hạn tool calls và token/context ≤16.384.

Kết quả lưu trong `/kaggle/working/lifemate_runs/<RUN_NAME>/`: `transcripts.jsonl`, `transcripts.manifest.json`, `asr.log`, `submission.jsonl`, `diagnostics.jsonl`, `inference.log`, `setup.log`, `models.log`, `validation.json`, `run_config.json`, `environment.txt`, và `lifemate_artifacts.zip`. Transcript/logs có nội dung người dùng; kiểm tra trước khi chia sẻ. Đây không phải chế độ bảo đảm giới hạn 30 giây/mẫu của cuộc thi.''')

code('from pathlib import Path\n' + (ROOT / 'code/lifemate/context.py').read_text() + '\n' + '''import os, sys, json, subprocess, datetime, shutil, zipfile, hashlib

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
# Loader đặt toàn bộ model trên GPU đã chọn; thiếu VRAM dừng bằng OOM, không auto offload.
TRAIN_ASR_TRANSCRIPT_RATIO = 0.5  # 50% sample dùng audio_text, 50% dùng ASR; chọn theo SEED
TRAIN_RENAME_VARIANTS = 2  # giữ mẫu gốc + tối đa 2 biến thể đổi tên app/tool
TRAIN_DISTRACTOR_PROBABILITY = 0.5  # thêm 1-3 app khác nhóm vào biến thể nếu có ứng viên
TRAIN_APP_GROUPS_FILE = None  # JSON group -> app IDs trong TRAIN catalog, nếu muốn tùy chỉnh
TRAIN_MISSING_TOOL_POLICY = "error"  # toàn bộ schema được giữ; lỗi schema không mặc định bỏ mẫu
TRAIN_DATA_ROOT = None  # None: tự tìm train/train.jsonl trong /kaggle/input
LOCAL_ADAPTER = None   # TRAIN_LLM=False + đường dẫn adapter để dùng bản đã train
TRAIN_EPOCHS = 3
TRAIN_MAX_STEPS = -1   # -1: theo epochs; đặt 2 cho kiểm tra ngắn, không phải model đủ chất lượng
TRAIN_MAX_LENGTH = MAX_CONTEXT_TOKENS
TRAIN_BATCH_SIZE = 1
TRAIN_QUANTIZATION = "none"
TRAIN_GRAD_ACCUMULATION = 8

ASR_REPO = "g-group-ai-lab/gipformer1.5-68M-rnnt"
ASR_FORMAT = "onnx"
LLM_REPO = "Qwen/Qwen3.5-4B"
ASR_REVISION = None  # None: resolve main thành SHA và ghi lại trong model_manifest.json
LLM_REVISION = None
LOCAL_ASR = None     # encoder/decoder/joiner ONNX + tokens.txt
LOCAL_LLM = None     # thư mục Transformers có config.json + tokenizer + trọng số
HF_SECRET_NAME = None  # tùy chọn: tên Kaggle Secret chứa HF token (public models không cần)

SAMPLE_SECONDS = float("inf")  # không giới hạn thời gian mỗi mẫu
MAX_INPUT_TOKENS = DEFAULT_MAX_INPUT_TOKENS
MAX_NEW_TOKENS = DEFAULT_MAX_NEW_TOKENS
SCHEMA_CACHE_MODE = "auto"  # auto / shared / off: cache system+tools giữa sample, không chứa yêu cầu
SCHEMA_CACHE_MB = 256       # ngân sách snapshot giữ lại; chưa gồm model/cache riêng của sample
MAX_CALLS = 8
MAX_INVALID = 3
ASR_PRECISION = "int8"
ASR_THREADS = 4
SEED = 42
FAIL_FAST = False      # False: lưu lỗi từng mẫu và tiếp tục, báo lỗi ở bước kiểm chứng
USE_TRAIN_TRANSCRIPT = False  # chỉ được bật cho split train
RUN_TRAIN_SCORE = True        # proxy F1, chỉ chạy khi SPLIT == "train"
RUN_NAME = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")

APP = Path("/kaggle/working/lifemate_app")
MODEL_ROOT = Path("/kaggle/working/lifemate_models")
OUT = Path("/kaggle/working/lifemate_runs") / RUN_NAME
for folder in (APP / "lifemate", APP / "tests", APP / "tests/fixtures", MODEL_ROOT, OUT):
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
assert ASR_FORMAT == "onnx"
assert ASR_PRECISION in {"int8", "fp32"}
assert ASR_THREADS > 0
assert isinstance(GPU_INDEX, int) and GPU_INDEX >= 0
validate_inference_budget(MAX_INPUT_TOKENS, MAX_NEW_TOKENS)
validate_training_length(TRAIN_MAX_LENGTH)
assert SCHEMA_CACHE_MODE in {"auto", "shared", "off"}
assert 0 < SCHEMA_CACHE_MB < float("inf")
print("Artifacts:", OUT)
''')

md('''## 1. Dependency và runtime

Giữ nguyên PyTorch/CUDA do Kaggle cung cấp, không cài đè torch. Các thư viện ASR/LLM được cài trong tiến trình riêng trước khi dùng. GIPformer chạy CPU bằng sherpa-onnx, không dùng CUDA; bước suy luận thực tế sẽ kiểm tra việc nạp model. Nếu GPU memory không đủ, giảm `MAX_INPUT_TOKENS`, chọn model nhỏ hơn hoặc đổi GPU.''')
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
import json, torch, sherpa_onnx, soundfile
from transformers import Qwen3_5ForConditionalGeneration

gpu = torch.cuda.is_available()
print(json.dumps(dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=gpu,
    gpu_name=torch.cuda.get_device_name(0) if gpu else None,
    bf16=torch.cuda.is_bf16_supported() if gpu else False,
    sherpa_onnx=sherpa_onnx.__version__)))
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
if HW["gpu"] and not HW["bf16"]:
    raise RuntimeError("Cấu hình BF16 cần GPU hỗ trợ BF16; không tự chuyển sang FP16.")
LLM_DTYPE = "bfloat16" if HW["gpu"] else "float32"
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
            'lifemate/gpu.py', 'lifemate/context.py', 'lifemate/text.py', 'lifemate/function_calling.py', 'lifemate/asr.py', 'lifemate/llm.py', 'lifemate/prefix_cache.py', 'lifemate/prompt_session.py', 'lifemate/runtime.py', 'lifemate/history.py', 'lifemate/scoring.py', 'lifemate/evaluation.py', 'lifemate/eval_diagnostics.py', 'lifemate/eval_trace.py',
            'lifemate/local_time.py', 'lifemate/temporal_text.py', 'lifemate/temporal_training.py', 'lifemate/batching.py',
            'lifemate/schema_prompt.py', 'lifemate/schema_cache.py', 'lifemate/numerics.py', 'lifemate/reasoning.py', 'lifemate/recovery.py',
            'lifemate/local_assist.py', 'lifemate/assist_training.py',
            'lifemate/itn.py', 'lifemate/local_helpers.py', 'lifemate/local_knowledge.py', 'lifemate/local_checks.py',
            'lifemate/execution_policy.py', 'lifemate/tool_execution.py', 'lifemate/specialist_training.py',
            'main.py', 'score_train.py', 'lifemate/sft.py', 'lifemate/data_generalization.py', 'prepare_sft.py', 'train_lora.py',
            'merge_lora.py', 'finetune.py', 'tests/test_baseline.py', 'tests/test_sft.py',
            'tests/test_finetune_integration.py', 'tests/test_audio.py', 'tests/test_catalog.py',
            'tests/test_staged_pipeline.py', 'tests/test_decoding.py', 'tests/test_stopping_filters.py', 'tests/test_scoring.py', 'tests/test_evaluation.py', 'tests/test_batching.py', 'tests/test_data_generalization.py', 'tests/test_rollout_cache_lifecycle.py',
            'tests/test_transcript_normalization.py', 'tests/test_text.py', 'tests/test_runtime_parity.py', 'tests/test_runtime_config.py',
            'tests/test_native_function_calling.py', 'tests/test_prefix_cache.py', 'tests/test_prompt_session.py', 'tests/test_transcript_mixing.py', 'tests/test_eval_diagnostics.py', 'tests/test_eval_trace.py',
            'tests/test_schema_prompt.py', 'tests/test_schema_cache.py', 'tests/test_numerics.py',
            'tests/test_tool_validation.py',
            'tests/test_local_time.py', 'tests/test_local_time_runtime.py',
            'tests/test_local_tools_agent.py', 'tests/test_temporal_training.py', 'tests/test_local_time_evaluation.py', 'tests/test_local_assist.py',
            'tests/test_itn.py', 'tests/test_local_knowledge.py', 'tests/test_local_checks.py', 'tests/test_execution_policy.py', 'tests/test_automatic_guards.py',
            'tests/test_specialist_training.py', 'tests/test_python_helpers_integration.py',
            'tests/fixtures/qwen35_chat_template.jinja']:
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

Hai model mặc định là public, không yêu cầu token. Với model cần xác thực, thêm Kaggle Secret rồi đặt `HF_SECRET_NAME`; token chỉ truyền qua environment, không ghi vào cấu hình/logs. SHA thực tế được lưu để tái lập lần chạy. `LOCAL_ASR` luôn phải trỏ tới model ONNX transducer. Không tải model trong bước inference.''')
code('''download_settings = dict(asr_repo=ASR_REPO, llm_repo=LLM_REPO, asr_format=ASR_FORMAT, asr_precision=ASR_PRECISION,
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
    options = {}
    if part == "asr":
        suffix = ".int8.onnx" if cfg["asr_precision"] == "int8" else ".onnx"
        options["allow_patterns"] = ["tokens.txt", "README.md", *[p + suffix for p in ("encoder", "decoder", "joiner")]]
    snapshot_download(repo_id=repo, revision=sha, local_dir=str(target), max_workers=4, **options)
    manifest[part] = {"path": str(target), "repo": repo, "revision": sha, "source": "huggingface"}
asr = Path(manifest["asr"]["path"])
llm = Path(manifest["llm"]["path"])
suffix = ".int8.onnx" if cfg["asr_precision"] == "int8" else ".onnx"
for file in (asr / "tokens.txt", *[asr / (p + suffix) for p in ("encoder", "decoder", "joiner")], llm / "config.json"):
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

md('''## 5. ASR trước — lưu toàn bộ transcript, giải phóng runtime CPU

Tiến trình ASR chỉ nạp GIPformer. Output được chuẩn hóa bằng cùng hàm dùng cho audio_text khi SFT: NFC, lowercase, gộp khoảng trắng, viết hoa chữ cái đầu lượt và thêm dấu chấm nếu chưa có dấu kết câu. Giữ nguyên dấu câu sẵn có; không tự tách câu/từ hoặc sửa lỗi ASR. Cache cũ cần REFRESH_TRANSCRIPTS=True hoặc đường dẫn mới; manifest hiện dùng version 4 và ghi chính sách chuẩn hóa. Kết thúc thành công mới có cache với manifest `complete` để LLM dùng. Cache lưu cả transcript hiện tại và toàn bộ lịch sử có audio/text; agent chỉ lấy phần lịch sử cần thiết vào prompt. Nếu lỗi ASR, cell dừng, giữ `.partial.jsonl` và báo lỗi; chạy lại cùng cấu hình sẽ tiếp tục từ mẫu chưa xong. Không biến lỗi ASR thành dự đoán rỗng.

Để chạy LLM lại mà không chạy ASR, giữ đường dẫn `TRANSCRIPT_CACHE` và manifest cạnh nó, đặt `RUN_ASR=False`. Nếu đổi ASR để ablation, dùng cache khác hoặc `REFRESH_TRANSCRIPTS=True`. Mỗi cache ghi model/settings/seed và dấu vết dữ liệu để tránh dùng nhầm transcript.''')
code('''TRANSCRIPTS = Path(TRANSCRIPT_CACHE) if TRANSCRIPT_CACHE else OUT / "transcripts.jsonl"
TRANSCRIPT_MANIFEST = TRANSCRIPTS.with_suffix(".manifest.json")
ASR_ENV = BASE_ENV.copy()
ASR_ENV["CUDA_VISIBLE_DEVICES"] = ""
ASR_ENV.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
if RUN_ASR:
    asr_command = [sys.executable, "-u", APP / "main.py", "--stage", "asr",
        "--data-root", SPLIT_DIR, "--split", SPLIT, "--asr-model", ASR_PATH,
        "--transcript-cache", TRANSCRIPTS, "--device", DEVICE,
        "--asr-precision", ASR_PRECISION, "--asr-threads", str(ASR_THREADS),
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

Mặc định `TRAIN_LLM=True`: dùng toàn bộ sample trong TRAIN để huấn luyện, không tạo tập validation hay app-held-out. `TRAIN_ASR_TRANSCRIPT_RATIO=0.5` phân bổ sample dùng audio_text và ASR theo tỉ lệ 50:50, có làm tròn theo số mẫu. SEED quyết định nguồn transcript; mọi bước SFT của một sample dùng cùng nguồn. ASR không tự quay về audio_text khi lỗi; nguồn và số lượng được ghi trong manifest. Transcript của cả hai nguồn và text của mọi lượt history được chuẩn hóa chung: NFC, lowercase, gộp khoảng trắng, viết hoa chữ cái đầu lượt, thêm dấu chấm nếu chưa có dấu kết câu. History giữ đầy đủ theo resolver chung. Sau đó huấn luyện LoRA/QLoRA, rồi tự chuyển sang **base model + adapter ở epoch cuối cùng** cho task chính. Nhãn phải là `label.tool_calls` theo thứ tự thực thi. Replay dùng `ToolEnv` thật của train; kết quả tool trước được đưa vào prompt bước sau. Chỉ tính loss trên assistant JSON, không đọc nhãn test.

TRAIN được bổ sung tối đa `TRAIN_RENAME_VARIANTS=2` biến thể đổi tên app/tool nhất quán; với xác suất `TRAIN_DISTRACTOR_PROBABILITY=0.5` mỗi biến thể thêm 1–3 app khác nhóm từ TRAIN. Không đổi tên tham số, ID, văn bản hay giá trị arguments; giữ tên helper/sys. Demo local_entity/local_context phụ thuộc tên app chỉ giữ ở bản gốc. Biến thể vượt token budget bị bỏ toàn bộ, không cắt nội dung. Canonical gold chỉ replay một lần mỗi sample, aliases không gọi ToolEnv. Số bước SFT có thể tăng gần 3 lần. Manifest lưu cấu hình, sample IDs và thống kê biến thể.

`tool_schemas` giữ toàn bộ schema của mọi tool được phép ngay từ bước đầu, gồm đầy đủ description, returns và metadata. History và mọi kết quả tool được giữ nguyên, không giới hạn lượt/ký tự hoặc giảm theo độ dài prompt. SFT tạo trực tiếp action call/done. Mọi schema đều có sẵn; local_schema.inspect chỉ kiểm tra arguments và góp ý, không mở khóa tool. Vượt giới hạn token sẽ báo lỗi, không tự cắt dữ liệu. Cần tạo lại SFT cũ trong thư mục mới; manifest phải có `prompt_policy=full-context-token-bounded-helpers-v11`. Mặc định `TRAIN_MISSING_TOOL_POLICY="error"`, không tự bỏ mẫu lỗi schema.

Mặc định agent có controller `local_batch.run`: một native call chứa các helper độc lập theo ngân sách token, không có trần số helper theo batch/sample; env read vẫn giới hạn 2–4 call/nhóm. Không trộn hai nhóm, không write/meta/batch lồng nhau. Mọi toán hạng phải có trước nhóm; write và ask_missing_info đứng riêng. Env read vẫn validate/call tuần tự trên đúng một ToolEnv, log đủ từng call. Kết quả rỗng/lỗi/ambiguous làm ngắt nhóm; response nêu executed và remaining_calls, không tự chạy tiếp. SFT gộp sau gold replay và augmentation, không dùng kết quả tương lai làm đầu vào. Manifest lưu batch_examples và ungrouped_examples; diagnostics lưu llm_steps/non_helper_steps/batches. Các lượt helper hợp lệ không tiêu tốn giới hạn 32 bước của env/lỗi; không có trần 8 helper. SFT bỏ các giới hạn số demo helper cũ; chỉ gộp helper khi có tokenizer để đo đúng độ dài target. Không có cờ bổ sung; cần SFT/run mới để train định dạng batch. Các cache CPU/GPU và kiểm tra dừng generation vẫn dùng như trước.

Muốn dùng adapter đã có: đặt `TRAIN_LLM=False`, `LOCAL_ADAPTER` tới thư mục adapter. Muốn dùng model đã merge: đặt `TRAIN_LLM=False`, `LOCAL_ADAPTER=None`, `LOCAL_LLM` tới model merged. Nếu thiếu train data khi bật training, cell sẽ dừng rõ ràng thay vì dùng model gốc âm thầm.

Backend huấn luyện mặc định là **Unsloth FastVisionModel (text-only LoRA)** với LoRA không lượng tử hóa, max length 16.384 token, batch size 1, LoRA dropout=0 và gradient checkpointing `unsloth`. Import Unsloth trước Transformers/PEFT trong subprocess riêng. GPU phải hỗ trợ BF16; không tự chuyển sang FP16. Cấu hình batch 1 × 16K cần nhiều VRAM, chưa được xác nhận chạy vừa GPU Kaggle/T4. Prompt hoặc JSON target quá dài sẽ bị báo lỗi, không bị cắt âm thầm. Không chạy validation loss hay rollout cuối mỗi epoch hoặc sau khi train. Checkpoint vẫn được lưu mỗi epoch; adapter xuất ra là trạng thái cuối quá trình training, không nạp lại checkpoint theo điểm validation. Khi đặt `TRAIN_MAX_STEPS > 0`, adapter lấy ở bước train cuối cùng theo giới hạn đó. Checkpoint/adapter nằm dưới `/kaggle/working/lifemate_finetunes`; không nhét model nhiều GB vào ZIP logs. Nếu mất session, gắn lại base weights + adapter hoặc dùng checkpoint để resume bằng `train_lora.py`.''')
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
        "--asr-transcript-ratio", str(TRAIN_ASR_TRANSCRIPT_RATIO),
        "--rename-variants", str(TRAIN_RENAME_VARIANTS),
        "--distractor-probability", str(TRAIN_DISTRACTOR_PROBABILITY),
        "--seed", str(SEED),
        "--missing-tool-policy", TRAIN_MISSING_TOOL_POLICY,
        "--epochs", str(TRAIN_EPOCHS), "--max-steps", str(TRAIN_MAX_STEPS),
        "--max-length", str(TRAIN_MAX_LENGTH),
        "--batch-size", str(TRAIN_BATCH_SIZE),
        "--gradient-accumulation", str(TRAIN_GRAD_ACCUMULATION),
        "--quantization", TRAIN_QUANTIZATION]
    train_command.extend(["--asr-model", ASR_PATH, "--asr-device", "cpu",
                          "--asr-precision", ASR_PRECISION, "--asr-threads", str(ASR_THREADS)])
    if TRAIN_APP_GROUPS_FILE:
        train_command.extend(["--app-groups-file", TRAIN_APP_GROUPS_FILE])
    try:
        run_logged(train_command, OUT / "training_pipeline.log", cwd=APP, env=FT_ENV)
    finally:
        if FT_RUN.is_dir():
            numerics_debug = FT_RUN / "training/debug_numerics"
            if numerics_debug.is_dir():
                shutil.copytree(numerics_debug, OUT / "training_debug_numerics", dirs_exist_ok=True)
            for filename in ("prepare_sft.log", "train_lora.log", "base_model.json",
                             "training/training_config.json",
                             "training/train_results.json", "training/trainer_state.json", "sft/manifest.json",
                             "sft/train_skipped.jsonl"):
                path = FT_RUN / filename
                if path.is_file():
                    shutil.copy2(path, OUT / ("training_" + filename.replace("/", "_")))
    ADAPTER_PATH = FT_RUN / "training" / "adapter"
if ADAPTER_PATH is not None and not (ADAPTER_PATH / "adapter_config.json").is_file():
    raise FileNotFoundError(f"Không có adapter hợp lệ: {ADAPTER_PATH}")
print("Inference LLM:", LLM_PATH, "adapter:", ADAPTER_PATH)
''')

md('''## 7. LLM từ transcript cache → kiểm chứng → lưu kết quả và logs

Inference chỉ nạp LLM trên một GPU (`GPU_INDEX`, `--llm-device-map single`), không nạp ASR, và mặc định không giới hạn thời gian từng mẫu. Qwen3.5 dùng function calling native của checkpoint: tools được đưa qua `tools`, assistant tool_calls và role tool được render bằng chat template. Greedy decoding chờ hết `</tool_call>`; JSON nằm trong parameter không gây dừng sớm. Khi hoàn tất, model trả marker `{"action":"done"}`. Model khác giữ protocol JSON. Vẫn giữ `MAX_NEW_TOKENS` làm giới hạn an toàn. Một GPU vẫn cần đủ VRAM cho trọng số và context. stdout/stderr được ghi vào `inference.log`; `diagnostics.jsonl` có `sample_seconds` (LLM), `asr_seconds`, `total_sample_seconds` (tổng hai giai đoạn). Notebook kiểm tra đủ ID và schema output; không tự suy diễn tính đúng của state hash. Chạy lại cell dùng cache cũ và tạo thư mục kết quả mới.''')
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
    "--schema-cache", SCHEMA_CACHE_MODE, "--schema-cache-mb", str(SCHEMA_CACHE_MB),
    "--data-root", SPLIT_DIR, "--split", SPLIT,
    "--asr-model", ASR_PATH, "--llm-model", LLM_PATH,
    "--output", SUBMISSION, "--log-path", DIAGNOSTICS,
    "--device", DEVICE, "--asr-precision", ASR_PRECISION, "--asr-threads", str(ASR_THREADS), "--llm-dtype", LLM_DTYPE,
    "--limit", str(LIMIT), "--sample-seconds", str(SAMPLE_SECONDS),
    "--max-input-tokens", str(MAX_INPUT_TOKENS), "--max-new-tokens", str(MAX_NEW_TOKENS),
    "--max-calls", str(MAX_CALLS), "--max-invalid", str(MAX_INVALID),
    "--seed", str(SEED)]
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
                  schema_cache=SCHEMA_CACHE_MODE, schema_cache_mb=SCHEMA_CACHE_MB,
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
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print(target, len(cells), 'cells')
