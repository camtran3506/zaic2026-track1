# HƯỚNG DẪN VẬN HÀNH TOÀN BỘ LIFEMATE PIPELINE (ASR + LLM)

Tài liệu này hướng dẫn chi tiết cách thiết lập môi trường, tải mô hình, huấn luyện (fine-tune) ASR và LLM, cũng như chạy suy luận (inference) toàn bộ pipeline LifeMate. Mọi tiến trình nặng đều được hướng dẫn thực thi chạy ngầm bằng lệnh `nohup`, cấu hình thư mục lưu cache tối ưu trên NAS để tránh tràn bộ nhớ đĩa hệ thống `/tmp`.

---

## MỤC LỤC
1. [Tổng quan kiến trúc & Cấu trúc thư mục](#1-tổng-quan-kiến-trúc--cấu-trúc-thư-mục)
2. [Cấu hình môi trường & Thiết lập bộ nhớ đệm (Bắt buộc)](#2-cấu-hình-môi-trường--thiết-lập-bộ-nhớ-đệm-bắt-buộc)
3. [Bước 1: Tải và chuẩn bị Pretrained Models](#3-bước-1-tải-và-chuẩn-bị-pretrained-models)
4. [Bước 2: Fine-tune mô hình ASR Gipformer (Tùy chọn)](#4-bước-2-fine-tune-mô-hình-asr-gipformer-tùy-chọn)
5. [Bước 3: Fine-tune mô hình LLM LoRA Adapter](#5-bước-3-fine-tune-mô-hình-llm-lora-adapter)
6. [Bước 4: Chạy suy luận (Inference Pipeline)](#6-bước-4-chạy-suy-luận-inference-pipeline)
7. [Bảng tra cứu tham số dòng lệnh (CLI Parameters)](#7-bảng-tra-cứu-tham-số-dòng-lệnh-cli-parameters)
8. [Giám sát & Quản lý tiến trình nền (nohup)](#8-giám-sát--quản-lý-tiến-trình-nền-nohup)
9. [Xử lý sự cố thường gặp (Troubleshooting)](#9-xử-lý-sự-cố-thường-gặp-troubleshooting)

---

## 1. TỔNG QUAN KIẾN TRÚC & CẤU TRÚC THƯ MỤC

### 1.1. Luồng xử lý End-to-End 1 tiến trình (Mặc định chuẩn BTC) & Chế độ Legacy Tách Giai đoạn
Mặc định (`--stage online` trong `main.py`, `run_server.sh`, `predict.py`, `benchmark_timing.py` và `predict_notebook.ipynb`), LifeMate sử dụng **1 tiến trình duy nhất nạp đồng thời ASR và LLM** ([`lifemate/predictor.py`](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/predictor.py)) để tuân thủ đúng chuẩn nộp bài Docker & đo thời gian của BTC:
1. **Khởi tạo cố định & nạp model đồng thời (Load Upfront):**
   - Cố định seed (`seed_everything(42)` + `PYTHONHASHSEED`) trước khi load bất kỳ model nào.
   - Nạp **`LocalASR`** (`Gipformer 1.5` 68M Zipformer-RNNT, chạy trên CPU ONNX Runtime) và **`LocalLLM`** (`Qwen/Qwen3.5-4B` `bfloat16` + optional LoRA Adapter, chạy trên GPU) cùng tồn tại trong RAM/VRAM.
   - Tự động phát hiện split trong `/data` (`--split auto`) và import `ToolEnv` trực tiếp từ `/data/<split>/zaic_tools`.
2. **Suy luận từng sample theo đúng thứ tự gốc (Sample-by-sample E2E):**
   - Bắt đầu bấm giờ `t1 = time.perf_counter()` **trước khi đọc file audio và khởi tạo `ToolEnv`**.
   - Tạo đúng 1 thực thể `ToolEnv(split, item["id"], root=env_root)` cho mỗi mẫu.
   - Chạy live `ASR (câu hiện tại + history) -> LLM -> ToolEnv` nhiều lượt thông qua `LifeMateAgent`.
   - Inference mặc định **tắt thinking** (`enable_thinking=False`, `thinking_budget=0`): tối đa **16.000 input / 384 output tokens** mỗi lượt. Dữ liệu SFT mặc định dùng **Step-wise SFT** (`--step-wise-sft`) và tắt Short Checked Reasoning (`--no-short-reasoning`) để đồng bộ prefix `<think>\n\n</think>\n\n` giữa train và inference.
   - Kết thúc mẫu, tính `ms = int((time.perf_counter() - t1) * 1000)` và ghi + `flush()` ngay từng dòng `{"id", "tool_calls", "state_hash", "time"}` ra `submission.jsonl`.

> [!NOTE]
> **Chế độ tách 2 tiến trình cũ (`--stage full` / `--stage asr` / `--stage llm`):** Vẫn được giữ lại trong `main.py` phục vụ thử nghiệm/ablation nội bộ khi muốn chạy ASR 1 lần ra `transcripts.jsonl` (kèm SHA-256 manifest) rồi thử nhiều adapter/prompt LLM mà không cần nhận dạng lại audio. Không dùng chế độ này cho Docker nộp bài BTC.

### 1.2. Cấu trúc cây thư mục chuẩn
```text
lifemate/
├── instruction.md                     # Tài liệu hướng dẫn này
├── data/                              # Toàn bộ dataset & dữ liệu huấn luyện (gom chung theo yêu cầu BTC)
│   ├── train_asr.jsonl                # Dữ liệu 1,536 sample ASR audio_text đã gán nhãn chuẩn
│   ├── train/                         # Split train (train.jsonl + audio/ + zaic_tools/)
│   └── public_test/                   # Split public_test (public_test.jsonl + audio/ + zaic_tools/)
├── runtime_cache/                     # Thư mục cache tốc độ cao trên NAS
│   ├── triton/                        # Cache cho Triton JIT kernels
│   ├── torchinductor/                 # Cache cho PyTorch Inductor compiler
│   ├── tmp/                           # Thư mục tạm hệ thống (thay thế /tmp)
│   └── logs/                          # Thư mục chứa log các tiến trình chạy nền
└── code/                              # Mã nguồn chính (tương ứng /code trong Docker theo Mục 5 BTC)
    ├── predict.py                     # Entrypoint BTC mặc định (/data -> /result/submission.jsonl)
    ├── predict.sh                     # Script BTC chạy không tham số (bash /code/predict.sh)
    ├── benchmark_timing.py            # Script 4 bước đo thời gian & kiểm tra parity theo Phụ lục BTC
    ├── predict_notebook.ipynb         # Jupyter Notebook 4 bước theo Phụ lục Mục 7 BTC
    ├── start_jupyter.sh               # Khởi chạy JupyterLab cổng 9777 (password/token: zac2026)
    ├── README.md                      # Mô tả giải pháp theo Mục 6.1 BTC (ASR, LLM, params, context, agent loop, data)
    ├── Dockerfile                     # Đóng gói container offline chuẩn ZAC 2026 (CUDA <= 13.2)
    ├── requirements.txt               # Danh sách thư viện runtime + jupyterlab
    ├── main.py                        # CLI suy luận đa chế độ (mặc định --stage online; hỗ trợ full/asr/llm)
    ├── run_server.sh                  # Shell script launcher (mặc định STAGE=online, relative ../data & saved_models/)
    ├── lifemate/
    │   ├── predictor.py               # Engine suy luận 1-process dùng chung cho predict.py/main.py/notebook
    │   └── ...                        # agent, asr, models, schema_cache, guards, ...
    ├── models/llm/                    # Mô hình LLM gốc (Qwen3.5-4B base dùng để train/merge; được .dockerignore bỏ qua)
    ├── saved_models/                  # Nơi lưu trữ checkpoint mô hình phục vụ inference & Docker
    │   ├── asr/                       # Mô hình ASR gốc (ONNX fp32, int8, model.pt)
    │   ├── asr_ft/                    # Mô hình ASR sau khi fine-tune (ONNX fp32)
    │   └── llm_ft/                    # LLM đã gộp LoRA run_sft_step (predict.py/run_server.sh/Docker tự nhận diện)
    ├── training_code/                 # Toàn bộ mã nguồn huấn luyện & README.md theo Mục 5 & 6.2 BTC
    │   ├── README.md                  # Hướng dẫn reproduce training, seed=42, link HF/data
    │   ├── finetune.py                # Script 1-lệnh tự động hóa SFT LLM
    │   ├── prepare_sft.py             # Chuẩn bị dữ liệu huấn luyện SFT cho LLM
    │   ├── train_lora.py              # Huấn luyện Unsloth LoRA trên LLM
    │   ├── merge_lora.py              # Trộn LoRA adapter vào base model (-> saved_models/llm_ft)
    │   ├── audit_initial_tokens.py    # Audit token budget
    │   ├── score_train.py             # Chấm điểm proxy trên train
    │   ├── download_models.py         # Script tải mô hình từ Hugging Face vào saved_models/
    │   └── asr_finetune/              # Bộ công cụ fine-tune ASR Gipformer
    │       ├── setup_icefall.sh       # Clone và kiểm tra Icefall dependency
    │       ├── prepare_gipformer_sft.py # Chuẩn bị tập train ASR từ ../data/train_asr.jsonl
    │       ├── train_gipformer.py     # Huấn luyện PyTorch RNNT Gipformer
    │       └── export_gipformer.py    # Xuất checkpoint sang ONNX FP32 -> saved_models/asr_ft
    └── result/                        # Thư mục lưu kết quả nộp bài và adapter
```

---

## 2. CẤU HÌNH MÔI TRƯỜNG & THIẾT LẬP BỘ NHỚ ĐỆM (BẮT BUỘC)

Hệ thống lưu trữ NAS cần được thiết lập các biến môi trường chỉ định thư mục tạm và cache. Điều này ngăn việc các trình biên dịch JIT (`Triton`, `TorchInductor`, `Unsloth`) ghi vào `/tmp` cục bộ gây tràn dung lượng ổ root máy chủ.

### 2.1. Khởi tạo thư mục đệm
Chạy lệnh tạo thư mục nếu chưa có:
```bash
mkdir -p /mnt/mmlab2024nas/trantran/lifemate/runtime_cache/{triton,torchinductor,tmp,logs}
```

### 2.2. Khai báo biến môi trường chuẩn
Trước khi chạy bất kỳ lệnh nào (huấn luyện hoặc suy luận), hãy xuất (export) toàn bộ các biến môi trường sau trong terminal (hoặc đặt ở đầu script shell / file `.bashrc`):

```bash
# --- CẤU HÌNH CACHE VÀ THƯ MỤC TẠM TRÊN NAS ---
export TRITON_CACHE_DIR="/mnt/mmlab2024nas/trantran/lifemate/runtime_cache/triton"
export TORCHINDUCTOR_CACHE_DIR="/mnt/mmlab2024nas/trantran/lifemate/runtime_cache/torchinductor"
export TMPDIR="/mnt/mmlab2024nas/trantran/lifemate/runtime_cache/tmp"

# --- CẤU HÌNH PYTHON VÀ CODE PATH ---
export PYTHONPATH="/mnt/mmlab2024nas/trantran/.venv3/lib/python3.12/site-packages:/mnt/mmlab2024nas/trantran/lifemate/code:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="1"
export TOKENIZERS_PARALLELISM="false"

# --- CẤU HÌNH GPU & CHẾ ĐỘ OFFLINE ---
export CUDA_VISIBLE_DEVICES="0"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"
```

> [!NOTE]
> **Lưu ý về Python Runtime & Chọn GPU:**
> - Trên môi trường hiện tại, đường dẫn binary Python chuẩn có đầy đủ thư viện là `/usr/bin/python3.12`. Khi chạy các lệnh, ta sẽ gọi trực tiếp `/usr/bin/python3.12` kết hợp `PYTHONPATH` đã export ở trên để đảm bảo nhận diện chính xác các gói: `torch`, `unsloth`, `transformers`, `sherpa_onnx`, `k2`, `kaldifeat`.
> - Các script Python (`train_gipformer.py`, `finetune.py`, `train_lora.py`, `main.py`) nhận cờ `--cuda-visible-devices <ID>` (mặc định tự kế thừa từ biến môi trường `CUDA_VISIBLE_DEVICES`, hoặc đặt `-1` để chạy CPU). Không dùng `--cuda-device`.

---

## 3. BƯỚC 1: TẢI VÀ CHUẨN BỊ PRETRAINED MODELS

Pipeline LifeMate hoạt động **hoàn toàn offline**. Toàn bộ mô hình phải được tải về sẵn trong thư mục [`code/saved_models`](file:///mnt/mmlab2024nas/trantran/lifemate/code/saved_models) trước khi bắt đầu.

Tạm thời bỏ cờ `HF_HUB_OFFLINE` để tải mô hình qua script [`download_models.py`](file:///mnt/mmlab2024nas/trantran/lifemate/code/training_code/download_models.py):

```bash
cd code

# Tải cả Base LLM (Qwen3.5-4B) và Base ASR (Gipformer 1.5 với đầy đủ ONNX FP32, INT8 và PyTorch weights)
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 /usr/bin/python3.12 training_code/download_models.py \
  --models-dir saved_models \
  --only both \
  --asr-precision all
```

### (Tùy chọn) Tải sẵn mô hình ASR Fine-tuned từ Hugging Face
Nếu bạn không muốn tự fine-tune lại mô hình ASR từ đầu mà muốn dùng ngay checkpoint đã được fine-tune sẵn trên tập dữ liệu cuộc thi:
```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 /usr/bin/python3.12 training_code/download_models.py \
  --models-dir saved_models \
  --only asr \
  --asr-ft-repo hugebenevolence/lifemate-final \
  --asr-ft-subdir asr-gipformer
```
Khi đó, các tệp ONNX của mô hình fine-tune sẽ được lưu tại [`code/saved_models/asr_ft`](file:///mnt/mmlab2024nas/trantran/lifemate/code/saved_models/asr_ft).

---

## 4. BƯỚC 2: FINE-TUNE MÔ HÌNH ASR GIPFORMER (TÙY CHỌN)

Bước này áp dụng khi bạn muốn tự huấn luyện lại mô hình nhận dạng giọng nói Gipformer bằng tệp [`data/train_asr.jsonl`](file:///mnt/mmlab2024nas/trantran/lifemate/data/train_asr.jsonl) (trong đó các câu audio text chính đã được con người chuyển số thành chữ chính xác).

### 4.1. Chuẩn bị thư viện Icefall
Chạy script cài đặt liên kết mã nguồn Icefall:
```bash
cd code
bash training_code/asr_finetune/setup_icefall.sh
```

### 4.2. Chuẩn bị dữ liệu huấn luyện ASR (`prepare_gipformer_sft.py`)
Script này áp dụng chính sách **Selective Normalization & Anti-Noise** và hiện mặc định khớp 100% với bộ dữ liệu đã tạo ra `saved_models/asr_ft` (`result/asr_sft_data`):
- **Tập `audio_text` chính (1,536 câu):** Con người đã kiểm tra và đảm bảo chuyển toàn bộ số thành chữ chính xác (`năm mươi ngàn`, `bảy rưỡi`...). Do đó, cách đọc số được giữ nguyên vẹn; còn lại **tất cả các phép chuẩn hóa khác** (xử lý ký hiệu `+`, `&`, `%`, `@`, `°`, dấu thời gian `h`, ngày tháng `tháng`, dấu phẩy số thập phân `phẩy`, viết hoa toàn bộ, loại bỏ dấu câu, căn chỉnh dấu thanh BPE `model_style`, kiểm tra bảng ký tự và BPE token) **đều được áp dụng đồng nhất như history audio**. Không bật `--verify-main`, `--max-main-wer`, `--max-history-wer` hay `--final` để bảo toàn 100% dữ liệu gốc.
- **Tập số còn sót (9 câu):** Một số ít câu còn sót chữ số chưa được con người chuyển (`2 triệu`, `tập 12`, `tháng 9`, `101`, `Omega-3`) sẽ được tự động chuyển số thành chữ qua `target_v3` (khớp âm học với hypothesis của Gipformer) giống như history.
- **Tập hội thoại lịch sử (`history`, 3,491 lượt user):** Áp dụng đầy đủ quy trình mở rộng số/ký hiệu qua `target_v3` và giữ nguyên toàn bộ (`--max-history-wer` mặc định tắt `None`, `skipped_rows=0`).

> [!NOTE]
> Các tùy chọn thử nghiệm thêm từ commit `d76a2c8` (`--verify-main`, `--max-main-wer`, `--eval-wer`, `--selection-metric wer`) đều ở chế độ **tắt mặc định (opt-in)** nên `prepare_gipformer_sft.py` tạo đúng 4,496 dòng `train.jsonl` + 540 dòng `dev.jsonl` khớp hoàn toàn bản gốc `result/asr_sft_data`.

Chạy lệnh (relative paths từ `code/`):
```bash
cd code

/usr/bin/python3.12 training_code/asr_finetune/prepare_gipformer_sft.py \
  --input-jsonl ../data/train_asr.jsonl \
  --data-root ../data \
  --model-dir saved_models/asr \
  --output-dir result/asr_sft_data \
  --val-fraction 0.1 \
  --seed 42
```
*Kết quả:* Tạo ra `train.jsonl` (4,496 mẫu), `dev.jsonl` (540 mẫu) và `manifest.json` trong `result/asr_sft_data` (`skipped_rows: 0`).

### 4.3. Chạy huấn luyện ASR ngầm bằng `nohup` (`train_gipformer.py`)
Huấn luyện Gipformer PyTorch RNNT loss trên GPU theo đúng cấu hình đã tạo `result/asr_sft/best.pt` (step 13,000, `dev_loss=1.1992`, chạy hết 3 epochs = 13,488 steps; nếu thư mục đích đã tồn tại thì đổi `--output-dir` sang tên mới như `result/asr_sft_repro`):
```bash
cd code

nohup /usr/bin/python3.12 training_code/asr_finetune/train_gipformer.py \
  --cuda-visible-devices 0 \
  --data-root ../data \
  --data-dir result/asr_sft_data \
  --model-dir saved_models/asr \
  --output-dir result/asr_sft \
  --epochs 3 \
  --learning-rate 3e-5 \
  --accum 4 \
  --max-dev-audio 300 \
  --augment-scope all \
  --selection-metric loss \
  --eval-every 1000 \
  --patience 5 \
  > ../runtime_cache/logs/train_asr.log 2>&1 &
```
*Kiểm tra tiến trình:* Xem log thời gian thực bằng `tail -f ../runtime_cache/logs/train_asr.log`.

### 4.4. Xuất mô hình fine-tuned sang ONNX FP32 (`export_gipformer.py`)
Sau khi huấn luyện xong, chuyển checkpoint tốt nhất `best.pt` thành định dạng ONNX FP32 đặt trong `saved_models/asr_ft`:
```bash
cd code

/usr/bin/python3.12 training_code/asr_finetune/export_gipformer.py \
  --best result/asr_sft/best.pt \
  --original-dir saved_models/asr \
  --output-dir saved_models/asr_ft \
  --data-root ../data
```
*Xác nhận:* Thư mục `saved_models/asr_ft` sẽ chứa đầy đủ `encoder.onnx`, `decoder.onnx`, `joiner.onnx`, `tokens.txt`, `bpe.model` và tệp kê khai `lifemate_manifest.json`.

Inference nạp các file ONNX đã xuất trong `saved_models/asr_ft`, không nạp trực tiếp
`result/asr_sft/best.pt`. Sau mỗi lần train ASR tạo checkpoint mới, cần chạy lại
bước export này để cập nhật ONNX trước khi tạo transcript mới.

---

## 5. BƯỚC 3: FINE-TUNE MÔ HÌNH LLM LORA ADAPTER

Quá trình fine-tune LLM huấn luyện mô hình `Qwen3.5-4B` thực hiện gọi công cụ (tool call) chính xác theo giọng nói và văn cảnh người dùng.

Pipeline đã tích hợp toàn bộ các kỹ thuật tối ưu hóa cao nhất:
- **Step-wise SFT (mặc định bật `--step-wise-sft`):** Mỗi lượt quyết định là một dòng SFT riêng biệt, đồng bộ prefix `<think>\n\n</think>\n\n` giữa toàn bộ các bước đọc/ghi trong lúc train và lúc suy luận (`--no-step-wise-sft` để chuyển về full-trajectory packing).
- **Dual Gold + ASR (`--transcript-mode dual`):** Nhân đôi dữ liệu với 100% bản chép lời chuẩn (Gold) và 100% bản nhận dạng giọng nói ASR. Prompt của mỗi bản có nhãn nguồn riêng để model phân biệt.
- **Weighted Loss Trainer (`--fuzzy-weight 0.3`):** Giảm trọng số phạt sai sót trên các tham số tìm kiếm mờ (`fuzzy`), ưu tiên độ chính xác của lệnh gọi công cụ (`tool_call`).
- **Lọc ASR lỗi nặng (`--filter-damaged-asr`):** Bỏ qua các bản sao ASR nếu bị ASR làm mất hoàn toàn tham số thực thi chuẩn.
- **Search Recovery Trajectories:** Bổ sung kịch bản xử lý tự phục hồi khi kết quả tìm kiếm rỗng.
- **Self-Correction Guard Recovery (`--guard-recovery-ratio 0.18`):** Bổ sung các mẫu phục hồi đơn bước khi gặp từ chối từ guard (`xref_guard`, schema `validate`, và hỏi thừa `missing_info_guard`). Chỉ tính loss trên lượt sửa đúng, không dạy sinh lệnh hỏi thừa.
- **Guard dùng chung cho SFT và inference:** Kiểm tra nguồn ID `x-ref`, giới hạn hai lượt đọc rỗng/lỗi cho mỗi tool khi trạng thái chưa đổi, chặn gọi lặp `local_*` helper với cùng tham số đã có kết quả (`loop_guard`), và chặn `sys.ask_missing_info` khi tham số sai tên hoặc giá trị đã xác định chắc chắn từ dữ kiện nhìn thấy.
- **Short Checked Reasoning mặc định tắt (`--no-short-reasoning`):** Giữ đồng bộ 100% với chế độ suy luận `enable_thinking=False`, `thinking_budget=0` (có thể bật lại khi thử nghiệm bằng `--short-reasoning`).
- **Pre-computed `date_reference`:** Tự động giải quyết thời gian biểu mà không cần sinh các lệnh gọi helper thời gian dư thừa.

### 5.1. Chạy Fine-tune LLM toàn diện bằng `nohup` (Cấu hình `run_sft_step` — Khuyên dùng)
Chạy kịch bản 1 lệnh [`finetune.py`](file:///mnt/mmlab2024nas/trantran/lifemate/code/training_code/finetune.py) để tái lập đúng cấu hình **`run_sft_step`** (được chuẩn bị từ commit `79fdce3` ngày 13/11, dùng Step-wise SFT 3 epochs, kích hoạt Guard Recovery, tắt Short Checked Reasoning để đồng bộ `thinking_budget=0`, và tạo transcript SFT chỉ bằng `asr_ft` cũ ở `beam=4`, tắt hotwords & post-normalize):

> [!IMPORTANT]
> Cả `training_code/finetune.py` và `training_code/prepare_sft.py` hiện đã đặt **mặc định theo cấu hình `run_sft_step`** (`--step-wise-sft`, `--asr-beam-size 4`, `--asr-hotwords off`, `--no-asr-post-normalize`, `--asr-threads 4`, `--sft-profile run_sft_step`, và `--filter-damaged-asr` mặc định bật trên `finetune.py`), còn `train_lora.py` giữ nguyên quy trình huấn luyện LoRA (`epochs=3`, `lr=1e-4`, `lora_r=16`, `lora_alpha=32`, `fuzzy_weight=0.3`, `seed=42`). Phần prompt tĩnh chia sẻ với inference (`SYSTEM_PROMPT` trong [agent.py](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/agent.py) và `date_reference` trong [local_time.py](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/local_time.py)) đã được đồng bộ đúng theo `run_sft_step`, còn khác biệt của các module guard/fallback mới dùng cho inference được cô lập qua profile `--sft-profile run_sft_step` trong [`sft_run_step_compat.py`](file:///mnt/mmlab2024nas/trantran/lifemate/code/training_code/sft_run_step_compat.py). Nhờ đó, chạy `prepare_sft.py` / `finetune.py` với cấu hình mặc định tái lập đúng từng byte `result/run_sft_step/sft/train.jsonl` (`10,724` dòng, `sha256: a32459e8a200626f846a519b17fffa33427905838fcd220a076f54b182ab0996`) mà không ảnh hưởng các cải tiến inference. Nếu thư mục `result/run_sft_step` đã tồn tại và không rỗng, đổi `--output-dir` sang tên thư mục mới (ví dụ `result/run_sft_step_repro`) vì script không ghi đè thư mục cũ.

`finetune.py` và `prepare_sft.py` mặc định dùng `saved_models/asr_ft` để tạo
transcript ở **cả dual và mix**, cùng nguồn ASR fine-tune với inference. Các lệnh
bên dưới dùng relative path từ `code/`. Nếu chưa có bộ ONNX fine-tune đầy đủ, chạy
bước export 4.4; chương trình báo lỗi thay vì tự dùng ASR gốc. Có thể chỉ định
model khác bằng `--asr-model` khi chủ động làm ablation.

Đổi model ASR cần dựng lại SFT trong thư mục output mới; dữ liệu SFT đã tạo bằng
ASR gốc không tự được cập nhật. Chế độ chỉ dùng `audio_text` (tỷ lệ ASR bằng 0,
không có audio history cần nhận dạng) và tập chỉ gồm synthetic không nạp ASR.

Khi dual thực sự tạo hai bản (`0 < --asr-transcript-ratio < 1`), prompt của mẫu
gốc có `request_transcript_source: "audio_text"` hoặc `"asr"`, cùng
`request_transcript_note` giải thích nguồn. Bản ASR nhắc có thể nhận dạng sai và
cần đối chiếu context/history/kết quả đọc; không hỏi thêm chỉ vì nguồn là ASR.
Mỗi bản chỉ chứa transcript của chính nó. Nhãn áp dụng cho yêu cầu hiện tại,
không gán nguồn cho toàn bộ history. Recovery, augmentation và full-trajectory
của bản dual giữ cùng nhãn. Chế độ mix, các ablation tỷ lệ 0/1, mẫu synthetic,
partition chẩn đoán và prompt inference không thêm nhãn này.

```bash
cd code

nohup /usr/bin/python3.12 training_code/finetune.py \
  --cuda-visible-devices 0 \
  --data-root ../data \
  --base-model models/llm \
  --output-dir result/run_sft_step \
  --asr-model saved_models/asr_ft \
  --transcript-mode dual \
  --asr-transcript-ratio 0.5 \
  --asr-precision fp32 \
  --asr-decoding-method modified_beam_search \
  --asr-beam-size 4 \
  --asr-hotwords off \
  --no-asr-post-normalize \
  --asr-threads 4 \
  --asr-batch-size 16 \
  --filter-damaged-asr \
  --guard-recovery-ratio 0.18 \
  --step-wise-sft \
  --no-short-reasoning \
  --validation-fraction 0.0 \
  --missing-tool-policy error \
  --guard-conflict-policy skip \
  --max-length 16384 \
  --seed 42 \
  --backend unsloth \
  --quantization none \
  --fuzzy-weight 0.3 \
  --reasoning-weight 0.25 \
  --epochs 3 \
  --batch-size 1 \
  --gradient-accumulation 8 \
  --learning-rate 1e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.0 \
  --save-strategy epoch \
  > ../runtime_cache/logs/train_llm_step.log 2>&1 &
```

*Kiểm tra tiến trình:*
```bash
tail -f ../runtime_cache/logs/train_llm_step.log
```
*Kết quả đầu ra:* Adapter hoàn chỉnh sẽ được lưu tại:
`result/run_sft_step/training/adapter`

### 5.2. Chạy tách riêng từng bước (Dành cho Debug hoặc tùy biến)
Nếu bạn muốn chia nhỏ để kiểm tra tập dữ liệu trước khi train:

#### Bước 3a: Chuẩn bị dữ liệu SFT
Lệnh sau chuẩn bị bộ SFT Step-wise chuẩn của `run_sft_step` (`dual`, tỷ lệ ASR `0.5`, toàn bộ tập train, `asr_ft` thuần `beam=4`, `--asr-hotwords off`, `--no-asr-post-normalize`). Nếu thư mục đích đã có dữ liệu, chọn tên output mới; `prepare_sft.py` không ghi đè SFT cũ (có thể thay `--transcript-mode mix` nếu muốn chạy ablation `mix`).

```bash
cd code

/usr/bin/python3.12 training_code/prepare_sft.py \
  --data-root ../data \
  --tokenizer models/llm \
  --output-dir result/sft_step_dual \
  --asr-model saved_models/asr_ft \
  --asr-precision fp32 \
  --asr-decoding-method modified_beam_search \
  --asr-beam-size 4 \
  --asr-hotwords off \
  --no-asr-post-normalize \
  --asr-threads 4 \
  --asr-batch-size 16 \
  --transcript-mode dual \
  --asr-transcript-ratio 0.5 \
  --filter-damaged-asr \
  --guard-recovery-ratio 0.18 \
  --step-wise-sft \
  --no-short-reasoning \
  --validation-fraction 0.0 \
  --missing-tool-policy error \
  --guard-conflict-policy skip \
  --max-length 16384 \
  --seed 42
```

Kiểm tra `asr` trong `result/sft_step_dual/manifest.json`: `model_path` trỏ vào
`saved_models/asr_ft`, `precision` là `fp32`, `max_active_paths` là `4`, và không bật `hotwords_mode`/`post_normalize`. Nhãn nguồn trong prompt chỉ bật cho `dual`; `mix` giữ định dạng prompt hiện có. Mẫu synthetic luôn dùng một bản `audio_text`, không qua ASR.

#### Bước 3b: Huấn luyện LoRA với Unsloth
```bash
cd code

nohup /usr/bin/python3.12 training_code/train_lora.py \
  --cuda-visible-devices 0 \
  --base-model models/llm \
  --sft-dir result/sft_step_dual \
  --output-dir result/training_lora_step \
  --backend unsloth \
  --quantization none \
  --max-length 16384 \
  --fuzzy-weight 0.3 \
  --reasoning-weight 0.25 \
  --epochs 3 \
  --batch-size 1 \
  --gradient-accumulation 8 \
  --learning-rate 1e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.0 \
  --save-strategy epoch \
  --seed 42 \
  > ../runtime_cache/logs/train_lora_only.log 2>&1 &
```

### 5.3. Gộp LoRA Adapter vào Base Model (`merge_lora.py`)
Model đã merge sẵn vào `code/saved_models/llm_ft/` từ adapter `run_sft_step`. Nếu muốn xuất lại từ adapter khác vào thư mục mới:
```bash
cd code

/usr/bin/python3.12 training_code/merge_lora.py \
  --base-model models/llm \
  --adapter result/run_sft_step/training/adapter \
  --output-dir saved_models/llm_ft \
  --dtype bfloat16
```

---

### 5.4. Bổ sung mẫu synthetic không có audio hoặc ToolEnv

Giữ cách chọn tập train và catalog hiện tại. Trong tập train đã ghép, mẫu được nhận diện
là synthetic nếu `id` bắt đầu bằng `synthetic_` hoặc có trường `"synthetic": true`.
Không dùng việc thiếu file WAV để nhận diện synthetic, nhằm giữ lỗi dữ liệu audio của
mẫu gốc được phát hiện như trước.

- Mẫu gốc vẫn dùng ToolEnv và chính sách `dual`/`mix` đang chọn. Tỷ lệ ASR chỉ tính
  trên mẫu gốc; thêm synthetic không thay đổi lịch chọn transcript của chúng.
- Synthetic phải có `audio_text` không rỗng, `label.tool_calls` và `label.tool_results`
  là hai danh sách cùng độ dài. Mỗi kết quả tool phải là một JSON object; các call
  phải phù hợp với schema trong catalog và `context.installed_apps` của mẫu.
- Synthetic chỉ tạo **một nguồn transcript `audio_text`**, kể cả khi bật `dual` hoặc
  tỷ lệ ASR bằng 1. Không chạy ASR cho audio hiện tại hay history synthetic. History
  cần có sẵn text hoặc `audio_text`; đường dẫn WAV giả trong synthetic được bỏ qua.
- Trajectory synthetic được dựng lần lượt từ call/result đã ghi: chỉ sau call thứ
  `i` mới thêm result thứ `i` vào observations. `label.final_state` không đưa vào prompt.
  Không khởi tạo ToolEnv, không yêu cầu/tạo state hash và không chạy search recovery
  hoặc guard recovery cho các mẫu này.
- Việc gộp call, augmentation và reasoning SFT vẫn áp dụng theo cấu hình, nên một
  nguồn transcript có thể tạo nhiều dòng huấn luyện. Kiểm tra schema cục bộ không
  chứng minh nghiệp vụ hoặc trạng thái cuối đã được thực thi đúng.

Synthetic chỉ nằm trong partition train, không được đưa vào rollout validation hoặc
app-heldout. Nếu synthetic lộ app đang giữ riêng cho app-heldout, bước prepare báo lỗi.
Manifest ghi `synthetic_training`, `synthetic_samples`, `trajectory_sources` và
`transcript_sources` để kiểm tra nguồn dữ liệu. Tập chỉ gồm synthetic dùng
`--asr-transcript-ratio 0` và không cần `--asr-model` hay module ToolEnv.

### 5.5. Guard hỏi thừa và dữ liệu mâu thuẫn

Guard `sys.ask_missing_info` chạy trước `env.validate/call` ở cả inference và lúc
dựng SFT. Nó đối chiếu tên tham số với schema, các giá trị cố định (hoặc mặc định khi
người dùng chọn dùng mặc định), phép gán
trường rõ ràng trong yêu cầu, entity khớp duy nhất từ context/history/kết quả đọc thành
công, và ngày giờ tính được bằng helper Python hiện có. Khi đã biết một lựa chọn
hợp lệ của `x-one-of`, không cần hỏi lựa chọn còn lại. Thông tin mơ hồ hoặc thực sự
chưa có vẫn được hỏi, kể cả trường optional cần thiết cho yêu cầu cụ thể.

Nếu bị chặn, model nhận `agent_feedback` gồm bằng chứng, `known_arguments` và phần
còn thiếu để tiếp tục sửa trong giới hạn `max_invalid` hiện có. Guard không tự gọi
write hay lấy đáp án từ nhãn gold. Các mẫu recovery hỏi thừa chỉ được chèn trên mẫu
gốc theo `--guard-recovery-ratio`; synthetic vẫn không chạy recovery qua ToolEnv.

Khi gold mâu thuẫn với guard, prepare mặc định **bỏ cả sample**, ghi rõ lý do và
phản hồi guard vào `<partition>_skipped.jsonl`. Dùng `--guard-conflict-policy error`
để dừng ngay khi gặp mâu thuẫn (hỗ trợ cả `finetune.py` và `prepare_sft.py`). Không
sửa nhãn hay sửa file train nguồn. Manifest ghi `training_guards` để kiểm tra cấu hình.

SFT (gốc và synthetic), recovery và inference đều tắt giới hạn số read trả rỗng
hoặc chưa hoàn tất (`empty_read_limit=0`). Model được đọc tiếp với arguments khác;
guard vẫn chặn lặp nguyên read + arguments khi trạng thái chưa đổi. Guard ngăn
tự đoán ID (`enforce_observed_xref=True`) và các giới hạn tổng call/invalid/step
vẫn giữ nguyên. Chạy lại inference để áp dụng; dựng lại SFT nếu muốn khôi phục các
mẫu từng bị bỏ vì chạm ngưỡng read rỗng cũ.

Sau thay đổi này, cần dựng lại SFT ở thư mục output mới với policy
`full-trajectory-date-apps-v17-dual-transcript-source`, rồi train adapter từ dữ liệu mới.
Guard inference có hiệu lực ngay khi chạy code mới; adapter cũ chưa học các mẫu
recovery mới. Đường dẫn dataset/catalog và cách chọn dual của mẫu gốc giữ nguyên.

### 5.6. Search Fallback ở tầng Guard cho tên riêng bị ASR phiên âm sai (Ưu tiên 2B, Zero-Retraining)
Cơ chế Search Fallback 2B chạy tự động trong tầng điều phối Python ([`code/lifemate/tool_execution.py`](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/tool_execution.py) và [`code/lifemate/execution_policy.py`](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/execution_policy.py)) và **không làm thay đổi các hằng số policy trong `lifemate_runtime.json` hay schema tool**, nên dùng được ngay với mọi checkpoint LoRA/LLM đã train trước đó mà **không cần train lại**:
- **Gợi ý Fallback tự động trong `verification` & `LoopMonitor`:** Khi lệnh `read` trả về rỗng (`status="empty"`) hoặc `stocks.get_quote` trả về `done=False`, Python tự đính kèm trường `search_fallback` (chỉ rút danh xưng đầu chuỗi nếu có, trích các cụm mốc địa điểm/sự kiện tiếng Việt từ `transcript` như `"quán ..."`, tra ứng viên gần đúng trong `context`/`history`, và gợi ý `fallback_call` bằng `{}` / `{"query": "*"}` hoặc `stocks.get_portfolio({})`). Khi model lặp lại nguyên call rỗng/lỗi đó, `LoopMonitor` chặn và lặp lại trực tiếp `search_fallback` để model không bị kẹt ở `max_invalid=3`.
- **Tự động chuyển hướng truy vấn chắc chắn rỗng (`prepare_call_arguments`):** Giữ nguyên hoàn toàn các lệnh lặp trùng khớp tuyệt đối (để `LoopMonitor` xử lý đúng chuẩn) và các truy vấn mới độc lập, nhưng tự động chuyển hướng sang mốc tiếng Việt trong `transcript` hoặc lệnh liệt kê (`{}` / `{"query": "*"}`) đối với các thử lại chắc chắn 100% thất bại: (1) nối dài thêm ngày giờ/từ vào một `query` đã rỗng trước đó, (2) tách nhỏ hoặc thêm `near` vào tập token đã chứng minh rỗng trên `maps.search_place` hay `_shop_match`, và (3) gọi lại `stocks.get_quote` chỉ khác chữ hoa/thường sau khi mã cổ phiếu đã báo lỗi.

## 6. BƯỚC 4: CHẠY SUY LUẬN (INFERENCE PIPELINE)

Giai đoạn suy luận đọc tập kiểm thử (`public_test` hoặc `private_test`, hoặc `--split auto`), phiên âm giọng nói qua ASR, gửi văn bản và schema công cụ cho LLM, thực thi môi trường `ToolEnv` (import trực tiếp từ thư mục split của BTC) và xuất tệp nộp bài `submission.jsonl` (mỗi dòng gồm `id`, `tool_calls`, `state_hash` và thêm `time` bằng mili-giây ở chế độ online/BTC).

Với `--thinking-budget 0` (mặc định), inference dùng **guard chọn token** để đầu ra
bắt đầu trực tiếp bằng `<tool_call>` ở native function calling hoặc `{` cho JSON.
Prefix mở `<think>` và lời suy luận đứng trước hành động bị chặn ngay khi sinh,
không chỉ xóa khỏi kết quả sau đó. Guard giữ nguyên nội dung tham số, kể cả chuỗi
literal `<think>...</think>` bên trong dữ liệu tool. Đây là kiểm tra đầu ra bắt đầu
đúng dạng; parser và schema tiếp tục kiểm tra phần JSON/XML còn lại.

Guard chạy ở cả đường có cache và fallback. Nếu chủ động đặt ngân sách thinking
lớn hơn 0 và còn đủ context/thời gian, cơ chế thinking có giới hạn vẫn hoạt động;
nếu ngân sách hiệu lực về 0 thì guard tự bật. SFT/Short Checked Reasoning giữ cấu
hình riêng. Khởi động lại tiến trình inference để áp dụng code mới với adapter hiện có.

Cách chọn ASR & model phụ thuộc entrypoint chạy:

- [`predict.py`](file:///mnt/mmlab2024nas/trantran/lifemate/code/predict.py) / [`predict.sh`](file:///mnt/mmlab2024nas/trantran/lifemate/code/predict.sh) / [`benchmark_timing.py`](file:///mnt/mmlab2024nas/trantran/lifemate/code/benchmark_timing.py) / [`predict_notebook.ipynb`](file:///mnt/mmlab2024nas/trantran/lifemate/code/predict_notebook.ipynb): tự chọn `saved_models/asr_ft` nếu `saved_models/asr_ft/tokens.txt` tồn tại và không rỗng (ngược lại dùng `saved_models/asr`); tự chọn `saved_models/llm_ft` nếu có `config.json` (ngược lại dùng `saved_models/llm`, và nạp thêm `saved_models/llm_adapter` nếu có `adapter_config.json`), hỗ trợ ghi đè qua `LIFEMATE_ASR_MODEL`, `LIFEMATE_LLM_MODEL`, `LIFEMATE_LLM_ADAPTER`. Toàn bộ tham số suy luận sản xuất (`seed=42`, `asr_beam_size=8`, `asr_hotwords=auto`, `asr_hotwords_score=1.5`, `asr_post_normalize=True`, `asr_threads=8`) được cố định đồng nhất giữa `predict.py`, `benchmark_timing.py` và `predict_notebook.ipynb`, nên **tự động kích hoạt Hybrid ASR** mới nhất trong [`lifemate/asr_biasing.py`](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/asr_biasing.py) mà không cần export lại ONNX.
- [`run_server.sh`](file:///mnt/mmlab2024nas/trantran/lifemate/code/run_server.sh): tự chọn `saved_models/asr_ft` theo điều kiện `tokens.txt` ở trên; nếu chưa có thì chọn `saved_models/asr`, và chọn `saved_models/llm_ft` (nếu có `config.json`) hoặc `saved_models/llm`. Mặc định `--data-root` là relative `../data`, `STAGE=online`, và tự bật Hybrid ASR (`ASR_BEAM_SIZE=8`, `ASR_HOTWORDS=auto`, `ASR_HOTWORDS_SCORE=1.5`, `ASR_POST_NORMALIZE=1`).
- `main.py` (`--stage online` mặc định, hoặc `--stage full` / `--stage asr`): dùng chính xác thư mục truyền qua `--asr-model`, không tự chọn checkpoint mới nhất hoặc tự export `best.pt`. Lưu ý bản thân `main.py` giữ mặc định bảo thủ (`--asr-beam-size 4`, `--asr-hotwords off`, `--no-asr-post-normalize`), nên khi gọi trực tiếp `main.py` cần truyền đủ `--asr-beam-size 8 --asr-hotwords auto --asr-hotwords-score 1.5 --asr-post-normalize --asr-threads 8` (như các lệnh ở Mục 6.4/6.5) để kích hoạt Hybrid ASR.
- `main.py --stage llm`: chỉ đọc transcript cache; không nạp hay chuyển model ASR.

> [!NOTE]
> - Hybrid ASR nằm hoàn toàn trong logic Python tại [`lifemate/asr_biasing.py`](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/asr_biasing.py) (áp dụng khi `hotwords=auto` và `asr_post_normalize=True`), nên các entrypoint `--stage online` (`predict.sh`, `predict.py`, `run_server.sh`, `benchmark_timing.py`, `predict_notebook.ipynb`) **tự động dùng Hybrid ASR ngay lập tức**.
> - Các cải tiến suy luận sau `79fdce3` gồm Hybrid ASR (`asr_biasing.py`, `asr.py`, `transcripts.py`) và các lớp guard / loop recovery / fallback (`tool_execution.py`, `execution_policy.py`, `missing_info_guard.py`, `predictor.py`, `runtime.py`) đều được giữ nguyên ở inference; riêng prompt tĩnh (`SYSTEM_PROMPT` trong [`lifemate/agent.py`](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/agent.py) và `date_reference()` trong [`lifemate/local_time.py`](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/local_time.py)) đã được đồng bộ khớp hoàn toàn với weights `saved_models/llm_ft` (`run_sft_step`).
> - Chỉ khi dùng chế độ legacy `--stage full` / `--stage asr` / `--stage llm` mới cần chú ý: `asr_identity` trong manifest chỉ hash file ONNX và cờ CLI (không hash mã Python `asr_biasing.py`). Vì vậy, nếu trong `result/` đã có cache cũ (ví dụ `result/transcripts.jsonl`), **bắt buộc truyền `--refresh-transcripts` hoặc đổi sang file `--transcript-cache` mới** (như `result/transcripts_asr_hybrid.jsonl`) để tránh tái sử dụng transcript trước khi cập nhật Hybrid ASR.

---

### 6.1. Kịch bản 1: Chạy chuẩn nộp bài BTC (`predict.sh` / `predict.py`) & Docker

Phục vụ chạy chuẩn theo tài liệu BTC (`bash /code/predict.sh`, `/data` mount read-only, tự nhận diện split, xuất `/result/submission.jsonl` và in thời gian load model + thống kê latency):

#### Chạy trực tiếp trên máy cục bộ (tự dùng `saved_models/asr_ft` + `saved_models/llm_ft` đã merge)
```bash
cd code

# Chạy predict.sh trỏ tới ../data để kiểm thử local (không cần truyền --llm-adapter vì saved_models/llm_ft đã merge run_sft_step)
PYTHON_BIN="/usr/bin/python3.12" \
DATA_ROOT="../data" \
SPLIT="public_test" \
OUTPUT="result/submission.jsonl" \
bash predict.sh
```

#### Đóng gói và kiểm thử bằng Docker offline theo BTC (`Dockerfile`)
> [!IMPORTANT]
> - `code/saved_models/llm_ft/` đã được gộp sẵn từ adapter `run_sft_step` thành **thư mục thật**, và symlink `code/saved_models/llm` cũ đã được xóa để `docker build` (`COPY saved_models/ /code/saved_models/`) không chép nhầm symlink hỏng (`code/models/llm/` đã được `.dockerignore` loại khỏi context).
> - Theo cấu hình mặc định chuẩn BTC, `predict.sh` / `predict.py` / `predict_notebook.ipynb` trong Docker tự dùng `saved_models/llm_ft` + `saved_models/asr_ft` với mặc định **Hybrid ASR** (`lifemate/asr_biasing.py` + `--asr-beam-size 8`, `--asr-hotwords auto`, `--asr-hotwords-score 1.5`, `--asr-post-normalize`, `--asr-threads 8`). Sau mỗi lần sửa `lifemate/asr_biasing.py`, cần `docker build` lại để cập nhật mã Python vào image.

```bash
cd code

# 1. Build image BTC sạch (chứa saved_models/asr_ft + saved_models/llm_ft đã merge run_sft_step + Hybrid ASR mới nhất)
docker build -t lifemate-zac2026:latest .

# 2a. Chạy chuẩn môi trường chấm bài BTC (--network none, /data read-only; tự dùng llm_ft + Hybrid ASR beam=8/auto/post_normalize)
docker run --rm --gpus '"device=0"' --network none \
  -v "$(cd ../data/public_test && pwd)":/data/public_test:ro \
  -v "$(pwd)/result":/result \
  lifemate-zac2026:latest bash /code/predict.sh

# 2b. (Tuỳ chọn) Test nhanh một adapter chưa merge trên Docker KHÔNG cần merge_lora / docker build lại
# (Mount base models/llm + adapter từ host và truyền LIFEMATE_LLM_MODEL + LIFEMATE_LLM_ADAPTER)
docker run --rm --gpus '"device=0"' --network none \
  -e LIFEMATE_LLM_MODEL=/llm \
  -e LIFEMATE_LLM_ADAPTER=/adapter \
  -v "$(cd models/llm && pwd)":/llm:ro \
  -v "$(cd result/run_sft_step/training/adapter && pwd)":/adapter:ro \
  -v "$(cd ../data/public_test && pwd)":/data/public_test:ro \
  -v "$(pwd)/result":/result \
  lifemate-zac2026:latest bash /code/predict.sh \
    --asr-beam-size 8 \
    --asr-hotwords auto \
    --asr-hotwords-score 1.5 \
    --asr-post-normalize \
    --asr-threads 8

# 3a. Khởi động JupyterLab trên image sạch đã merge (BTC chạy predict_notebook.ipynb tại cổng 9777, token/pass: zac2026)
docker run --rm --gpus '"device=0"' --network none -p 9777:9777 \
  -v "$(cd ../data/public_test && pwd)":/data/public_test:ro \
  -v "$(pwd)/result":/result \
  lifemate-zac2026:latest bash /code/start_jupyter.sh

# 3b. (Tuỳ chọn) Khởi động JupyterLab để test nhanh một adapter chưa merge
docker run --rm --gpus '"device=0"' --network none -p 9777:9777 \
  -e LIFEMATE_LLM_MODEL=/llm \
  -e LIFEMATE_LLM_ADAPTER=/adapter \
  -v "$(cd models/llm && pwd)":/llm:ro \
  -v "$(cd result/run_sft_step/training/adapter && pwd)":/adapter:ro \
  -v "$(cd ../data/public_test && pwd)":/data/public_test:ro \
  -v "$(pwd)/result":/result \
  lifemate-zac2026:latest bash /code/start_jupyter.sh
```

---

### 6.2. Kịch bản 2: Đo lường thời gian & kiểm tra Parity (`benchmark_timing.py` & `predict_notebook.ipynb`)

Thực hiện chính xác 4 bước theo Phụ lục Mục 7 của BTC (Set Seed -> Load ASR + LLM -> Đọc dataset & import `ToolEnv` từ thư mục split -> Loop từng sample bấm giờ từ trước khi đọc audio, ghi & flush ra `/result/jupyter_submission.jsonl`) và kiểm tra đối chiếu tuyệt đối `id`, `tool_calls`, `state_hash` với `submission.jsonl`:

```bash
cd code

/usr/bin/python3.12 benchmark_timing.py \
  --cuda-visible-devices 0 \
  --data-root ../data \
  --split public_test \
  --output result/jupyter_submission.jsonl \
  --compare result/submission.jsonl \
  --strict-compare \
  --report-json result/timing_report.json
```
*Ghi chú:* `benchmark_timing.py` tự nạp `saved_models/llm_ft` đã merge từ `run_sft_step` (không cần `--llm-adapter`). Nếu muốn test một adapter chưa merge khác, truyền thêm `--llm-model models/llm --llm-adapter <adapter_dir>`. Trong Docker của BTC, mở [`predict_notebook.ipynb`](file:///mnt/mmlab2024nas/trantran/lifemate/code/predict_notebook.ipynb) trên JupyterLab (`http://<host>:9777`, mật khẩu/token `zac2026`) và chạy tuần tự 4 cell code để sinh `/result/jupyter_submission.jsonl` có kết quả giống hệt `submission.jsonl`.

---

### 6.3. Kịch bản 3: Chạy nhanh qua launcher `run_server.sh` (Mặc định `--stage online`)
Script [`code/run_server.sh`](file:///mnt/mmlab2024nas/trantran/lifemate/code/run_server.sh) mặc định:

- Dùng `DATA_ROOT=../data` (relative), ưu tiên chọn `saved_models/asr_ft` theo điều kiện `tokens.txt` ở trên; nếu chưa có sẽ dùng `saved_models/asr`.
- Tự ưu tiên chọn `saved_models/llm_ft` khi `saved_models/llm_ft/config.json` tồn tại (do đó **không** truyền thêm `--llm-adapter` khi `llm_ft` đã merge, tránh lệch fingerprint/chồng adapter).
- Tự động bật cấu hình tối ưu hóa **Hybrid ASR** (`1A + 1B + Hybrid 2A` trong [`lifemate/asr_biasing.py`](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/asr_biasing.py)): `ASR_BEAM_SIZE=8`, `ASR_HOTWORDS=auto`, `ASR_HOTWORDS_SCORE=1.5`, `--asr-post-normalize` (`ASR_POST_NORMALIZE=1`, đặt `0` để tắt).
- Chạy chế độ mặc định `STAGE=online` (nạp đồng thời ASR + LLM trong 1 tiến trình, nhận dạng live từng sample theo Hybrid ASR và ghi thêm `"time"` ms). Nếu chuyển sang `STAGE=full`, cần kèm `--refresh-transcripts` hoặc đặt `TRANSCRIPT_CACHE="result/transcripts_hybrid.jsonl"` để không dùng nhầm `result/transcripts.jsonl` cũ.

Chạy ngầm với `nohup` (dùng đúng weights `saved_models/llm_ft` đã merge từ `run_sft_step` + Hybrid ASR mặc định của `run_server.sh`):
```bash
cd code

nohup env \
  PYTHON_BIN="/usr/bin/python3.12" \
  DATA_ROOT="../data" \
  SPLIT="public_test" \
  OUTPUT="result/submission_step_hybrid.jsonl" \
  DEVICE="cuda" \
  bash run_server.sh \
  --asr-precision fp32 \
  --asr-decoding-method modified_beam_search \
  --asr-beam-size 8 \
  --asr-threads 8 \
  --thinking-budget 0 \
  --prefix-cache auto \
  --schema-cache auto \
  > ../runtime_cache/logs/inference_step.log 2>&1 &
```

*(Tuỳ chọn)* Nếu muốn chạy qua `run_server.sh` với **adapter chưa merge** (`result/run_sft_step/training/adapter`) + Hybrid ASR:
```bash
cd code

nohup env \
  PYTHON_BIN="/usr/bin/python3.12" \
  DATA_ROOT="../data" \
  SPLIT="public_test" \
  LLM_MODEL="models/llm" \
  OUTPUT="result/submission_adapter_hybrid.jsonl" \
  DEVICE="cuda" \
  bash run_server.sh \
  --llm-adapter result/run_sft_step/training/adapter \
  --asr-precision fp32 \
  --asr-decoding-method modified_beam_search \
  --asr-beam-size 8 \
  --asr-threads 8 \
  --thinking-budget 0 \
  --prefix-cache auto \
  --schema-cache auto \
  > ../runtime_cache/logs/inference_step_adapter.log 2>&1 &
```

*Theo dõi tiến độ:*
```bash
tail -f ../runtime_cache/logs/inference_step.log
```

---

### 6.4. Kịch bản 4: Chạy trực tiếp bằng `main.py` (`--stage online`, tự kích hoạt Hybrid ASR)
Nếu muốn kiểm soát tuyệt đối từng tham số trên model đã merge (`saved_models/llm_ft`) hoặc adapter chưa merge (`models/llm` + `--llm-adapter`):
```bash
cd code

# 4a. Chạy trực tiếp với model đã merge (saved_models/llm_ft) + Hybrid ASR
nohup /usr/bin/python3.12 main.py \
  --cuda-visible-devices 0 \
  --stage online \
  --data-root ../data \
  --split public_test \
  --output result/submission.jsonl \
  --asr-model saved_models/asr_ft \
  --asr-precision fp32 \
  --asr-decoding-method modified_beam_search \
  --asr-beam-size 8 \
  --asr-hotwords auto \
  --asr-hotwords-score 1.5 \
  --asr-post-normalize \
  --asr-threads 8 \
  --llm-model saved_models/llm_ft \
  --llm-dtype bfloat16 \
  --prefix-cache auto \
  --schema-cache auto \
  --sample-schedule original \
  --max-input-tokens 16000 \
  --max-new-tokens 384 \
  --thinking-budget 0 \
  --max-calls 8 \
  --max-invalid 3 \
  --sample-seconds inf \
  --seed 42 \
  > ../runtime_cache/logs/inference_main.log 2>&1 &

# 4b. (Tuỳ chọn) Chạy với base model + LoRA adapter chưa merge (result/run_sft_step/training/adapter) + Hybrid ASR
nohup /usr/bin/python3.12 main.py \
  --cuda-visible-devices 0 \
  --stage online \
  --data-root ../data \
  --split public_test \
  --output result/submission_step_hybrid.jsonl \
  --asr-model saved_models/asr_ft \
  --asr-precision fp32 \
  --asr-decoding-method modified_beam_search \
  --asr-beam-size 8 \
  --asr-hotwords auto \
  --asr-hotwords-score 1.5 \
  --asr-post-normalize \
  --asr-threads 8 \
  --llm-model models/llm \
  --llm-adapter result/run_sft_step/training/adapter \
  --llm-dtype bfloat16 \
  --prefix-cache auto \
  --schema-cache auto \
  --sample-schedule original \
  --max-input-tokens 16000 \
  --max-new-tokens 384 \
  --thinking-budget 0 \
  --max-calls 8 \
  --max-invalid 3 \
  --sample-seconds inf \
  --seed 42 \
  > ../runtime_cache/logs/inference_main_adapter.log 2>&1 &
```

---

### 6.5. Kịch bản 5: Tách rời công đoạn ASR và LLM (`--stage asr` / `--stage llm`, Legacy Ablation)
Chỉ dùng cho thử nghiệm nội bộ khi muốn chạy Hybrid ASR một lần ra `transcripts_asr_hybrid.jsonl` rồi đánh giá nhiều adapter hoặc prompt khác nhau (không dùng cho Docker BTC). Truyền `--refresh-transcripts` để đảm bảo tạo mới theo [`lifemate/asr_biasing.py`](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/asr_biasing.py) hiện tại thay vì dùng lại cache cũ:

#### Công đoạn 1: Chỉ chạy ASR (Hybrid ASR)
```bash
cd code

nohup /usr/bin/python3.12 main.py \
  --stage asr \
  --data-root ../data \
  --split public_test \
  --transcript-cache result/transcripts_asr_hybrid.jsonl \
  --refresh-transcripts \
  --asr-model saved_models/asr_ft \
  --asr-precision fp32 \
  --asr-decoding-method modified_beam_search \
  --asr-beam-size 8 \
  --asr-hotwords auto \
  --asr-hotwords-score 1.5 \
  --asr-post-normalize \
  --asr-threads 8 \
  --asr-batch-size 16 \
  > ../runtime_cache/logs/stage_asr.log 2>&1 &
```

Kiểm tra nguồn model và cấu hình giải mã đã ghi vào manifest của cache:
```bash
rg -n '"model_path"|"precision"|"decoding_method"|"max_active_paths"|"hotwords_mode"|"hotwords_score"|"post_normalize"' \
  result/transcripts_asr_hybrid.manifest.json
```

#### Công đoạn 2: Chỉ chạy LLM (sử dụng lại transcript cache Hybrid ASR ở trên)
*(Dùng `--llm-model saved_models/llm_ft`, hoặc thay bằng `--llm-model models/llm --llm-adapter result/run_sft_step/training/adapter` nếu chạy trực tiếp trên adapter chưa merge.)*
```bash
cd code

nohup /usr/bin/python3.12 main.py \
  --cuda-visible-devices 0 \
  --stage llm \
  --data-root ../data \
  --split public_test \
  --transcript-cache result/transcripts_asr_hybrid.jsonl \
  --llm-model saved_models/llm_ft \
  --output result/submission.jsonl \
  --prefix-cache auto \
  --schema-cache auto \
  --sample-schedule schema \
  --max-input-tokens 16000 \
  --max-new-tokens 384 \
  --thinking-budget 0 \
  > ../runtime_cache/logs/stage_llm.log 2>&1 &
```

---

### 6.6. Kịch bản 6: Chạy Zero-Shot Inference (Không có Adapter)
Để đánh giá khả năng gốc của mô hình `Qwen3.5-4B` khi chưa qua fine-tune, bỏ tham số `--llm-adapter`:
```bash
cd code

nohup /usr/bin/python3.12 main.py \
  --cuda-visible-devices 0 \
  --stage online \
  --data-root ../data \
  --split public_test \
  --asr-model saved_models/asr_ft \
  --asr-precision fp32 \
  --asr-decoding-method modified_beam_search \
  --asr-beam-size 8 \
  --asr-hotwords auto \
  --asr-hotwords-score 1.5 \
  --asr-post-normalize \
  --asr-threads 8 \
  --llm-model models/llm \
  --output result/submission_zeroshot.jsonl \
  > ../runtime_cache/logs/inference_zeroshot.log 2>&1 &
```

---

## 7. BẢNG TRA CỨU THAM SỐ DÒNG LỆNH (CLI PARAMETERS)

### 7.1. Tham số ASR (Nhận dạng âm thanh)
| Tham số | Giá trị khuyên dùng | Các giá trị khả dĩ | Ý nghĩa & Mô tả |
| :--- | :--- | :--- | :--- |
| `--asr-model` | `saved_models/asr_ft` | Đường dẫn thư mục | Mặc định của `training_code/finetune.py` & `prepare_sft.py` cho dual và mix, không tự dùng ASR gốc khi thiếu weights. `main.py`/`predict.py` & launcher ưu tiên `saved_models/asr_ft` rồi mới dùng `saved_models/asr`. |
| `--asr-precision` | `fp32` | `auto`, `fp32`, `int8` | Độ chính xác ONNX. `fp32` cho WER tiếng Việt tốt nhất (~4.1% so với 4.8% của int8). |
| `--asr-decoding-method` | `modified_beam_search` | `modified_beam_search`, `greedy_search` | Thuật toán giải mã RNNT. Modified beam search vượt trội trên các từ hiếm. |
| `--asr-beam-size` | `8` (inference) / `4` (train `run_sft_step`) | Số nguyên >= 1 | Độ rộng chùm beam search (`max_active_paths`). Inference (`predict.py`, `run_server.sh`, `benchmark_timing.py`, `predict_notebook.ipynb`) mặc định `8`; toàn bộ training (`finetune.py`, `prepare_sft.py`) và `main.py` mặc định `4`. |
| `--asr-hotwords` | `auto` (inference) / `off` (train `run_sft_step`) | `off`, `auto` | Chế độ **1A** contextual BPE hotwords theo từng sample/turn (phiên bản **Hybrid ASR** trong `lifemate/asr_biasing.py`, mở rộng ngữ cảnh và kiểm soát thực thi cho tên app/loanword). Mặc định bật `auto` ở `predict.py`/`run_server.sh`/`predict_notebook.ipynb` và `off` ở `finetune.py`/`prepare_sft.py`/`main.py`. |
| `--asr-hotwords-score` | `1.5` | Số thực | Trọng số boost hotwords BPE khi `--asr-hotwords auto` (tránh >= `2.0` vì dễ ép nhầm hoặc cắt ngắn câu dài). |
| `--asr-post-normalize` | Bật (inference) / Tắt (train `run_sft_step`) | Bật / Tắt | Chế độ **Hybrid 2A** chuẩn hóa hậu ASR trong `lifemate/asr_biasing.py` (kết hợp quy tắc mã/tên riêng quen thuộc như `SE3`, `RON 95`, `VOV1`/`VOV2`/`VOV5`, `Compact K2` với sửa OOV loanword/biến thể hình thái tổng quát) trước `normalize_transcript`. Mặc định bật ở `predict.py`/`run_server.sh`/`predict_notebook.ipynb` và tắt (`False`, bật/tắt bằng `--no-asr-post-normalize`) ở `finetune.py`/`prepare_sft.py`/`main.py`. |
| `--asr-batch-size` | `16` | Số nguyên >= 1 | Kích thước batch gộp file âm thanh để decode song song offline. |
| `--asr-threads` | `8` (inference) / `4` (train `run_sft_step`) | Số nguyên >= 1 | Số luồng CPU sử dụng cho ONNX Runtime của Gipformer (`predict.py` mặc định `8`; `main.py`/`prepare_sft.py`/`finetune.py` mặc định `4`). |
| `--max-history-wer` | `None` (tái lập `asr_ft`) | `None` hoặc float `0..1` | Ngưỡng WER tối đa để lọc bỏ câu history nhiễu trong `prepare_gipformer_sft.py` / `train_gipformer.py` (`None` giữ toàn bộ 3,491 câu history như bản `asr_ft` gốc). |

### 7.2. Tham số Huấn luyện LLM SFT & LoRA
| Tham số | Giá trị khuyên dùng | Các giá trị khả dĩ | Ý nghĩa & Mô tả |
| :--- | :--- | :--- | :--- |
| `--transcript-mode` | `dual` | `dual`, `mix` | `dual`: nhân bản 100% Gold + 100% ASR; `mix`: chia tỷ lệ. |
| `--fuzzy-weight` | `0.3` | `0.0` đến `1.0` | Trọng số phạt loss trên các tham số `fuzzy` (giá trị < 1.0 giúp tránh overfitting chuỗi tìm kiếm). |
| `--reasoning-weight` | `0.25` | `0.0` đến `1.0` | Trọng số loss của token reasoning trong dữ liệu SFT; chỉ áp dụng khi chủ động bật `--short-reasoning` (mặc định tắt). |
| `--guard-recovery-ratio` | `0.18` | `0.0` đến `1.0` | Tỷ lệ mẫu chèn Guard Recovery (`xref_guard`, schema `validate`, hỏi thừa `missing_info_guard`). |
| `--reasoning-stepwise-ratio` | `0.5` | `0.0` đến `1.0` | Tỷ lệ sinh thêm mẫu reasoning từng bước khi bật `--short-reasoning` ở chế độ full-trajectory (`--no-step-wise-sft`). |
| `--no-inject-guard-recovery` | Tắt (cờ) | Bật / Tắt | Cờ tắt hoàn toàn việc chèn các mẫu Guard Recovery trong `prepare_sft.py`. |
| `--step-wise-sft` | Đã bật mặc định | Bật / Tắt | Xuất mỗi lượt quyết định thành một dòng SFT riêng (Step-wise SFT), giữ đồng bộ `<think>\n\n</think>\n\n` ở mọi bước trung gian; dùng `--no-step-wise-sft` để chuyển về full-trajectory packing. |
| `--short-reasoning` | Đã tắt mặc định | Bật / Tắt | Mặc định tắt (`--no-short-reasoning`) để đồng bộ `thinking_budget=0`; truyền `--short-reasoning` để sinh reasoning trong `prepare_sft.py` và `finetune.py`. |
| `--filter-damaged-asr` | Đã bật mặc định trên `finetune.py` | Bật / Tắt | Loại bỏ bản sao ASR khỏi SFT nếu ASR làm hỏng các đối số chuẩn của tool call. `finetune.py` mặc định bật (`--no-filter-damaged-asr` để tắt); `prepare_sft.py` bật bằng cờ `--filter-damaged-asr`. |
| `--sft-profile` | `run_sft_step` | `run_sft_step`, `current` | Chọn profile replay cho `prepare_sft.py` và `finetune.py`. `run_sft_step` (mặc định) kích hoạt hook tương thích trong `sft_run_step_compat.py` chỉ trong phạm vi replay để tái lập đúng từng byte `result/run_sft_step/sft/train.jsonl`; `current` dùng logic guard mới nhất ở `lifemate/`. |
| `--epochs` | `3` | Số thực > 0 | Số epoch huấn luyện LoRA (mặc định `3` cho Step-wise SFT). |
| `--batch-size` | `1` | Số nguyên | Batch size cục bộ trên mỗi GPU. |
| `--gradient-accumulation` | `8` | Số nguyên | Tích lũy gradient (tương đương effective batch size = 8). |
| `--learning-rate` | `1e-4` | Số thực | Tốc độ học của optimizer AdamW cho LoRA. |
| `--lora-r` | `16` | `8`, `16`, `32`, `64` | Hạng LoRA (Rank). |
| `--lora-alpha` | `32` | `16`, `32`, `64` | Hệ số tỷ lệ LoRA Alpha (thường đặt bằng `2 * r`). |
| `--lora-dropout` | `0.0` | `0.0` | Bắt buộc `0.0` để kích hoạt tối ưu hóa Unsloth Fast LoRA kernels. |

### 7.3. Tham số Suy luận LLM & Benchmarking (Inference)
| Tham số | Giá trị khuyên dùng | Các giá trị khả dĩ | Ý nghĩa & Mô tả |
| :--- | :--- | :--- | :--- |
| `--stage` | `online` (mặc định `main.py`) | `online`, `full`, `asr`, `llm` | `online` nạp đồng thời ASR + LLM trong 1 tiến trình theo chuẩn BTC; `full`/`asr`/`llm` chạy luồng tách cache cũ. |
| `--split` | `auto` (`predict.py`) / `public_test` (`main.py`) | `auto`, `train`, `public_test`, `private_test` | `auto` tự phát hiện thư mục split duy nhất trong `--data-root` (ví dụ `/data/private_test`). |
| `--prefix-cache` | `auto` | `auto`, `off` | Kích hoạt bộ nhớ đệm KV đa lượt tự động (tăng tốc độ suy luận gấp ~2.5 lần). |
| `--schema-cache` | `auto` | `auto`, `shared`, `off` | Tái sử dụng KV-cache các system prompt và định nghĩa JSON schema của các app. |
| `--sample-schedule` | `original` | `original`, `schema` | `online`/`predict.py` luôn giữ nguyên `original` theo thứ tự file gốc đúng chuẩn BTC; `schema` chỉ áp dụng khi chạy `--stage llm`. |
| `--max-input-tokens` | `16000` | Số nguyên | Giới hạn token đầu vào cho prompt; tổng input + output tối đa 16.384 tokens. |
| `--max-new-tokens` | `384` | Số nguyên | Toàn bộ ngân sách token đầu ra cho quyết định hoặc tool call mỗi lượt. |
| `--thinking-budget` | `0` | Số nguyên >= 0 | 0 tắt thinking và chặn prefix reasoning bằng logits guard, yêu cầu mở đầu JSON/tool call. Giá trị lớn hơn 0 cho phép thinking khi còn đủ ngân sách. |
| `--thinking-reserve-seconds` | `2.0` | Số thực >= 0 | Chỉ có tác dụng nếu chủ động bật thinking; không dùng với mặc định `--thinking-budget 0`. |
| `--max-calls` | `8` | Số nguyên | Giới hạn tối đa số bước gọi công cụ trên mỗi mẫu để tránh vòng lặp vô hạn. |
| `--max-invalid` | `3` | Số nguyên | Số lần tối đa môi trường cho phép gọi sai cú pháp trước khi hủy lượt. |
| `--sample-seconds` | `inf` | `inf` hoặc số giây | Giới hạn thời gian tối đa cho mỗi mẫu (mặc định `inf` = không giới hạn). |
| `--compare` / `--strict-compare` | *(chỉ `benchmark_timing.py`)* | Đường dẫn JSONL / Cờ | Đối chiếu `id`, `tool_calls`, `state_hash` với `submission.jsonl`; `--strict-compare` thoát mã lỗi khác 0 nếu lệch. |
| `--report-json` | *(chỉ `benchmark_timing.py`)* | Đường dẫn JSON | Lưu thống kê tổng (`mean`, `median`, `p95`, `max`, `asr_mean_ms`, `llm_agent_mean_ms`) và chi tiết từng sample. |

---

## 8. GIÁM SÁT & QUẢN LÝ TIẾN TRÌNH NỀN (NOHUP)

Vì các công đoạn huấn luyện và suy luận có thể mất từ vài chục phút đến vài giờ, việc chạy ngầm bằng `nohup` giúp tiến trình không bị gián đoạn khi ngắt kết nối SSH.

### 8.1. Kiểm tra log thời gian thực
Mỗi lệnh chạy nền đã được điều hướng ghi log vào thư mục `/mnt/mmlab2024nas/trantran/lifemate/runtime_cache/logs/`:
```bash
# Xem log huấn luyện ASR
tail -n 50 -f /mnt/mmlab2024nas/trantran/lifemate/runtime_cache/logs/train_asr.log

# Xem log huấn luyện LLM LoRA (run_sft_step)
tail -n 50 -f /mnt/mmlab2024nas/trantran/lifemate/runtime_cache/logs/train_llm_step.log

# Xem log suy luận
tail -n 50 -f /mnt/mmlab2024nas/trantran/lifemate/runtime_cache/logs/inference_step.log
```

### 8.2. Kiểm tra trạng thái GPU và tiến trình
```bash
# Giám sát trạng thái GPU, VRAM và tiến trình đang chiếm GPU
watch -n 1 nvidia-smi

# Tìm PID của các tiến trình Python đang chạy liên quan đến lifemate
ps aux | grep -E "predict.py|benchmark_timing.py|main.py|finetune.py|train_lora.py|train_gipformer.py" | grep -v grep
```

### 8.3. Dừng hoặc hủy tiến trình an toàn
Khi muốn dừng một tiến trình đang chạy ngầm:
```bash
# Gửi tín hiệu dừng an toàn (SIGTERM) qua PID
kill -15 <PID>

# Hoặc dừng toàn bộ tiến trình python liên quan đến tác vụ
pkill -f "train_gipformer.py"
pkill -f "main.py"
```

---

## 9. XỬ LÝ SỰ CỐ THƯỜNG GẶP (TROUBLESHOOTING)

### 9.1. Lỗi tràn bộ nhớ đĩa ổ cứng gốc (`No space left on device` tại `/tmp`)
- **Nguyên nhân:** Các thư viện `triton`, `torch` biên dịch JIT mặc định ghi vào thư mục `/tmp` của hệ điều hành. Khi thư mục này nhỏ sẽ gây crash toàn bộ tiến trình.
- **Cách khắc phục:** Đảm bảo bạn đã export chính xác 3 biến môi trường chỉ định cache sang ổ đĩa NAS:
  ```bash
  export TRITON_CACHE_DIR="/mnt/mmlab2024nas/trantran/lifemate/runtime_cache/triton"
  export TORCHINDUCTOR_CACHE_DIR="/mnt/mmlab2024nas/trantran/lifemate/runtime_cache/torchinductor"
  export TMPDIR="/mnt/mmlab2024nas/trantran/lifemate/runtime_cache/tmp"
  ```

### 9.2. Lỗi `FileNotFoundError` khi chạy `python` trong `.venv3`
- **Nguyên nhân:** Symlink `bin/python3` bên trong `.venv3` trỏ tới `/opt/miniconda3/bin/python3` không tồn tại trên máy host.
- **Cách khắc phục:** Không gọi `source .venv3/bin/activate`. Thay vào đó, gọi trực tiếp `/usr/bin/python3.12` kết hợp `PYTHONPATH` trỏ vào thư viện của `.venv3` như hướng dẫn tại [Mục 2](#2-cấu-hình-môi-trường--thiết-lập-bộ-nhớ-đệm-bắt-buộc).

### 9.3. Lỗi `CUDA Out of Memory (OOM)` trong quá trình train LLM
- **Nguyên nhân:** GPU đang có tiến trình khác chiếm dụng VRAM hoặc batch size quá lớn.
- **Cách khắc phục:**
  1. Kiểm tra tiến trình chiếm GPU bằng lệnh `nvidia-smi`.
  2. Đảm bảo `--batch-size 1` và tăng `--gradient-accumulation 8` hoặc `16`.
  3. Đảm bảo `--lora-dropout 0.0` để kích hoạt Unsloth GPU memory patch.

### 9.4. Lỗi `Missing fine-tuned GIPformer ONNX artifacts`
- **Nguyên nhân:** Chạy `main.py` với đường dẫn `--asr-model saved_models/asr_ft` nhưng thư mục này chưa được xuất file bằng `export_gipformer.py`.
- **Cách khắc phục:** Export lại theo mục 4.4 và kiểm tra đủ `encoder.onnx`, `decoder.onnx`, `joiner.onnx`, `tokens.txt`. Nếu chưa có ASR fine-tune, chạy `main.py` với `--asr-model saved_models/asr`. Launcher chỉ kiểm tra `saved_models/asr_ft/tokens.txt` tồn tại và không rỗng để chọn thư mục; nếu ONNX còn thiếu trong thư mục được chọn thì bước ASR báo lỗi, không tự chuyển sang model gốc. Khi đổi nguồn ASR, dùng cache mới hoặc `--refresh-transcripts` ở bước `asr/full`.

---
*Tài liệu được cập nhật tự động và đồng bộ với kiến trúc LifeMate Baseline mới nhất.*
