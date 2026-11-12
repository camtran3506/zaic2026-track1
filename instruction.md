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

### 1.1. Luồng xử lý tách biệt (Two-Stage Offline Pipeline)
LifeMate hoạt động theo nguyên lý **2 giai đoạn độc lập tuyệt đối (Stage Separation)** để tối ưu hóa bộ nhớ VRAM:
1. **Giai đoạn ASR (Âm thanh -> Chữ viết):**
   - Sử dụng mô hình `Gipformer 1.5` (68M Zipformer-RNNT).
   - Chạy trên CPU ONNX Runtime đa luồng (hoặc GPU), giải mã batch audio hiện tại và audio history.
   - Kết quả xuất ra tệp bộ nhớ đệm `transcripts.jsonl` kèm mã băm kiểm tra (SHA-256 manifest). Tiến trình ASR giải phóng hoàn toàn bộ nhớ sau khi hoàn thành.
2. **Giai đoạn LLM (Chữ viết -> Tool Execution Agent):**
   - Sử dụng mô hình `Qwen/Qwen3.5-4B` tải ở định dạng `bfloat16` kết hợp LoRA Adapter.
   - Tự động nạp `user_context` (bao gồm `date_reference` tính sẵn và danh mục app), kết hợp dynamic prefix caching, schema snapshot caching.
   - Inference mặc định **tắt thinking** (`enable_thinking=False`, `thinking_budget=0`): tối đa **16.000 input / 384 output tokens** mỗi lượt, không cấp riêng 256 token thinking. Dữ liệu SFT mặc định **bật Short Checked Reasoning** trong thẻ `<think>` để huấn luyện.
   - Tương tác nhiều lượt với môi trường công cụ (Tool Environment) cho đến khi ra kết quả cuối cùng hoặc đạt giới hạn số bước.

### 1.2. Cấu trúc cây thư mục chuẩn
```text
/mnt/mmlab2024nas/trantran/lifemate/
├── instruction.md                     # Tài liệu hướng dẫn này
├── train_asr.jsonl                    # Dữ liệu 1,536 sample ASR audio_text đã gán nhãn chuẩn
├── train+public_test/                 # Thư mục gốc chứa dataset
│   ├── train/                         # Split train (train.jsonl + audio/)
│   └── public_test/                   # Split test (public_test.jsonl + audio/)
├── runtime_cache/                     # Thư mục cache tốc độ cao trên NAS
│   ├── triton/                        # Cache cho Triton JIT kernels
│   ├── torchinductor/                 # Cache cho PyTorch Inductor compiler
│   ├── tmp/                           # Thư mục tạm hệ thống (thay thế /tmp)
│   └── logs/                          # Thư mục chứa log các tiến trình chạy nền
└── code/                              # Mã nguồn chính của pipeline
    ├── main.py                        # Điểm khởi động chính cho Inference
    ├── run_server.sh                  # Shell script chạy nhanh inference chuẩn
    ├── finetune.py                    # Script 1-lệnh tự động hóa SFT LLM
    ├── prepare_sft.py                 # Chuẩn bị dữ liệu huấn luyện SFT cho LLM
    ├── train_lora.py                  # Huấn luyện Unsloth LoRA trên LLM
    ├── merge_lora.py                  # Trộn LoRA adapter vào base model
    ├── models/                        # Nơi lưu trữ checkpoint mô hình
    │   ├── asr/                       # Mô hình ASR gốc (ONNX fp32, int8, model.pt)
    │   ├── asr_ft/                    # Mô hình ASR sau khi fine-tune (ONNX fp32)
    │   └── llm/                       # Mô hình LLM gốc (Qwen3.5-4B weights)
    ├── scripts/                       # Các scripts phụ trợ
    │   ├── download_models.py         # Script tải mô hình từ Hugging Face
    │   └── asr_finetune/              # Bộ công cụ fine-tune ASR Gipformer
    │       ├── setup_icefall.sh       # Clone và kiểm tra Icefall dependency
    │       ├── prepare_gipformer_sft.py # Chuẩn bị tập train ASR
    │       ├── train_gipformer.py     # Huấn luyện PyTorch RNNT Gipformer
    │       └── export_gipformer.py    # Xuất checkpoint sang ONNX FP32
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

Pipeline LifeMate hoạt động **hoàn toàn offline**. Toàn bộ mô hình phải được tải về sẵn trong thư mục [`code/models`](file:///mnt/mmlab2024nas/trantran/lifemate/code/models) trước khi bắt đầu.

Tạm thời bỏ cờ `HF_HUB_OFFLINE` để tải mô hình qua script [`download_models.py`](file:///mnt/mmlab2024nas/trantran/lifemate/code/scripts/download_models.py):

```bash
cd /mnt/mmlab2024nas/trantran/lifemate/code

# Tải cả Base LLM (Qwen3.5-4B) và Base ASR (Gipformer 1.5 với đầy đủ ONNX FP32, INT8 và PyTorch weights)
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 /usr/bin/python3.12 scripts/download_models.py \
  --only both \
  --asr-precision all
```

### (Tùy chọn) Tải sẵn mô hình ASR Fine-tuned từ Hugging Face
Nếu bạn không muốn tự fine-tune lại mô hình ASR từ đầu mà muốn dùng ngay checkpoint đã được fine-tune sẵn trên tập dữ liệu cuộc thi:
```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 /usr/bin/python3.12 scripts/download_models.py \
  --only asr \
  --asr-ft-repo hugebenevolence/lifemate-final \
  --asr-ft-subdir asr-gipformer
```
Khi đó, các tệp ONNX của mô hình fine-tune sẽ được lưu tại [`code/models/asr_ft`](file:///mnt/mmlab2024nas/trantran/lifemate/code/models/asr_ft).

---

## 4. BƯỚC 2: FINE-TUNE MÔ HÌNH ASR GIPFORMER (TÙY CHỌN)

Bước này áp dụng khi bạn muốn tự huấn luyện lại mô hình nhận dạng giọng nói Gipformer bằng tệp [`train_asr.jsonl`](file:///mnt/mmlab2024nas/trantran/lifemate/train_asr.jsonl) (trong đó các câu audio text chính đã được con người chuyển số thành chữ chính xác).

### 4.1. Chuẩn bị thư viện Icefall
Chạy script cài đặt liên kết mã nguồn Icefall:
```bash
cd /mnt/mmlab2024nas/trantran/lifemate/code
bash scripts/asr_finetune/setup_icefall.sh
```

### 4.2. Chuẩn bị dữ liệu huấn luyện ASR (`prepare_gipformer_sft.py`)
Script này áp dụng chính sách **Selective Normalization & Anti-Noise**:
- **Tập `audio_text` chính (1,538 câu):** Con người đã kiểm tra và đảm bảo chuyển toàn bộ số thành chữ chính xác (`năm mươi ngàn`, `bảy rưỡi`...). Do đó, cách đọc số được giữ nguyên vẹn; còn lại **tất cả các phép chuẩn hóa khác** (xử lý ký hiệu `+`, `&`, `%`, `@`, `°`, dấu thời gian `h`, ngày tháng `tháng`, dấu phẩy số thập phân `phẩy`, viết hoa toàn bộ, loại bỏ dấu câu, căn chỉnh dấu thanh BPE `model_style`, kiểm tra bảng ký tự và BPE token) **đều được áp dụng đồng nhất như history audio**. Vì các câu này đã được con người nghe và thẩm định, pipeline không áp dụng bộ lọc WER (`--max-history-wer`) để bảo toàn 100% dữ liệu.
- **Tập số còn sót (7 câu):** Một số ít câu còn sót chữ số chưa được con người chuyển (`2 triệu`, `tập 12`, `tháng 9`, `101`, `Omega-3`) sẽ được tự động chuyển số thành chữ qua `target_v3` (khớp âm học với hypothesis của Gipformer) giống như history.
- **Tập hội thoại lịch sử (`history`, 3,491 lượt user):** Chưa qua thẩm định của con người, áp dụng đầy đủ quy trình mở rộng số/ký hiệu qua `target_v3` và tùy chọn lọc bỏ các câu có WER quá cao so với hypothesis âm học (`--max-history-wer`).

Chạy lệnh:
```bash
cd /mnt/mmlab2024nas/trantran/lifemate/code

/usr/bin/python3.12 scripts/asr_finetune/prepare_gipformer_sft.py \
  --input-jsonl /mnt/mmlab2024nas/trantran/lifemate/train_asr.jsonl \
  --data-root /mnt/mmlab2024nas/trantran/lifemate/train+public_test \
  --model-dir /mnt/mmlab2024nas/trantran/lifemate/code/models/asr \
  --output-dir /mnt/mmlab2024nas/trantran/lifemate/code/result/asr_sft_data \
  --val-fraction 0.1 \
  --seed 42
```
*Kết quả:* Tạo ra `train.jsonl`, `dev.jsonl` và `manifest.json` trong `result/asr_sft_data`.

### 4.3. Chạy huấn luyện ASR ngầm bằng `nohup` (`train_gipformer.py`)
Huấn luyện Gipformer PyTorch RNNT loss trên GPU:
```bash
cd /mnt/mmlab2024nas/trantran/lifemate/code

nohup /usr/bin/python3.12 scripts/asr_finetune/train_gipformer.py \
  --cuda-visible-devices 0 \
  --data-root /mnt/mmlab2024nas/trantran/lifemate/train+public_test \
  --data-dir /mnt/mmlab2024nas/trantran/lifemate/code/result/asr_sft_data \
  --model-dir /mnt/mmlab2024nas/trantran/lifemate/code/models/asr \
  --output-dir /mnt/mmlab2024nas/trantran/lifemate/code/result/asr_sft \
  --epochs 3 \
  --learning-rate 3e-5 \
  --accum 4 \
  --eval-every 250 \
  --patience 3 \
  > /mnt/mmlab2024nas/trantran/lifemate/runtime_cache/logs/train_asr.log 2>&1 &
```
*Kiểm tra tiến trình:* Xem log thời gian thực bằng `tail -f /mnt/mmlab2024nas/trantran/lifemate/runtime_cache/logs/train_asr.log`.

### 4.4. Xuất mô hình fine-tuned sang ONNX FP32 (`export_gipformer.py`)
Sau khi huấn luyện xong, chuyển checkpoint tốt nhất `best.pt` thành định dạng ONNX FP32 đặt trong `code/models/asr_ft`:
```bash
cd /mnt/mmlab2024nas/trantran/lifemate/code

/usr/bin/python3.12 scripts/asr_finetune/export_gipformer.py \
  --best /mnt/mmlab2024nas/trantran/lifemate/code/result/asr_sft/best.pt \
  --original-dir /mnt/mmlab2024nas/trantran/lifemate/code/models/asr \
  --output-dir /mnt/mmlab2024nas/trantran/lifemate/code/models/asr_ft \
  --data-root /mnt/mmlab2024nas/trantran/lifemate/train+public_test
```
*Xác nhận:* Thư mục `code/models/asr_ft` sẽ chứa đầy đủ `encoder.onnx`, `decoder.onnx`, `joiner.onnx`, `tokens.txt`, `bpe.model` và tệp kê khai `lifemate_manifest.json`.

---

## 5. BƯỚC 3: FINE-TUNE MÔ HÌNH LLM LORA ADAPTER

Quá trình fine-tune LLM huấn luyện mô hình `Qwen3.5-4B` thực hiện gọi công cụ (tool call) chính xác theo giọng nói và văn cảnh người dùng.

Pipeline đã tích hợp toàn bộ các kỹ thuật tối ưu hóa cao nhất:
- **Full-Trajectory SFT:** Gộp toàn bộ các lượt gọi công cụ của 1 mẫu thành 1 trajectory liền mạch.
- **Dual Gold + ASR (`--transcript-mode dual`):** Nhân đôi dữ liệu với 100% bản chép lời chuẩn (Gold) và 100% bản nhận dạng giọng nói ASR.
- **Weighted Loss Trainer (`--fuzzy-weight 0.3`, `--reasoning-weight 0.25`):** Giảm trọng số phạt sai sót trên các tham số tìm kiếm mờ (`fuzzy`) và token reasoning, ưu tiên độ chính xác của lệnh gọi công cụ (`tool_call`).
- **Lọc ASR lỗi nặng (`--filter-damaged-asr`):** Bỏ qua các bản sao ASR nếu bị ASR làm mất hoàn toàn tham số thực thi chuẩn.
- **Search Recovery Trajectories:** Bổ sung kịch bản xử lý tự phục hồi khi kết quả tìm kiếm rỗng.
- **Self-Correction Guard Recovery (`--guard-recovery-ratio 0.18`):** Bổ sung các mẫu phục hồi đơn bước khi gặp từ chối từ guard (`xref_guard` và schema `validate`), dạy mô hình phản xạ sửa lỗi chính xác trong 3 lượt `max_invalid` khi suy luận.
- **Short Checked Reasoning mặc định bật (`--short-reasoning`):** Dữ liệu SFT thêm dòng reasoning ngắn đã kiểm chứng và các mẫu reasoning từng bước. Khi inference vẫn dùng `enable_thinking=False`, `thinking_budget=0`.
- **Pre-computed `date_reference`:** Tự động giải quyết thời gian biểu mà không cần sinh các lệnh gọi helper thời gian dư thừa.

### 5.1. Chạy Fine-tune LLM toàn diện bằng `nohup` (Khuyên dùng)
Chạy kịch bản 1 lệnh [`finetune.py`](code/finetune.py) (mặc định kích hoạt Guard Recovery và Short Checked Reasoning):

```bash
cd /mnt/mmlab2024nas/trantran/lifemate/code

nohup /usr/bin/python3.12 finetune.py \
  --cuda-visible-devices 0 \
  --data-root /mnt/mmlab2024nas/trantran/lifemate/train+public_test \
  --base-model /mnt/mmlab2024nas/trantran/lifemate/code/models/llm \
  --output-dir /mnt/mmlab2024nas/trantran/lifemate/code/result/run_sft_noreason \
  --asr-model /mnt/mmlab2024nas/trantran/lifemate/code/models/asr \
  --transcript-mode dual \
  --asr-transcript-ratio 0.5 \
  --asr-precision auto \
  --asr-decoding-method modified_beam_search \
  --asr-beam-size 4 \
  --filter-damaged-asr \
  --fuzzy-weight 0.3 \
  --guard-recovery-ratio 0.18 \
  --no-short-reasoning \
  --epochs 2 \
  --batch-size 1 \
  --gradient-accumulation 8 \
  --learning-rate 1e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.0 \
  --save-strategy epoch \
  > /mnt/mmlab2024nas/trantran/lifemate/runtime_cache/logs/train_llm3.log 2>&1 &
```

*Kiểm tra tiến trình:*
```bash
tail -f /mnt/mmlab2024nas/trantran/lifemate/runtime_cache/logs/train_llm.log
```
*Kết quả đầu ra:* Adapter hoàn chỉnh sẽ được lưu tại:
`/mnt/mmlab2024nas/trantran/lifemate/code/result/run_sft_dual/training/adapter`

### 5.2. Chạy tách riêng từng bước (Dành cho Debug hoặc tùy biến)
Nếu bạn muốn chia nhỏ để kiểm tra tập dữ liệu trước khi train:

#### Bước 3a: Chuẩn bị dữ liệu SFT
```bash
cd /mnt/mmlab2024nas/trantran/lifemate/code

/usr/bin/python3.12 prepare_sft.py \
  --data-root /mnt/mmlab2024nas/trantran/lifemate/train+public_test \
  --tokenizer /mnt/mmlab2024nas/trantran/lifemate/code/models/llm \
  --output-dir /mnt/mmlab2024nas/trantran/lifemate/code/result/sft_prepared \
  --asr-model /mnt/mmlab2024nas/trantran/lifemate/code/models/asr \
  --transcript-mode dual \
  --asr-transcript-ratio 0.5 \
  --filter-damaged-asr \
  --guard-recovery-ratio 0.18 \
  --short-reasoning \
  --validation-fraction 0.0
```

#### Bước 3b: Huấn luyện LoRA với Unsloth
```bash
cd /mnt/mmlab2024nas/trantran/lifemate/code

nohup /usr/bin/python3.12 train_lora.py \
  --cuda-visible-devices 0 \
  --base-model /mnt/mmlab2024nas/trantran/lifemate/code/models/llm \
  --sft-dir /mnt/mmlab2024nas/trantran/lifemate/code/result/sft_prepared \
  --output-dir /mnt/mmlab2024nas/trantran/lifemate/code/result/training_lora \
  --backend unsloth \
  --quantization none \
  --fuzzy-weight 0.3 \
  --epochs 2 \
  --batch-size 1 \
  --gradient-accumulation 8 \
  --learning-rate 1e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.0 \
  --save-strategy epoch \
  > /mnt/mmlab2024nas/trantran/lifemate/runtime_cache/logs/train_lora_only.log 2>&1 &
```

### 5.3. (Tùy chọn) Gộp LoRA Adapter vào Base Model (`merge_lora.py`)
Nếu muốn xuất ra một thư mục checkpoint mô hình đầy đủ không cần nạp PEFT adapter lúc suy luận:
```bash
cd /mnt/mmlab2024nas/trantran/lifemate/code

/usr/bin/python3.12 merge_lora.py \
  --base-model /mnt/mmlab2024nas/trantran/lifemate/code/models/llm \
  --adapter /mnt/mmlab2024nas/trantran/lifemate/code/result/run_sft_dual/training/adapter \
  --output-dir /mnt/mmlab2024nas/trantran/lifemate/code/models/llm_merged \
  --dtype bfloat16
```

---

## 6. BƯỚC 4: CHẠY SUY LUẬN (INFERENCE PIPELINE)

Giai đoạn suy luận đọc tập kiểm thử (`public_test` hoặc `private_test`), phiên âm giọng nói qua ASR, gửi văn bản và schema công cụ cho LLM, thực thi môi trường mô phỏng và xuất tệp nộp bài `submission.jsonl`.

### 6.1. Kịch bản 1: Chạy chuẩn qua launcher `run_server.sh` (Khuyên dùng)
Script [`code/run_server.sh`](file:///mnt/mmlab2024nas/trantran/lifemate/code/run_server.sh) đã cấu hình sẵn tự động:
- Ưu tiên chọn `models/asr_ft` nếu có; nếu không có sẽ tự động dùng `models/asr`.
- Tự động truyền các cờ tối ưu hóa runtime tốt nhất.
- Chạy tách bạch 2 tiến trình ASR và LLM.

Chạy ngầm với `nohup`:
```bash
cd /mnt/mmlab2024nas/trantran/lifemate/code

# Gán adapter vừa huấn luyện vào biến LLM_ADAPTER (nếu có)
nohup env \
  PYTHON_BIN="/usr/bin/python3.12" \
  DATA_ROOT="/mnt/mmlab2024nas/trantran/lifemate/train+public_test" \
  SPLIT="public_test" \
  OUTPUT="/mnt/mmlab2024nas/trantran/lifemate/code/result/submission1.jsonl" \
  TRANSCRIPT_CACHE="/mnt/mmlab2024nas/trantran/lifemate/code/result/transcripts.jsonl" \
  DEVICE="cuda" \
  bash run_server.sh \
  --llm-adapter /mnt/mmlab2024nas/trantran/lifemate/code/result/run_sft_dual/training/adapter \
  --asr-precision fp32 \
  --asr-decoding-method modified_beam_search \
  --asr-beam-size 4 \
  --asr-batch-size 16 \
  --prefix-cache auto \
  --schema-cache auto \
  --sample-schedule schema \
  > /mnt/mmlab2024nas/trantran/lifemate/runtime_cache/logs/inference_server1.log 2>&1 &
```

*Theo dõi tiến độ:*
```bash
tail -f /mnt/mmlab2024nas/trantran/lifemate/runtime_cache/logs/inference_server.log
```

---

### 6.2. Kịch bản 2: Chạy trực tiếp bằng `main.py`
Nếu muốn kiểm soát tuyệt đối từng tham số hoặc chạy trên split khác:
```bash
cd /mnt/mmlab2024nas/trantran/lifemate/code

nohup /usr/bin/python3.12 main.py \
  --cuda-visible-devices 0 \
  --data-root /mnt/mmlab2024nas/trantran/lifemate/train+public_test \
  --split public_test \
  --stage full \
  --output /mnt/mmlab2024nas/trantran/lifemate/code/result/submission.jsonl \
  --transcript-cache /mnt/mmlab2024nas/trantran/lifemate/code/result/transcripts.jsonl \
  --asr-model /mnt/mmlab2024nas/trantran/lifemate/code/models/asr_ft \
  --asr-precision fp32 \
  --asr-decoding-method modified_beam_search \
  --asr-beam-size 4 \
  --asr-batch-size 16 \
  --asr-threads 4 \
  --llm-model /mnt/mmlab2024nas/trantran/lifemate/code/models/llm \
  --llm-adapter /mnt/mmlab2024nas/trantran/lifemate/code/result/run_sft_dual/training/adapter \
  --llm-dtype bfloat16 \
  --prefix-cache auto \
  --schema-cache auto \
  --sample-schedule schema \
  --max-input-tokens 16000 \
  --max-new-tokens 384 \
  --thinking-budget 0 \
  --max-calls 8 \
  --max-invalid 3 \
  --sample-seconds inf \
  --seed 42 \
  > /mnt/mmlab2024nas/trantran/lifemate/runtime_cache/logs/inference_main.log 2>&1 &
```

---

### 6.3. Kịch bản 3: Tách rời công đoạn ASR và LLM
Rất hữu ích khi bạn muốn chạy ASR một lần duy nhất, sau đó có thể thử nghiệm nhiều adapter hoặc prompt của LLM mà không phải tốn thời gian chạy lại nhận dạng giọng nói:

#### Công đoạn 1: Chỉ chạy ASR
```bash
cd /mnt/mmlab2024nas/trantran/lifemate/code

nohup /usr/bin/python3.12 main.py \
  --stage asr \
  --data-root /mnt/mmlab2024nas/trantran/lifemate/train+public_test \
  --split public_test \
  --transcript-cache /mnt/mmlab2024nas/trantran/lifemate/code/result/transcripts.jsonl \
  --asr-model /mnt/mmlab2024nas/trantran/lifemate/code/models/asr_ft \
  --asr-precision fp32 \
  --asr-decoding-method modified_beam_search \
  --asr-beam-size 4 \
  --asr-batch-size 16 \
  > /mnt/mmlab2024nas/trantran/lifemate/runtime_cache/logs/stage_asr.log 2>&1 &
```

#### Công đoạn 2: Chỉ chạy LLM (sử dụng lại transcript cache ở trên)
```bash
cd /mnt/mmlab2024nas/trantran/lifemate/code

nohup /usr/bin/python3.12 main.py \
  --cuda-visible-devices 0 \
  --stage llm \
  --data-root /mnt/mmlab2024nas/trantran/lifemate/train+public_test \
  --split public_test \
  --transcript-cache /mnt/mmlab2024nas/trantran/lifemate/code/result/transcripts.jsonl \
  --llm-model /mnt/mmlab2024nas/trantran/lifemate/code/models/llm \
  --llm-adapter /mnt/mmlab2024nas/trantran/lifemate/code/result/run_sft_dual/training/adapter \
  --output /mnt/mmlab2024nas/trantran/lifemate/code/result/submission.jsonl \
  --prefix-cache auto \
  --schema-cache auto \
  --sample-schedule schema \
  --max-input-tokens 16000 \
  --max-new-tokens 384 \
  --thinking-budget 0 \
  > /mnt/mmlab2024nas/trantran/lifemate/runtime_cache/logs/stage_llm.log 2>&1 &
```

---

### 6.4. Kịch bản 4: Chạy Zero-Shot Inference (Không có Adapter)
Để đánh giá khả năng gốc của mô hình `Qwen3.5-4B` khi chưa qua fine-tune:
Bỏ tham số `--llm-adapter`:
```bash
cd /mnt/mmlab2024nas/trantran/lifemate/code

nohup /usr/bin/python3.12 main.py \
  --cuda-visible-devices 0 \
  --data-root /mnt/mmlab2024nas/trantran/lifemate/train+public_test \
  --split public_test \
  --stage full \
  --asr-model /mnt/mmlab2024nas/trantran/lifemate/code/models/asr \
  --llm-model /mnt/mmlab2024nas/trantran/lifemate/code/models/llm \
  --output /mnt/mmlab2024nas/trantran/lifemate/code/result/submission_zeroshot.jsonl \
  --transcript-cache /mnt/mmlab2024nas/trantran/lifemate/code/result/transcripts_zeroshot.jsonl \
  > /mnt/mmlab2024nas/trantran/lifemate/runtime_cache/logs/inference_zeroshot.log 2>&1 &
```

---

## 7. BẢNG TRA CỨU THAM SỐ DÒNG LỆNH (CLI PARAMETERS)

### 7.1. Tham số ASR (Nhận dạng âm thanh)
| Tham số | Giá trị khuyên dùng | Các giá trị khả dĩ | Ý nghĩa & Mô tả |
| :--- | :--- | :--- | :--- |
| `--asr-model` | `models/asr_ft` | Đường dẫn thư mục | Thư mục chứa ONNX ASR (`asr_ft` ưu tiên hơn `asr`). |
| `--asr-precision` | `fp32` | `auto`, `fp32`, `int8` | Độ chính xác ONNX. `fp32` cho WER tiếng Việt tốt nhất (~4.1% so với 4.8% của int8). |
| `--asr-decoding-method` | `modified_beam_search` | `modified_beam_search`, `greedy_search` | Thuật toán giải mã RNNT. Modified beam search vượt trội trên các từ hiếm. |
| `--asr-beam-size` | `4` | Số nguyên >= 1 | Độ rộng chùm beam search (`max_active_paths`). Mức 4 cân bằng giữa độ chính xác và tốc độ. |
| `--asr-batch-size` | `16` | Số nguyên >= 1 | Kích thước batch gộp file âm thanh để decode song song offline. |
| `--asr-threads` | `4` | Số nguyên >= 1 | Số luồng CPU sử dụng cho ONNX Runtime của Gipformer. |
| `--max-history-wer` | `0.5` | `None` hoặc float `0..1` | Ngưỡng WER tối đa để lọc bỏ câu history nhiễu trong `prepare_gipformer_sft.py`. |

### 7.2. Tham số Huấn luyện LLM SFT & LoRA
| Tham số | Giá trị khuyên dùng | Các giá trị khả dĩ | Ý nghĩa & Mô tả |
| :--- | :--- | :--- | :--- |
| `--transcript-mode` | `dual` | `dual`, `mix` | `dual`: nhân bản 100% Gold + 100% ASR; `mix`: chia tỷ lệ. |
| `--fuzzy-weight` | `0.3` | `0.0` đến `1.0` | Trọng số phạt loss trên các tham số `fuzzy` (giá trị < 1.0 giúp tránh overfitting chuỗi tìm kiếm). |
| `--reasoning-weight` | `0.25` | `0.0` đến `1.0` | Trọng số loss của token reasoning trong dữ liệu SFT; reasoning mặc định bật khi huấn luyện. |
| `--guard-recovery-ratio` | `0.18` | `0.0` đến `1.0` | Tỷ lệ mẫu chèn quỹ đạo tự sửa lỗi Guard Recovery (`xref_guard` và schema `validate` trong `prepare_sft.py`). |
| `--reasoning-stepwise-ratio` | `0.5` | `0.0` đến `1.0` | Tỷ lệ sinh thêm mẫu reasoning từng bước khi bật `--short-reasoning` (mặc định bật). |
| `--no-inject-guard-recovery` | Tắt (cờ) | Bật / Tắt | Cờ tắt hoàn toàn việc chèn các mẫu Guard Recovery trong `prepare_sft.py`. |
| `--short-reasoning` | Đã bật mặc định | Bật / Tắt | Sinh reasoning trong `prepare_sft.py` và `finetune.py`; dùng `--no-short-reasoning` để tắt. |
| `--filter-damaged-asr` | Đã bật (cờ) | Bật / Tắt | Loại bỏ bản sao ASR khỏi SFT nếu ASR làm hỏng các đối số chuẩn của tool call. |
| `--epochs` | `2` | Số thực > 0 | Số epoch huấn luyện LoRA (2 epoch là tối ưu chống suy giảm chất lượng). |
| `--batch-size` | `1` | Số nguyên | Batch size cục bộ trên mỗi GPU. |
| `--gradient-accumulation` | `8` | Số nguyên | Tích lũy gradient (tương đương effective batch size = 8). |
| `--learning-rate` | `1e-4` | Số thực | Tốc độ học của optimizer AdamW cho LoRA. |
| `--lora-r` | `16` | `8`, `16`, `32`, `64` | Hạng LoRA (Rank). |
| `--lora-alpha` | `32` | `16`, `32`, `64` | Hệ số tỷ lệ LoRA Alpha (thường đặt bằng `2 * r`). |
| `--lora-dropout` | `0.0` | `0.0` | Bắt buộc `0.0` để kích hoạt tối ưu hóa Unsloth Fast LoRA kernels. |

### 7.3. Tham số Suy luận LLM (Inference)
| Tham số | Giá trị khuyên dùng | Các giá trị khả dĩ | Ý nghĩa & Mô tả |
| :--- | :--- | :--- | :--- |
| `--prefix-cache` | `auto` | `auto`, `off` | Kích hoạt bộ nhớ đệm KV đa lượt tự động (tăng tốc độ suy luận gấp ~2.5 lần). |
| `--schema-cache` | `auto` | `auto`, `shared`, `off` | Tái sử dụng KV-cache các system prompt và định nghĩa JSON schema của các app. |
| `--sample-schedule` | `schema` | `schema`, `original` | Sắp xếp các mẫu có cùng tập installed apps chạy liền kề nhau để tối đa hóa tỷ lệ trúng schema cache. |
| `--max-input-tokens` | `16000` | Số nguyên | Giới hạn token đầu vào cho prompt; tổng input + output tối đa 16.384 tokens. |
| `--max-new-tokens` | `384` | Số nguyên | Toàn bộ ngân sách token đầu ra cho quyết định hoặc tool call mỗi lượt. |
| `--thinking-budget` | `0` | Số nguyên >= 0 | Mặc định tắt thinking, không dành 256 token riêng cho suy nghĩ. |
| `--thinking-reserve-seconds` | `2.0` | Số thực >= 0 | Chỉ có tác dụng nếu chủ động bật thinking; không dùng với mặc định `--thinking-budget 0`. |
| `--max-calls` | `8` | Số nguyên | Giới hạn tối đa số bước gọi công cụ trên mỗi mẫu để tránh vòng lặp vô hạn. |
| `--max-invalid` | `3` | Số nguyên | Số lần tối đa môi trường cho phép gọi sai cú pháp trước khi hủy lượt. |
| `--sample-seconds` | `inf` | `inf` hoặc số giây | Giới hạn thời gian tối đa cho mỗi mẫu (mặc định `inf` = không giới hạn). |

---

## 8. GIÁM SÁT & QUẢN LÝ TIẾN TRÌNH NỀN (NOHUP)

Vì các công đoạn huấn luyện và suy luận có thể mất từ vài chục phút đến vài giờ, việc chạy ngầm bằng `nohup` giúp tiến trình không bị gián đoạn khi ngắt kết nối SSH.

### 8.1. Kiểm tra log thời gian thực
Mỗi lệnh chạy nền đã được điều hướng ghi log vào thư mục `/mnt/mmlab2024nas/trantran/lifemate/runtime_cache/logs/`:
```bash
# Xem log huấn luyện ASR
tail -n 50 -f /mnt/mmlab2024nas/trantran/lifemate/runtime_cache/logs/train_asr.log

# Xem log huấn luyện LLM LoRA
tail -n 50 -f /mnt/mmlab2024nas/trantran/lifemate/runtime_cache/logs/train_llm.log

# Xem log suy luận
tail -n 50 -f /mnt/mmlab2024nas/trantran/lifemate/runtime_cache/logs/inference_server.log
```

### 8.2. Kiểm tra trạng thái GPU và tiến trình
```bash
# Giám sát trạng thái GPU, VRAM và tiến trình đang chiếm GPU
watch -n 1 nvidia-smi

# Tìm PID của các tiến trình Python đang chạy liên quan đến lifemate
ps aux | grep -E "main.py|finetune.py|train_lora.py|train_gipformer.py" | grep -v grep
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
- **Nguyên nhân:** Chạy `main.py` với đường dẫn `--asr-model models/asr_ft` nhưng thư mục này chưa được xuất file bằng `export_gipformer.py`.
- **Cách khắc phục:** Nếu chưa fine-tune ASR, hãy trỏ `--asr-model models/asr` (mô hình gốc tải về từ Hugging Face luôn đầy đủ ONNX). Nếu dùng `run_server.sh`, script sẽ tự động kiểm tra xem `models/asr_ft/tokens.txt` có tồn tại hay không để chọn thư mục thích hợp.

---
*Tài liệu được cập nhật tự động và đồng bộ với kiến trúc LifeMate Baseline mới nhất.*
