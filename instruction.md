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
   - Inference mặc định **tắt thinking** (`enable_thinking=False`, `thinking_budget=0`): tối đa **16.000 input / 384 output tokens** mỗi lượt, không cấp riêng 256 token thinking. Dữ liệu SFT mặc định dùng **Step-wise SFT** (`--step-wise-sft`) và tắt Short Checked Reasoning (`--no-short-reasoning`) để đồng bộ prefix `<think>\n\n</think>\n\n` giữa train và inference (có thể bật lại reasoning khi thử nghiệm bằng `--short-reasoning`).
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

Inference nạp các file ONNX đã xuất trong `models/asr_ft`, không nạp trực tiếp
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

### 5.1. Chạy Fine-tune LLM toàn diện bằng `nohup` (Khuyên dùng)
Chạy kịch bản 1 lệnh [`finetune.py`](code/finetune.py) (mặc định sử dụng Step-wise SFT, 3 epochs, kích hoạt Guard Recovery và tắt Short Checked Reasoning để đồng bộ với inference `thinking_budget=0`):

`finetune.py` và `prepare_sft.py` mặc định dùng `code/models/asr_ft` để tạo
transcript ở **cả dual và mix**, cùng nguồn ASR fine-tune với inference. Các lệnh
bên dưới cũng ghi rõ đường dẫn này. Nếu chưa có bộ ONNX fine-tune đầy đủ, chạy
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
cd /mnt/mmlab2024nas/trantran/lifemate/code

nohup /usr/bin/python3.12 finetune.py \
  --cuda-visible-devices 0 \
  --data-root /mnt/mmlab2024nas/trantran/lifemate/train+public_test \
  --base-model /mnt/mmlab2024nas/trantran/lifemate/code/models/llm \
  --output-dir /mnt/mmlab2024nas/trantran/lifemate/code/result/run_sft_final \
  --asr-model /mnt/mmlab2024nas/trantran/lifemate/code/models/asr_ft \
  --transcript-mode dual \
  --asr-transcript-ratio 0.5 \
  --asr-precision auto \
  --asr-decoding-method modified_beam_search \
  --asr-beam-size 4 \
  --filter-damaged-asr \
  --fuzzy-weight 0.3 \
  --guard-recovery-ratio 0.18 \
  --step-wise-sft \
  --no-short-reasoning \
  --epochs 3 \
  --batch-size 1 \
  --gradient-accumulation 8 \
  --learning-rate 1e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.0 \
  --save-strategy epoch \
  > /mnt/mmlab2024nas/trantran/lifemate/runtime_cache/logs/train_llm_final.log 2>&1 &
```

*Kiểm tra tiến trình:*
```bash
tail -f /mnt/mmlab2024nas/trantran/lifemate/runtime_cache/logs/train_llm_stepwise.log
```
*Kết quả đầu ra:* Adapter hoàn chỉnh sẽ được lưu tại:
`/mnt/mmlab2024nas/trantran/lifemate/code/result/run_sft_stepwise/training/adapter`

### 5.2. Chạy tách riêng từng bước (Dành cho Debug hoặc tùy biến)
Nếu bạn muốn chia nhỏ để kiểm tra tập dữ liệu trước khi train:

#### Bước 3a: Chuẩn bị dữ liệu SFT
Lệnh sau tạo hai bộ riêng `sft_dual_asr_ft` và `sft_mix_asr_ft`, dùng cùng ASR
fine-tune, tỷ lệ 0.5 và toàn bộ tập train. Nếu các thư mục đã có dữ liệu, chọn tên
output mới; prepare không ghi đè SFT cũ.

```bash
cd /mnt/mmlab2024nas/trantran/lifemate/code

for SFT_TRANSCRIPT_MODE in dual mix; do
  /usr/bin/python3.12 prepare_sft.py \
    --data-root /mnt/mmlab2024nas/trantran/lifemate/train+public_test \
    --tokenizer /mnt/mmlab2024nas/trantran/lifemate/code/models/llm \
    --output-dir "/mnt/mmlab2024nas/trantran/lifemate/code/result/sft_${SFT_TRANSCRIPT_MODE}_asr_ft" \
    --asr-model /mnt/mmlab2024nas/trantran/lifemate/code/models/asr_ft \
    --asr-precision fp32 \
    --asr-decoding-method modified_beam_search \
    --asr-beam-size 4 \
    --transcript-mode "$SFT_TRANSCRIPT_MODE" \
    --asr-transcript-ratio 0.5 \
    --filter-damaged-asr \
    --guard-recovery-ratio 0.18 \
    --step-wise-sft \
    --no-short-reasoning \
    --validation-fraction 0.0 || break
done
```

Kiểm tra `asr.model_path` trong `manifest.json` của mỗi bộ: phải trỏ vào
`code/models/asr_ft`. Nhãn nguồn trong prompt vẫn chỉ bật cho dual; mix giữ định
dạng prompt hiện có. Mẫu synthetic luôn dùng một bản `audio_text`, không qua ASR.

#### Bước 3b: Huấn luyện LoRA với Unsloth
```bash
cd /mnt/mmlab2024nas/trantran/lifemate/code

nohup /usr/bin/python3.12 train_lora.py \
  --cuda-visible-devices 0 \
  --base-model /mnt/mmlab2024nas/trantran/lifemate/code/models/llm \
  --sft-dir /mnt/mmlab2024nas/trantran/lifemate/code/result/sft_dual_asr_ft \
  --output-dir /mnt/mmlab2024nas/trantran/lifemate/code/result/training_lora \
  --backend unsloth \
  --quantization none \
  --fuzzy-weight 0.3 \
  --epochs 3 \
  --batch-size 1 \
  --gradient-accumulation 8 \
  --learning-rate 1e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.0 \
  --save-strategy epoch \
  > /mnt/mmlab2024nas/trantran/lifemate/runtime_cache/logs/train_lora_only.log 2>&1 &
```

Để train bộ mix, thay `--sft-dir` bằng `.../result/sft_mix_asr_ft` và chọn
`--output-dir` huấn luyện riêng.

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

## 6. BƯỚC 4: CHẠY SUY LUẬN (INFERENCE PIPELINE)

Giai đoạn suy luận đọc tập kiểm thử (`public_test` hoặc `private_test`), phiên âm giọng nói qua ASR, gửi văn bản và schema công cụ cho LLM, thực thi môi trường mô phỏng và xuất tệp nộp bài `submission.jsonl`.

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

Cách chọn ASR phụ thuộc lệnh chạy:

- `run_server.sh`: tự chọn `models/asr_ft` khi `models/asr_ft/tokens.txt` tồn tại
  và không rỗng; nếu chưa có thì chọn `models/asr`. Đây là điều kiện chọn thư mục;
  bộ ONNX trong thư mục được chọn vẫn phải đầy đủ.
- `main.py --stage full` hoặc `--stage asr`: dùng chính xác thư mục truyền qua
  `--asr-model`, không tự chọn checkpoint mới nhất hoặc tự export `best.pt`.
- `main.py --stage llm`: chỉ đọc transcript cache; không nạp hay chuyển model ASR.

Các lệnh bên dưới dùng cache riêng `transcripts_asr_ft.jsonl` cho ASR fine-tune.
Nếu đổi checkpoint, model hoặc cấu hình giải mã mà vẫn dùng cùng đường dẫn cache,
hãy chạy lại bước ASR với `--refresh-transcripts` hoặc chọn một đường dẫn cache mới.
Bước `full/asr` báo lỗi khi cache không khớp model/settings, không tự thay cache cũ.
Chỉ chạy `--stage llm` sẽ tiếp tục dùng nguồn ASR ghi trong manifest của cache đó.

Sau khi bước ASR hoàn tất, kiểm tra nguồn model và cấu hình giải mã của cache:

```bash
rg -n '"model_path"|"precision"|"decoding_method"|"max_active_paths"' \
  /mnt/mmlab2024nas/trantran/lifemate/code/result/transcripts_asr_ft.manifest.json
```

`model_path` phải trỏ tới `code/models/asr_ft` khi muốn dùng ASR đã fine-tune.

### 6.1. Kịch bản 1: Chạy chuẩn qua launcher `run_server.sh` (Khuyên dùng)
Script [`code/run_server.sh`](code/run_server.sh) đã cấu hình sẵn tự động:

- Ưu tiên chọn `models/asr_ft` theo điều kiện `tokens.txt` ở trên; nếu chưa có sẽ dùng `models/asr`.
- Tự động truyền các cờ tối ưu hóa runtime tốt nhất.
- Chạy tách bạch 2 tiến trình ASR và LLM.

Launcher cố định đường dẫn model trong script và truyền chúng sau các tham số CLI.
Biến môi trường `ASR_MODEL`/`LLM_MODEL` hoặc cờ `--asr-model`/`--llm-model` truyền
vào launcher không ghi đè lựa chọn này. Dùng `main.py` ở mục 6.2 nếu cần chỉ định
thư mục model khác. Log khởi động ghi `[run_server] ASR_MODEL=...` để kiểm tra.

Chạy ngầm với `nohup`:
```bash
cd /mnt/mmlab2024nas/trantran/lifemate/code

# Chọn adapter vừa huấn luyện qua --llm-adapter
nohup env \
  PYTHON_BIN="/usr/bin/python3.12" \
  DATA_ROOT="/mnt/mmlab2024nas/trantran/lifemate/train+public_test" \
  SPLIT="public_test" \
  OUTPUT="/mnt/mmlab2024nas/trantran/lifemate/code/result/submission5.jsonl" \
  TRANSCRIPT_CACHE="/mnt/mmlab2024nas/trantran/lifemate/code/result/transcripts.jsonl" \
  DEVICE="cuda" \
  bash run_server.sh \
  --llm-adapter /mnt/mmlab2024nas/trantran/lifemate/code/result/run_sft_synth/training/adapter \
  --asr-precision fp32 \
  --asr-decoding-method modified_beam_search \
  --asr-beam-size 4 \
  --asr-batch-size 16 \
  --thinking-budget 0 \
  --prefix-cache auto \
  --schema-cache auto \
  --sample-schedule schema \
  > /mnt/mmlab2024nas/trantran/lifemate/runtime_cache/logs/inference_server5.log 2>&1 &
```

*Theo dõi tiến độ:*
```bash
tail -f /mnt/mmlab2024nas/trantran/lifemate/runtime_cache/logs/inference_server1.log
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
  --transcript-cache /mnt/mmlab2024nas/trantran/lifemate/code/result/transcripts_asr_ft.jsonl \
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
  --transcript-cache /mnt/mmlab2024nas/trantran/lifemate/code/result/transcripts_asr_ft.jsonl \
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
  --transcript-cache /mnt/mmlab2024nas/trantran/lifemate/code/result/transcripts_asr_ft.jsonl \
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
Để đánh giá khả năng gốc của mô hình `Qwen3.5-4B` khi chưa qua fine-tune, dùng cùng
ASR fine-tune và cấu hình giải mã như các lệnh có adapter ở trên để đối chiếu LLM:
Bỏ tham số `--llm-adapter`:
```bash
cd /mnt/mmlab2024nas/trantran/lifemate/code

nohup /usr/bin/python3.12 main.py \
  --cuda-visible-devices 0 \
  --data-root /mnt/mmlab2024nas/trantran/lifemate/train+public_test \
  --split public_test \
  --stage full \
  --asr-model /mnt/mmlab2024nas/trantran/lifemate/code/models/asr_ft \
  --asr-precision fp32 \
  --asr-decoding-method modified_beam_search \
  --asr-beam-size 4 \
  --llm-model /mnt/mmlab2024nas/trantran/lifemate/code/models/llm \
  --output /mnt/mmlab2024nas/trantran/lifemate/code/result/submission_zeroshot.jsonl \
  --transcript-cache /mnt/mmlab2024nas/trantran/lifemate/code/result/transcripts_asr_ft_zeroshot.jsonl \
  > /mnt/mmlab2024nas/trantran/lifemate/runtime_cache/logs/inference_zeroshot.log 2>&1 &
```

---

## 7. BẢNG TRA CỨU THAM SỐ DÒNG LỆNH (CLI PARAMETERS)

### 7.1. Tham số ASR (Nhận dạng âm thanh)
| Tham số | Giá trị khuyên dùng | Các giá trị khả dĩ | Ý nghĩa & Mô tả |
| :--- | :--- | :--- | :--- |
| `--asr-model` | `models/asr_ft` | Đường dẫn thư mục | Mặc định của `finetune.py`/`prepare_sft.py` cho dual và mix, không tự dùng ASR gốc khi thiếu weights. `main.py` cần truyền đường dẫn; launcher ưu tiên `asr_ft` rồi mới dùng `asr`. |
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
| `--reasoning-weight` | `0.25` | `0.0` đến `1.0` | Trọng số loss của token reasoning trong dữ liệu SFT; chỉ áp dụng khi chủ động bật `--short-reasoning` (mặc định tắt). |
| `--guard-recovery-ratio` | `0.18` | `0.0` đến `1.0` | Tỷ lệ mẫu chèn Guard Recovery (`xref_guard`, schema `validate`, hỏi thừa `missing_info_guard`). |
| `--reasoning-stepwise-ratio` | `0.5` | `0.0` đến `1.0` | Tỷ lệ sinh thêm mẫu reasoning từng bước khi bật `--short-reasoning` ở chế độ full-trajectory (`--no-step-wise-sft`). |
| `--no-inject-guard-recovery` | Tắt (cờ) | Bật / Tắt | Cờ tắt hoàn toàn việc chèn các mẫu Guard Recovery trong `prepare_sft.py`. |
| `--step-wise-sft` | Đã bật mặc định | Bật / Tắt | Xuất mỗi lượt quyết định thành một dòng SFT riêng (Step-wise SFT), giữ đồng bộ `<think>\n\n</think>\n\n` ở mọi bước trung gian; dùng `--no-step-wise-sft` để chuyển về full-trajectory packing. |
| `--short-reasoning` | Đã tắt mặc định | Bật / Tắt | Mặc định tắt (`--no-short-reasoning`) để đồng bộ `thinking_budget=0`; truyền `--short-reasoning` để sinh reasoning trong `prepare_sft.py` và `finetune.py`. |
| `--filter-damaged-asr` | Đã bật (cờ) | Bật / Tắt | Loại bỏ bản sao ASR khỏi SFT nếu ASR làm hỏng các đối số chuẩn của tool call. |
| `--epochs` | `3` | Số thực > 0 | Số epoch huấn luyện LoRA (mặc định `3` cho Step-wise SFT). |
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
| `--thinking-budget` | `0` | Số nguyên >= 0 | 0 tắt thinking và chặn prefix reasoning bằng logits guard, yêu cầu mở đầu JSON/tool call. Giá trị lớn hơn 0 cho phép thinking khi còn đủ ngân sách. |
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
tail -n 50 -f /mnt/mmlab2024nas/trantran/lifemate/runtime_cache/logs/inference_server1.log
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
- **Cách khắc phục:** Export lại theo mục 4.4 và kiểm tra đủ `encoder.onnx`, `decoder.onnx`, `joiner.onnx`, `tokens.txt`. Nếu chưa có ASR fine-tune, chạy `main.py` với `--asr-model models/asr`. Launcher chỉ kiểm tra `models/asr_ft/tokens.txt` tồn tại và không rỗng để chọn thư mục; nếu ONNX còn thiếu trong thư mục được chọn thì bước ASR báo lỗi, không tự chuyển sang model gốc. Khi đổi nguồn ASR, dùng cache mới hoặc `--refresh-transcripts` ở bước `asr/full`.

---
*Tài liệu được cập nhật tự động và đồng bộ với kiến trúc LifeMate Baseline mới nhất.*
