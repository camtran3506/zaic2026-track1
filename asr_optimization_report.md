# Báo cáo Đánh giá & Đề xuất Tối ưu ASR (LifeMate Pipeline)

Báo cáo này tổng hợp hiện trạng chất lượng ASR (`Gipformer 1.5 asr_ft`) trên tập `public_test` (lượt chạy mới nhất `submission_traj`) và xếp hạng các phương án tối ưu theo **mức độ ưu tiên**, trong đó ưu tiên cao nhất là tận dụng tối đa mô hình `Gipformer 1.5` sẵn có (không cần nạp thêm model ASR thứ hai), và đặt phương án thêm model ASR ngoài ở mức ưu tiên thấp nhất.

---

## 1. Tổng quan hiện trạng ASR (`Gipformer 1.5 asr_ft`) trên `public_test` & `submission_traj`

- **Cấu hình lượt chạy mới nhất:** [submission_traj.jsonl](file:///mnt/mmlab2024nas/trantran/lifemate/code/result/submission_traj.jsonl) và [submission_traj.diagnostics.jsonl](file:///mnt/mmlab2024nas/trantran/lifemate/code/result/submission_traj.diagnostics.jsonl) (`634` mẫu `public_test`, `done=535/634`, `ask_missing_info=81`, `max_invalid=15`, `max_calls=3`) sử dụng cache [transcripts.jsonl](file:///mnt/mmlab2024nas/trantran/lifemate/code/result/transcripts.jsonl) sinh từ [models/asr_ft](file:///mnt/mmlab2024nas/trantran/lifemate/code/models/asr_ft) (`fp32`, `modified_beam_search`, `max_active_paths=4`), xác nhận qua [transcripts.manifest.json](file:///mnt/mmlab2024nas/trantran/lifemate/code/result/transcripts.manifest.json) và [inference_traj.log](file:///mnt/mmlab2024nas/trantran/lifemate/runtime_cache/logs/inference_traj.log).

### Điểm mạnh của `asr_ft`
- **Trên tập `train` (có ground truth; lưu ý `asr_ft` đã fine-tune 3 epoch trên tập này với `train_all=true`):**
  - WER gộp trên `1516` cặp đối chiếu giữ lại trong [sft_dual_asr_ft/train.jsonl](file:///mnt/mmlab2024nas/trantran/lifemate/code/result/sft_dual_asr_ft/train.jsonl) đạt **`6.27%`** (`2134 / 34025` từ), với `28 / 1544` mẫu bị lọc qua `asr_damaged_filtered` trong [preparation_audit.json](file:///mnt/mmlab2024nas/trantran/lifemate/code/result/sft_dual_asr_ft/preparation_audit.json).
- **Trên tập `public_test` (`634` mẫu unseen — so sánh với ASR gốc `int8 greedy` ở [infer_adapter_20261104_123833/transcripts.jsonl](file:///mnt/mmlab2024nas/trantran/lifemate/code/result/infer_adapter_20261104_123833/transcripts.jsonl)):**
  - `asr_ft` sửa được hàng loạt từ khóa app/domain và khẩu ngữ tiếng Việt:
    - `"gia lô"` giảm `7 → 0` (`"zalo"` tăng `23 → 35`), `"sjc"` tăng `5 → 36`, `"check"` tăng `12 → 20`, `"mail"` tăng `26 → 30`.
    - Cứu nhiều câu sai nghĩa ở bản gốc: `"nổi tiếng"` → `"nửa tiếng"` (`public_0008`), `"Gửi cho minh"` → `"Gọi cho minh"` (`public_0015`), `"trọng điểm ... thiếu thị trường"` → `"tiền điện ... thiếu thì trừ"` (`public_0024`), `"hai vét ... lấy truyện"` → `"hai vé ... lấy chuyến"` (`public_0027`), `"trưởng khoa vĩ"` → `"chuyển qua ví"` (`public_0034`), `"hãy lịch đã vẫn"` → `"hủy lịch đá bóng"` (`public_0130`), `"tải xe đang tính thuyền"` → `"bãi xe đang tính tiền"` (`public_0504`).

### Điểm yếu còn tồn tại
1. **Nuốt hoặc trượt từ khóa đầu câu / từ ghép domain khi nói nhanh:**
   - Ví dụ `public_0391`: `"So giúp anh đi ô tô với xe buýt ra ga xem nếu chuyến buýt sớm nhất..."` bị nhận dạng thành `"Xoa giúp anh đi ô tô với xe buýt ga xem nếu chuyến big sớm nhất..."`.
2. **Tên riêng, tên quán/sự kiện, từ tiếng Anh (OOV) bị bẻ sang âm tiết tiếng Việt gần âm:**
   - Do bộ BPE của `Gipformer 1.5` tối ưu cho tiếng Việt, các tên ngoại lai/tên riêng dễ bị vỡ chữ: `public_0619` (`"cree vill hold"`), `public_0672` (`"note win chưa đi"`), `public_0336` (`"chals"`), `public_0357` (`"Hoa Anh Tuấn"`). Khi search tên bị sai, LLM dễ rơi vào vòng lặp `search` hoặc chạm `max_calls` / `max_invalid`.
3. **Mã chuyến bay/tàu hoặc ký hiệu chữ-số bị tách rời thành từng âm tiết đọc:**
   - Ví dụ `"s e ba"` (thay vì `se3`), `"rv sáu một hai"` (thay vì `rv612`), `"e năm"` (thay vì `e5`).

---

## 2. Bảng xếp hạng các mức ưu tiên tối ưu ASR

| Mức ưu tiên | Giải pháp | Đụng tới code / model | Chi phí VRAM & Rủi ro | Kỳ vọng tác động |
| :--- | :--- | :--- | :--- | :--- |
| **Ưu tiên 1A (Cao nhất)** | Bật **Contextual Biasing (BPE Hotwords động theo sample)** có sẵn trong `sherpa-onnx` + `Gipformer 1.5` | Sửa nhỏ [asr.py](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/asr.py) + trích hotwords từ context/history | `0` VRAM, rủi ro rất thấp | **Cao** — sửa trực tiếp từ khóa domain, tên riêng có trong context/history |
| **Ưu tiên 1B (Cao nhất)** | Tăng `--asr-beam-size` (`max_active_paths`) từ `4` → `8` | Chỉ đổi tham số CLI lúc chạy ASR | `0` VRAM, chỉ chạy CPU 1 lần rồi cache | **Trung bình–Cao** — giữ thêm nhánh BPE cho tên riêng/từ ghép |
| **Ưu tiên 2A (Trung bình–Cao)** | **Post-ASR Normalization** gộp mã chữ-số (`s e ba` → `se3`, `rv sáu một hai` → `rv612`) & chuẩn hóa từ khóa cố định | Rule hậu xử lý text nhẹ trên output của ASR | `0` VRAM, rủi ro thấp nếu rule chặt chẽ | **Trung bình** — giúp LLM parse chính xác mã chuyến/ký hiệu |
| **Ưu tiên 2B (Trung bình–Cao)** | **Search Fallback** ở tầng Prompt / Guard khi tên riêng bị phiên âm sai | Prompt / recovery rule phía LLM | `0` VRAM, không sửa ASR | **Trung bình–Cao** — giảm `max_calls` và vòng lặp `search` rỗng |
| **Ưu tiên 3 (Trung bình)** | Tinh chỉnh sâu hơn trên chính `Gipformer 1.5` (`blank_penalty`, `hotwords_score`, oversample tên riêng/mã chuyến, đo WER trên `dev`) | Retrain / tune tham số decoding ASR | Tốn thời gian chạy train/eval ASR trên GPU/CPU | **Trung bình** — cải thiện gốc mô hình acoustic/BPE |
| **Ưu tiên 4 (Thấp nhất — Dự phòng)** | Ghép thêm model ASR thứ 2 (`whisper-large-v3-turbo` / `PhoWhisper`) chạy chọn lọc cho ca nghi ngờ OOV tiếng Anh | Tải thêm model & sửa luồng Stage ASR | Tăng độ phức tạp code, cần kiểm tra quy chế | **Cao cho OOV tiếng Anh**, nhưng chỉ dùng nếu thực sự cần |

---

## 3. Chi tiết triển khai từng mức ưu tiên

### Ưu tiên 1 (Cao nhất): Tận dụng tính năng có sẵn của `Gipformer 1.5` + `sherpa-onnx`

#### 1A. Bật Contextual Biasing / BPE Hotwords động theo từng mẫu (`modeling_unit="bpe"`)
Hiện tại [LocalASR](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/asr.py#L59-L94) khởi tạo `sherpa_onnx.OfflineRecognizer.from_transducer` với `modified_beam_search` nhưng chưa bật `modeling_unit="bpe"`, `bpe_vocab` và `create_stream(hotwords=...)`. Trong khi đó [models/asr_ft](file:///mnt/mmlab2024nas/trantran/lifemate/code/models/asr_ft) đã có sẵn `bpe.model` (SentencePiece):

- **Cơ chế kỹ thuật:**
  1. Sinh tệp `bpe.vocab` từ `bpe.model` theo chuẩn `<piece>\t<score>` (chỉ tạo 1 lần hoặc sinh tạm trong cache).
  2. Truyền `modeling_unit="bpe"`, `bpe_vocab=...` và `hotwords_score` phù hợp (ví dụ `1.5`–`2.5`) vào `OfflineRecognizer.from_transducer`.
  3. Tại hàm `transcribe` / `transcribe_batch`, truyền chuỗi `hotwords` động riêng cho từng sample qua `self.model.create_stream(hotwords=...)` (định dạng `MODEL_STYLE_PHRASE :score` phân cách bằng `/`).
- **Nguồn trích hotwords tự động cho từng sample (hoàn toàn từ input hợp lệ):**
  1. **Các lượt `assistant` trong `history`:** Text do assistant trả lời là văn bản chuẩn, thường chứa tên riêng, mã chuyến (`SE3`), địa điểm hoặc tên sự kiện từ lượt trước.
  2. **`context.user_profile`:** `name`, tên đường/quận/phường trong `home_address`, `work_address`, `city`.
  3. **Domain lexicon ngắn gọn theo `context.installed_apps`:** Ví dụ nếu sample có `public_transit` + `train_bus_tickets` thì gắn nhẹ các cụm dễ trượt như `"xe buýt ra ga"`, `"chuyến buýt"`, `"ga hà nội"`; nếu có ứng dụng tài chính/vàng thì gắn `"vàng sjc"`, `"zalo"`, v.v. (giữ danh sách ngắn gọn, có chọn lọc để tránh bias quá tay gây nhận nhầm từ bình thường).
- **Kết quả kiểm chứng thực tế trên `public_0391.wav` với `models/asr_ft`:**
  - Khi chạy thử nghiệm BPE hotwords (`"so giúp anh"`, `"xe buýt ra ga"`, `"chuyến buýt"`, điểm `:3.0`, `hotwords_score=2.5`):
    - **Không bật hotwords (`NO HW`):** `XOA GIÚP ANH ĐI Ô TÔ VỚI XE BUÝT GA XEM NẾU CHUYẾN BIG SỚM NHẤT...`
    - **Bật hotwords (`WITH HW`):** `SO GIÚP ANH ĐI Ô TÔ VỚI XE BUÝT RA GA XEM NẾU CHUYẾN BUÝT SỚM NHẤT...` *(khắc phục trực tiếp cả 3 cụm lỗi trọng yếu)*.

#### 1B. Tăng `--asr-beam-size` (`max_active_paths`) từ `4` lên `8`
- Stage ASR chạy hoàn toàn trên CPU và ghi ra cache `transcripts.jsonl`, hoàn toàn tách biệt với Stage LLM nên **không ảnh hưởng VRAM**.
- Tăng `max_active_paths` lên `8` (khi chạy với `modified_beam_search`) giúp beam giữ được nhiều giả thuyết BPE hơn đối với các từ ghép ít gặp và tên riêng trước khi chốt nhánh tốt nhất, đồng thời hỗ trợ contextual biasing hoạt động hiệu quả hơn.

---

### Ưu tiên 2 (Trung bình–Cao): Hậu xử lý nhẹ trên đầu ra & Search Fallback (Không sửa model)

#### 2A. Post-ASR Deterministic Normalization
Xử lý hậu kỳ xác định trên chuỗi sau khi decode (hoặc trước khi nạp vào LLM/SFT với điều kiện áp dụng đồng bộ):
- **Gộp mã chuyến / mã vé / ký hiệu chữ-số bị tách âm tiết:**
  - Áp dụng regex an toàn theo ngữ cảnh đứng sau các từ khóa như `"tàu"`, `"chuyến"`, `"mã"`, `"xe"`, `"đổ"`, `"xăng"`: ví dụ `"s e ba" → "se3"`, `"rv sáu một hai" → "rv612"`, `"e năm" → "e5"`.
- **Từ điển sửa lỗi phiên âm cố định (bổ trợ nhẹ):**
  - Chuẩn hóa một số biến thể phiên âm chắc chắn sai thành tên app/thuật ngữ chuẩn nếu còn lọt qua (ví dụ `"gia lô" → "zalo"`, `"meo" → "mail"` trong ngữ cảnh gửi/đọc thư).

#### 2B. Search Fallback ở tầng Prompt / Guard cho tên riêng bị phiên âm
- Nhiều mẫu gặp tên riêng tiếng Anh/tên quán lạ không nằm trong `history` hay `user_profile` (ví dụ `public_0619`: *"cree vill hold theo cái **đối diện quán mộc lan**"*). Khi ấy dù bật hotwords cũng không thể đoán trước mọi tên quán lạ trong `public_test`.
- **Giải pháp:** Bổ sung hướng dẫn ở prompt/recovery để khi lần `search` đầu tiên bằng cụm tên ngoại lai phiên âm trả về rỗng:
  1. Thử tìm ngay bằng **mốc tiếng Việt rõ nghĩa đi kèm** (như `"quán mộc lan"` / `"Mộc Lan"`), hoặc
  2. Thử tách **1 từ khóa ngắn có độ tin cậy cao nhất**, thay vì lặp lại 7–8 biến thể đoán mò dẫn tới `max_calls`.

---

### Ưu tiên 3 (Trung bình): Tinh chỉnh sâu hơn trên chính `Gipformer 1.5`

Nếu cần cải thiện chất lượng acoustic gốc của `Gipformer 1.5` mà vẫn giữ 1 model duy nhất:
1. **Đo WER chuẩn hóa trên tập `dev` ([result/asr_sft_data/dev.jsonl](file:///mnt/mmlab2024nas/trantran/lifemate/code/result/asr_sft_data/dev.jsonl)):** Hiện tại log train [train_asr.log](file:///mnt/mmlab2024nas/trantran/lifemate/runtime_cache/logs/train_asr.log) mới theo dõi `dev_loss` để chọn `best.pt`. Nên bổ sung bước đánh giá WER/CER trực tiếp trên `dev.jsonl` để tinh chỉnh các siêu tham số decoding như `hotwords_score`, `blank_penalty` và `max_active_paths` một cách khách quan.
2. **Oversample / Cân bằng dữ liệu tên riêng & mã chữ-số khi fine-tune ASR:** Tăng trọng số lấy mẫu đối với các câu chứa từ tiếng Anh ngắn, tên riêng, mã chuyến tàu/bay trong tập train trước khi export sang ONNX `models/asr_ft`.

---

### Ưu tiên 4 (Thấp nhất — Chỉ dùng làm phương án dự phòng): Bổ sung model ASR thứ 2 cho tên riêng

> Chỉ cân nhắc nếu sau khi áp dụng Ưu tiên 1–3 mà tỷ lệ hỏng mẫu vì tên tiếng Anh / OOV vẫn cao và quy chế cho phép dùng thêm model pre-trained ngoài `Gipformer`.

- **Model đề xuất:** `openai/whisper-large-v3-turbo` (qua `faster-whisper` chạy nhanh ở `fp16`/`int8` cho tên tiếng Anh, thương hiệu, code-switching Anh–Việt) hoặc `vinai/PhoWhisper-medium` (nếu thiên về tên riêng tiếng Việt).
- **Cách ghép an toàn, không đụng VRAM của LLM:**
  - Vì Stage ASR và Stage LLM chạy ở 2 process tuần tự, có thể giữ nguyên **`asr_ft` làm transcript chính** (giữ văn phong khẩu ngữ, phương ngữ và cách đọc số khớp với SFT LLM).
  - Chỉ chạy Whisper chọn lọc (**Selective Decode**) trên khoảng `10–15%` mẫu có dấu hiệu nghi ngờ OOV / vỡ tên riêng (từ tiếng Anh ngắn rời rạc, chuỗi âm vô nghĩa không nằm trong từ điển tiếng Việt) và đính kèm phần tên riêng gợi ý (**Dual-Hypothesis Hint**) cho LLM tham khảo khi điền tham số `query`.
