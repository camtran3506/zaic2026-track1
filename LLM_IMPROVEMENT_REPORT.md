# Báo cáo Lộ trình Cải tiến LLM Pipeline (`lifemate`)

> **Trọng tâm:** Ưu tiên các cải tiến **không cần train lại LoRA** ở **Tier 1** để chạy được ngay trên adapter hiện tại ([adapter](file:///mnt/mmlab2024nas/trantran/lifemate/code/result/run_sft_traj/training/adapter)), sau đó mới đến các hạng mục **Tier 2** dành cho vòng chuẩn bị dữ liệu SFT / fine-tune tiếp theo.
> **Trạng thái mã nguồn:** Tài liệu báo cáo thuần phân tích — không sửa đổi mã nguồn trong lúc tiến trình inference hiện tại ([inference_traj4.log](file:///mnt/mmlab2024nas/trantran/lifemate/runtime_cache/logs/inference_traj4.log)) đang chạy.

---

## 1. Tổng quan Chẩn đoán Thực nghiệm (`submission_traj3`)

Đối chiếu kết quả rollout trên **634 mẫu Public Test** ([submission_traj3.jsonl](file:///mnt/mmlab2024nas/trantran/lifemate/code/result/submission_traj3.jsonl), [submission_traj3.diagnostics.jsonl](file:///mnt/mmlab2024nas/trantran/lifemate/code/result/submission_traj3.diagnostics.jsonl)) dùng adapter hiện tại (`result/run_sft_traj/training/adapter`):

| Chỉ số | Giá trị quan sát | Nhận xét |
| :--- | :--- | :--- |
| Tổng số mẫu Public Test | `634` | `stop_reasons`: `done: 538` (84.86%), `ask_missing_info: 86` (13.56%), `max_invalid: 7` (1.10%), `max_calls: 3` (0.47%) |
| Phân bố `executed` | `{0: 37, 1: 136, 2: 202, 3: 135, 4: 63, 5: 46, 6: 11, 7: 1, 8: 3}` | Toàn bộ 3 ca chạm trần `executed=8` đều là vòng lặp tìm kiếm không thoát ra được |
| Tổng `invalid_proposals` | `42` lần trên `20` mẫu | Trong đó `7` mẫu chạm trần `max_invalid=3` và bị ngắt phiên giữa chừng |
| Tổng `local_executed` | `108` lượt | Tỉ lệ dùng helper còn thưa; riêng lỗi helper đang bị tính chung vào trần `max_invalid=3` |
| Phân bố Local Helper trong SFT ([manifest.json](file:///mnt/mmlab2024nas/trantran/lifemate/code/result/run_sft_traj/sft/manifest.json)) | `local_context.lookup: 461`, `local_assist.solve: 156`, `local_entity.resolve: 72`, `local_checks.evaluate: 6`, `local_time.departure_time: 2`, `local_time.duration_between: 2`, `local_time.calendar_reference: 0`, `local_text.normalize: 0`, `local_schema.inspect: 0` | Mất cân bằng mạnh; helper thời gian và kiểm tra số học ít được model dùng đúng lúc |

### Ràng buộc Kỹ thuật Bắt buộc để Giữ Tương thích với Adapter Cũ

Trong [runtime.py](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/runtime.py#L157-L196), hàm `load_runtime_manifest` kiểm tra khớp chặt các hằng số phiên bản ghi trong [lifemate_runtime.json](file:///mnt/mmlab2024nas/trantran/lifemate/code/result/run_sft_traj/training/adapter/lifemate_runtime.json#L4-L11). Vì vậy, toàn bộ các thay đổi thuộc **Tier 1** **không được sửa giá trị** các hằng số phiên bản sau (nếu đổi chuỗi, loader sẽ từ chối nạp adapter cũ):

* `TEXT_NORMALIZATION = "nfc-whitespace-lower-initial-terminal-v1"` ([runtime.py](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/runtime.py#L175))
* `LOCAL_TIME_POLICY = "local-time-tools-v1"` ([runtime.py](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/runtime.py#L177))
* `LOCAL_HELPER_POLICY = "python-specialists-itn-v2"` ([runtime.py](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/runtime.py#L179))
* `ASSIST_POLICY = "fixed-python-task-workflows-v1"` ([runtime.py](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/runtime.py#L182))
* `SCHEMA_PROMPT_POLICY = "redundant-schema-metadata-v1"` ([runtime.py](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/runtime.py#L184))
* `ITN_POLICY = "vi-explicit-spans-v1"` ([runtime.py](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/runtime.py#L186))
* `EXECUTION_POLICY = "automatic-tool-guards-v1"` ([runtime.py](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/runtime.py#L188))
* `REASONING_POLICY = "deterministic-short-checked-reasoning-v1"` ([runtime.py](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/runtime.py#L190))

---

## 2. TIER 1 — Ưu tiên Cao Nhất: Cải tiến Guardrail, Helper & Hậu xử lý (Không cần train lại, dùng ngay với Adapter cũ)

Các hạng mục dưới đây đều vận hành ở tầng tiền/hậu xử lý đề xuất tool-call (`prepare_call_arguments`, `precheck_call`, `build_search_fallback`, `LoopMonitor`, `missing_info_guard`, `agent.py`), giữ nguyên format prompt/schema lúc train nên chạy được ngay với adapter hiện tại.

### P1.1 — Chuẩn hóa & Bắt lỗi Mã Sân bay IATA + Gợi ý dịch ngày `"thứ ... tuần sau"` trong `flights.search` / `train_bus_tickets.search`

* **Mức ưu tiên:** Cao nhất (Tác động trực tiếp 4/10 ca chết trần `max_calls` / `max_invalid` và tiết kiệm lượt gọi ở nhiều ca khác).
* **File liên quan:** [tool_execution.py](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/tool_execution.py) — `build_search_fallback` ([L366-L439](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/tool_execution.py#L366-L439)), `prepare_call_arguments` ([L474-L494](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/tool_execution.py#L474-L494)), `precheck_call` ([L529-L555](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/tool_execution.py#L529-L555)).
* **Bằng chứng lỗi thực tế:**
  1. **`public_0213`** (`stop=max_calls`, `executed=8`): Transcript *"đặt giúp chị hai vé chuyến rv157 đi đà lạt thứ năm tuần sau nhé rồi giữ luôn một chỗ ở bãi khu b không năm sân bay nội bài..."*. Trong DB chuyến `RV157` đi Đà Lạt (`destination='DLI'`), nhưng model truyền `destination='DAD'` (Đà Nẵng) → trả về `flights: []`, sau đó lặp cộng từng ngày `2026-10-01` → `2026-10-08` hết 8 lượt gọi, bỏ lỡ hoàn toàn cả vé máy bay lẫn đặt bãi xe `parking.reserve`.
  2. **`public_0252`** (`stop=max_invalid`, `inv=3`): Transcript *"Đặt cho em hai vé chuyến rv23253 đi quy nhơn hôm mùng chín tháng mười..."*. Trong DB chuyến `RV253` đi Quy Nhơn (`destination='UIH'`), nhưng model truyền `destination='HUI'` (Huế) → trả về `flights: []` rồi chạm trần `max_invalid=3`.
  3. **`public_0025`** (`stop=max_invalid`, `inv=3`): Bước 1 model truyền tên thành phố `{'origin': 'TP.HCM', 'destination': 'Hà Nội'}` thay vì mã IATA `SGN`/`HAN` → trả về `flights: []`, mất oan 1 lượt mới tự đổi sang `SGN`/`HAN`.
  4. **`public_0287`** (`stop=max_invalid`, `inv=3`): `current_time = 2026-10-01` (Thứ Năm). Transcript *"sáng thứ bảy tuần sau"* → chuyến `RV612` trong DB nằm ở `2026-10-10`, nhưng model tìm `2026-10-03` (Thứ Bảy tuần này) → `flights: []` → kéo hỏng luôn chuỗi gọi xe (`ride_hailing.book`) và tạo lịch (`calendar.create_event`).
* **Giải pháp đề xuất (0-train):**
  1. Trong `prepare_call_arguments` ([tool_execution.py](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/tool_execution.py#L474-L494)), ánh xạ chuẩn tên thành phố/sân bay sang mã IATA của bộ dữ liệu cho `flights.search` (`TP.HCM`/`Sài Gòn` → `SGN`, `Hà Nội`/`Nội Bài` → `HAN`, `Đà Nẵng` → `DAD`, `Đà Lạt`/`Liên Khương` → `DLI`, `Quy Nhơn`/`Phù Cát` → `UIH`, `Huế`/`Phú Bài` → `HUI`, `Cần Thơ` → `VCA`, `Nha Trang`/`Cam Ranh` → `CXR`, `Phú Quốc` → `PQC`, `Hải Phòng`/`Cát Bi` → `HPH`, `Vinh` → `VII`, `Đồng Hới` → `VDH`, `Thanh Hóa`/`Thọ Xuân` → `THD`, `Tuy Hòa` → `TBB`, `Pleiku` → `PXU`, `Buôn Ma Thuột` → `BMV`, `Chu Lai` → `VCL`, `Cà Mau` → `CAH`, `Côn Đảo` → `VCS`).
  2. Đối chiếu trực tiếp với tên điểm đi/điểm đến có trong `transcript` (ví dụ thấy `"đi đà lạt"` mà argument là `DAD`, hoặc `"đi quy nhơn"` mà argument là `HUI`): tự động hiệu chỉnh sang `DLI` / `UIH` ở `prepare_call_arguments` (hoặc chặn ở `precheck_call` với chỉ dẫn mã IATA đúng).
  3. Mở rộng `build_search_fallback` ([tool_execution.py](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/tool_execution.py#L366-L439)) cho `flights.search` và `train_bus_tickets.search` (hiện hàm này mới xử lý `stocks.get_quote` và các tool có tham số `query`): khi tìm kiếm trả về `[]`:
     * Nếu `transcript` có cụm `"tuần sau"` / `"tuần tới"`, gợi ý rõ ngày dịch `+7` (`date + 7 days`, ví dụ `2026-10-03` → `2026-10-10`).
     * Nếu `transcript` nêu ngày cụ thể (`public_0061`: *"ngày hai mươi tháng mười một"* mà model tìm `2026-11-21` thay vì `2026-11-20`), kiểm tra chênh lệch ngày so với số ngày nêu trong transcript và gợi ý ngày chính xác thay vì cộng dồn tuyến tính.

---

### P1.2 — Chặn Vòng lặp Tìm kiếm Rỗng / Trùng lặp (`Search Loop Cap`)

* **Mức ưu tiên:** Cao (Cứu trực tiếp toàn bộ 3/3 ca `max_calls=8`: `public_0061`, `public_0213`, `public_0574`).
* **File liên quan:** [execution_policy.py](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/execution_policy.py#L15) (`DEFAULT_EMPTY_READ_LIMIT`), [L148-L154](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/execution_policy.py#L148-L154) (`LoopMonitor.check`); [agent.py](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/agent.py#L166); [tool_execution.py](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/tool_execution.py#L225-L284).
* **Bằng chứng lỗi thực tế:**
  1. **`public_0061`** & **`public_0213`**: Model gọi `flights.search` trả về `flights: []` 8 lần liên tiếp với ngày tăng dần `+1`. Nguyên nhân trực tiếp là [execution_policy.py#L15](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/execution_policy.py#L15) đặt `DEFAULT_EMPTY_READ_LIMIT = 0`, khiến nhánh chặn `read_empty_retry_cap` tại [execution_policy.py#L148-L154](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/execution_policy.py#L148-L154) hoàn toàn không hoạt động.
  2. **`public_0574`** (`stop=max_calls`, `executed=8`): `context.user_profile.work_address` có sẵn `"Tiệm tạp hóa Thảo, 203 Lê Văn Sỹ, Quận 3"`, nhưng model gọi `maps.search_place` 7 lần liên tiếp (`'tiệm tạp hóa Thảo'`, `'tạp hóa Thảo'`, `'Thảo'`, `'tiệm tạp hóa'`, `'tạp hóa'`, `'tiệm'`, `'*'`). Vì môi trường trả về 1 bản ghi không khớp tên (`Phở Hoà Pasteur`), mảng `places` **không rỗng**, nên `_dead_tokens_from_empty` ([tool_execution.py#L225-L240](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/tool_execution.py#L225-L240)) không kích hoạt. Kết quả là đốt sạch `max_calls=8`, không kịp gọi `grocery.checkout` và `tasks.complete(task_id='task_7381')`.
* **Giải pháp đề xuất (0-train):**
  1. Bật `DEFAULT_EMPTY_READ_LIMIT = 2` trong [execution_policy.py#L15](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/execution_policy.py#L15) (tối đa 2 lần read rỗng liên tiếp cho cùng một tool trong cùng state epoch).
  2. Bổ sung theo dõi **kết quả lặp lại (`repeated_search_results`)** trong `LoopMonitor` / `tool_execution.py`: khi một tool tìm kiếm có `query` (`maps.search_place`, `grocery.search`, `ecommerce.search`, `recipes.search`) được gọi ≥ 2 lần trong cùng epoch mà tập ID trả về **trùng hệt lần trước** (không sinh ID mới nào hoặc kết quả không chứa token cốt lõi của `query`), coi nhánh truy vấn biến thể con đó là bão hòa; chặn lặp tiếp và trả thông báo nhắc dùng `context.user_profile` (`home_address` / `work_address`) hoặc chuyển sang bước hành động kế tiếp.

---

### P1.3 — Cô lập lỗi của Local Helper (`local_*`) khỏi trần `max_invalid_proposals`

* **Mức ưu tiên:** Cao (Ngăn helper nội bộ làm sập oan cả phiên chạy môi trường).
* **File liên quan:** [agent.py](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/agent.py#L266-L324) và [local_checks.py](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/local_checks.py#L27-L51), [L247-L318](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/local_checks.py#L247-L318).
* **Bằng chứng lỗi thực tế:**
  * **`public_0482`** (`stop=max_invalid`, `inv=3`, `executed=2`): Transcript *"Rứa hai thứ vợ dặn mua trên app đi chợ hết nhiêu tiền em ví anh còn đủ trả không."* Model đã gọi đúng `grocery.search({'query': 'cà chua'})` (`32000`) và `grocery.search({'query': 'cải ngọt'})` (`18000`), nhưng sau đó gặp 3 lỗi đề xuất liên tiếp (`inv=3`) nên không kịp gọi `ewallet.get_balance({})`.
  * Về mặt cơ chế trong [agent.py#L295-L323](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/agent.py#L295-L323):
    * Lỗi gọi trùng local helper ([agent.py#L296-L307](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/agent.py#L296-L307)) hoặc lỗi tham số `local_*` (`status == 'invalid_request'` ở [agent.py#L313-L314](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/agent.py#L313-L314) hay `ValueError` ở [agent.py#L315-L317](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/agent.py#L315-L317)) đều gọi `reject(...)`, **cộng trực tiếp vào biến `invalid` dùng chung với trần `max_invalid=3`** ([agent.py#L345](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/agent.py#L345)).
    * Trong [local_checks.py#L280-L286](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/local_checks.py#L280-L286), `_evaluate` siết cứng `if args.keys() != required` với `operation` ∈ `{"sum", "minimum", "compare", "filter"}` và không chấp nhận tham số dạng chuỗi số hoặc thừa/thiếu khóa phụ.
* **Giải pháp đề xuất (0-train):**
  1. Trong [agent.py](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/agent.py), tách bộ đếm lỗi riêng cho local helper (`local_invalid`), **không cộng vào `invalid` của `max_invalid=3`**. Nếu `local_invalid` vượt ngưỡng (ví dụ `2`), từ chối thực thi helper đó với thông báo ngắn yêu cầu model bỏ qua helper và gọi trực tiếp app tool cần thiết (ví dụ `ewallet.get_balance`).
  2. Trong [local_checks.py](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/local_checks.py) (`_arguments` và `_evaluate`), chuẩn hóa an toàn chuỗi số thuần túy (ví dụ `"32000"` → `32000`) trước khi kiểm tra `_decimal`, đồng thời giữ chặt semantic kiểm toán học để phản hồi lỗi rõ khóa cần truyền khi thiếu.

---

### P1.4 — Cải thiện `missing_info_guard.py` khi ID đã có sẵn ở `history` & Điều hướng `contacts` → `phone.recent_calls`

* **Mức ưu tiên:** Trung bình – Cao (Nhắm vào nhóm `20` mẫu gọi `sys.ask_missing_info` ngay bước 1 và lỗi chọn nhầm app danh bạ).
* **File liên quan:** [missing_info_guard.py](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/missing_info_guard.py#L153-L262) (`_entity_value`, `check_missing_info`), [execution_policy.py#L37-L49](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/execution_policy.py#L37-L49) (`check_tool_access`), [tool_execution.py#L529-L555](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/tool_execution.py#L529-L555) (`precheck_call`).
* **Bằng chứng lỗi thực tế:**
  1. **`public_0511`** (`stop=max_invalid`, `inv=3`): Transcript *"Mẹ chuyển tiền tháng này chưa rồi đi nạp thẻ thêm mười lượt metro kia gọi xong gọi điện cho mẹ nha"*. Mẫu này **không cài app `contacts`**, nhưng có cài `phone` (trong `phone.recent_calls` có sẵn bản ghi `'Mẹ'` với `'phone_number': '09126743080'`). Model gọi `contacts.search` (bị `check_tool_access` chặn vì không thuộc app đã cài) hoặc gọi `phone.call(contact_id=...)` rồi chạm trần `inv=3`.
  2. **`public_0529`** (`stop=ask_missing_info`, `executed=1`): Trong `history` đã có `banking.get_balance` trả về duy nhất 1 tài khoản thanh toán (`acc_2807`, `24,600,000`) và 1 tài khoản tiết kiệm (`acc_555`). User nói *"Ờ tiền bên thanh toán đâu có đủ rưỡi hai cái tôi gửi một cái thôi nha"*, nhưng model gọi ngay `sys.ask_missing_info(tool_name='banking.open_savings', params=['from_account_id'])`.
  3. **`public_0528`** (`stop=ask_missing_info`, `executed=1`): Trong `history` đã có `flights.my_bookings` với 2 vé (`bk_318` bay ngày Thứ Bảy `2026-12-19`, `bk_74` bay ngày Thứ Bảy `2026-12-26`). Transcript nói *"cái vé thứ bảy này đó hủy giùm tôi nghen"*, nhưng model gọi ngay `sys.ask_missing_info(tool_name='flights.cancel', params=['booking_id'])` mà không đối chiếu `bk_318` (Thứ Bảy tuần hiện tại `2026-12-19` so với `current_time = 2026-12-16`).
  4. **`public_0517`** (`stop=ask_missing_info`, `executed=1`): *"tắt cái v năm gửi xe giúp em nghe"* (lỗi ASR đọc *"phiên"* thành *"v năm"*), trong `history` đã có 2 phiên từ `parking.list_sessions` nhưng model hỏi lại ngay ở bước 1.
* **Giải pháp đề xuất (0-train):**
  1. Trong `check_tool_access` ([execution_policy.py](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/execution_policy.py#L37-L49)) hoặc `precheck_call`: khi model gọi `contacts.search` / `contacts.get` mà app `contacts` không được cài, kiểm tra `enabled`: nếu có `phone.recent_calls` hoặc `chat_ott.list_chats`, chèn thẳng gợi ý điều hướng: *"App `contacts` không được cài; hãy gọi `phone.recent_calls({})` để lấy `phone_number` cho `phone.call`, hoặc `chat_ott.list_chats({})` cho `chat_ott`."*
  2. Mở rộng `check_missing_info` ([missing_info_guard.py#L191-L262](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/missing_info_guard.py#L191-L262)):
     * Bổ sung fallback collection khi không có `contacts`: trong `_entity_value`, nếu `argument == 'phone_number'` của `phone.call` (hoặc tương tự), cho phép tra thêm từ `phone.recent_calls` đã quan sát.
     * Đối với `from_account_id` trong `banking.*` hoặc các `x-ref` chỉ có **1 bản ghi duy nhất khớp cụm từ khóa ngữ cảnh** (như `"tài khoản thanh toán"` khớp duy nhất `acc_2807` ở `public_0529`, hay `"thứ bảy này"` khớp ngày `2026-12-19` của `bk_318` ở `public_0528`): trả về rejection `information_already_available` kèm `known_arguments` và `unresolved_params` để buộc model tiếp tục thay vì dừng bằng `sys.ask_missing_info`.

---

### P1.5 — Hướng dẫn "Bỏ qua nhánh con rỗng / Dừng sạch" trong `check_observed_xref`

* **Mức ưu tiên:** Trung bình (Ngăn `max_invalid=3` khi dữ liệu nguồn thực sự không có kết quả).
* **File liên quan:** [tool_execution.py](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/tool_execution.py#L522-L526) (`check_observed_xref`) và `attempt_env_call` ([L568-L651](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/tool_execution.py#L568-L651)).
* **Bằng chứng lỗi thực tế:**
  * **`public_0025`** (`stop=max_invalid`, `inv=3`): Transcript yêu cầu *"đặt giúp tôi vé đi hà nội ngày mùng một tháng mười hai ... hẹn báo thức trước ba tiếng mà nhắn zalo cô mai báo giờ bay nhé"*. Thực tế `flights.search({'origin': 'SGN', 'destination': 'HAN', 'date': '2026-12-01'})` trả về `flights: []`. Vì không tìm thấy chuyến bay nào, model lại cố đề xuất `flights.book` / các call tiếp theo với ID không có trong `observed_ids`, bị `xref_guard` chặn 3 lần đến `max_invalid=3`.
* **Giải pháp đề xuất (0-train):**
  Khi `check_observed_xref` phát hiện tham số `x-ref` không nằm trong `observed_ids[ref_app]` **và đồng thời read tool của `ref_app` đã được gọi trong phiên hiện tại nhưng chỉ trả về danh sách rỗng `[]`**, bổ sung hướng dẫn dứt khoát vào thông báo lỗi: *"Collection `<ref>` đã đọc và rỗng `[]`; không tồn tại ID hợp lệ để thực thi `<name>`. Không thử lại `<name>`; hãy xử lý các yêu cầu độc lập còn lại hoặc kết thúc bằng `done`."*

---

### P1.6 — Hướng dẫn tham số bắt buộc `near` cho `parking.find` khi bị `precheck` / môi trường từ chối

* **Mức ưu tiên:** Trung bình – Thấp (Bổ trợ cho P1.1 ở các câu đa nhiệm vé máy bay + đỗ xe).
* **File liên quan:** [tool_execution.py](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/tool_execution.py#L529-L555) (`precheck_call`) và [execution_policy.py](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/execution_policy.py).
* **Bằng chứng lỗi thực tế:**
  * Ở cả **`public_0213`** và **`public_0252`**, yêu cầu đều có vế thứ hai *"giữ luôn một chỗ ở bãi khu b không năm sân bay nội bài"*. Khi gọi thử trực tiếp vào `ToolEnv`, `parking.find({'query': 'Nội Bài'})` trả về `{'ok': False, 'error': 'thiếu tham số bắt buộc: near'}` (khác với các app search khác vốn dùng `query`). Nếu model đoán mò `lot_id='khu b không năm'` trong `parking.reserve`, sẽ bị `xref_guard` chặn.
* **Giải pháp đề xuất (0-train):**
  Khi `check_observed_xref` chặn `parking.reserve` hoặc khi `parking.find` trả lỗi thiếu `near`, bổ sung gợi ý chuẩn cấu trúc tham số: gọi `parking.find` với tham số `near` (ví dụ tên sân bay/địa điểm từ transcript) để quan sát `lot_id` hợp lệ trước khi gọi `parking.reserve`.

---

## 3. TIER 2 — Cải tiến Dữ liệu SFT & Training LLM (Dành cho vòng Fine-tune tiếp theo)

Các hạng mục này đòi hỏi chạy lại `prepare_sft.py` và train lại LoRA adapter (`train_lora.py`), phù hợp khi bạn sẵn sàng chạy một chu kỳ huấn luyện mới sau khi đánh giá hiệu quả của Tier 1:

1. **P2.1 — Xử lý `GoldGuardConflict` (`train_1696`) thay vì bỏ toàn bộ mẫu ([sft.py](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/sft.py)):**
   * Trong `run_sft_traj` ([manifest.json](file:///mnt/mmlab2024nas/trantran/lifemate/code/result/run_sft_traj/sft/manifest.json)), mẫu `train_1696` bị skip hoàn toàn (`kept 1544/1545`) vì bước thứ 3 trong nhãn Gold là `chat_ott.send_message(chat_id='chat_6142', ...)` vốn là một ID giả không có trong `chat_ott.list_chats`, dẫn tới kết quả Gold `{'ok': True, 'done': False, 'message': 'không tìm thấy chat_id'}` và xung đột với `xref_guard`.
   * **Hướng cải tiến:** Khi bước cuối của Gold có `done: False` không làm đổi `final_state` và bị `xref_guard` chặn, chỉ cắt bỏ bước no-op `done: False` đó để giữ lại 2 bước hợp lệ phía trước (`banking.open_savings`, `chat_ott.list_chats`) thay vì loại cả sample.
2. **P2.2 — Tối ưu hóa hiển thị `user_context.date_reference` hoặc bật `--include-date-helpers` khi cần ([local_time.py](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/local_time.py#L15-L64), [prompt_session.py](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/prompt_session.py#L41-L44), [prepare_sft.py](file:///mnt/mmlab2024nas/trantran/lifemate/code/prepare_sft.py#L78-L79)):**
   * **Lưu ý hiện trạng:** `user_context.date_reference` **đã có sẵn và đã được bật mặc định** qua `prompt_session.py` cả lúc chuẩn bị SFT lẫn lúc inference. Chỉ có cờ `--include-date-helpers` (dạy gọi tool `local_time.calendar_reference`) là đang tắt mặc định (`False`) vì prompt đã có `user_context.date_reference` (dẫn tới `local_time.calendar_reference: 0` trong [manifest.json](file:///mnt/mmlab2024nas/trantran/lifemate/code/result/run_sft_traj/sft/manifest.json)).
   * **Nguyên nhân vẫn sót lỗi (`public_0287`):** `date_reference()` trả về mảng phẳng `days` (15 ngày) kèm `this_week` / `next_week` dạng chuỗi khoảng (`"2026-10-05 .. 2026-10-11"`). Ở chế độ zero-thinking (`thinking_budget=0`), model đôi khi chọn nhầm `Thứ Bảy` đầu tiên (`2026-10-03`) thay vì `Thứ Bảy` thuộc `next_week` (`2026-10-10`). Khi train lại SFT, có thể gắn nhãn trực tiếp `(tuần này)` / `(tuần sau)` vào từng phần tử trong `days` của `date_reference` (hoặc tăng cường ví dụ `tuần sau` trong dữ liệu).
3. **P2.3 — Bổ sung ví dụ `assist_training` / `local_checks.evaluate` cho câu điều kiện dựa trên kết quả nằm trong `history` ([assist_training.py](file:///mnt/mmlab2024nas/trantran/lifemate/code/lifemate/assist_training.py)):**
   * **Bằng chứng (`public_0043`, `executed=0`):** Trong `history`, assistant đã gọi `gold_price.get_price` và trả lời giá vàng SJC mua vào là `131.7 triệu`; lượt hiện tại yêu cầu *"Vàng SJC mà mua vào hôm nay dưới một trăm ba hai triệu thì sáng mai chín giờ nhắc tôi đi mua vàng nha"*. Do `131.7 < 132`, điều kiện **ĐÚNG** và cần gọi `reminders.create`, nhưng model thấy câu trả lời read ở `history` nên dừng ngay ở `executed=0`.
   * Cần bổ sung các mẫu huấn luyện kết hợp dữ kiện số ở `history` + kiểm tra điều kiện → kích hoạt write tool.
4. **P2.4 — Đồng bộ tập SFT dual-transcript bằng `transcripts.jsonl` mới (`B1 + A1 + A2`):**
   * Chạy lại `prepare_sft.py` trên bản `transcripts.jsonl` mới để phân phối lỗi ASR khi train LoRA khớp hoàn toàn với phân phối đầu vào thực tế ở bước inference, và cân nhắc bật tỷ lệ nhỏ `reasoning_stepwise_examples` (hiện bằng `0`) cho nhóm bài toán so sánh số/thời gian.

---

## 4. Tóm tắt Thứ tự Triển khai Khuyến nghị

| Thứ tự | Hạng mục | Cần train lại? | Nhóm ca lỗi khắc phục trực tiếp |
| :---: | :--- | :---: | :--- |
| **1** | **P1.1** IATA normalization + `"tuần sau"` (+7d) search fallback | Không | `public_0213`, `public_0252`, `public_0025`, `public_0287` |
| **2** | **P1.2** Search loop cap (`DEFAULT_EMPTY_READ_LIMIT=2` + lặp kết quả `maps.search_place`) | Không | `public_0061`, `public_0213`, `public_0574` |
| **3** | **P1.3** Cô lập lỗi `local_*` khỏi trần `max_invalid=3` | Không | `public_0482` |
| **4** | **P1.4** Siết `missing_info_guard` với ID từ `history` + điều hướng `contacts` → `phone.recent_calls` | Không | `public_0511`, `public_0528`, `public_0529` |
| **5** | **P1.5** Thông báo dừng sạch trong `check_observed_xref` khi read trả `[]` | Không | `public_0025` (sau khi search thực sự rỗng) |
| **6** | **P1.6** Gợi ý `parking.find(near=...)` | Không | `public_0213`, `public_0252` |
| **7** | **P2.1–P2.4** Sửa `GoldGuardConflict`, `date_reference`, câu điều kiện từ `history`, retrain với ASR mới | Có | Vòng fine-tune SFT + LoRA tiếp theo |
