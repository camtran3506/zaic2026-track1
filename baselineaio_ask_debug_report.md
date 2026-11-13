**Debug `sys.ask_missing_info`: `2ffdfa4` → `1475bfb` (`baselineaio`)**

Phạm vi là đúng diff với commit cha, không gồm `1b21b5b` hoặc hỗ trợ synthetic
được thêm sau đó. Không thay đổi adapter, tham số inference hay dữ liệu train.

**Kết luận đã kiểm chứng**

- Không có guard kiểm tra một yêu cầu hỏi thêm có thực sự thiếu thông tin trong
  transcript, context, history, ITN, kết quả read hoặc default/const của schema.
  Lỗ hổng này đã tồn tại ở commit cha; nó không phải guard mới của baselineaio.
- Middleware chỉ kiểm tra target `tool_name` của ask là một application tool được
  phép. Validator kiểm tra schema của `sys.ask_missing_info`, không đối chiếu
  `params` với schema của application tool được hỏi.
- Với `tool_name="alarm.create"`, cả `params=["time"]` và
  `params=["not_a_real_argument"]` đều qua precheck của snapshot baselineaio,
  advisory schema inspection và validator công khai của public_test.
- `zaic_tools/apps/sys.py::Sys.ask_missing_info` chỉ trả
  `{"ok": true, "waiting_for_user": true}`, không kiểm tra thông tin đã biết.
  `agent.py` dừng ngay sau khi ask được thực thi thành công.
- SFT vẫn phạt dự đoán sai bằng cross entropy so với gold. Không có loss riêng
  cho hỏi thừa, không có RL reward hay bước tối ưu rollout để phạt hỏi thừa.
  Điểm state/tool khi evaluation là cách chấm kết quả; không phải guard tự sửa
  hành động hoặc phần thưởng được đưa vào backward của SFT.

**Các thay đổi mới đáng điều tra, theo mức ưu tiên**

| Thay đổi so với `2ffdfa4` | Cơ chế có thể liên quan | Mức kết luận |
| --- | --- | --- |
| `finetune.py`: `add_arguments(p, enabled=True)` → `enabled=False` | Launcher mặc định đổi từ 2 rename variants + distractor probability 0.5 sang 0 + 0. Mất các ví dụ buộc đọc schema độc lập với tên tool. | Thay đổi đã xác nhận; tác động đến chất lượng cần A/B. Nếu gọi `prepare_sft.py` trực tiếp thì mặc định augmentation của script này vốn đã tắt ở commit cha. |
| Bỏ date/datetime helper demonstrations theo mặc định, thay bằng `user_context.date_reference` | Model phải suy ra timestamp trực tiếp nhiều hơn, giảm các ví dụ trung gian biến ngày/giờ thành giá trị có thể dùng. | Nghi ngờ phù hợp với hỏi lại time/date, chưa chứng minh nguyên nhân. |
| Step-wise → full-trajectory; thêm weighted loss và sparse output-head computation | Đổi cách gom mẫu, số optimizer steps và phân bổ loss. | Audit masking/token weights bên dưới không phát hiện mất supervision của call thực thi. Chưa kiểm tra ngang bằng forward/backward Unsloth trên GPU. |
| Bỏ validation/score callback và best-checkpoint selection; xuất checkpoint cuối | Không còn đo chất lượng rollout để chọn checkpoint tránh hỏi thừa. | Thay đổi đã xác nhận; không chứng minh riêng checkpoint cuối là nguyên nhân. |
| Inference bật `enforce_observed_xref=True` và `empty_read_limit=2` | Chặn call có ID chưa được ghi nhận, hoặc chặn đọc tiếp sau hai kết quả rỗng/thất bại; feedback sau ngưỡng read đề cập ask_missing_info. | Có thể thúc đẩy fallback trong các rollout bị từ chối; không giải thích trực tiếp ask xảy ra trước mọi rejection. |

Các thay đổi khác: mặc định epochs 3 → 2, learning rate 2e-4 → 1e-4,
mix một transcript/mẫu → dual hai nguồn, bổ sung search recovery, thay đổi ASR
precision/decoding và incremental prefix-cache chain/schema scheduling.
Các biến này không được thay đồng thời trong một thí nghiệm đối chứng, nên không
thể kết luận dual là nguyên nhân duy nhất. Cache mới cũng chưa được A/B bằng
model thật trên GPU trong lần debug này.

**Bất đối xứng guard mới giữa inference và SFT**

Tại `1475bfb`, `agent.py` tạo `LoopMonitor(empty_read_limit=2)` và truyền
`spec`/`observed_ids` cho `attempt_env_call`. Trong khi đó `sft.py::replay_sample`
vẫn tạo `LoopMonitor()` (empty_read_limit=0) và không truyền hai tham số x-ref.
Gold replay không tạo các trạng thái bị guard mới từ chối để model học cách
đọc lại, tìm ID và tiếp tục thực thi. Search recovery chỉ dạy trường hợp query
trả rỗng, không tương đương guard/schema recovery.

Guard là middleware inference, không phải một hình phạt được model tối ưu trong
SFT. Vì vậy không thể nói model đã "học sợ guard" chỉ từ việc bật guard mới.
Nó có thể chọn ask sau feedback khi chạy thực tế; còn ask ngay từ đầu phải được
điều tra thêm ở dữ liệu, việc học schema, checkpoint và runtime.

**Audit masking bằng đúng snapshot và tokenizer thật**

Đã giải nén file thường của `git archive 1475bfb code` vào thư mục tạm riêng,
import chính mã của snapshot đó và dùng tokenizer local Qwen3.5. Chạy trên CPU,
không nạp trọng số model và không sử dụng GPU.

Dữ liệu được kiểm tra: `code/result/run_sft_dual/sft/train.jsonl`.

| Chỉ số | Kết quả |
| --- | ---: |
| Dòng SFT | 3.078 |
| Lượt assistant được supervise | 10.072 |
| Lượt ask được supervise | 428 |
| Lỗi thiếu/thừa lượt được supervise theo assistant_train_mask | 0 |
| Token target trước weighting | 341.046 |
| Tổng trọng số token target sau weighting | 330.669,2 |
| Token target của ask | 19.181 |
| Tỷ trọng token ask trước weighting | 5,624% |
| Tỷ trọng token ask sau weighting | 5,801% |

Đây là tỷ trọng token/trọng số, không phải tỷ trọng giá trị cross entropy thực
tế. Fuzzy weighting tăng tương đối phần ask rất nhẹ trong dữ liệu này; chưa có
bằng chứng nó là nguyên nhân chính của mức ask 75,4% khi inference. Audit này
không thay thế kiểm tra numerical equivalence của sparse Unsloth loss.

**Đối chiếu run thực tế**

- Gold: 217/1.545 mẫu có ask (14,0%).
- SFT dual: 428/3.078 trajectory có target cuối là ask (13,9%).
- `submission1.jsonl`: 478/634 mẫu có ask (75,4%); 369 mẫu chỉ gọi ask.
- 393/478 mẫu ask có `invalid_proposals=0`; 85 mẫu có ít nhất một rejection.
  Chưa có trace từng rejection để quy 85 mẫu đó riêng cho xref_guard hoặc read cap.
- 219/478 ask nhắm đến tool không có trong catalog train. Catalog coverage này
  vốn là vấn đề tổng quát hóa; điểm mới trong diff là launcher tắt augmentation.
- Ví dụ `public_0015`: transcript có "đặt báo thức sáu giờ sáng mai", model vẫn
  hỏi `alarm.create` với `params=["time"]`, không có rejection trước đó.
- Manifest run dual xác nhận rename_variants=0, distractor_probability=0.0,
  variants_kept=0 và distractor_variants=0.

Các câu chỉ dẫn "không đoán tham số" và "thiếu thông tin thì hỏi lại" đã có
trước baselineaio. Chúng không phải thay đổi mới đủ để giải thích regression.

**Guard hỏi thừa hiện còn thiếu những gì**

1. `params` phải là tham số có thật của tool đích; cần xét required và x-one-of,
   thay vì chỉ kiểm tra đây là một list string.
2. Đối chiếu với giá trị đã được xác định chắc chắn từ transcript/ITN, context,
   history, read results, date/time helper hoặc schema const/default hợp lệ.
3. Nếu đã có giá trị, phản hồi cho model tiếp tục hành động thay vì thực thi ask
   rồi kết thúc. Với x-ref chưa có, có thể hướng về read/resolve phù hợp.
4. Chỉ chặn khi có bằng chứng rõ; enum có nhiều lựa chọn hay nhiều entity chưa
   phân giải không tự động có nghĩa thông tin đã đầy đủ.

Đây là đề xuất, chưa được triển khai trong lần debug này. Guard chỉ là chốt
kiểm tra; để model tự chọn đúng cần cả ví dụ sửa hỏi thừa hoặc preference pairs.

**Thứ tự kiểm chứng tiếp theo**

- A/B inference trên cùng một nhóm mẫu lỗi, cùng adapter và transcript, tắt các
cache để loại trừ ảnh hưởng của fast path mới. Không cần train lại cho bước này.
- Kiểm tra riêng date helpers đối với nhóm hỏi time/date.
- Nếu vẫn regression, so sánh sparse weighted loss với loss gốc trên cùng batch
  và trọng số, rồi mới làm thí nghiệm training thay full-trajectory/step-wise.

Không kết luận chắc nguyên nhân nhân quả khi chưa chạy các A/B này.
