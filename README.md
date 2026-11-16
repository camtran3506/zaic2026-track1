# LifeMate — Offline Vietnamese Voice-to-Tools Agent

**LifeMate** là hệ thống chuyển giọng nói tiếng Việt thành hành động sử dụng công cụ, được phát triển cho **Zalo AI Challenge 2026 – LifeMate track**. Hệ thống chạy offline trong một tiến trình: mô hình, tokenizer và các thành phần runtime được nạp từ thư mục cục bộ, không cần kết nối mạng khi chạy Docker (`--network none`).

## Điểm nổi bật

- **Nhận dạng tiếng nói (ASR):** GIPformer 1.5-RNNT, fine-tune theo miền dữ liệu và chạy CPU qua ONNX/sherpa-onnx.
- **Hiểu yêu cầu và gọi công cụ:** Qwen3.5-4B fine-tune, chạy GPU ở BF16; agent xử lý vòng lặp gọi công cụ với các giới hạn và kiểm tra an toàn.
- **Ưu tiên chạy offline:** checkpoint và cấu hình inference được đóng gói trong `code/saved_models/`.
- **Tái lập và kiểm chứng:** seed cố định, bộ kiểm thử, cùng công cụ đối chiếu kết quả giữa các entrypoint inference.

## Bố cục repository

```text
.
├── data/                 # Dữ liệu dùng cho huấn luyện và đánh giá
└── code/                 # Mã nguồn inference, huấn luyện và Docker
    ├── predict.py        # Entrypoint inference
    ├── predict.sh        # Chạy batch inference
    ├── Dockerfile        # Môi trường inference offline
    ├── lifemate/         # ASR, LLM, agent và các thành phần hỗ trợ
    ├── saved_models/     # Trọng số inference
    ├── training_code/    # Pipeline huấn luyện
    └── tests/            # Automated tests
```

## Chạy inference

Trong môi trường BTC/Docker đã chuẩn bị dữ liệu ở `/data` và mã nguồn tại `/code`:

```bash
bash /code/predict.sh
```

Lệnh trên chạy batch inference và ghi kết quả vào `/result/submission.jsonl`. Để xem cách chạy local, cấu hình chi tiết, kiến trúc và kiểm chứng parity, xem [tài liệu solution](code/README.md). Hướng dẫn pipeline huấn luyện nằm tại [`code/training_code/README.md`](code/training_code/README.md).

## Tài liệu

- [Solution overview và hướng dẫn inference](code/README.md)
- [Pipeline huấn luyện](code/training_code/README.md)

---

*Zalo AI Challenge 2026 — LifeMate track*