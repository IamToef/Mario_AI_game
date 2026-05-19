# Mario Deep Q-Network (DQN) Agent

![Graphical Abstract](C:\Users\phong.tran\.gemini\antigravity\brain\07bbd3a6-17f1-495a-aba0-b7e802df6832\graphical_abstract_1779181167148.png)

Dự án này cung cấp một hệ thống học tăng cường (reinforcement learning) có khả năng tự động chơi trò chơi Super Mario Bros. Hệ thống sử dụng kiến trúc Deep Q-Network được tối ưu hóa đặc biệt cho quá trình thực thi trên CPU, giúp các nhà phát triển tiến hành huấn luyện mà không yêu cầu phần cứng GPU chuyên dụng. Mã nguồn được cấu trúc theo hướng mô-đun hóa nhằm tạo điều kiện thuận lợi cho việc nghiên cứu và kiểm thử thuật toán.

## Kiến trúc Hệ thống

![System Architecture](C:\Users\phong.tran\.gemini\antigravity\brain\07bbd3a6-17f1-495a-aba0-b7e802df6832\system_architecture_1779181189976.png)

Kiến trúc phần mềm được chia thành bốn thành phần chính hoạt động độc lập và liên kết chặt chẽ với nhau. Lớp tiền xử lý môi trường (wrappers) làm nhiệm vụ tối ưu hóa hình ảnh đầu vào bằng cách bỏ qua các khung hình thừa, chuyển đổi màu sắc sang ảnh xám, thu nhỏ độ phân giải xuống 84x84 pixel và gộp bốn khung hình liên tiếp để cung cấp thông tin về chuyển động. Bộ nhớ kinh nghiệm (replay memory) lưu trữ các tương tác quá khứ và lấy mẫu ngẫu nhiên nhằm giảm thiểu sự phụ thuộc tuyến tính giữa các dữ liệu huấn luyện.

Mô hình mạng nơ-ron tích chập (CNN) đóng vai trò trích xuất đặc trưng không gian từ hình ảnh trước khi sử dụng các lớp kết nối đầy đủ để đánh giá giá trị Q cho từng hành động. Cuối cùng, tác tử (agent) điều phối chiến lược khám phá epsilon-greedy và thực hiện đồng bộ hóa giữa mạng trực tuyến (online network) và mạng mục tiêu (target network) để duy trì sự ổn định trong suốt quá trình hội tụ của thuật toán.

## Yêu cầu Cài đặt

Môi trường phát triển yêu cầu sử dụng phiên bản Python từ 3.12 trở lên. Để hệ thống hoạt động chính xác, bạn cần cài đặt bộ thư viện Gymnasium kết hợp với giả lập NES và Pytorch phiên bản CPU.

Danh sách các thư viện cần thiết đã được tổng hợp đầy đủ trong tập tin cấu hình. Bạn có thể tự động cài đặt toàn bộ bằng cách sử dụng trình quản lý gói pip thông qua lệnh sau:

```bash
pip install -r requirements.txt
```

## Hướng dẫn Sử dụng

Để khởi động quá trình huấn luyện mô hình, bạn cần thực thi tập tin điều khiển chính. Tập tin này sẽ thiết lập môi trường giả lập, khởi tạo tác tử và bắt đầu vòng lặp học tập. Quá trình này sẽ tự động ghi nhận các chỉ số hiệu suất vào một tập tin nhật ký định dạng CSV, đồng thời lưu trữ trọng số mạng nơ-ron định kỳ vào thư mục lưu trữ cục bộ.

```bash
python main.py
```

Khi bạn muốn kiểm tra khả năng chơi thực tế của mô hình đã được huấn luyện, hãy chạy tập tin đánh giá trực quan. Kịch bản này tự động tìm kiếm và tải bộ trọng số tốt nhất từ thư mục lưu trữ, loại bỏ cơ chế di chuyển ngẫu nhiên và hiển thị trực tiếp quá trình điều khiển nhân vật Mario trên màn hình.

```bash
python play.py
```
