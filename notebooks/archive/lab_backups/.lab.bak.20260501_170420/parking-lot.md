# Parking Lot

## Ideas chưa thử

- **Archive injection**: sau mỗi gen, replace N genome fitness thấp nhất trong NEAT pop bằng genome từ high ht_rate archive cells. Archive genomes thật sự sinh sản.
- **Pareto multi-objective**: giữ genome Pareto-optimal trên (solvability, ht_rate) — không genome nào bị dominate cả 2 chiều
- **Adaptive ht weight**: khi sol > 0.8, tăng ht weight tự động (sol đã đủ tốt, push ht)
- **Multiplicative sol×ht**: dùng geometric mean thay vì weighted sum — penalize imbalance nặng
- **Separate sol floor**: chỉ count ht vào archive cell khi sol ≥ 0.7 threshold
