# Hypothesis Backlog — BC Vector Variants

Ranked by directness vs L-pattern. Each variant = 1 experiment.

## H1 — Add `num_turns_norm` to BC (most direct)

**Change:** BC dim 6 → 7. Add `num_turns / max_possible_turns` where max = path_len.

**Rationale:** L-pattern definition itself = ≤1 turn. Encoding turns directly into BC means 2 maze cùng L sẽ có BC component này gần nhau → bị phạt.

**Risk:** Có thể quá targeted, làm BC vector "biết câu trả lời" của metric → metric sẽ tăng nhưng không đảm bảo improve thực sự về structural diversity.

## H2 — Replace `path_norm` (length-only) with `path_complexity` triple

**Change:** Thay 1 dim path_norm bằng 3 dims:
- path_len_norm
- num_turns_norm
- dead_ends_along_path_norm

**Rationale:** Capture full shape của path, không chỉ độ dài. Nhiều thông tin hơn H1.

**Risk:** Tăng dim BC → distance noise hơn nếu các dim mới không informative.

## H3 — Weighted BC (boost structural dims)

**Change:** bc_distance dùng weighted norm: `sqrt(sum(w_i * (a_i - b_i)^2))` với w cho structural cao hơn.
- wall_dens: 0.5
- path_norm: 1.5
- dead_norm: 2.0
- branch_norm: 2.0
- regions_norm: 1.0
- diff: 1.0

**Rationale:** Dead-ends và branches là structural features mạnh; nếu boost weight thì 2 maze giống nhau ở chỗ này (cùng có 0 dead-ends, 0 branches như L-pattern) sẽ bị "đẩy gần nhau" mạnh hơn → bị phạt nặng hơn.

## H4 — Path 2D histogram (where the path goes)

**Change:** Sample 30 points trên BFS path, quantize vào grid 4×4 → 16-dim histogram. BC = 16 dims này.

**Rationale:** Maze cùng L-pattern có path đi qua cùng vùng (top-left và bottom-right) → histogram giống nhau → bị phạt. Capture *vị trí* path đi qua chứ không chỉ shape.

**Risk:** Dim cao (16) có thể làm distance ít discriminative.

## H5 — Drop wall_dens, add `path_efficiency`

**Change:** Bỏ wall_dens (quá coarse), thêm path_efficiency = manhattan(start, end) / actual_path_len. L-pattern có efficiency cao (≈1) vì path gần với đường thẳng-rẽ-thẳng.

**Rationale:** L-pattern thường efficient → low diversity ở dim này. Maze zigzag → low efficiency. Phân biệt rõ.

## H6 — Combine top 2 (after H1-H5 done)

Combine winning ideas. E.g., weighted BC + path complexity triple.

---

## Order of execution

1. **#1 = H1** — bắt đầu với targeted nhất
2. **#2 = H5** — đối lập H1, test xem indirect feature có work không
3. **#3 = H3** — không thêm dim, chỉ reweight, ablation pure
4. **#4 = H2** — extend H1 với multi-dim
5. **#5 = H4** — radical new direction (positional)
6. **#6+** — combine winners
