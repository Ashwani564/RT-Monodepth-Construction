# LLM Supervisor Evaluation (llama3.1:8b)

- Depth log: `depth_logs/depth_log_20260427_015450.csv` (1031 rows, 137 valid detections)
- Distance log: `depth_logs/distance_log_20260427_015450.csv` (19 rows)
- Total questions: **30**
- Correct: **10** (33.3%)
- Mean latency / question: 3.949s

## Per-category accuracy

| Category | Correct | Total | Accuracy |
| --- | ---: | ---: | ---: |
| counts | 1 | 5 | 20.0% |
| aggregate-depth | 1 | 4 | 25.0% |
| aggregate-confidence | 1 | 1 | 100.0% |
| lookup | 4 | 5 | 80.0% |
| proximity-zone | 0 | 6 | 0.0% |
| reasoning | 2 | 6 | 33.3% |
| safety-decision | 1 | 3 | 33.3% |

## Per-question results

| QID | Category | Ground truth | Predicted | Correct | Latency (s) |
| --- | --- | --- | --- | :-: | ---: |
| Q01 | counts | 96 | 67 | ✗ | 1.389 |
| Q02 | counts | 30 | 34 | ✗ | 1.427 |
| Q03 | counts | 11 | 17 | ✗ | 1.549 |
| Q04 | counts | 3 | 3 | ✓ | 2.867 |
| Q05 | counts | 19 | 23 | ✗ | 0.929 |
| Q06 | aggregate-depth | 2.73 | 2.053 | ✗ | 6.154 |
| Q07 | aggregate-depth | 7.47 | 6.3 | ✗ | 2.223 |
| Q08 | aggregate-depth | 1.24 | 1.24 | ✓ | 3.591 |
| Q09 | aggregate-depth | 2.63 | 2.053 | ✗ | 6.17 |
| Q10 | aggregate-confidence | 0.457 | 0.486 | ✓ | 6.177 |
| Q11 | lookup | 2.05 | 2.05 | ✓ | 2.342 |
| Q12 | lookup | 193 | 193 | ✓ | 2.917 |
| Q13 | lookup | 124 | 124 | ✓ | 3.332 |
| Q14 | lookup | 0.016 | 0.016 | ✓ | 2.547 |
| Q15 | lookup | 2.833 | 2.625 | ✗ | 3.147 |
| Q16 | proximity-zone | 11 | 0 | ✗ | 1.864 |
| Q17 | proximity-zone | 8 | 2833 | ✗ | 6.251 |
| Q18 | proximity-zone | 0 | 1183 | ✗ | 6.19 |
| Q19 | proximity-zone | 0.83 | 0.501 | ✗ | 2.402 |
| Q20 | proximity-zone | 0.793 | 0.474 | ✗ | 5.636 |
| Q21 | proximity-zone | 1.094 | 0.021 | ✗ | 6.172 |
| Q22 | reasoning | Vehicle | vehicle | ✓ | 6.413 |
| Q23 | reasoning | Vehicle | vehicle | ✓ | 5.809 |
| Q24 | reasoning | Machinery | person | ✗ | 6.041 |
| Q25 | reasoning | 118 | 43 | ✗ | 3.02 |
| Q26 | reasoning | 1 | 157 | ✗ | 6.315 |
| Q27 | reasoning | 2 | 1765 | ✗ | 6.172 |
| Q28 | safety-decision | yes | no | ✗ | 1.97 |
| Q29 | safety-decision | 1 | 2 | ✗ | 4.758 |
| Q30 | safety-decision | yes | yes | ✓ | 2.684 |
