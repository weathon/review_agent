# PRKV:Page Restruct KV Cache for High Accuracy and  Efficiency LLM Generation

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 2, 6

## Abstract
As the key-value(KV) cache size scales with context length, accessing large KV
cache each step and substantial GPU memory demand challenge us to deploy
LLMs with long contexts.Various sparse attention methods have been proposed
and offloading-based KV retrieval preserves entire KV cache in CPU memory
and dynamically retrieves most relevant KV pairs for each decoding step, which
performs higher quality and effectively reduces GPU memory consumption than
other line works. However, exiting KV retrieval performs page-level to reduce
estimation overhead, which introduces inaccurate KV selection and significant re-
trieval overhead. We propose PRKV, a framework that both-optimizes algorithm
and system for page-level KV retrieval with KV offloading. On the algorithm side,
PRKV introduces hybrid KV selection that combines both static and dynamic KV
selection strategies. On the system side, PRKV employs contiguous memory in-
dexing and batched transfer optimizations to improve retrieval efficiency. Exper-
iments demonstrate that PRKV improve accuracy across various scenarios and
models, delivering up to 6.75× speedup compared to SOTA KV retrieval methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces PRKV, a hybrid key value retrieval framework for long context LLM inference that combines token level reuse with dynamic page level selection.
PRKV periodically reuses top k tokens from the previous step and a local window, then fills the remaining budget with page selection guided by compact page representations.
It restructures the cache so the reused tokens are stored contiguously, which simplifies indexing and reduces memory traffic.
On the systems side, PRKV uses a head major layout and a batched copy pipeline from CPU to GPU to cut transfer overheads.
Experiments on retrieval and reasoning benchmarks show accuracy close to full cache and up to 6.75x end to end speedups over prior offloading methods.

### Strengths
* The paper propose simple and effective hybrid design which combines prior-step token reuse with dynamic page selection to close the recall gap of page-only methods while keeping estimation overhead low.
* HND layout and batched KV transfer are broadly reusable optimizations that materially reduce TPOT and drive the reported speedups.
* Broad and convincing evaluation across multiple models and benchmarks with clear wins in both accuracy and latency.

### Weaknesses
* The paper relocates static tokens to the front and refreshes every T steps (Alg. 1), but the overhead (CPU time and additional PCIe traffic) of reordering KV pages/indices is not reported.
* Results stop at 14B‑parameter general models and 8B for reasoning. It remains unclear how PRKV behaves for larger size model, and models with different attention architecture (e.g. DeepSeek).
* The LongBench evaluation setup seems a bit tricky: The evaluation truncates inputs longer than the model’s context by taking half from the beginning and half from the end, then concatenating. This setup disadvantages the “full cache” baseline and can artificially make PRKV look better than “full attention” because PRKV can still select tokens from the entire document while the baseline never sees the middle. Could you elaborate on the setup?

### Questions
* Additionally, the score in the needle-in-a-haystack seems under-specified, making it hard to interpret results. Could you specify how you measure the success, why use Kimi API as the judge model (what about other models as the judge results)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces PRKV, a framework for page-level KV retrieval that combines static reuse and dynamic selection through a “hybrid KV selection” strategy. The authors also propose system optimizations, namely contiguous memory indexing using the HND layout and batched KV transfers to improve retrieval efficiency. PRKV reports up to 6.75× speedup and near-full-cache accuracy across multiple benchmarks.

### Strengths
1. Clarity and completeness:
The paper is reasonably well written and provides a comprehensive experimental evaluation across several standard long-context benchmarks.
2. Incremental algorithmic variation:
The proposed periodic static token update introduces a minor variation to existing retrieval schemes. While not particularly innovative, it reflects an effort to stabilize long-sequence performance by refreshing the static token set.

### Weaknesses
1. Limited algorithmic novelty:
The proposed method largely concatenates two existing paradigms of KV-cache compression, static dropping and dynamic retrieval, without introducing a fundamentally new principle. As a result, the algorithmic contribution is incremental and lacks clear conceptual advancement.
2. Questionable system design and justification:
The claimed system optimizations are not well substantiated. Although the HND layout may simplify head-wise indexing, it is unclear how this layout integrates with modern GPU attention kernels, which typically assume NHD for coalesced access and contiguous memory traversal (as used in FlashAttention and vLLM). Similarly, the batched KV transfer optimization resembles standard engineering practice rather than a substantive research contribution.

### Questions
1. Please justify how the HND layout efficiently supports GPU attention kernels. Most state-of-the-art implementations (e.g., FlashAttention, FlashInfer) assume NHD layout for coalesced tensor access. Has PRKV been benchmarked with real kernel implementations to confirm compatibility?
2. The proposed “static token selection” is periodically updated. Does this imply that the entire KV cache must still be retained in CPU memory to enable re-selection? If so, how does this affect the memory footprint compared to fully static methods such as SnapKV?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes PRKV, a hybrid sparse-attention framework for long-context LLM decoding with KV offloading. It combines a static token-level set built from an observation window (reused and periodically updated) and a dynamic page-level selection. And it reorganizes the KV layout and uses batched transfers to reduce PCIe traffic and kernel launch overhead. Experimental results report state-of-the-art quality under fixed KV budgets and impressive end-to-end speedups over prior offloading baselines.

### Strengths
- Addresses a practical bottleneck in long-context serving: reducing KV retrieval latency and GPU memory pressure during decoding with CPU offloading, while maintaining quality.  ￼
- Well-motivated hybrid design by reusing prior top-k with local window to boost attention recall
- System optimizations are concrete and codesigned with the proposed algorithm

### Weaknesses
- Some method knobs are central but their sensitivity and generalization across models or tasks are not fully explored   ￼  ￼
- A few writing issues slightly obscure otherwise solid ideas.  ￼

### Questions
Thank you for the submission. I like the paper overall, the hybrid selection insight is compelling, and the system side is thoughtfully engineered.  However, several descriptions are still vague or underspecified. Clarifications that would strengthen the paper:
- The authors show higher recall with a hybrid split and reuse and local window. Could you provide principled guidance for choosing S and the local window size across different models/tasks? Extra sensitivity curves beyond the current S-only ablation will be appreciated to better estimate the impact of the proposed method.  ￼  ￼
- The algorithm periodically recomputes the static set and restructures pages. What is the measured overhead of this reorganization?   ￼
- In some scenarios, PRKV matches or even beats full attention. What is the reason for this?  ￼
- The HND layout speeds up head-wise indexing, and the batched-transfer path reduces kernel launches. Is there any restriction to apply this layout? ￼

### Soundness
2

### Presentation
2

### Contribution
2
