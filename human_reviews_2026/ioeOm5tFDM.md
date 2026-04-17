# Alleviating Forgetfulness of Linear Attention by Hybrid Sparse Attention and Contextualized Learnable Token Eviction

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 6, 2

## Abstract
Linear-attention models that compress the entire input sequence into a fixed-size recurrent state offer an efficient alternative to Transformers, but their finite memory induces forgetfulness that harms retrieval-intensive tasks.
To mitigate the issue, we explore a series of hybrid models that restore direct access to past tokens. We interleave token mixers of intermediate time and space complexity between linear attention and full attention, including the query-aware native sparse attention, and sparse attention with token eviction. We further propose a novel learnable token eviction module. Combined with sliding-window attention, an end-to-end trainable lightweight CNN-based eviction module aggregates information from both past and future adjacent tokens to adaptively retain a limited set of critical KV-pairs per head, maintaining linear attention's constant time and space complexity. Empirical evaluations on retrieval-intensive benchmarks support the effectiveness of our approaches.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose a form of learned token eviction which uses a CNN to learn the token eviction scores within a sliding window. The proposed method requires training from scratch.

### Strengths
- Sparse Attention is an important topic for long context models as the attention operation is quadratic
- The authors utilize many recent works as a motivation and backbone for their method.

### Weaknesses
- L165: The probing step remains quadratic due to the constant setting of the block size $M$. As this is part of the attention computation as a precursor operation, I don't think you claim that the attention computation is a constant $MK$. 

- How can the method presented in figure 2 ever learn to retain a token in a simple task such as needle-in-a-haystack? For example, datasets like RULER have some tasks where the retrieval target can be any number of key value pairs which is not known until the query is given at the end of the prompt? How could this scheme know what to retain in this situation?

- Figure 3 says the out of window KV's have a capped capacity, but it looks like the second row extends to all previous KV's. Shouldn't this be capped instead of storing all of them?

- I don't quite understand equation 6. If the mask was applied, meaning that there was no computation in the forward pass for $v^\prime_{j,h}$, as this computation was skipped. Is the last term on the RHS supposed to be $\langle v_{j,h}, \frac{\partial \mathcal{L}}{\partial v^\prime_{j,h} \rangle$?

- L301 states that unimportant tokens will receive negative gradients as a matter of fact. How can you be so sure of this?

- Table 2 is very hard to make sense of with all numbers above 70% being bolded.  

- This is supposed to be for linear (and therefore efficient) attention, but there is no latency comparison between models. 

---

Overall, I find the method and the results to be underwhelming at best. The results seem mixed and inconclusive between methods which makes it hard to pin down the exact strengths of the method. As there is no latency comparison, readers have no idea of the overall efficiency compared to the other baseline methods. It is also hard to see the exact benefit of the learned token eviction when it is not compared to a simple baseline such as aggregated attention scores within the window. 

---
 
## Minor

L107: stemm --> stemming?

L130: "tokens can be self-evidently crucial" --> I find "self-evidently" to be problematic here. If they were self evidently crucial, then they would be easy to identify and retain. But in the case of a passkey, it is not self evident that they are crucial for the task without perfect knowledge of the query which will be asked. I think these words should be rephrased.

### Questions
- RULER was only considered for NIAH single tasks up to 4K. Can you extend this to the more complicated tasks with longer contexts? The model that uses NSA should be capable of handling this at least, right?

- Could it be possible that getting ride of the LTE module and only tracking token scores could lead to a similar or better eviction policy?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes the Learned Token Eviction (LTE) algorithm. The authors combine LTE with linear attention, specifically sliding window attention, and refer to it as laLTE.

### Strengths
- The method is simple to implement.

### Weaknesses
## Lacks of novelties
The method is somewhat akin to combining existing elements.

## Lacks of efficiency analysis

Even if the method is claimed as linear complexity, the linear complexity does not always mean faster and efficient than flash attention. The author lacks a critical analysis of efficiency in real-world hardware. Any latency seconds were not reported.

Especially about CNN, the latency analysis is really crucial, since the small size of Conv operation is known to be slower than normal vector-matrix and matrix-matrix operations. I want to hear the answer and a detailed analysis of the following things:
 - Really need CNN?
 - How much does CNN slow down in wall clock latency? Please compare it to HW efficient alternatives (Mamba, LightningAttention2)

## Eviction is still a critical problem in a multi-turn request scenario, especially for tool calling

Since the KV cache is more and more critical in an agentic AI scenario, we cannot drop the KV cache without a precise study.

 - Is there any analysis about tool callings?
 - What is the agentic LLM performance?

## Where is the training curve?
 - I cannot be sure this model is sufficiently trained or scalable

## Lacks of some strong performance baselines:
- Mamba2
- Lightening attention https://github.com/MiniMax-AI/MiniMax-M1

I think we need to include such alternatives to build a competitive method for deployable methods.

### Questions
## Typo 
- line 259: se -> sequence (?)

## Questions
- How does the inverse rotation affect the precision errors? on fp8? fp16? fp4?
- Table 2, you must put the latency.

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper addresses the recall gap of linear-attention and recurrent LMs by re-introducing targeted access to past tokens without abandoning efficiency. It studies two hybrid designs: (1) laNSA: interleaves linear attention with Native Sparse Attention (NSA) that combines query-aware block probing, a compressed branch, and SWA-style gating. This improves retrieval but keeps an O(N) KV cache. (2) laLTE: interleaves linear attention with Learnable Token Eviction (LTE). A tiny per-token, per-head 1D-CNN predicts whether to retain a KV as it leaves a large SWA window; combined with an attention sink and SWA, this aims for O(1) time and space per step while preserving long-range evidence. Experiments at roughly 0.4B and 1.4B parameters trained on 10B/30B tokens (FineWeb-Edu) evaluate short-context language tasks and long-context retrieval (RULER S-NIAH, EVAPORATE). Both laLTE and laNSA outperform strong linear baselines; laNSA is strongest among linear-time models on EVAPORATE, while laLTE is the best constant-space option and approaches hybrid full-attention variants in some settings.

### Strengths
### Novelty: 
This work proposes two complementary mixers interleaved with linear attention—NSA for query-aware sparse access over the full past and LTE for learned keep/evict under a strict cache budget; introduces per-token, per-head retention via a tiny 1D-CNN with SWA-enabled look-ahead and an attention sink to maintain near-constant KV memory; provides deployment-minded decoding/KV design (two-segment cache, lazy batched scoring) and frames a clear accuracy–efficiency Pareto frontier (laLTE for constant-space, laNSA for higher accuracy under linear time).

### Differentiation from prior works: 
This work moves beyond fixed windows and global/uniform/time-decay heuristics by making head-aware, context-conditioned retention decisions; regains direct token-level access to salient long-range evidence without reverting to O(N²) attention, contrasting with state-space/recurrence approaches that compress history into fixed states; offers a stronger NSA baseline (query-aware probing + compressive branch + SWA gating) that sharpens comparisons.

### Weaknesses
### Scale and generality: 
Results are limited to 0.4B/1.4B. It is unclear whether the trends hold for larger, modern LLM families (e.g., Qwen2.5/3, DeepSeek) or for multilingual/code models.
### Benchmark breadth: 
The evaluation focuses on long-context retrieval. Broader benchmarks commonly used today (e.g., instruction following, math, and code such as AlpacaEval, GSM8K, HumanEval) are absent, making it hard to gauge side effects beyond retrieval.

### Efficiency reporting: 
The paper argues constant time/space for laLTE by design, but it does not provide systematic, measured wall-clock throughput and GPU memory usage across mixers (GDN, +SWA, laLTE, laNSA, +full-attention).

### NSA dependency: 
laNSA still requires O(N) KV. The trade-off versus laLTE is discussed conceptually; clearer guidance on when laNSA is preferable in practice would help.

### Questions
1. Can you report measured GPU memory (GB), effective KV size, tokens/sec, and latency for GDN, GDN+SWA, laLTE, laNSA, and a GDN+full-attention hybrid at 4K and at least 8K contexts on the same hardware?

2. Since laLTE/laNSA are trained modules, could you also include an inference-only comparison on the same GPU (with matched context length and batch size) against training-free efficient attention approaches such as HiP [1]? Reporting retrieval accuracy (e.g., RULER/EVAPORATE), throughput (tokens/s), and peak memory would help clarify whether the additional training cost yields meaningful gains over training-free sparse attention methods.

--- 
[1] Lee et al. A Training-free Sub-quadratic Cost Transformer Model Serving Framework With Hierarchically Pruned Attention. ICLR 2025.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper targets the “forgetfulness” of linear-attention models by interleaving Gated DeltaNet layers with stronger token mixers. Two variants are proposed: (1) laLTE, which introduces a learnable token eviction (LTE) module that scores each KV pair per head using a tiny 3-layer 1D CNN with a short receptive field and then retains only a capped number of out-of-window tokens. (2) laNSA, which swaps in Native Sparse Attention (NSA) layers that perform query-aware block, offering more direct access to the past but requiring O(N) KV memory.
The authors position these mixers on a complexity–access hierarchy. Empirically, on EVAPORATE and RULER, the hybrids often outperform pure GDN/GDN+SWA, while full-attention interleaves remain the strongest but are also the most costly.

### Strengths
1. Per-head, per-token scoring from short local context with grouped 1D convs is simple, parallel, and adds ~1% params. The design is a constant budget and predictable latency.
2. Results are reported on both synthetic (S-NIAH) and realistic (EVAPORATE) retrieval benchmarks, showing consistent gains over pure GDN/GDN+SWA in many settings.

### Weaknesses
1. The novelty is limited. The laNSA component is adopted from prior NSA work, and the overall recipe seems an alternation of existing hybrid attention rather than a fundamentally new mechanism. The idea of LTE also sits close to the broader family of token-eviction methods, making the novelty feel incremental.
2. Evidence does not decisively beat the common practice. The improvements on EVAPORATE are modest averages (e.g., laLTE/laNSA only several points over GDN/GDN+SWA), while interleaving full attention (GDN+Attn.) remains stronger in many settings.
3. No end-to-end latency measurements (prefill & decode) under the claimed constant budgets, nor e2e latency comparisons against strong kernels (e.g., Flash-/Flex-Attention baselines) make it hard to assess the practical gains of LTE beyond proxy complexity.  
4. The evaluated models are relatively small (0.4B/1.4B on 10B/30B tokens FineWeb-Edu), which limits the strength of conclusions about scalability to modern LLMs. 
5. Ablation study on LTE is insufficient. The paper motivates head-wise independence and a short receptive field, but there is limited analysis on (i) sensitivity to the cap b, window w, and receptive field R; (ii) alternatives to CNN scoring (MLP/other efficient attention predictors).

### Questions
1. Could authors report e2e prefill and decode latency vs. GDN, GDN+SWA, and GDN+Attn., all using the same optimized kernels?
2. How does laLTE compare to other recent learnable-eviction or head-aware KV-budgeting methods under a matched training budget?

### Soundness
3

### Presentation
3

### Contribution
2
