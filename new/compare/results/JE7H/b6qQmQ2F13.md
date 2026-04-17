---
job_id: 440bccbb-72ab-4fda-8535-50a77ed15a09
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: b6qQmQ2F13.pdf
paper: Not All Bits Are Equal: Scale-Dependent Memory Optimization Strategies for Reasoning Models
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper studies memory–accuracy tradeoffs for quantized reasoning LLMs, including KV-cache compression and test-time scaling, which clearly falls under efficient inference, representation learning for language, and scaling laws, all within ICLR’s scope.

## Minimum Quality
Pass ✅.  
All required sections are present (Abstract, Introduction, Related Work in Section A, Methodology/Setup, Experiments, Results, Conclusion). The paper is in English, technically coherent, with substantial empirical work and no obvious fatal methodological flaws. Weaknesses are about positioning and depth, not baseline scientific validity.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
No hidden prompts or attempts to manipulate automated reviewing are present in the main paper content.

---

# Expected Review Outcome:

## Summary

The paper empirically studies memory–accuracy trade-offs for reasoning-oriented LLMs, focusing on how to allocate a fixed memory budget between model weights and KV cache. Using Qwen3 (0.6B–32B), DeepSeek-R1-Distill, and OpenReasoning-Nemotron across math, code, and knowledge-intensive benchmarks, the authors sweep over weight precisions, token budgets, parallel sampling group sizes, and KV cache compression strategies (eviction and quantization). They report several scale-dependent guidelines, most notably that for “effectively small” models (below roughly an 8-bit 4B in weight memory), extra memory should favor higher-precision weights, while for larger models, memory is better spent on longer generations, parallel sampling, and KV cache.

## Strengths

1. **Clear, practically relevant question and scope.**  
   The work asks a concrete deployment-oriented question: given a fixed memory budget, how should practitioners trade off model size, weight precision, KV cache length/compression, and parallel sampling for reasoning tasks. This is directly relevant to many real-world scenarios where VRAM, not FLOPs, is the bottleneck.

2. **Large, systematic experimental sweep with explicit axes.**  
   The experimental design in Section 3 is well-structured around five factors: model parameters \(N\), weight precision \(P_W\), token budget \(T\), group size \(G\), and KV policy \(\pi_{\mathrm{kv}}\). The authors explore about 1,700 configurations, covering 0.6B–32B Qwen3 models under 4/8/16-bit GPTQ, token budgets 2k–30k, \(G\in\{1,3,4,6,8,12,16\}\), and several KV-compression strategies (R-KV, StreamingLLM, HQQ). This breadth gives the paper some authority, even though not every combination is exhaustively covered.

3. **Well-articulated scale-dependent findings, backed by figures.**  
   The central “scale threshold” finding is supported by several visualizations:
   - **Figure 1** (AIME25, Qwen3) and **Figure 2** show that for total memory below roughly 8–10 GB, Pareto-optimal points increase effective model size rather than token budget, whereas at higher memory budgets the Pareto frontier advances primarily through longer generations. The decomposition into token budget (Fig. 2a) and effective model size (Fig. 2b) makes this pattern easy to see.
   - **Figures 3 and 4** show similar analyses for LiveCodeBench and GPQA-Diamond, respectively, highlighting task-dependent precision trade-offs (8/16-bit for math/code vs 4-bit for knowledge tasks).
   - **Figure 5** and **Figure 6** convincingly show when parallel scaling improves the global memory–accuracy Pareto frontier (only for sufficiently large effective sizes).

4. **Useful concrete guidelines and negative results.**  
   The “Findings 1–6” summarized in Sections 4, 5, and C.1 are specific and operational:
   - For math and code, 4-bit weights are *not* memory-optimal; 8-/16-bit dominates (Finding 2), in contrast to Dettmers & Zettlemoyer (2023).
   - Parallel scaling with majority voting only helps memory–accuracy trade-offs for effectively large models (Finding 3).
   - KV cache compression (eviction or quantization) consistently improves memory–accuracy frontiers, showing weight-only quantization is insufficient (Finding 4).
   - Eviction beats KV quantization for effectively small models, but quantization becomes competitive at larger scales (Finding 5).
   - For latency and throughput, 4-bit also never lies on the Pareto frontier (Finding 6, **Figure 10** and **Figure 11**), again countering common deployment wisdom.  
   These are precisely the type of “rule-of-thumb” results practitioners can act on.

5. **Explicit memory modeling with clear equations and tables.**  
   Section B provides explicit formulas for weight memory and KV cache memory:
   - The weight memory equation  
     \[
     M_{\rm weights}\approx\left(N_{\rm quant}\cdot\frac{P_{W}}{8}+\frac{N_{\rm quant}}{g_{W}}\cdot\frac{P_{S}+P_{Z}}{8}\right)+\left(N_{\rm unquant}\cdot\frac{P_{\rm native}}{8}\right)
     \]
     clearly separates quantized and unquantized parameters and includes scale overhead.  
   - The KV cache equations for different strategies (full, eviction, quantization) in Section B are clear and consistent with the architecture specs in **Table 2**.  
   **Table 1** and **Table 2** give concrete per-model memory footprints and per-token KV size, which grounds the empirical plots and helps readers reproduce or adapt the analysis.

6. **Careful use of figures to dissect KV strategies.**  
   The KV compression section is particularly well presented:
   - **Figure 8** shows that both eviction and quantization push the global Pareto frontier beyond full KV cache, especially under 10 GB.  
   - **Figure 9** (multi-panel per-model curves), together with **Figures 17–18**, distinguishes the vertical “constant memory, increasing accuracy” behavior of eviction from the left-shift “reduced per-token cost at reduced accuracy” behavior of quantization. This visualization is effective in illustrating why eviction is superior at small scales: small models are sensitive to precision loss but benefit strongly from capped KV memory.

7. **Cross-model-family generalization checks.**  
   Section 4 and Appendix C.6 repeat the core scaling experiments on DeepSeek-R1-Distill and OpenReasoning-Nemotron. **Figure 6** and **Figure 16** show that the same “small models: invest in weights; large models: invest in KV cache” pattern appears in these families as well, increasing confidence that the conclusions are not idiosyncratic to Qwen3.

8. **Positioning relative to some recent work on quantization and reasoning.**  
   The related work (Section A) cites several directly relevant studies, including Dettmers & Zettlemoyer (4-bit scaling laws), Li et al. (2025a) and Liu et al. (2025b) on quantization and reasoning, and KV compression work such as Lexico, SnapKV, Scissorhands, Kivi, KVQuant, etc. The paper clearly explains how its contribution is complementary: previous works focus mainly on accuracy degradation at fixed context/max length, whereas this paper frames the problem in terms of memory-optimal allocation across multiple inference axes.

## Weaknesses

1. **The “8-bit 4B effective size” threshold is largely heuristic and can be overinterpreted.**  
   The paper repeatedly emphasizes a critical threshold at “effective size below 8-bit 4B” (≈4.2 GB of weights) at which strategy flips (e.g., Finding 1, Findings 3 & 5). However:
   - There is no attempt to explain *why* this particular threshold should be special beyond the empirical observation on AIME25 with Qwen3, and later qualitative mentions that the threshold “shifts” for MATH500 (Figure 14) and for other families (Appendix C.6).  
   - The figures show a more gradual transition: in **Figure 2**, the change from “weights-first” to “tokens-first” is smooth as total memory increases, not clearly pinned to a single model size/precision combination. On MATH500 (**Figure 14**) and GPQA (**Figure 4**), the frontier behavior appears different again.  
   - DeepSeek and Nemotron experiments in C.6 similarly describe threshold-like behavior but with shifted regimes (e.g., 7B vs 14B).  
   This makes the repeated use of “8-bit 4B threshold” a bit misleading; the underlying phenomenon is real, but the specific numerical cut-off is not universal. The paper should tone down the universality and either give a more principled rationale (e.g., based on bits-per-parameter vs bits-per-token trade-offs) or consistently present it as a rough heuristic that varies with architecture and task.

2. **Limited theoretical or analytic understanding of the trade-off, despite clear equations.**  
   The memory equations in Section B and the latency/throughput observations in C.1 set up a structure that *could* support some simple analytic insights (e.g., expressing a threshold where marginal accuracy gain per additional KV byte falls below that per additional weight byte). Instead, conclusions are purely empirical.  
   For instance, one could attempt to estimate a function \(A(N, P_W, T)\) and analyse marginal gains \(\partial A/\partial M_{\mathrm{weights}}\) vs \(\partial A/\partial M_{\mathrm{kv}}\) and at least articulate a functional dependence. Without any such abstraction, the paper is more of a thorough benchmarking study than a scaling-law analysis, and it becomes less clear how to extrapolate beyond the tested regime (e.g., larger than 32B, different architectures, different PRM designs).

3. **KV compression comparison is somewhat narrow and misses a deeper analysis of *when* each method helps.**  
   While **Figure 8** and **Figure 9** nicely show that both quantization and eviction identify Pareto improvements, the analysis is fairly coarse:
   - Only one quantization backend (HQQ) is systematically used for KV (KVQuant, Kivi, etc., are only cited). There is no sensitivity analysis for group size \(g_{\rm kv}\), residual buffer size, or per-sample variability. Given how much the conclusions depend on “quantization hurts small models,” it would be helpful to see whether a slightly less aggressive configuration (e.g., 4-bit with larger residual buffer) narrows the gap with eviction for small models.  
   - The comparison between eviction policies (R-KV and StreamingLLM) is relegated to Appendix C.7 (**Figures 17–18**) with minimal commentary. In some panels, StreamingLLM appears closer to the eviction frontier than R-KV for certain budgets, but the main text does not discuss this nuance.  
   - The analysis focuses on memory vs. accuracy, but KV eviction also affects *compute* and may change throughput or latency in non-trivial ways (less attention cost). This is acknowledged in Section 5 but not explored, even though the same GPU setup is used in C.1.  
   Overall, the presented evidence supports “compression is broadly good” and “eviction tends to help more for small models,” but a more detailed dissection of regimes where quantization could be *preferable* (e.g., very large models with high batch) is missing.

4. **Parallel scaling analysis assumes ideal batching and ignores practical scheduling/serving aspects.**  
   The memory model for parallel scaling in Section 4 and Figure 5 assumes a fully batched serving context where the KV cache scales linearly with group size \(G\), and model weights are fully shared. This is reasonable, but:
   - In practice, many deployments have mixed workloads, non-uniform prompt lengths, and prefill vs decode optimization (e.g., with paged attention). These aspects can change memory profiles and potentially alter which strategies are Pareto-optimal.  
   - The analysis treats group size and token budget as independent knobs but does not examine diminishing returns in accuracy for very large \(G\) at fixed memory. While **Figure 5** indicates that higher \(G\) helps at high memory, the curves for \(G=12,16\) on large models often show very small marginal accuracy improvements relative to large KV footprint.  
   - External verifier analysis (Section 4.1, **Figure 7**) only uses one PRM (ActPRM-X, 7B). Given the strong conclusion that verifiers are “consistently memory-inefficient,” exploring a lightweight PRM or verifying on a sub-sample would be informative, or at least this conclusion should be qualified.  
   These omissions do not invalidate the current results, but they limit how broadly one can apply the parallel scaling recommendations.

5. **Task coverage is narrow in type and structure.**  
   The benchmarks are AIME25, MATH500, LiveCodeBench, and GPQA-Diamond. These are all *single-turn*, heavily chain-of-thought–based tasks with long generations. Missing are:
   - Multi-turn conversational reasoning or tool-augmented settings where KV reuse patterns differ (e.g., partial reuse across turns, non-monotonic conversation length).  
   - Non-English or multimodal reasoning tasks, which could have different tokenization behavior and thus different KV/cache scaling.  
   - More “everyday” reasoning tasks (e.g., GSM8K, StrategyQA) where required chain lengths are shorter.  
   As a result, some conclusions (especially about optimal token budgets of 20–30k in the large-model regime) may not translate to many deployment settings where such extreme lengths are unnecessary or impossible due to prompt construction.

6. **Uncertainty reporting and variability across seeds are largely missing.**  
   The paper reports accuracies averaged over 32 generations (or 8 in KV experiments) but does not provide confidence intervals, error bars, or any explicit measure of variability in figures like **Figure 1**, **Figure 3**, **Figure 5**, or **Figure 9**. On small benchmarks like AIME25 and GPQA-Diamond, variance across seeds or sampling runs can be substantial. Without variability estimates, small differences along the Pareto front (often just a few percentage points) may not be statistically meaningful, which weakens some of the fine-grained ordering used to decide “strictly dominated” configurations.

7. **Some equations and memory accounting assumptions are under-specified or not entirely consistent.**  
   While Section B is generally well done, there are a few points that should be clarified:
   - The weight memory formula includes \(N_{\rm quant}\) and \(N_{\rm unquant}\), but the split is described only qualitatively (“large linear layers” vs “token embedding, layernorms, head” in BF16). It would be useful to state the proportion of unquantized weights per model (or at least per family) and show, for example, how much 4→8-bit changes total \(M_{\rm weights}\) when unquantized layers dominate.  
   - Equation for KV quantization memory:
     \[
     M_{\rm kv}=(G\cdot T\cdot n_{\rm layers}\cdot n_{\rm kv\_heads}\cdot d_{\rm head}\cdot 2)\cdot\left(\frac{P_{\rm kv}}{8}+\frac{1}{g_{\rm kv}}\frac{P_{S}+P_{Z}}{8}\right)
     \]
     implicitly assumes that scale parameters are stored per group for *each* token and head, which is standard, but the overhead term might double count if the HQQ backend compresses scale storage more aggressively (e.g., per-block or per-layer). It would help to cross-check with HQQ’s exact memory accounting and say explicitly if the formula is approximate.  
   - The memory values in **Table 1** and the KB-per-token in **Table 2** look self-consistent, but the text sometimes refers to “32B 4-bit weights occupy 18.01 GB” without clarifying whether this includes optimizer states, vocab embedding shard, etc. A note that all memory numbers are *inference only* and exclude optimizer or activation overhead would avoid confusion.  
   These are not fatal issues but create some friction if a reader wants to plug their own model constants into the formulas.

8. **Some important related works on memory orchestration and reasoning efficiency are missing or under-discussed.**  
   There is a large body of very recent work on memory orchestration and efficient reasoning that is not cited, including:
   - CogMem: a cognitive memory architecture for multi-turn reasoning.  
   - Work on orchestrating KV/cache reuse and external memory layers for efficient inference (“reuse, don’t recompute”–style approaches).  
   - Selective attention / “not all thoughts matter” approaches that also trade off memory and reasoning quality.  
   These works directly address the same conceptual question of “where to spend memory for reasoning,” but from more architectural or algorithmic angles. Their absence in the main text weakens the positioning of this paper as the primary or first systematic treatment of the topic. (See “Potentially Missing Related Work” below for details.)

9. **The paper sometimes overstates generality and underplays limitations.**  
   Section 6 and the bullet-point findings are written as fairly broad prescriptions, but Section 7 admits significant scope limitations: only inference-side methods, a small set of compression techniques, and largely chain-of-thought math/knowledge tasks. The main text occasionally glosses over this and risks readers overgeneralizing. For instance, Finding 2 is stated as “For knowledge-intensive tasks, 4-bit is broadly memory-optimal,” based primarily on **Figure 4** (GPQA-Diamond) without exploring other knowledge tasks or different retrieval/tool-augmented settings. These claims would be more balanced if repeatedly framed as “under our experimental setup and for these benchmarks.”

## Potentially Missing Related Work

1. **Zhang, Y., Hu, J., Dras, M. (2025): CogMem: A Cognitive Memory Architecture for Sustained Multi-Turn Reasoning in Large Language Models.**  
   - Relevance: Proposes a specialized memory architecture for long multi-turn reasoning, essentially another way of trading memory between short-term KV cache and more persistent memory. This is directly relevant to how one might extend the present analysis to conversational settings.  
   - Suggestion: Cite and briefly discuss in Related Work (Section A) where train-time and inference-time memory mechanisms are surveyed. It would also be useful to mention in the Conclusion / Limitations that CogMem-style architectural changes represent an orthogonal axis to the compression knobs studied here.

2. **Patel, D., Patel, S. (2025): Reuse, Don't Recompute: Efficient Large Reasoning Model Inference via Memory Orchestration.**  
   - Relevance: Introduces memory orchestration techniques that selectively reuse past computations and KV entries to reduce memory and compute, targeting reasoning models specifically. This overlaps strongly with this paper’s focus on KV cache efficiency and could serve as an alternative or complementary approach to eviction/quantization.  
   - Suggestion: Discuss in Section 2 or Section A under “Efficient inference” and compare their notion of orchestration with the fixed policies (R-KV, StreamingLLM) used here.

3. **Xiong, Z., Garg, S., Shrivastava, V. (2025): Superposition Reasoning Model.**  
   - Relevance: Proposes an architecture for efficient reasoning that *superposes* multiple reasoning paths, effectively changing the memory footprint of parallel reasoning. Since this paper studies parallel scaling via multiple KV caches and majority voting, Superposition RM is a directly related alternative.  
   - Suggestion: Add to the discussion on parallel scaling in Section 4. It would be appropriate to note that architectural solutions (e.g., shared internal state across samples) may shift the memory–accuracy Pareto frontier compared to independent KV caches.

4. **Shrivastava, V., Awadallah, A. H., Balachandran, V. (2025): Sample More to Think Less: Group Filtered Policy Optimization for Concise Reasoning.**  
   - Relevance: Explores how to obtain concise reasoning while maintaining accuracy, which interacts with the token budget dimension central to this paper. Their findings about “thinking less” under certain policies could change the optimal token budgets assumed in Figures 1–3.  
   - Suggestion: Cite when discussing serial scaling and budget forcing (Section 4) and in Related Work on test-time scaling laws (Section A).

5. **Xiong, Z., Garg, S., Shrivastava, V. (2025): Not All Thoughts Matter: Selective Attention for Efficient Reasoning.**  
   - Relevance: Proposes selective attention mechanisms that discard unimportant reasoning tokens, essentially a more learned version of R-KV/StreamingLLM eviction. This is highly relevant to the KV eviction discussion.  
   - Suggestion: Add to Section 5 and Appendix C.7 for KV compression, contrasting heuristic redundancy-aware policies (R-KV) with more sophisticated learned selection, and clarify how such methods might further improve the memory–accuracy frontier.

## Questions

1. **On the “8-bit 4B” threshold and analytic characterization:**
   - Can the authors provide a more formal explanation or model for why the empirical transition seems to occur around the effective size of an 8-bit 4B Qwen3 on AIME25?  
   - For example, do they observe that beyond this point, the marginal benefit per additional parameter (bits) is lower than per additional KV token (bits), and if so can they quantify this from the data?

2. **Variability and statistical significance:**
   - For key plots like **Figure 1**, **Figure 3**, **Figure 4**, and **Figure 5**, could the authors report confidence intervals or standard errors across seeds or generation batches?  
   - Are the observed differences along the Pareto frontier (often on the order of 2–5 percentage points) statistically robust, especially on small datasets like AIME25?

3. **KV quantization configuration sensitivity:**
   - In **Figure 9** and **Figures 17–18**, results are shown for group size \(g_{\rm kv}=64\) and residual buffer 128. How do the conclusions change with different group sizes or residual lengths, especially for small models where quantization hurts?  
   - Is there a configuration of, say, 4-bit KV quantization that becomes competitive with eviction even at small scales, perhaps trading slightly more memory overhead for better accuracy?

4. **Broader task and setting generalization:**
   - Have the authors experimented with more “typical” chain lengths (e.g., 2k–4k tokens) on tasks like GSM8K or mixed reasoning/QA tasks, and if so does the same “weights vs KV” threshold behavior occur?  
   - How do the recommendations change in multi-turn conversational settings where KV reuse and pruning across turns becomes critical?

5. **Parallel scaling under realistic serving conditions:**
   - The analysis assumes batched inference with uniform group size and token budget. Could the authors comment on how results might change under non-uniform prompts or a serving system with paged attention and heterogeneous requests?  
   - Is the group size \(G\) that appears optimal in **Figure 5** and **Figure 15** still realistic when accounting for queueing and SLAs?

6. **External verifier configurations:**
   - In **Figure 7**, Best-of-N with ActPRM-X is found to be memory-inefficient. Would smaller PRMs (e.g., 1–2B) or verifying a subset of candidate samples significantly change this conclusion?  
   - It would be helpful if the authors could share any pilot experiments or reasoning why such variants are unlikely to be competitive in memory terms.

7. **Clarification on weight memory composition:**
   - For the weight memory equation in Section B, could the authors provide approximate values of \(N_{\rm quant}\) and \(N_{\rm unquant}\) (or at least their ratio) for one or two representative models, so that readers can better understand the contribution of unquantized components?  
   - This would also help clarify how close the “effective size” notion is to a simple \(N \cdot P_W\) approximation.

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A. The work is methodological and empirical on existing LLMs and public benchmarks, with no new data collection or sensitive applications.

## Soundness Rating

3: good.  
The empirical methodology is extensive and generally careful, with clear memory formulas and consistent use of figures and tables. However, the lack of uncertainty quantification, the heuristic nature of some “threshold” claims, and limited exploration of alternative KV quantization configurations prevent a top soundness rating.

## Presentation Rating

3: good.  
The paper is well written, the figures (especially Figures 1–5, 8–9, 10–11, 17–18) are informative, and the structure is clear. Some claims are overstated relative to the evidence, related work is missing several very recent directly relevant papers, and certain technical details (e.g., exact composition of unquantized weights) could be clearer.

## Contribution Rating

3: good.  
The paper does not introduce a new algorithm, but it provides a substantial, carefully organized empirical study and yields actionable guidelines that challenge the conventional “4-bit is always best” deployment narrative. Its focus on reasoning LLMs and KV cache makes it timely and useful, though the lack of a more principled theoretical framework and some missed related work limit the contribution to “strong empirical study” rather than a foundational scaling-law paper.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  

The paper offers a well-executed, memory-centric analysis of reasoning LLM deployment, with clear equations, informative figures, and several practically important findings (e.g., 4-bit is often suboptimal for reasoning, KV compression is crucial, parallel scaling helps only at larger scales). At the same time, the work is primarily empirical, some of the most prominent claims (universal thresholds, universal 4-bit optimality for knowledge) are somewhat overstated, and several closely related recent works on memory and reasoning are missing. Overall, strengths outweigh weaknesses and the paper is a useful contribution, but it is better viewed as a strong empirical guideline paper than a definitive theory of memory-optimal reasoning.

## Reviewer Confidence

4: confident.  
I am familiar with quantization, KV cache compression, and test-time scaling literature, and I carefully checked the memory equations, figures, and tables. Some thresholds and empirical choices could still be debated, but my overall assessment is unlikely to change dramatically.