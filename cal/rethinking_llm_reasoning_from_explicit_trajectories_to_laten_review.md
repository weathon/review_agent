=== CALIBRATION EXAMPLE 26 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title accurately captures the paper’s core premise, and the abstract efficiently summarizes the motivation, method, and results. However, the claim that the approach *“surpasses the state-of-the-art Qwen3 hybrid reasoning framework”* is overstated. The experiments in Section 4.2 and Table 2 explicitly compare against **Qwen3’s non-thinking mode**, not its hybrid thinking mode. Hybrid reasoning typically refers to dynamic routing or interleaved explicit/latent computation; comparing solely to the disabled-thinking baseline risks misleading readers about the magnitude of the contribution. The abstract should be revised to precisely state which mode is being compared, and frame the Qwen3 results as enabling a competitive *latent alternative* to prompt-controlled skipping rather than surpassing the full hybrid system.

### Introduction & Motivation
The introduction effectively motivates the “overthinking” problem and surveys two prevalent mitigation strategies: RL-based compression and prompt-based skipping. The proposed Latent Reasoning Tuning (LRT) is positioned as a middle ground that avoids autoregressive token generation while retaining adaptive reasoning. This framing is clear.

However, the introduction lacks precise positioning against **parameter-efficient tuning (PEFT)** and **soft prompt/prefix tuning** literature. The core mechanism introduced later (Section 3, Appendix C) maps input hidden states to a sequence of latent vectors via a lightweight network and Hadamard product with learnable embeddings. This is architecturally analogous to conditional prompt tuning or hyper-network-based prefix generation. Since ICLR places a high premium on novelty, the introduction must explicitly differentiate LRT from these established methods early on. Claiming that prior latent reasoning methods *“require retraining or substantial fine-tuning”* overlooks adapter-based and prefix-prompt methods that achieve similar non-intrusive modularity. Clarifying this distinction upfront will strengthen the contribution narrative.

### Section 2: Reasoning Trajectory Analysis
This section provides the empirical motivation for replacing explicit trajectories with latent representations by randomly dropping tokens or steps from existing chains and measuring answer accuracy. While the observation that models remain robust to ~50% random deletion is interesting, **the experimental design does not support the claimed conclusion**.

Random deletion fundamentally breaks syntactic structure, positional encodings, and attention causality. An LLM’s ability to maintain accuracy under such noise likely reflects pre-trained robustness to corrupted context or the presence of strong lexical cues in math/logic prompts, not evidence that “reasoning trajectories contain significantly more information than required.” More importantly, this setup does not validate that a *learned, compact latent representation* can effectively substitute for compressed reasoning. The leap from “random deletion degrades performance gracefully” to “a single forward-pass latent network can capture the essential reasoning” is conceptually large. A stronger motivation would involve information-theoretic analysis (e.g., measuring attention concentration on specific steps, or using probing classifiers to quantify step-wise necessity), rather than arbitrary token dropping. As it stands, Section 2 risks being viewed as a weak foundation for the method.

### Section 3: Method
The formalization (Eq 1–3) and Algorithm 1 are clearly presented. The two-stage training (SFT followed by GRPO) freezing the base model $P_\theta$ is a practical design choice. However, several technical and conceptual concerns require addressing:

1. **Alignment Objective vs. Reasoning Surrogate:** Equation 4 optimizes $-\log P_\theta(Y | [X, G_\phi(H_X)])$ directly on the final answer $Y$. While computationally efficient, this objective only ensures that $z = G_\phi(X)$ acts as a *context vector* that improves answer prediction. There is no explicit constraint that $z$ encodes *reasoning structure* or intermediate logic. Consequently, $G_\phi$ may simply learn dataset-specific shortcuts or heuristic mappings rather than true latent reasoning traces. Without an auxiliary loss (e.g., distilling attention patterns, step consistency, or reasoning graph structure), it is unclear how the latent space captures “reasoning” beyond acting as a learned soft prompt.

2. **Gradient Flow & Frozen Base Model:** Since $P_\theta$ is frozen, $G_\phi$ must adapt the latent vectors to match the pre-existing receptive field of $P_\theta$’s hidden states. This is a highly constrained optimization landscape. The paper should report convergence behavior: Does the SFT stage plateau quickly? Does GRPO reliably improve upon SFT, or does it suffer from high variance due to the lack of fine-grained reward shaping? Including training curves or KL-divergence tracking between $P_\theta(\cdot|X,z)$ and the original explicit reasoning distribution would strengthen reproducibility claims.

3. **Architecture Choice (Appendix C):** Equation 5 applies a Hadamard product between input projections and learnable vectors $\hat{r}_i$. This is effectively a **conditional token-prefix injection**. While functional, the novelty relative to established prompt-tuning or adapter-based CoT skipping is minimal. The paper should explicitly acknowledge this relationship and justify why this specific formulation is superior to simpler alternatives (e.g., LoRA applied to the attention layers, or direct prefix tokens conditioned on a small MLP).

### Section 4: Experiments & Results
The experimental design covers relevant benchmarks and compares against strong RL-compression baselines. Table 1 and Table 2 demonstrate consistent gains, which is encouraging. However, several empirical gaps limit the strength of the claims:

1. **Baseline Fairness & Missing Comparisons:** The baselines focus on explicit trajectory compression or prompt skipping. Given the architectural similarity to PEFT methods, the evaluation should include **prompt-tuning or prefix-tuning baselines** trained under identical budgets and datasets. Without this, it is difficult to isolate whether gains come from “latent reasoning” or simply from better input conditioning via frozen-base fine-tuning.

2. **Pass@k Comparison Interpretation (Table 2):** The Qwen3 comparison highlights superior pass@4 performance. However, pass@k heavily depends on sampling diversity and variance. If LRT conditions the base model on 256 stochastic latent vectors, it may naturally produce more diverse answer distributions than a deterministic non-thinking prompt. The paper should report **mean accuracy across multiple seeds** and compare pass@4 under matched sampling protocols (e.g., using diversity metrics or controlling temperature/top-p identically across methods). Otherwise, the pass@4 advantage could reflect sampling dynamics rather than improved reasoning capability.

3. **Efficiency Claims vs. Real-World Costs (Table 7 & D.3):** The paper claims substantial efficiency gains, but Table 7 shows:
   - Latency improvement over non-thinking is marginal (~2.8 sec/question).
   - Peak memory is **higher** (6528 MB vs 3946 MB for non-thinking).
   - The reasoning network ($G_\phi$) is a 0.6B parameter model. Generating its output and materializing 256 latent vectors in the KV cache incurs non-trivial FLOPs and memory bandwidth costs.
   The “effective throughput” metric that divides by latent tokens processed in parallel is theoretically sound but misaligned with real-world deployment, where latency and memory footprint dominate. A more rigorous efficiency analysis should report **FLOPs per sample**, **end-to-end wall-clock time on standard hardware**, and **memory trade-offs vs. explicit CoT with short budgets**. As presented, the efficiency advantage is ambiguous.

4. **Token Count Ablation (Table 3):** Performance peaks at 256 tokens and drops at 512, attributed to “larger training scales may be necessary.” This is speculative. The drop could indicate overfitting, capacity mismatch between $G_\phi$ and $P_\theta$, or degradation in the attention-to-latent mapping. A deeper diagnostic (e.g., analyzing latent norm growth, attention saturation, or gradient norms) would clarify whether 256 is an architectural limit or a training artifact.

### Appendix & Additional Analysis
Appendix C clearly details the projection layers and initialization from Qwen3-Embedding-0.6B, which aids reproducibility. Appendix D.1 (Qwen3-8B extension) is valuable and shows scalability. However, Appendix D.4 (cosine similarity of latent vectors) is purely descriptive. While it shows domain clustering, it does not establish a causal link between geometric structure and reasoning quality. For ICLR, a more impactful analysis would probe whether intervening on specific latent dimensions alters error types (e.g., arithmetic vs. logical failures). Appendix D.5 addresses statistical significance via standard deviations, which is good practice, but formal hypothesis testing (e.g., bootstrap confidence intervals or paired t-tests against baselines) would strengthen the claims.

## Overall Assessment
The paper tackles a highly relevant problem in LLM reasoning: reducing the computational overhead of explicit chain-of-thought generation while preserving accuracy. The proposed Latent Reasoning Tuning framework offers a clean, modular alternative that freezes the base model and learns auxiliary latent vectors via SFT+RL. Empirical results across multiple benchmarks are promising, and the non-intrusive design is practically appealing.

However, for ICLR’s acceptance bar, several substantive issues must be addressed:
1. The foundational motivation (Section 2) relies on random trajectory deletion, which does not logically validate the effectiveness of learned latent compression.
2. The core architecture closely aligns with conditional prompt/prefix tuning, yet the paper frames it as a novel latent reasoning paradigm without sufficiently differentiating from established PEFT literature.
3. The SFT objective optimizes final answer likelihood without enforcing that latent vectors capture reasoning structure, raising questions about whether the method learns true reasoning or dataset shortcuts.
4. Efficiency claims are undermined by higher peak memory and marginal latency gains relative to non-thinking baselines. The 0.6B auxiliary model’s overhead is not fully accounted for in practical deployment terms.

With these concerns addressed—particularly by tightening the novelty positioning, adding architectural baselines, clarifying the alignment objective, and providing more rigorous efficiency/FLOP analysis—the paper could represent a solid contribution to efficient reasoning. As it stands, the empirical results are encouraging, but the conceptual and methodological framing requires substantial refinement to meet ICLR’s standards for novelty and empirical rigor.

# Neutral Reviewer
## Balanced Review

### Summary
This paper addresses the high inference overhead of explicit, autoregressive reasoning chains in LLMs by proposing Latent Reasoning Tuning (LRT), a framework that replaces step-by-step token generation with compact, learnable latent vectors produced by a lightweight auxiliary network. Motivated by empirical evidence that reasoning LLMs remain highly accurate even when conditioned on randomly fragmented trajectories, LRT trains the auxiliary reasoning network via a two-stage SFT and GRPO pipeline while keeping the base LLM frozen. Experiments across mathematical and logical reasoning benchmarks demonstrate that LRT consistently outperforms existing efficient reasoning baselines and native hybrid reasoning modes, while maintaining flexible switching between latent and explicit reasoning.

### Strengths
1. **Strong empirical motivation and analysis**: The trajectory fragmentation experiments in Section 2 rigorously demonstrate that reasoning chains contain substantial redundancy and that models robustly extract correct answers from highly degraded inputs. This provides a clear, data-driven justification for bypassing explicit token-by-token generation.
2. **Practical, modular architecture design**: By freezing the base reasoning LLM and training a separate reasoning network $G_\phi$, the method avoids catastrophic forgetting, preserves the base model's original capabilities, and enables flexible mode-switching at inference. This non-intrusive design is highly valuable for real-world deployment (Sections 3.2 & 6).
3. **Comprehensive and well-executed empirical evaluation**: The paper evaluates on five diverse benchmarks (AMC, MATH-500, GSM8K, LSAT, GPQA) across multiple model scales (1.5B to 8B) and token budgets. LRT consistently surpasses strong efficient reasoning baselines (NoThinking, ShorterBetter, LC-R1) and Qwen3's native non-thinking mode (Tables 1 & 2). The two-stage training ablation (Table 4) provides clear evidence that RL is necessary to transcend dataset quality limits.
4. **High reproducibility**: The paper provides a public code repository, detailed hardware specifications, explicit hyperparameters for both SFT and RL stages, and clear inference protocols (Appendix B). This aligns well with ICLR's reproducibility standards.

### Weaknesses
1. **Efficiency claims are only partially substantiated**: While the paper highlights reduced autoregressive decoding, Table 7 shows that peak memory usage (6528 MB) is significantly higher than the non-thinking mode (3946 MB), and latency improvement is marginal (11.79s vs 14.62s). The computational cost of the 0.6B embedding model forward pass plus processing 256 parallel latent tokens is not disentangled from the final answer generation, making the true efficiency trade-off unclear.
2. **Lack of direct empirical comparison with latent reasoning literature**: The paper discusses methods like Coconut, Pause Tokens, and internalized CoT in Section 5 and Appendix E, but provides no head-to-head experimental comparison. Given the rapid progress in latent reasoning, this omission makes it difficult to assess LRT's relative standing and novelty in a rigorous manner.
3. **Underexplored architectural justification**: The reasoning network is initialized from a pre-trained text embedding model (Qwen3-Embedding-0.6B) without ablation or comparison against simpler alternatives (e.g., random initialization, lightweight transformers, or MLPs on frozen hidden states). It remains unclear whether performance gains stem from the latent formulation itself or from strong semantic priors in the embedding model.
4. **Superficial mechanistic analysis of latent representations**: Section D.4 shows domain-level clustering via cosine similarity but does not investigate how latent tokens interact with the base model's attention mechanism, whether attention patterns differ from explicit CoT, or if representation collapse occurs during training. Deeper mechanistic insight is needed to fully validate the "latent reasoning" claim.

### Novelty & Significance
**Novelty**: Moderate-High. Latent reasoning and CoT compression are active areas, and the core idea of mapping inputs to fixed-length latent vectors is not fundamentally new. However, LRT introduces a novel, well-motivated training pipeline (frozen base + two-stage SFT/GRPO on an auxiliary network) and provides a clear empirical bridge from trajectory redundancy analysis to latent representation learning. The modular design that preserves base model flexibility is a practical innovation.
**Clarity**: High. The paper is well-structured, motivation flows logically into methodology, and algorithmic descriptions are precise. Mathematical notation and training objectives are clearly stated.
**Reproducibility**: High. Code is provided, hyperparameters and hardware are specified, datasets and baselines are standard and accessible.
**Significance**: High for the efficient reasoning subfield. As reasoning models scale toward longer chains (o1, R1, QwQ), methods that compress or internalize reasoning without costly full-model fine-tuning are highly relevant. LRT offers a strong, empirically validated baseline that pushes beyond current prompt-based and RL-length-penalty approaches, making it a meaningful contribution for ICLR.

### Suggestions for Improvement
1. **Disentangle efficiency metrics**: Report compute breakdowns (e.g., FLOPs or GPU-hours) separately for the auxiliary network forward pass, latent sequence processing, and final answer decoding. Compare against baselines using a unified metric like *throughput-per-correct-answer* or *energy-per-solved-problem* to substantiate efficiency claims rigorously.
2. **Include direct comparisons with latent reasoning baselines**: Implement and benchmark at least one representative method from the latent CoT literature (e.g., Coconut, Pause Tokens, or ICoT) under identical training budgets and evaluation protocols to clearly position LRT's empirical advantages.
3. **Ablate the reasoning network architecture and initialization**: Test the framework with random initialization, a lightweight transformer, and an MLP to isolate the impact of the latent formulation from pre-trained embedding priors. This will clarify design choices and strengthen methodological claims.
4. **Deepen latent space analysis**: Include attention visualizations or gradient-based attribution methods to show how the base model utilizes the latent tokens during answer generation. Investigate whether latent vectors encode logical structure, step boundaries, or merely act as compressed prompts. Adding a probing task (e.g., predicting problem difficulty or domain) would further validate the geometric analysis in Section D.4.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Benchmark against SOTA continuous/latent reasoning methods (e.g., Coconut, PAUSE Tokens, ICoT-SI) rather than only text-compression RL baselines. Without these, the claim of advancing "latent reasoning" lacks essential empirical context and comparative rigor.
2. Provide an end-to-end latency and FLOP breakdown that explicitly accounts for the 0.6B $G_\phi$ forward pass and context-injection overhead. Current efficiency claims are undermined because the true compute cost of generating latent vectors is not compared against baselines on an equal footing.
3. Run a direct equal-wall-clock or equal-FLOP comparison against standard test-time scaling. Comparing solely under fixed token budgets ignores the auxiliary inference cost of $G_\phi$, making latency and throughput advantages speculative.

### Deeper Analysis Needed (top 3-5 only)
1. Ablate against a standard continuous prefix or soft prompt trained under the identical RL pipeline. Without this, performance gains may stem from generic representation expansion rather than the proposed "latent reasoning" mechanism.
2. Analyze sensitivity to the extraction depth of $H_X$ instead of only using final hidden states. Terminal states may discard intermediate logical states required for complex problem-solving, weakening the architectural justification for bypassing explicit traces.
3. Probe the latent vectors for alignment with interpretable reasoning behaviors (e.g., decomposition, verification, backtracking) instead of relying on pairwise cosine similarity. This is required to prove that logical structure is actually encoded rather than acting as high-capacity memorization triggers.

### Visualizations & Case Studies
1. Plot cross-attention maps of the base LLM over the latent sequence $z$ during answer decoding. This is necessary to verify that $z$ actively steers generation logic rather than being passively averaged out by the attention mechanism.
2. Provide qualitative case studies contrasting a complex problem where explicit CoT is strictly required but LRT fails. Bounding failure modes is essential to prove the method isn't just exploiting dataset-specific patterns.
3. Render a continuous performance-compute Pareto frontier across fine-grained latent lengths and RL iterations. Sparse table points raise cherry-picking concerns; a curve is required to validate the robustness of the claimed efficiency-accuracy trade-off.

### Obvious Next Steps
1. Formalize and detail the positional encoding and attention masking strategy for injected latent tokens. Without explaining how $z$ interacts with $P_\theta$'s positional scaling, claims regarding scalability and architectural safety are incomplete.
2. Report training variance and convergence stability across multiple RL random seeds, as the reported 100-step GRPO run is highly vulnerable to policy gradient noise. Stability analysis should be included to confirm the RL stage isn't a fluke.
3. Evaluate on non-mathematical tasks such as code synthesis, multi-hop reasoning, or multi-turn instruction following. Mathematical QA dominance alone is insufficient to claim broad reasoning capabilities for a novel latent framework at ICLR.

# Final Consolidated Review
## Summary
The paper proposes Latent Reasoning Tuning (LRT), a framework that replaces explicit, token-by-token chain-of-thought generation with compact, continuous latent representations produced by a lightweight auxiliary network. By keeping the base LLM frozen and optimizing the auxiliary network via a two-stage SFT and GRPO pipeline, LRT demonstrates improved accuracy over RL-compression and prompt-skipping baselines across mathematical, logical, and scientific benchmarks while enabling flexible switching between explicit and latent reasoning modes.

## Strengths
- **Non-intrusive, modular architecture:** Training a separate reasoning network while freezing the base LLM avoids catastrophic forgetting, preserves original model capabilities, and allows seamless toggling between reasoning modes at inference without weight modifications.
- **Robust empirical motivation:** Section 2’s trajectory fragmentation experiments convincingly demonstrate that explicit reasoning chains contain substantial redundancy and that LLMs maintain high accuracy even when conditioned on heavily degraded inputs, providing a clear data-driven foundation for bypassing autoregressive step generation.
- **Consistent cross-benchmark improvements:** LRT outperforms strong efficient-reasoning baselines (ShorterBetter, LC-R1, NoThinking) across five diverse tasks (AMC, MATH-500, GSM8K, LSAT, GPQA) and scales effectively from 1.5B to 8B models, with the SFT+RL ablation confirming the necessity of reinforcement learning to surpass dataset imitation limits.
- **High reproducibility:** The manuscript provides public code, explicit training/inference hyperparameters, hardware specifications, and clear dataset/baseline protocols, aligning well with rigorous reproducibility standards.

## Weaknesses
- **Incomplete efficiency accounting:** Claims of substantial efficiency gains are partially undermined by unaccounted computational overhead. The 0.6B auxiliary network forward pass and KV-cache materialization for 256 latent vectors increase peak memory (6528 MB vs. 3946 MB for non-thinking) and are not fully disentangled from end-to-end latency or FLOPs. Without disaggregated compute metrics, the true efficiency trade-off versus explicit reasoning under matched budgets remains ambiguous.
- **Missing empirical comparisons to latent/continuous reasoning baselines:** While the paper situates itself within latent reasoning literature and discusses methods like Coconut, ICoT-SI, and Pause Tokens, it provides no head-to-head experimental evaluation against them. This omission makes it difficult to determine whether performance improvements stem from the latent formulation itself or simply from the specific SFT+GRPO training regimen.
- **Architectural overlap with established PEFT methods:** The core mechanism (projecting hidden states to a sequence of continuous conditioning vectors combined via Hadamard product with learnable embeddings) closely mirrors conditional prompt/prefix tuning. The paper frames LRT as a distinct latent reasoning paradigm but does not explicitly differentiate it from or empirically compare against simpler continuous prefix-tuning approaches trained under identical RL objectives.
- **Lack of structural constraints in the training objective:** Optimizing solely for final answer likelihood ($-\log P_\theta(Y|X, z)$) provides no explicit incentive for the latent vectors $z$ to encode intermediate reasoning logic or structural dependencies. While this avoids costly KL-divergence computation, it raises the possibility that the network learns dataset-specific heuristics or shortcuts rather than generalizable reasoning traces, limiting claims about latent reasoning structure and out-of-distribution robustness.

## Nice-to-Haves
- Clarify the exact positional encoding and attention masking scheme applied to the injected latent sequence to guarantee proper causal alignment with the base model's attention mechanics.
- Provide a continuous Pareto curve plotting accuracy against latent token length (or compute budget) rather than discrete table points to better visualize the robustness of the efficiency-accuracy tradeoff.
- Include cross-attention visualizations or gradient-based attribution maps showing how specific latent tokens interact with the base model's attention heads during answer generation.

## Novel Insights
LRT effectively decouples reasoning from autoregressive text generation, reframing it as a parallel, continuous conditioning problem. The empirical bridge drawn from trajectory fragmentation robustness to latent representation learning challenges the assumption that step-by-step token decoding is strictly necessary for complex inference. Furthermore, the emergent geometric structure of the learned latent vectors (Appendix D.4)—showing clear domain clustering and semantic separation between competition math, logic, and scientific reasoning—suggests that the auxiliary network learns to map distinct problem classes into specialized regions of a compressed reasoning space, functioning as task-specific continuous priors rather than mere soft prompts.

## Suggestions
- Report a disaggregated latency/FLOP/memory breakdown for the $G_\phi$ forward pass, latent KV-cache injection, and final answer decoding. Compare against baselines using a unified metric such as *compute-per-correct-answer* or *wall-clock throughput under equal FLOP budgets* to rigorously substantiate efficiency claims.
- Add empirical comparisons against at least one representative latent or continuous CoT baseline (e.g., ICoT-SI or Coconut) trained on identical datasets and compute budgets to clearly position LRT’s contributions within the latent reasoning landscape.
- Conduct an ablation on the reasoning network’s architecture and initialization (e.g., random initialization, small MLP, or lightweight transformer) to isolate whether performance gains derive from the continuous conditioning formulation or from the strong semantic priors of the pre-trained Qwen3-Embedding model.

# Actual Human Scores
Individual reviewer scores: [4.0, 4.0, 8.0, 4.0]
Average score: 5.0
Binary outcome: Accept
