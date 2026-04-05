=== CALIBRATION EXAMPLE 52 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title is broadly accurate: the paper does propose moving from explicit reasoning trajectories to latent representations. That said, “Rethinking LLM Reasoning” is quite general, while the actual contribution is narrower: an auxiliary reasoning network that conditions a frozen base model.
- The abstract clearly states the problem (long reasoning trajectories are expensive), the proposed method (Latent Reasoning Tuning, LRT), and the claimed result (better efficiency/performance than selected efficient reasoning methods).
- A concern is that the abstract overstates novelty and scope. It implies a fundamental replacement of explicit reasoning with latent reasoning, but the method still depends on the base model’s answer generation and uses training data extracted from explicit trajectories. It also claims LRT “surpasses the state-of-the-art Qwen3 hybrid reasoning framework,” but the paper’s evidence is limited to specific Qwen3 sizes and budgets; this is not enough to establish a broad SOTA claim.

### Introduction & Motivation
- The motivation is relevant for ICLR: reasoning efficiency is a real and important problem, and the paper connects it to overthinking and inference cost.
- The introduction does identify a gap between shortening explicit reasoning and bypassing it with fixed prompts. However, the novelty gap is somewhat overstated. The paper cites prior latent reasoning work in Related Work, but the introduction does not sufficiently distinguish LRT from earlier latent-thought or continuous-latent reasoning approaches beyond saying it uses an auxiliary network rather than retraining the base LLM.
- The contributions are stated clearly, but one claim is too strong: “reasoning trajectories are not essential.” The evidence later shows the base model can answer with fragmented trajectories, but that does not by itself establish that a learned latent trajectory is the right or necessary replacement. It shows redundancy, not equivalence.
- The introduction under-explains the central conceptual leap: why an auxiliary network that outputs latent vectors should be expected to preserve reasoning structure better than direct answer prediction, especially when the base model is frozen.

### Method / Approach
- The method is described at a high level, but several key implementation and conceptual details are not fully reproducible from the paper alone.
- In Section 3.2 and Algorithm 1, the training objective is not fully coherent. Equation (4) uses `-log f_θ(Y | [X, G_ϕ(H_X)])`, but `f_θ` is not clearly defined relative to `P_θ`, and it is not explicit how latent vectors are injected into the frozen base model’s computation graph.
- The method hinges on greedy decoding to justify treating the reasoning trajectory as a deterministic function `h(X, θ)` in Equation (2). But the experiments later use sampling temperature 0.6 and top-p 0.95. This creates a mismatch between the theory/derivation and the actual inference/training regime. If the reasoning trajectory is stochastic in practice, the deterministic framing is not well justified.
- Algorithm 1 suggests the RL stage samples `K` candidate answers from `P_θ(. | [E_X, z])` and updates `G_ϕ` using GRPO-like policy gradients. But the policy being optimized is the auxiliary network through the base model’s outputs, not a conventional policy with clearly specified action space. The derivation of the policy ratio and credit assignment is not explained enough to assess correctness.
- The architecture description in Appendix C is important, but it raises unresolved design questions: the latent sequence is produced via a Hadamard product with learnable vectors, then a Qwen3-Embedding-0.6B-based network, then projected back. This is a substantial engineering choice, yet there is little justification for why this specific structure should work or how sensitive results are to it.
- A key assumption is that a frozen reasoning LLM can interpret latent vectors in a meaningful way with no parameter updates. That assumption is central, but the paper does not analyze when or why it holds, nor failure modes when the base model’s hidden states are out of distribution.
- Edge cases are not discussed. For example, what happens on tasks requiring long deliberative search, tool use, or exact symbolic intermediate states? Since the latent network is producing a fixed-length representation, there may be severe expressivity limits, but this is not analyzed.
- The paper’s “non-intrusive design” claim is only partially true: while the base model is frozen, the system still requires architectural interfacing and a separately trained reasoner. That is fine, but it is not as seamless as the framing suggests.

### Experiments & Results
- The experiments do address the paper’s core claim: they compare against efficient reasoning baselines, evaluate on several math and out-of-domain benchmarks, and include an ablation on latent token count and training strategy.
- However, the experimental design leaves important questions unresolved for ICLR-level standards.
- The strongest comparisons are against NoThinking, ShorterBetter, and LC-R1, but these are not exhaustive baselines for “efficient reasoning.” The paper does not compare against more direct latent-reasoning or compact-deliberation methods in a way that convincingly isolates its advantage.
- Table 1 mixes budgets and models in a way that is somewhat hard to interpret. The paper claims fair comparisons under the same token budget, but the efficiency metric is not consistently reported in Table 1, so it is difficult to know whether performance gains come from genuinely better reasoning or from differing effective compute usage.
- The most important ablation missing is a direct comparison to simpler alternatives: e.g., using the same latent network but training it to predict the final answer directly without the intermediate “reasoning trajectory” framing, or using the same parameter budget to fine-tune the base model directly. Without this, it is hard to know whether the auxiliary latent trajectory is actually the source of the gains.
- Table 2 is interesting, but the comparison to Qwen3 non-thinking mode is not entirely apples-to-apples. The “ours@1 / ours@4” results are based on latent reasoning on top of thinking-mode Qwen3, but the paper does not fully clarify whether the compute budget and decoding settings are matched to the non-thinking baseline in a way that isolates the method’s contribution.
- The paper reports standard deviations only in Appendix D.5 for one setting. That is better than nothing, but ICLR reviewers would likely expect variance analysis across repeated runs for the main tables, especially because the method uses stochastic decoding and RL.
- The result claims are somewhat cherry-picked in presentation. For example, Table 2 shows cases where `pass@1` is equal or worse than the baseline on some benchmarks, while the text emphasizes average gains. The method appears strongest in `pass@4`, but that metric is not directly comparable to latency-focused claims.
- D.3 is important and shows latency improvements, but the measurement protocol is limited: it uses 64 random MATH-500 problems. That is too small to make strong general claims about efficiency, especially since memory and throughput can be workload-dependent.
- The ablation on latent token count (Table 3) is useful, but it does not establish why 256 is optimal or whether the curve is stable across datasets and base models. More importantly, Table 6 suggests the latent-token optimum depends on model scale, which implies the method may require significant retuning across backbones.
- The paper should report stronger comparisons on cost-normalized performance, not just accuracy under fixed budgets. For an ICLR paper on efficient reasoning, the key claim is not just accuracy but accuracy per unit compute.

### Writing & Clarity
- The paper is generally readable, and the structure is clear enough to follow the overall idea.
- The main clarity issue is conceptual rather than grammatical: the relation among explicit trajectories, latent trajectories, hidden states, and the frozen base model is not always spelled out in a way that makes the full computation graph obvious.
- Figure 1 and Figure 3 convey the high-level idea, but they do not fully explain how the latent vectors are inserted into the LLM or how the auxiliary network interfaces with the base model.
- Algorithm 1 is helpful, but the notation is overloaded and some operations are underspecified, especially in the RL stage. This impedes reproducibility.
- The paper also does not clearly separate what is empirical observation from what is theoretical justification. Section 2’s “reasoning trajectory analysis” reads like a hypothesis confirmation, but the evidence is presented in a way that makes the causal claim stronger than the data supports.

### Limitations & Broader Impact
- The paper acknowledges some limitations indirectly, such as the dependence on latent token count and the fact that larger base models may benefit more from longer latent trajectories. But it does not explicitly discuss the main limitations of the approach.
- A major limitation is expressivity: a fixed-length latent trajectory may be insufficient for tasks requiring long, branching, or verifiable reasoning. This is especially important because the paper’s motivation is to replace long explicit reasoning.
- Another limitation is reliance on a strong frozen base model. The method appears to act more like a controller or conditioner than a standalone reasoner. Its effectiveness may not transfer well to weaker or differently trained backbones.
- The paper does not discuss potential failure modes such as adversarial prompts, distribution shift, or interpretability problems introduced by latent reasoning.
- Broader impact discussion is absent. While this is not mandatory for every paper, for an ICLR submission on reasoning systems it would be valuable to acknowledge that latent internal reasoning can further reduce transparency and make model behavior harder to audit, even as it improves efficiency.

### Overall Assessment
This is an interesting and timely attempt to reduce reasoning cost by replacing explicit chain-of-thought generation with a learned latent trajectory produced by an auxiliary network. The empirical results suggest that the idea can work on several benchmarks and can improve efficiency relative to a handful of efficient-reasoning baselines. However, for ICLR standards, the paper still leaves important questions unresolved: the method is not fully specified or theoretically grounded, the comparison to prior latent-reasoning work is not sufficiently sharp, and the experiments do not yet isolate the mechanism well enough to establish that latent reasoning itself—not just an added learned controller—drives the gains. I think the contribution is promising, but the current evidence is not yet strong enough to make the paper feel decisive.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes Latent Reasoning Tuning (LRT), a framework that replaces explicit token-by-token reasoning trajectories with a learned latent representation produced by an auxiliary reasoning network. The core claim is that many reasoning tasks do not require full CoT generation: the model can answer accurately from fragmented trajectories, and this insight is used to motivate a compact latent “trajectory” that conditions the base LLM. Experiments on math and out-of-domain benchmarks report improved accuracy over several efficient-reasoning baselines and stronger performance than Qwen3’s non-thinking mode under matched token budgets.

### Strengths
1. **Timely problem and relevant target for ICLR.**  
   The paper addresses the increasingly important issue of inference-time reasoning efficiency in LLMs, which is highly aligned with ICLR interests in efficient architectures, representation learning, and practical scaling.

2. **Clear empirical motivation for the method.**  
   The trajectory analysis in Section 2 is a useful diagnostic: the authors show that the base reasoning model retains substantial accuracy even when a large fraction of reasoning tokens or steps are randomly removed. This supports their claim that explicit trajectories contain redundancy and motivates a compact alternative.

3. **Modular design is attractive.**  
   A notable design choice is that the base LLM is frozen and the reasoning ability is moved into an auxiliary network. This makes the approach potentially easier to adopt than methods requiring full retraining of the reasoning model, and the paper emphasizes the possibility of switching between latent and explicit reasoning modes.

4. **Reasonably broad benchmark coverage.**  
   The evaluation includes both in-domain math benchmarks and out-of-domain reasoning tasks such as LSAT and GPQA, which is a stronger setup than evaluating only on arithmetic or competition math.

5. **Ablations and efficiency measurements are included.**  
   The paper does not only report accuracy; it also studies latent-token count, SFT vs. SFT+RL, latency, throughput, memory, and some geometric properties of latent representations. This improves the paper’s completeness relative to a minimal benchmark-only submission.

6. **Claims are positioned against relevant efficiency baselines.**  
   Comparisons to NoThinking, ShorterBetter, and LC-R1 are appropriate for the stated problem of making reasoning cheaper, and the paper also compares against Qwen3’s native non-thinking mode.

### Weaknesses
1. **Novelty is moderate rather than clearly strong by ICLR standards.**  
   The main idea—moving reasoning from explicit text into latent space—is conceptually interesting, but it is closely related to a substantial prior line of latent-CoT / continuous reasoning work cited in the paper (e.g., Coconut, latent thoughts, recurrent depth, reasoning in latent space). The paper’s specific novelty is the use of an auxiliary network to produce a full latent trajectory for a frozen reasoning LLM, but the conceptual leap feels incremental relative to the existing latent-reasoning literature.

2. **The core analysis may not establish the strongest causal claim.**  
   The “fragmental reasoning” experiment shows that the model can often answer correctly when parts of the trajectory are removed, but random deletion is a coarse perturbation. This does not necessarily prove that full reasoning is unnecessary in general, only that some trajectories contain redundancy and the model is robust to degradation. The paper somewhat over-interprets this as evidence that explicit trajectories are not essential.

3. **Limited evidence that the latent network truly models reasoning rather than shortcuts.**  
   Because training uses only question-answer supervision in SFT and reward-based correctness in RL, it is unclear what the latent vectors encode beyond an answer-predictive prompt embedding. The paper does not provide strong mechanistic evidence that the latent representations correspond to meaningful reasoning computations rather than compressed task conditioning.

4. **Fairness and comparability of baselines are not fully convincing.**  
   Some baselines are prompt-based, others are RL-based, and the paper’s own method adds a separate reasoning network with its own computational cost. The comparison on accuracy alone may obscure that the auxiliary network introduces parameters and inference overhead not present in the simplest “direct answer” baselines. The efficiency story would be stronger with a unified FLOPs / wall-clock / parameter-cost comparison.

5. **Potential ambiguity in training and inference details hurts reproducibility.**  
   The method description leaves several important points underspecified: how the latent sequence is consumed by the base model, how many layers/parameters the reasoning network has, whether the latent tokens attend bidirectionally or causally, how the projection layers are integrated, and how the frozen base model is exactly conditioned on the latent vectors. The appendix helps, but not enough for straightforward reproduction.

6. **Some results are mixed and the gains are not uniformly large.**  
   In Table 2, pass@1 occasionally matches or underperforms the non-thinking baseline, and Table 3 shows a non-monotonic dependence on latent length. The paper does discuss these cases, but the overall performance improvements are often modest on some benchmarks, which makes the claim of “surpassing state of the art Qwen3 hybrid reasoning” feel somewhat selective.

7. **The efficiency claim needs a more rigorous apples-to-apples accounting.**  
   The paper reports latency, throughput, and peak memory, but the auxiliary reasoner itself is not cost-free. For ICLR-level acceptance, one would expect a clearer breakdown of total compute including the latent-network forward pass, and ideally comparison against matched-budget baselines using identical model families and decoding strategies.

8. **Interpretability analysis is suggestive but not very strong.**  
   The latent-space similarity table is interesting, but the conclusions about “domain clustering” and “complexity stratification” are somewhat qualitative and based on averaged cosine similarities. This is not enough to substantiate deeper claims about the structure of latent reasoning.

### Novelty & Significance
**Novelty: Moderate.** The paper repackages latent reasoning in a pragmatic, modular form by adding an auxiliary network to generate compact reasoning vectors for a frozen LLM. This is a sensible and potentially useful design, but it is not a major conceptual departure from prior latent-reasoning and reasoning-compression work.

**Significance: Moderate.** If validated robustly, the method could be practically valuable because it targets inference efficiency without requiring the base model to be retrained. However, the current evidence suggests a useful engineering contribution more than a breakthrough that clearly clears the higher novelty and rigor bar often expected at ICLR.

**Clarity: Good overall, with some important missing details.** The paper is generally easy to follow, and the motivation-method-results structure is coherent. That said, several architectural and training specifics are insufficiently precise.

**Reproducibility: Medium.** The appendix provides training hyperparameters and some implementation notes, but key details of the latent network, conditioning interface, and exact baseline configurations are not fully specified. This would likely make faithful reproduction challenging.

**Significance relative to ICLR bar: Borderline.** The topic is timely and the empirical story is plausible, but the paper would need stronger evidence of conceptual novelty, a more rigorous efficiency accounting, and deeper mechanistic validation to be clearly above the ICLR acceptance threshold.

### Suggestions for Improvement
1. **Strengthen the causal evidence for the main claim.**  
   Add more controlled experiments beyond random token/step deletion, such as importance-based removal, adversarial removal, and comparisons to semantically preserved compressed trajectories. Show more directly when explicit reasoning is and is not necessary.

2. **Provide a more rigorous compute-accounting analysis.**  
   Report end-to-end latency, FLOPs, memory, and parameter counts for the full system, including the reasoning network. Compare against baselines under identical decoding setups and include overhead from generating latent tokens.

3. **Clarify the architecture and conditioning mechanism in detail.**  
   Specify the exact shape of latent tokens, how they are inserted into the base model, whether they are prepended as prefix embeddings or handled via cross-attention, and how the projection layers interact with the frozen model. Include pseudocode or diagrams with tensor shapes.

4. **Add stronger mechanistic analysis of the latent space.**  
   Go beyond average cosine similarity. For example, probe whether latent vectors encode problem type, difficulty, or solution strategy; test linear separability; and compare against simple learned prompt embeddings or direct soft prompts.

5. **Compare against stronger and more directly relevant latent-reasoning baselines.**  
   Since the paper claims novelty over latent reasoning, it should include more explicit comparisons to recent latent-CoT systems or adapter/prefix-tuning approaches where possible, not only token-compression and no-thinking baselines.

6. **Report more detailed per-task and per-budget results.**  
   Some improvements are uneven across benchmarks. Provide confidence intervals or multiple seeds per benchmark, and include cases where the method underperforms or matches the baseline to present a balanced view.

7. **Discuss limitations more honestly.**  
   In particular, acknowledge that the method still relies on a separate learned module, may not preserve interpretability, and may not always beat explicit thinking when enough test-time budget is available.

8. **Improve the theoretical framing.**  
   The current argument that a complete explicit trajectory is unnecessary is empirically plausible but theoretically thin. A more careful framing would help: e.g., the method can be presented as learning a compressed task-conditioned latent program rather than eliminating reasoning altogether.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add strong comparisons to the most relevant latent-reasoning baselines (e.g., Coconut, recurrent/continuous CoT methods, and any recent “latent thoughts” models) on the same backbones and budgets. Without this, the claim that LRT is a new latent-reasoning advance is not convincing, especially at ICLR where novelty must be positioned against the right prior art.

2. Add a direct comparison to simple distillation / direct-answer training of the same base model with matched compute and data. The paper claims the reasoning network “replaces” explicit trajectories, but it is unclear whether gains come from latent reasoning or just another supervised answer-prediction pipeline.

3. Add an ablation that removes the RL stage and another that removes the SFT stage across all reported backbones. The current Table 4 only partially supports the two-stage design; ICLR reviewers will want to know whether the reported gains depend on RL, or whether the method is mostly an SFT effect.

4. Add compute-normalized comparisons: wall-clock training cost, inference latency, tokens processed, and quality-per-FLOP against thinking/non-thinking baselines. The paper’s main claim is efficiency, but the current evidence mixes accuracy with token budgets and does not fully establish a better accuracy–cost tradeoff.

5. Add robustness tests on longer/harder reasoning benchmarks beyond the current five-task set, ideally including multi-hop QA and harder math sets with explicit length pressure. The paper’s central claim is about eliminating explicit reasoning trajectories, so it needs to show this still holds when reasoning chains are truly necessary, not just on benchmarks where short answers often suffice.

### Deeper Analysis Needed (top 3-5 only)
1. Analyze whether the latent vectors actually encode reasoning or are just compressed task identifiers / answer priors. The cosine-similarity analysis is too coarse; without probing the information content of latent states, the claim that they “encapsulate reasoning logic” is not established.

2. Analyze failure cases where fragmented trajectories work but LRT fails, and vice versa. The paper argues that full trajectories are unnecessary, but it never shows where latent reasoning breaks down, which is essential to trust the method’s true scope.

3. Quantify how performance varies with latent length, base-model size, and reasoning difficulty in a single unified analysis. Right now these appear as disconnected ablations; ICLR would expect a clearer scaling story that explains when latent reasoning helps and when it saturates.

4. Analyze whether the method preserves factuality and reasoning faithfulness, not just final-answer accuracy. Since LRT removes explicit trajectories, readers need evidence that it does not simply improve shortcutting or answer memorization.

5. Provide a principled explanation of why the base LLM can decode from learnable latent embeddings without a mismatch in representation space. The method hinges on this interface, but the paper does not analyze whether performance is sensitive to projection choice, latent initialization, or latent-token parameterization.

### Visualizations & Case Studies
1. Show side-by-side case studies of the original reasoning trajectory, the latent tokens, and the resulting answer on both success and failure examples. This would reveal whether LRT is genuinely preserving intermediate reasoning structure or merely learning hidden shortcuts.

2. Visualize latent-space structure with probes or clustering by problem type, difficulty, and answer class. The current cosine-similarity table is too indirect; a clearer visualization is needed to support the claim that latent reasoning organizes meaningful computation.

3. Plot accuracy versus latency / token-count / FLOPs curves for all main methods on the same axes. The paper claims efficiency gains, so reviewers need a direct Pareto-style visualization showing where LRT dominates and where it does not.

4. Include qualitative examples where explicit reasoning is long but fragmented trajectories still succeed, alongside the corresponding LRT outputs. This would directly support the paper’s motivating hypothesis that only partial reasoning content is sufficient.

### Obvious Next Steps
1. Evaluate whether the auxiliary reasoning network generalizes across tasks without retraining, e.g., train on math and test on logic/science. This is the most direct test of whether LRT learns transferable reasoning representations rather than benchmark-specific compression.

2. Test whether the same reasoning network can transfer to different base models or different embedding backbones. Since the method claims modularity without parameter modification, cross-model transfer is a necessary next step.

3. Study adaptive latent budgeting instead of fixed-length latent trajectories. A fixed 256-token latent length looks ad hoc; ICLR reviewers would expect a mechanism that allocates more or fewer latent tokens based on problem difficulty.

4. Compare against hybrid schemes that use latent reasoning only when the model is uncertain, and direct answers otherwise. This would test whether LRT is practical as a deployable reasoning controller rather than just a new full-time inference mode.

# Final Consolidated Review
## Summary
This paper proposes Latent Reasoning Tuning (LRT), which replaces explicit token-by-token reasoning trajectories with a learned latent trajectory produced by an auxiliary reasoning network, while keeping the base reasoning LLM frozen. The paper is motivated by an empirical observation that reasoning models remain fairly robust even when parts of their chain-of-thought are randomly removed, and it claims improved efficiency/performance over a small set of efficient-reasoning baselines and Qwen3 non-thinking mode.

## Strengths
- The paper targets a timely and important problem: reasoning efficiency in slow-thinking LLMs, where inference cost is dominated by long explicit trajectories.
- The core empirical observation in Section 2 is useful: DeepSeek-R1-Distill-Qwen-7B retains surprisingly strong accuracy even under substantial token- or step-level skipping, suggesting that explicit trajectories contain a lot of redundancy.
- The framework is modular: the base LLM is frozen and the reasoning function is moved into a separate auxiliary network, which is a practical design choice and makes the system easier to slot into an existing reasoning model than full retraining.

## Weaknesses
- The novelty is only moderate. The paper is clearly related to prior latent-reasoning / continuous-CoT work, and the main contribution is an auxiliary network that produces a latent trajectory for a frozen LLM rather than a fundamentally new reasoning principle. The positioning against Coconut-like and other latent-thought methods is not sharp enough to make the advance feel decisive.
- The central causal claim is overstated. The fragmentation experiment shows redundancy and robustness to degradation, but it does not establish that explicit reasoning is unnecessary in general, nor that the learned latent trajectory is the uniquely right replacement. The paper stretches this empirical observation into a stronger theory than the evidence supports.
- The method description leaves important technical ambiguity. In particular, how the latent vectors are injected into the frozen base model, how the RL objective is implemented over the auxiliary module, and how the learned latent sequence interacts with the model’s hidden states are not fully specified in a way that makes the system straightforward to reproduce or verify.
- The efficiency story is incomplete. The paper emphasizes latency and throughput gains, but the auxiliary reasoning network is not free; a proper end-to-end compute comparison, including FLOPs and wall-clock cost for the full pipeline, is missing. As a result, the “more efficient reasoning” claim is only partially substantiated.
- The evaluation does not yet isolate the mechanism well enough. There is no strong ablation showing whether the gains come from latent reasoning itself versus just another learned conditioner or soft prompt-like controller, and the comparison set is not broad enough to convincingly rule out simpler alternatives.

## Nice-to-Haves
- Stronger controlled experiments on trajectory removal, beyond random skipping, would make the motivation more convincing.
- A clearer per-task and per-budget Pareto analysis of accuracy versus compute would improve the efficiency claim.
- More mechanistic probing of the latent space would help show whether the learned vectors encode reasoning structure or mostly compressed task priors.

## Novel Insights
The most interesting aspect of the paper is not that it “eliminates” reasoning, but that it reframes reasoning as a compact conditioning problem: the auxiliary network learns a latent trajectory that appears to preserve enough information for the frozen base model to answer correctly. This suggests that a substantial portion of slow-thinking computation may be redundant or compressible, but the current evidence more strongly supports the idea of learned compression than a full replacement for explicit reasoning. In other words, the paper’s best insight is that reasoning efficiency may be improved by learning a task-conditioned latent interface, not by assuming chain-of-thought is universally dispensable.

## Potentially Missed Related Work
- Coconut / continuous latent CoT / latent thought methods — directly relevant because they study reasoning in latent space and are closer than token-compression baselines.
- Recurrent depth / continuous reasoning approaches — relevant because they also replace explicit step-by-step decoding with latent iterative computation.
- Soft-prompt / prefix-tuning style conditioning — relevant as a simpler alternative mechanism that could explain some of the gains.

## Suggestions
- Add a direct ablation against a matched direct-answer or soft-prompt baseline using the same frozen backbone, data, and compute budget.
- Include a stronger latent-reasoning baseline comparison set on the same model family and budgets, especially Coconut-like and other continuous-thought methods.
- Report end-to-end compute accounting for the full LRT system, not just output latency, so the efficiency claim is fully grounded.

# Actual Human Scores
Individual reviewer scores: [4.0, 4.0, 8.0, 4.0]
Average score: 5.0
Binary outcome: Accept
