# Latent Refinement Decoding: Enhancing Diffusion-Based Language Models by Refining Belief States

- Avg Score: 6.50
- Decision: Reject
- Scores: 6, 6, 8, 6

## Abstract
Autoregressive (AR) models remain the standard for natural language generation but still suffer from high latency due to strictly sequential decoding. Recent diffusion-inspired approaches, such as LLaDA and Dream, mitigate this by generating in parallel, yet they suffer from two core limitations: information loss, as predictive distributions for non-finalised tokens are discarded at each step, and a lack of well-behaved commitment dynamics, where local decisions are not properly coordinated at the global level. We introduce Latent Refinement Decoding (LRD), a two-stage framework with Latent Refinement and a Predictive Feedback Loop. The first stage maintains masked positions as distributional mixtures of predicted tokens and the mask embedding, allowing the model to establish more globally consistent beliefs. The second stage progressively finalises confident tokens while retaining uncertain ones for iterative feedback. KL-divergence dynamics provide a principled and reliable criterion for convergence and early stopping. Experiments across coding (HumanEval +6.3, MBPP +2.6) and reasoning (GSM8K +2.9, MATH500 +3.8) benchmarks show that LRD improves accuracy while delivering speedups of up to 10.6×. Moreover, LRD is orthogonal to system-level optimisation: when combined with KV-cache and parallel-based accelerators (e.g., Fast-dLLM), it improves accuracy and yields up to 2.4× additional speedup, making it a strong and versatile alternative for parallel sequence generation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Latent Refinement Decoding (LRD), a two-stage decoding framework for diffusion-based language models that addresses information loss from hard masking and inefficient convergence. The method operates by: (1) Phase 1 - maintaining masked positions as distributional mixtures of predicted tokens and mask embeddings in continuous space (Latent Refinement), and (2) Phase 2 - progressively finalizing confident tokens while retaining uncertain ones as soft embeddings (Predictive Feedback Loop). KL-divergence is used to monitor convergence and enable adaptive early stopping. Experiments on LLaDA and Dream models across coding (HumanEval, MBPP) and reasoning (GSM8K, Math500) benchmarks demonstrate accuracy improvements of 0.7-6.3 points with speedups up to 10.6×.

### Strengths
1. Identifies specific failure modes of hard masking (information loss, infinite KL divergence under misprediction) with mathematical justification
2. KL-divergence monitoring provides an adaptive, task-dependent stopping criterion rather than fixed iteration counts
3. 5 model variants × 4 benchmarks × 3 sequence lengths = 60 experimental configurations with consistent improvements
4. Tables 2-4 and Figures 4-5 systematically validate each component's contribution

### Weaknesses
1. Only evaluated on two diffusion LM families (LLaDA, Dream). Generalization to other discrete diffusion approaches (e.g., score-based methods, D3PM) is unclear.
2. While HumanEval shows strong gains, improvements on GSM8K and MATH500 are sometimes quite small (e.g., +0.7 points).
3. Related work mentions dLLM-Cache, Fast-dLLM, Prophet, etc., but no direct comparisons are provided.

### Questions
1. How should practitioners set hyperparameters for new tasks without grid search?
2. How does LRD compare directly with dLLM-Cache, Fast-dLLM, or Prophet? Can they be combined?
3. Table 4 shows removing latent refinement sometimes helps, when is Phase 1 actually beneficial?
4. How does the method scale beyond 1024 tokens?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Latent Refinement Decoding (LRD) for diffusion-based language models (dLLMs). LRD replaces purely hard masking/commitment with a two-phase process: (i) a latent refinement phase that mixes the [MASK] embedding with an entropy-normalized expectation of top-p token embeddings (carrying forward uncertainty rather than discarding it), followed by (ii) a predictive feedback phase that progressively commits low-entropy positions while keeping uncertain positions soft, with KL-divergence dynamics used to trigger phase transition and early stopping. On GSM8K, MATH500, MBPP, and HumanEval, LRD yields consistent accuracy improvements and notable speedups.

### Strengths
- Clear, intuitive motivation. The paper correctly identifies two practical issues in current discrete diffusion decoding—information loss from hard remasking and premature commitment—and directly addresses both with a principled soft-to-hard schedule. The mechanics of mixing token and mask embeddings are described crisply and tied to entropy. 
- Method design is coherent. The KL-based stability monitor is a sensible criterion both for the phase switch and for early stopping; it connects the algorithm’s control flow to a measurable distributional convergence signal.
- Empirical results are broad and consistent. Gains appear across two families (Dream-7B, LLaDA-8B/1.5), multiple lengths (256–1024), and both reasoning and coding. The tables show +1–6 points accuracy and sizable speedups, especially at long contexts.
- Ablations are thoughtful. The paper isolates the contribution of hot-start refinement, mixed embeddings, and early stopping; it also explores nucleus size/top-p and max mix ratio. This helps validate that mixing drives much of the gain, while early stopping drives most speedup.

### Weaknesses
- Limited baselines beyond “standard” discrete diffusion decoding. While the paper thoroughly compares to its own “hard” baseline, I’d like to see head-to-head against stronger decoding/selection variants for dLLMs under identical compute budgets. Some are discussed in related work, but the empirical comparison is not fully exhaustive.
- Scope: reasoning/coding only. Results are on GSM8K, MATH500, HumanEval, MBPP. It would be helpful to see a general-purpose generation benchmark (e.g., long-form QA, summarization) to test whether the latent mixing compromises stylistic fidelity or introduces artifacts in open-ended text.
- Theoretical story is partial. The KL-monitor is motivated (and the hard-mask KL blow-up is a nice point), but the paper does not provide deeper convergence guarantees for the two-phase schedule beyond a qualitative justification. A simplified setting with a contraction argument or fixed-point analysis would strengthen the case.
- Cost accounting and fairness. Reported “Speed” is relative runtime (baseline = 1.0×), but details of per-step FLOPs and wall-clock contributions of KL computation, extra forward passes in latent mixing, and cache behaviors (if any) are not fully disentangled. A standardized tokens-per-second at equal quality or quality-at-fixed-time curve would make the speed claims crisper.

### Questions
- How does LRD interact with KV caching or block-reuse schemes for dLLMs? Does the latent mixing impede cache reusability, or can you cache attention over mixed embeddings safely?

- Can you show quality-vs-time Pareto curves across decoding policies to confirm LRD’s advantage at matched budgets?

- For the semi-AR block setting, do beliefs transfer between blocks (e.g., previous block’s soft state informing the next), or is each block re-initialized?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work introduces Latent Refinement Decoding (LRD), a two-stage framework that improves diffusion language model speed and generative quality on coding and math. 

In Phase 1: Latent Refinement, [MASK] token embeddings are mixed with the embeddings of the top-p predicted tokens until distributional shift decreases below a threshold. 

In Phase 2: Predictive Feedback Loop, standard low entropy denoising is applied to decode lowest entropy tokens, while mixing other tokens.

The Kullback-Liebler (KL) Divergence between iterative predictive steps is used to determine which tokens are finally decoded, enabling an early stopping mechanism for sampling.

Additional ablations are performed on variants of the decoding framework, revealing the importance of both stages of the framework as well as demonstrating the decoding speed improvement from early stopping.

### Strengths
- The presentation of this paper is very clear and easy to read
- The proposed method demonstrates improvements in both speed and generative performance on math and coding
- The figures give insight into method complexity and hyperparameter choices

### Weaknesses
- The evaluation is limited to two 7-8B parameter models and code and math, leaving models of different sizes and other natural language tasks unexamined
- The KL divergence is not compared to other statistical distances for monitoring saturation

### Questions
- Did you explore any other distances besides KL divergence?
- Instead of decoding the single lowest entropy token at each step, how is the performance of decoding $k$ tokens at a time in phase 2, such as if $i \in \textrm{top-k}(\\{H_t^{(j)}\\}_{j=1}^L)$?
- In Table 4 $E_\textrm{token}$, is the color-coding inverted? (Lower effective token count should be better/green)

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper considers the diffusion-based LMs and proposes an approach to speed up dLLM decoding. The proposed Latent Refinement Decoding (LRD) approach aims to address two core limitations of previous dLLM decoding methods (e.g., LLaDa, Dream), namely, the information loss and the lack of well-behaved commitment dynamics. In particular, the method contains two stages, where the first stage maintains mixtures of predicted tokens and mask embedding and the second stage progressively finalizes confident tokens. Experimental results demonstrate the effectiveness and efficiency of the proposed LRD approach.

### Strengths
Overall, the paper is well-written. The material is presented in a clear and organized way. The method specifications, experimental results, ablation studies are informative. The method itself is well-motivated, with improvements in terms of effectiveness and efficiency over previous approaches.

### Weaknesses
The paper can further benefit from clarifications on following points (detailed in "Questions" section):

1. How does LRD prevent/avoid "mispredictions"?

1. What are hyper-parameter setups that recover previous methods with hard assignments?

1. (Minor issue) It might worth considering how to better phrase the second "core limitation."

### Questions
### 1. How does LRD prevent/avoid "mispredictions"?

LRD maintains the full density mixture over (top-p) token embedding and mask token embedding, and therefore, it is a not a hard masking/assignment. While it is very clear that hard assignment can potentially have an infinite $d_{KL}$ from the true posterior due to the lack of ability to correct for "misspecification" (line 48), it is not entirely clear to me how this soft assignment approach can prevent or avoid this issue. In other word, is there a guarantee (e.g., on the $d_{KL}$) on how close the convergence state is to the true posterior? I understand that a rigorous bound might be nontrivial to derive, but having some discussion (at least providing some intuition) could be very helpful.

### 1. What are hyper-parameter setups that recover previous methods with hard assignments?

Is there a way to use specific values of $(\alpha, \tau_{\text{refine}}, \tau_{\text{decode}})$ to recover previous hard masking/assignment approaches? I can imagine when $\alpha$ is set to 0, the mixture will be de-activated. Having this additional discussion (if feasible) can provide a clearer explanation of the relationship between LRD and previous approaches.

### 1. (Minor issue) It might worth considering how to better phrase the second "core limitation."

In Abstract, the second "core limitation" is "premature commitment." In Introduction, the second limitation is "inefficient convergence dynamics" which may involve premature (when being aggressive) or overdue (when being conservative) commitments. I am wondering if sth like the lack of well-behaved commitment dynamics would be a better fit.

### Soundness
3

### Presentation
3

### Contribution
3
