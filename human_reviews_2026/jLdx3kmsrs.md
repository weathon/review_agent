# Think-at-Hard: Selective Latent Iterations to Improve Reasoning Language Models

- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Improving reasoning capabilities of Large Language Models (LLMs), especially under parameter constraints, is crucial for real-world applications. Prior work proposes recurrent transformers, which allocate a fixed number of extra iterations per token to improve generation quality. After the first, standard forward pass, instead of verbalization, last-layer hidden states are fed back as inputs for additional iterations to refine token predictions. Yet we identify a latent overthinking phenomenon: easy token predictions that are already correct after the first pass are sometimes revised into errors in additional iterations. To address this, we propose Think-at-Hard (TaH), a dynamic latent thinking method that iterates deeper only at hard tokens. It employs a lightweight neural decider to trigger latent iterations only at tokens that are likely incorrect after the standard forward pass. During latent iterations, Low-Rank Adaptation (LoRA) modules shift the LLM objective from general next-token prediction to focused hard-token refinement. We further introduce a duo-causal attention mechanism that extends attention from the token sequence dimension to an additional iteration depth dimension. This enables cross-iteration information flow while maintaining full sequential parallelism. Experiments show that TaH boosts LLM reasoning performance across five challenging benchmarks while maintaining the same parameter count. Compared with baselines that iterate twice for all output tokens, TaH delivers 8.1-11.3% accuracy gains while exempting 94% of tokens from the second iteration. Against strong single-iteration Qwen3 models finetuned with the same data, it also delivers 4.0-5.0% accuracy gains. When allowing less than 3% additional parameters from LoRA and the iteration decider, the gains increase to 8.5-12.6% and 5.3-5.4%, respectively.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a model architecture and training method that improve upon prior recurrent Transformer designs. It first identifies a key limitation of uniform latent iteration—processing all tokens indiscriminately—which can degrade already-correct predictions. To address this, it introduces a learned router that selectively re-encodes only difficult tokens, reducing redundant computation and preventing overthinking. In addition, the paper presents duo-causal attention, enabling tokens to attend not only to earlier positions but also to representations from shallower iteration depths, promoting cross-depth information flow.

The training is conducted in two stages: first, the model is trained under an oracle policy that uses ground-truth token difficulty (from an oracle language model) to supervise selective iteration; then, a lightweight decider network is trained to imitate this oracle routing policy. This staged approach stabilizes optimization and ensures that the backbone learns useful latent refinements before the router is introduced.

Experiments on multiple reasoning benchmarks (GSM8K, MATH500, AMC23, AIME25, OlympiadBench) demonstrate consistent gains over standard and uniform recurrent baselines, with only ~6% of tokens undergoing extra iterations. Ablation studies show that LoRA for subsequent iteration models, residues and duo-attention all contribute to the final performance.

### Strengths
- The paper presents a clear and well-motivated improvement to recurrent Transformer models by introducing selective latent iteration through a learned router, addressing the issue of “latent overthinking” in prior methods.
- The proposed two-stage training scheme (oracle-guided backbone training followed by decider imitation) is well designed and empirically effective, leading to stable optimization and efficient use of computation.
- Experimental results are comprehensive across multiple reasoning benchmarks, showing consistent gains with minimal extra compute and parameter overhead.

### Weaknesses
To tackle the overthinking by token selectivity, the paper introduces: (i) duo-causal attention; (ii) router and a corresponding training paradigm.

1. The two designs are largely orthogonal — duo-causal attention can function independently without token eviction and itself induces a form of soft selectivity through attention scores, while the router performs hard selection. It can be seen as a way to achieve sparsity and computational efficiency for duo-casual attention. Given that both introduce additional compute, memory, and operational complexity, their relationship and potential redundancy should be analyzed more thoroughly.

2. The rationale for adopting duo-causal attention, rather than retaining the simpler scheme of attending only to previous tokens within the same depth as in prior recurrent Transformers, is not well justified. Since subsequent iterations already employ LoRA, the existing attention formulation could potentially suffice without introducing a new structure.

The ablation study only contrasts full duo-causal attention with a degenerate variant that attends solely to the first layer, omitting intermediate alternatives. The reported results show that routing alone achieves performance comparable to soft-thinking, whereas duo-attention brings most of the additional gains. This leaves unclear what the standalone contribution of duo-causal attention is and whether its benefits justify the added complexity.

Besides, it is beneficial to include a brief explanation on how the 2d tokens are flatten into 1d order and how positional encoding is applied in the main text.

### Questions
1. How are these two methods related given that duo-attention itself provides a form of soft selectivity? What is the standalone contribution of duo-causal attention without the router?
2. What is the specific rationale for introducing duo-causal attention instead of retaining the simpler causal attention over previous tokens within the same depth, as in prior recurrent Transformers? Given that later iterations use LoRA adaptation, could similar improvements be achieved without adding duo-attention?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes Think-at-Hard (TaH), a selective latent iteration method to address "latent overthinking" in Large Language Models (LLMs)—a flaw where uniform extra iterations corrupt correct predictions of easy tokens. TaH uses a neural decider to identify "hard tokens" (mispredicted in the first pass) for targeted latent refinement, with key designs including duo-causal attention (cross-iteration context flow), depth-specific LoRA adapters (objective shift support), and a two-stage training scheme (oracle-aligned backbone/decider training). Experiments on 5 mathematical reasoning benchmarks (GSM8K, MATH500, etc.) with Qwen3-0.6B/1.7B show 4.0–5.4% accuracy gains with <3% extra parameters, outperforming uniform baselines like AlwaysThink. However, the work suffers from weak baselines, limited model size validation, narrow domain coverage, and incomplete efficiency analysis—undermining its validity.

### Strengths
* The paper clearly defines "latent overthinking" (uniform iterations revising correct easy-token predictions) as a critical gap in prior latent reasoning methods (e.g., AlwaysThink). This insight directly addresses inefficiencies in existing designs and provides a clear motivation for selective iteration .
* The two-stage training scheme (backbone optimization via oracle policy first, then decider imitation on frozen backbone) decouples the tightly coupled backbone and decider, resolving distribution shift issues and achieving faster convergence (lower validation perplexity than Standard) .

### Weaknesses
1. TaH’s comparisons are limited to underpowered baselines that do not represent current state-of-the-art (SOTA) dynamic computation methods. The core baselines—Standard (vanilla Qwen3), AlwaysThink (uniform iterations), SoftThink (logits-weighted latent iterations), and simple routing (between small models)—lack comparisons with stronger alternatives: (1) MoR (Bae et al., 2025), a concurrent selective recursion method, is only claimed to require "complete retraining" without direct performance/efficiency comparisons on the same Qwen3 backbone or datasets ; (2) SOTA layer-skipping methods (e.g., MoD, FlexiDepth) are mentioned but not evaluated, leaving unclear how TaH outperforms dynamic compute methods beyond latent iteration ; (3) even when comparing with Ponder (a latent reasoning baseline), the paper uses a weak PonderingPythia-1.4B backbone (which scores only 2.0% on MATH500, far below Qwen3-0.6B’s 47.2%), making the comparison uninformative . These weak baselines mean TaH’s gains cannot be contextualized against real-world alternatives.
2. The paper only evaluates TaH on Qwen3-0.6B and 1.7B—two small-scale models (≤2B parameters)—with no validation on medium/large LLMs (e.g., 7B, 13B, 34B). This limitation is critical because: (1) small models have low performance ceilings; the 4.0–5.4% gains may stem from fixing basic prediction errors (e.g., logical connectives) rather than improving complex reasoning, which is the core goal of latent iteration ; (2) larger LLMs often exhibit different reasoning behaviors (e.g., more stable CoT generation) and may not suffer from "latent overthinking" to the same degree—TaH’s effectiveness on these models (where efficiency gains matter more for deployment) remains unproven ; (3) the paper highlights edge deployment (a key use case for small models) but does not test TaH on even smaller models (e.g., 360M MobileLLM-R1), limiting insights into its scalability for resource-constrained scenarios .
3. All core experiments are restricted to mathematical reasoning (GSM8K, MATH500, AMC23, etc.), with only a single, underdeveloped cross-domain test: a 1.7B TaH model trained on Open-R1’s science subset and evaluated on GPQA-diamond (a graduate-level Q&A benchmark). This experiment provides minimal evidence of domain agnosticism—non-mathematical tasks (e.g., code debugging, logic puzzles, scientific problem-solving) have distinct hard-token patterns (e.g., syntax vs. mathematical operators), and TaH’s decider may fail to identify hard tokens in these domains . Without broader validation, the claim that TaH is "domain-agnostic" is unsupported.
4. TaH’s oracle policy (used to label hard tokens) relies on a frozen SFT model’s top-1 prediction mismatch, but the paper never validates this labeling’s reliability: (1) it assumes the SFT model’s predictions are accurate enough to distinguish easy/hard tokens, but SFT models may misclassify easy tokens (e.g., simple coherence tokens like "the"), leading to unnecessary iterations or missed refinements ; (2) no comparison is made with alternative hardness metrics (e.g., token entropy, prediction cross-entropy), which are widely used in prior work—this leaves uncertainty about whether the oracle’s labels are optimal or consistent .

These weaknesses—especially weak baselines, small model size limits, and narrow domain validation—mean TaH’s contributions are not yet sufficiently robust for acceptance. The work would need to (1) compare with SOTA dynamic compute baselines, (2) validate on medium/large models, (3) expand to non-mathematical tasks, and (4) address oracle reliability and memory overhead to strengthen its case.

### Questions
See the weakness part.

### Soundness
2

### Presentation
3

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
This paper introduces Think-at-Hard (TaH), a method that selectively applies latent iterations only to hard tokens in language model reasoning tasks, rather than applying uniform multiple iterations to all tokens as in prior approaches. TaH applies LoRA modules and residual connections to the LLM backbone to better facilitate such objective shift. It also design a duo-causal attention mechanism for cross-depth contextualization. Experiments on five math-focused benchmarks (GSM8K, MATH500, AMC23, AIME25, OlympiadBench) show the effectiveness of the proposed TaH.

### Strengths
1. The idea that from latent overthinking to think-at-hard is reasonable, and the experiments show the good improvement.
2. The paper is well-written, making it understandable and easy to follow. It has a well-organized structure, and the proposed method is presented clearly.
3. The figures are well constructed, visually appealing, and help to understand the proposed methodology.

### Weaknesses
1.The core mechanism depends on an oracle policy π derived from a “frozen reference LLM” (the SFT variant of the base model) to label “hard” tokens (Eq. 7). This raises two problems: 1) Robustness: the “hardness” signal is tightly coupled to a particular reference model and training distribution; label quality and transfer to new domains are unclear. 2) Generalization: Main results are math-dominated; outside of a brief science subset note in the appendix, there is limited main-paper evidence that the oracle-imitation decider works for non-math reasoning (commonsense, code, multimodal). The method’s central promise that token-wise adaptive compute needs broader validation to justify venue-level impact.

2. In table 1, TaH surpass TaH+ (unpruned) in some benchmark (0.6B: AMC23, 1.7B: MATH500, AIME25). So it is hard to isolate whether gains come from TaH versus architectural changes in depth/capacity.

3. In Table 2, removing LoRA sometimes improves a task (MATH500 51.6 vs 51.2), and overall benefits fluctuate per dataset; this weakens the case that depth-specific adapters are consistently necessary.

### Questions
1. Could you report main-paper results on at least one non-math benchmark (e.g., GPQA-diamond in the main tables, CommonsenseQA, code tasks) with the same training setup, including ablations?
2. Could you provide a Standard-pruned baseline with the same depth as TaH (after pruning) to isolate TaH’s effect?
3. Please report latency/throughput/memory under realistic batching for Standard vs. AlwaysThink vs. TaH on the same hardware; include cache behavior with duo-causal masking.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes TaH, a selective latent iteration framework that applies deeper internal reasoning only to $hard$ tokens, which are initially incorrectly predicted by a reference model, and directly generating $easy$ tokens. The method introduces a duo-causal attention mechanism to enable cross-iteration information flow while preserving training parallelism. It also uses LoRA adapters for refinement in deeper iterations. For training, it employs a two-stage training scheme: first training the backbone under a static oracle policy derived from a supervised fine-tuned (SFT) reference model, then training a lightweight neural decider to imitate this policy. Experiments on five reasoning benchmarks show consistent improvements over strong baselines with minimal computational overhead.

### Strengths
1. The paper is well-motivated. It identifies the issue of latent overthinking, and proposes a selective method for solution.
2. The duo-causal attention mechanism is proposed to enable depth-wise information flow. The two-stage decoupled training is also reasonable.
3. The authors conduct extensive studies and validate the method’s performance.
4. The paper is overall well-written and easy to follow.

### Weaknesses
Though the paper is overall reasonable to me, there may exist several potential weaknesses.
1. The proposed method seems to heavily rely on the supervised oracle policy. The oracle policy depends on ground-truth tokens and a frozen SFT reference model, limiting applicability to settings without high-quality supervision. It also remains unclear how TaH would perform if the reference model itself is biased or inaccurate on certain hard examples.
2. The experiments are mostly conducted on mathematical reasoning. The method’s effectiveness on more diverse tasks, such as commonsense reasoning, dialogue, or code generation, where $hardness$ is less local and more context-dependent, is not demonstrated.
3. There may exist potential robustness issue in the decider. The decider is trained in a specific data domain. It may produce less effective behavior on out-of-distribution or adversarial inputs. A single misclassification may permanently forfeit refinement opportunity, unlike AlwaysThink which provides a safety strategy.

### Questions
Please see strengths and weaknesses. 

Besides, it would be beneficial to further clarify how the approach differs from other adaptive-computation methods in the latent reasoning literature, as a discussion in the paper.

### Soundness
3

### Presentation
3

### Contribution
3
