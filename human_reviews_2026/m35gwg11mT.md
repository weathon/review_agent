# Boosting Large Language Models with Mask Fine-Tuning

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
The large language model (LLM) is usually kept integral in the mainstream optimization protocol. No works have questioned whether maintaining the integrity of the model is indispensable for promising performance. In this work, we introduce Mask Fine-Tuning (MFT), a brand-new LLM fine-tuning paradigm to show that properly breaking the structural integrity of the model can surprisingly lead to improved performance without model weights update. Specifically, MFT learns and applies a set of binary masks on well-optimized models supervised by the typical LLM fine-tuning objective. Based on full fine-tuned models, MFT uses the same fine-tuning datasets to gain consistent performance boosts across various domains and backbones (e.g., 2.60 / 4.15 average gain in IFEval with LLaMA2-7B / 3.1-8B). Detailed ablations and analyses study the proposed MFT from different perspectives such as sparse ratio, loss surface, etc. Additionally, MFT is compatible for collaborating with other LLM optimization procedures for general model enhancement by deploying it on well-trained models. Further, this study extends the functionality of masking operation from its conventional network pruning context for model compression into a general model capability scope.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper points out that existing fine-tuning methods, such as full-model fine-tuning and LoRA, often suffer from issues like overfitting, leading to performance degradation after fine-tuning. This paper proposes a Mask Fine-Tuning (MFT) method, which further improves the model's performance by fine-tuning a binary matrix after the initial model fine-tuning.

### Strengths
1. This paper is well-written, with clear logic, and it fully conveys the motivation and methodology of the research.

2. The paper conducts extensive experiments, including comparative experiments on data of different types and scales.

### Weaknesses
1. First, the paper points out that performing MFT fine-tuning based on "best fine-tuning" can improve model performance. However, challenges such as identifying the "best time point," along with the additional computational costs and training data required, hinder the practical application of MFT and increase the overall cost of fine-tuning.

2. Second, MFT is rather heuristic in nature. Currently, we still lack clarity on the rationality of applying MFT after "best fine-tuning" and the true reasons behind the resulting model performance improvement. The key factors that influence performance enhancement remain unknown, and it is worth exploring whether an analysis can be conducted from a theoretical or fundamental perspective.

3. Finally, there is the issue of MFT’s generalizability. The models used in existing experiments are limited to those with a scale of less than 8B parameters, and it is questionable whether the experimental conclusions are valid for larger-scale models.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes **Mask Fine-Tuning (MFT)** — a simple post-training step applied *after* standard supervised fine-tuning (SFT or FFT) of large language models.
During MFT, the model parameters $W$ are **frozen**, and a binary mask $M \in {0,1}$ is learned using the **straight-through estimator (STE)**.
The effective weights are obtained by element-wise multiplication:
$$
\tilde{W} = W \odot M
$$

Through experiments on **LLaMA2-7B** and **LLaMA3.1-8B**, the authors show consistent gains across instruction-following (IF-Eval), math (GSM8K), and code (HumanEval) benchmarks — typically $+2$–$6$ points compared to the best fine-tuned baseline.
Ablations reveal that shallow and late layers benefit most from masking, and visualizations indicate flatter loss landscapes with lower PAC-Bayes bounds, suggesting improved generalization.

### Strengths
* **Elegant simplicity and practicality**
  The method is conceptually minimal — no new modules or objectives, just a learnable binary mask applied post-SFT. It can be easily integrated into existing fine-tuning pipelines.

* **Systematic ablations**
  Layer-wise masking, masking ratio, data ratio, and local vs. global masking are all explored. These ablations reveal that masking early and late layers yields the largest gains.

### Weaknesses
1. **Limited generality — only LLaMA-based experiments**
   All experiments are performed exclusively on **LLaMA2-7B** and **LLaMA3.1-8B**.
   No evidence is provided that the method generalizes to other architectures such as Mistral, Falcon, GPT-NeoX, or encoder–decoder models.
   The results might exploit architectural features unique to LLaMA (e.g., SwiGLU gating, RMSNorm, rotary embeddings).

2. **Mechanistic opacity**
   The paper does not clearly explain *why* MFT improves performance.

   * Are the masked connections genuinely redundant or overfitted?
   * Does masking act as a form of regularization or noise smoothing?
     The presented PAC-Bayes argument is mathematically valid but does not illuminate the underlying mechanism.

3. **Weak baselines**
   MFT is compared only to standard fine-tuning and LoRA.
   There is no comparison to sparsity-based fine-tuning techniques such as Movement Pruning, Diff-Pruning, $L_0$-regularization, or Sparse-FT.
   Without these, the novelty relative to prior sparsity literature remains unclear.

4. **Lack of statistical robustness**
   All results are single-seed. Small benchmarks like HumanEval or IF-Eval require multiple seeds or confidence intervals (e.g., bootstrap estimates) to validate significance.

5. **Interpretability missing**
   There is no visualization or qualitative analysis showing *which* neurons or connections are masked, nor how masking alters attention or activation patterns.

### Questions
1. **Cross-architecture validation**
   Apply MFT to other architectures (e.g., Mistral-7B, Falcon-7B, T5-11B) to verify generality beyond the LLaMA family.

2. **Mechanistic analysis**

   * Visualize layer- and head-level mask distributions.
   * Measure changes in activation sparsity, gradient norms, or representational similarity before and after MFT.
   * Distinguish between pruning-like and regularization-like behavior.

3. **Add stronger baselines**
   Include comparisons to Movement Pruning, Diff-Pruning, and $L_0$-masking under equal compute budgets.

4. **Improve statistical reliability**
   Report mean ± std over multiple seeds, and possibly provide 95% confidence intervals for HumanEval / GSM8K.

5. **Integrated training variant (future work)**
   Explore alternating SFT and MFT epochs (e.g., epoch 1 full-SFT, epoch 2 MFT),
   or a joint objective:
   $$
   \mathcal{L}*{\text{joint}} = \mathcal{L}*{\text{SFT}}(W,M) + \lambda |M - 1|_1
   $$
   to treat masking as a regularization process *during* fine-tuning rather than as a post-hoc step.

6. **Relation to gating, dropout, and LoRA**
   The authors should explicitly situate MFT within this broader landscape:

   * **Dropout vs. MFT:** both apply multiplicative masks $r$ or $M$, but dropout uses random Bernoulli masks ($r_{ij}!\sim!\text{Bernoulli}(p)$) for stochastic regularization during training, whereas MFT learns a *deterministic* binary mask that persists at inference. Hence MFT can be viewed as a “learned deterministic dropout.”
   * **Gating vs. MFT:** gating mechanisms (e.g., SwiGLU) use *continuous* gates., MFT, in contrast, enforces *hard* selection on edges or neurons, producing structural sparsity rather than soft modulation.
   * **LoRA vs. MFT:** LoRA modifies the parameter space additively ($W' = W + BA$), introducing low-rank updates that expand the representational subspace. MFT modifies it multiplicatively ($W' = W \odot M$), effectively contracting the subspace by removing redundant connections. Interestingly, a well-trained LoRA could partially *cancel* certain weight directions ($BA\approx -W_{\text{unwanted}}$), producing a masking-like effect. An explicit comparative experiment—measuring cosine similarity between LoRA updates and MFT masks—would clarify whether both methods converge toward complementary adaptation patterns.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Mask Fine-Tuning, a post-training method that improves fully fine-tuned LLMs by learning which parameters to remove rather than updating weights. The key claim is counterintuitive: you can improve a well-trained model by carefully masking out 10% of its parameters in specific layers.  Main results show modest but consistent improvements over the best FFT checkpoint - typically 0.3-6 points depending on the task.  The paper discovers that you can improve models by carefully removing parameters but doesn't explain why this helps, when it will work, or how it differs meaningfully from existing pruning methods. The practical utility is unclear given the modest gains, manual tuning requirements, and cross-domain degradation.

### Strengths
1. The finding that you can improve a converged model by removing parameters (not updating them) challenges common assumptions about neural network training. This is worth investigating.

2. Experimental scope is reasonable, include Two model families (LLaMA2, LLaMA3.1); Three diverse domains (math, code, instruction-following); Multiple training scenarios (domain-specific FFT, mixed-domain FFT)

4. Includes both local and global masking experiments. The global masking results (Table 3) are mostly negative, but including them shows intellectual honesty about where the method fails.

5. Figure 5 suggests MFT moves models to flatter minima, which aligns with better generalization. The visualization is clear and the trend is consistent across domains.

6. Training cost analysis is included. Figure 4 breaks down memory, tokens, and time, making it clear what overhead MFT adds beyond FFT.

### Weaknesses
1. The improvements are small and sometimes within noise

Looking at Tables 1-2 with error bars: many gains are 0.3-2 points, and standard deviations often overlap between Best FFT and MFT. For example, LLaMA3.1-8B Math domain shows 77.0±0.88 vs 77.3±0.97 - not convincing. No statistical significance tests are provided to confirm these differences are real.

2. **The distinction from pruning is unconvincing**

The paper claims to differ from pruning because the goal is "improvement not compression," but technically it's doing the same thing - learning which parameters to remove using training data and gradients. Modern pruning methods like Wanda or SparseGPT also aim to maintain or improve performance while reducing parameters. The conceptual distinction feels forced.

More damaging: the paper doesn't compare against any actual pruning methods. The baselines are just random masking and L1 magnitude masking (which doesn't even use training). Where's the comparison to gradient-based pruning, lottery tickets, or recent LLM pruning work?

3. Cross-domain results reveal a problem

Tables 5-6 show that MFT often hurts performance on non-target domains. For instance, training MFT on math improves GSM8K but degrades HumanEval. This suggests the method may be overfitting to the target domain rather than genuinely improving the model. This contradicts the generalization improvement narrative.

### Questions
Q1: Is this just preventing overfitting through capacity reduction? The paper shows continued FFT hurts performance (overfitting) but MFT helps. The obvious explanation: MFT reduces capacity, making overfitting harder. But that's not really "improvement" - it's just better regularization than doing nothing. How does MFT compare to :(1) Continued FFT with dropout (2) Continued FFT with stronger weight decay. Including these comparisons would make it clearer whether MFT offers unique benefits beyond standard regularization approaches.

Q2: Why these specific layers? Figure 3 shows different layers work for different settings. What determines this? Is there something about these layers' representations? Their gradient statistics? Their weight magnitudes? The paper identifies which layers work but not *why*, making it hard to apply the method to new models.

Q3: Continued FFT comparison seems unfair. Continued FFT is trained for the full 4 epochs and evaluated at the end (showing degradation). But MFT can choose its best checkpoint within 2 epochs. Wouldn't continued FFT also improve if you picked its best checkpoint from the same epoch range?

Q4: Why does masking ratio vary by domain? Figure 6 shows coding prefers 10% but instruction following prefers lower ratios. What property of the domain determines this? Task complexity? Dataset size? Base model capability?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Mask Fine-Tuning (MFT), a novel post-fine-tuning approach that learns binary masks on already fine-tuned LLMs to further improve performance. The key insight is that removing certain parameters through learned masks can enhance model capability rather than merely maintaining it. The authors validate MFT on LLaMA2-7B and LLaMA3.1-8B across three domains (math, coding, and instruction-following), showing consistent improvements over fully fine-tuned baselines. The method freezes model weights and only learns which parameters to mask out, using the same training objective and datasets as standard fine-tuning. Theoretical analysis via PAC-Bayes bounds and empirical loss landscape visualizations are provided to support the approach.

### Strengths
1. A good perspective on model sparsity: The paper presents an interesting conceptual shift by using masking not for compression but for capability enhancement. This counter-intuitive finding that "subtraction leads to addition" is thought-provoking and extends the conventional understanding of sparse networks beyond efficiency concerns.
2. Comprehensive experimental validation: The authors conduct thorough ablations including layer-wise sensitivity analysis (Figure 3), masking ratio studies (Figure 6), and data ratio experiments (Figure 7). The proof-of-concept studies systematically identify which model components benefit most from MFT.
3. Theoretical grounding: The inclusion of PAC-Bayes generalization bounds (Section 3.3) and Hessian-based loss landscape analysis provides theoretical justification beyond empirical results. The analysis showing that both training loss and model complexity terms decrease is valuable.

### Weaknesses
1. **Limited task complexity and diversity:** The evaluation focuses on relatively standard benchmarks (GSM8K, HumanEval, IFEval) that may not fully demonstrate the method's effectiveness on more challenging or specialized tasks. The paper would benefit from:
More complex reasoning tasks (e.g., multi-hop reasoning, mathematical proof generation)
Domain-specific applications (legal document analysis, medical diagnosis, scientific literature understanding)
Longer-context tasks that stress different model capabilities

2. **Marginal performance gains:** While consistent, the improvements are often modest:
Many gains are within 1-3 points, raising questions about practical significance
Error bars overlap in several cases, suggesting some improvements may not be statistically significant
No discussion of whether these gains justify the additional training phase

### Questions
1. **Generalization to modern training paradigms:** Can you provide any preliminary results or theoretical analysis on how MFT would work with DPO, PPO, or other policy-based training methods? Given the growing importance of RL, this seems critical for practical adoption.


2. **Model diversity experiments:** What prevents extending the evaluation to other model families like Qwen? Are there architectural requirements that limit applicability? Results on at least one additional model family would significantly strengthen the claims.


3. **Efficiency quantification:** What is the exact wall-clock time overhead of MFT compared to continued FFT?

### Soundness
2

### Presentation
2

### Contribution
2
