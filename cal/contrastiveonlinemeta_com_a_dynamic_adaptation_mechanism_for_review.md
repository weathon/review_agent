=== CALIBRATION EXAMPLE 11 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title is descriptive but slightly over-promises. More critically, the abstract contains specific quantitative claims—"3-5× fewer updates than conventional meta-learning approaches" and "12-18% outperforming instruction-tuned baselines on unseen programming languages"—that are unsupported by any results in the paper (see Experiments below). The abstract also contains a garbled sentence ("a framework for instruction-tuned CodeLLMs that *coefficients to the issues* of catastrophic forgetting") that is symptomatic of a broader writing problem throughout.

The final line of the paper's "LLM use" section (Section 8) states: *"We use LLM polish writing based on our original paper."* The quality of the prose suggests this polishing was applied unevenly and without careful human review. Multiple paragraphs contain clearly corrupted or hallucinated text (detailed below), which raises serious questions about the care taken in preparing this submission.

---

### Introduction & Motivation

The problem motivation is reasonable—catastrophic forgetting and noisy feedback during deployment of CodeLLMs is a genuine and important challenge. However, several issues arise:

1. **Logical framing**: The paper presents continual learning and meta-learning as complementary but the introductory framing doesn't make clear *why* they must be combined (as opposed to, say, meta-learning alone). This argument is never rigorously developed.

2. **Claimed contribution specificity**: The third claimed contribution is: *"COM achieves significantly higher robustness than standard fine-tuning when tested on mixed-domain programming tasks, while requiring 3-5× fewer updates than conventional meta-learning approaches (Nichols et al., 2024)."* The cited paper (Nichols et al., 2024) is actually about performance-aligned LLMs for generating fast code—it is neither a meta-learning approach nor a relevant baseline for update-count comparisons. This citation appears to be misattributed.

3. **"No current solution boosts these strengths"**: The claim that *no* prior work combines contrastive learning with online meta-learning for code models is stated without sufficient engagement with the broader literature (e.g., online contrastive meta-learning in NLP beyond code).

---

### Related Work

The related work section is adequate in breadth but thin in depth. References [1]–[9] in the final paragraph of Section 2.3 are cited as numbered placeholders rather than the author-year format used throughout the rest of the paper, suggesting this paragraph was not carefully integrated. More substantively:

- The paper claims to differ from Qin et al. (2023) because their approach uses "static item embeddings" whereas COM handles "dynamic instruction-to-code relationships." This distinction is asserted rather than technically demonstrated.
- The treatment of EWC (Kirkpatrick et al., 2017) and other regularization-based continual learning methods is superficial. The paper dismisses architectural constraints as "computationally expensive and inflexible" without engaging seriously with methods like PackNet, PNN, or more recent parameter-efficient continual learning approaches.
- Ahmad et al. (2025) is cited for "large-scale datasets specifically designed for code instruction tuning" as well as for "meta-learning frameworks enabling efficient few-shot adaptation"—these appear to be two very different things attributed to the same paper, suggesting a citation error.

---

### Method (Section 4) — Most Critical Section

The framework description has several fundamental technical problems:

**1. The meta-learner architecture is underspecified and likely insufficient.** The meta-learner $g_\phi$ is described as a "2-layer MLP" (Section 5.4) that operates on 768-dimensional instruction embeddings. The frozen base model $h_\psi$ is CodeGen-16B. The composition $p(y|x) = h_\psi(g_\phi(f_\theta(x)))$ (Eq. 8) implies that the MLP modifies embeddings that are then passed to the frozen 16B model. This raises a critical question: **how does a 2-layer MLP transforming a 768-dim embedding vector provide meaningful task-specific adaptation to a frozen 16B-parameter autoregressive model?** CodeGen-16B generates code token-by-token conditioned on the full input context. Simply transforming the instruction embedding at the input stage is unlikely to yield the nuanced behavioral changes claimed (e.g., adapting to new APIs, different programming languages). The paper never justifies this architectural choice or provides ablations to verify that this bottleneck does not severely limit expressiveness.

**2. The online adaptation mechanism conflicts with inference efficiency claims.** Equation (5) requires computing $\nabla_\phi \mathcal{L}$ at each inference step. For a streaming deployment scenario, this means backpropagation through the instruction encoder and meta-learner at every user interaction. The paper claims this requires "<5% of the base model's parameters to be trainable," which is technically true, but does not address the wall-clock cost of per-step gradient computation during deployment. No latency figures are provided.

**3. The contrastive loss (Eq. 4) is non-standard in a potentially problematic way.** The denominator of Eq. (4) sums only over explicitly designated *negative* samples $x_k^-$, whereas the standard InfoNCE formulation (Oord et al., 2018) includes *all* samples (positives and negatives) in the partition function. This design choice is not discussed or justified. If negative samples are separately curated rather than drawn from the same batch (including positives), the loss has a different gradient structure and the temperature $\tau = 0.1$ may need different calibration.

**4. Equation (1) is missing.** Section 3.1 reads: *"The standard continual learning objective minimizes the cumulative loss of all the tasks:"* followed by a blank, with Eq. (1) appearing only later (after Eq. 2) as a rendering artifact. More substantively, the paper never returns to explain how this objective is modified or replaced in COM—it is presented as background but never connected to the COM training procedure.

**5. No explanation of positive/negative pair construction.** The quality of contrastive learning depends critically on how positive and negative pairs are defined. The paper states that "positive pairs might include different implementations of the same algorithm" but never specifies how these pairs are constructed for the CodeAlpaca-20k dataset or StreamCode benchmark in practice. This is a reproducibility failure.

**6. The projection loss (Eq. 10) penalizes $\|z_t - z_{t-1}\|^2$ at every step.** This discourages *any* representation change over time—which is in direct tension with the goal of online adaptation. The paper does not discuss how this regularization is balanced against adaptation, nor does it provide ablations on $\lambda$ to show the system is not simply being prevented from adapting at all.

---

### Experiments & Results — Fatal Flaw

**There are no experimental results in this paper.** Section 5 describes the experimental setup (datasets, baselines, metrics, implementation details) in reasonable detail, and then the paper jumps directly to Section 6 (Discussion). No tables, no figures with quantitative results, no statistical comparisons against baselines are presented anywhere in the paper.

This is a fatal deficiency. The abstract and introduction make multiple specific quantitative claims:
- "3-5× fewer updates than conventional meta-learning approaches"
- "12-18% on unseen programming languages"
- "significantly higher robustness than standard fine-tuning"

None of these claims are substantiated with any data. The paper is, in effect, a framework proposal with no empirical validation.

Additional experimental design concerns (assuming results were eventually added):

- **StreamCode** is described as a self-constructed benchmark, but no details of its construction, annotation methodology, or quality control are provided. There is no way to reproduce it.
- **Forgetting Rate (FR)** is defined as $1 - \frac{acc_{after}}{acc_{before}}$, which is undefined when $acc_{before} = 0$ and gives negative values when the model improves on old tasks. The standard backward transfer metric is more robust.
- **Adaptation Accuracy (AA)** is described only as "success rate on newly introduced tasks immediately after adaptation." For code generation, success rate typically means pass@k on unit tests—the paper does not specify k, the test suite source, or whether execution-based evaluation is used.
- The four baselines are reasonable but **no LoRA-based or prefix-tuning continual learning baselines** are included, which would be more natural comparators for parameter-efficient adaptation than full fine-tuning (SFT).
- **No ablations** of any kind are presented or even promised (e.g., removing the memory buffer, removing the contrastive pre-training, or using a learned rather than FIFO buffer policy).

---

### Writing & Clarity

Beyond the missing results, the paper contains multiple passages of clearly corrupted text that impede understanding and raise concerns about quality control:

- Section 4 intro: *"maintain the just minimal programming knowledge in the model, still enabling us to modulate task specific behavior"* and *"programming England's instructions"*
- Section 6.1: *"scope for improvementCivil War, though, in terms of..."* — garbled mid-sentence
- Section 7: *"where Headquarters and reagents of statements and feedback are still pushing and changing"*
- Acknowledgments: The template placeholder text (*"All the acknowledgments such as those to funding agencies go at the end of the paper"*) was never replaced with actual acknowledgment content.

These are not OCR artifacts—they are semantic corruptions suggesting heavy LLM-generation with inadequate human review.

---

### Limitations & Broader Impact

Section 6 discusses limitations (noisy feedback, FIFO buffer, data curation) and ethics (bias amplification) in a superficial but structurally appropriate way. However, the most significant limitation—that the frozen base model + small MLP adapter architecture may fundamentally lack the representational power needed for meaningful adaptation—is not acknowledged. The claim that COM handles "noisy feedback" is stated in the abstract as a solved problem, yet the limitations section lists it as an open challenge, which is contradictory.

---

## Overall Assessment

This paper cannot be accepted in its current state, primarily because **no experimental results are presented**—the entire results section is absent. The abstract and introduction make strong quantitative claims that are entirely unsubstantiated. Beyond this fatal flaw, the technical method has significant unresolved issues: the meta-learner architecture is likely too weak to induce meaningful behavioral changes in a frozen 16B-parameter model, the positive/negative pair construction strategy is unspecified preventing reproducibility, the projection regularization is in direct tension with adaptation goals, and the contrastive loss formulation deviates from standard InfoNCE without justification. The writing exhibits multiple instances of clearly corrupted or hallucinated text, and the acknowledgments contain unfilled template text. Even if results were added, the methodological gaps would require substantial revision to meet ICLR standards. The core problem statement is legitimate and the architectural decomposition idea has merit, but the paper requires fundamental rework before it is ready for a venue of this caliber.

# Neutral Reviewer
## Balanced Review

### Summary
The paper proposes "Contrastive-Online-Meta (COM)," a framework designed to mitigate catastrophic forgetting and noisy feedback in instruction-tuned CodeLLMs during online deployment. It combines contrastive pre-training for task-invariant representations with an online meta-learning mechanism for lightweight parameter adaptation. Experimental results claim that COM outperforms static fine-tuning and other continual learning baselines on specific adaptation and generalization benchmarks.

### Strengths
1.  **Relevance to CodeLLM Deployment:** The paper addresses a critical gap in CodeLLM utility—continuous adaptation in dynamic environments where new code patterns and feedback emerge post-deployment. This focus on "deployment-time" learning is highly relevant to ICLR's scope.
2.  **Modular Architectural Design:** The separation of the frozen base CodeLLM from learnable adapters (contrastive embeddings and meta-parameters) is a sound theoretical approach to balancing stability and plasticity. This explicit decoupling allows for efficient updates (claimed to require ~5% of base parameters) without retraining the entire model.
3.  **Comprehensive Evaluation Metrics:** The authors define clear metrics for continual learning scenarios, specifically **Adaptation Accuracy (AA)**, **Forgetting Rate (FR)**, **Generalization Gap (GG)**, and **Update Efficiency (UE)**. Using multiple dimensions (accuracy vs. forgetting vs. cost) provides a more robust assessment than accuracy alone.

### Weaknesses
1.  **Clarity and Writing Quality:** There are significant issues with prose quality and clarity that undermine the technical presentation. For example, the manuscript explicitly states in **Section 8** ("We use LLM polish writing based on our original paper"), which raises concerns regarding the originality of the text and adherence to submission guidelines regarding AI tool usage. Additionally, phrases such as "behavior-effective thing" (Abstract) and "coefficients to the issues" (Abstract) indicate a lack of rigorous proofreading.
2.  **Methodological Ambiguity:** The distinction between the model components' trainability is confusing. The Abstract describes a "contrastive pre-training module," but **Section 4.3** states "Gradients flow only through $g_\phi$ and $f_\theta$," implying $f_\theta$ (the instruction encoder) is updated online. This contradicts the standard notion of a pre-trained encoder being frozen during the meta-learning phase, making the training dynamics unclear.
3.  **Novelty Claims:** The claim in the Introduction that this is the "first principled merging of contrastive objectives and the meta-learning... of CodeLLMs" appears overstated. Related work in Section 2 acknowledges similar combinations (e.g., Qin et al., 2023; Wang et al., 2023), and the specific contribution of "contrastive regularization" on meta-parameters is not sufficiently differentiated from existing meta-continual learning literature.
4.  **Data Construction Transparency:** The **StreamCode** benchmark used for continual learning evaluation is described as "constructed" in **Section 5.1**, but the paper provides no details on how task boundaries are defined or how the "non-stationary streams" are simulated. Without this reproducibility data, the validity of the "Forgetting Rate" claims is difficult to assess.

### Novelty & Significance
*   **Novelty:** **Moderate.** While combining contrastive learning and meta-learning for continual adaptation is a coherent idea, the integration with CodeLLMs feels incremental. The "first principled merging" claim needs stronger justification against existing meta-continual learning work (e.g., in recommendation systems or general NLP).
*   **Significance:** **High.** Solving catastrophic forgetting in CodeLLMs is a significant practical challenge. If the method works as claimed, it could enable scalable, persistent coding assistants. However, the current writing quality and ambiguity reduce confidence in the feasibility of the claims.
*   **Clarity:** **Low.** The text requires substantial editing. The admission of LLM polishing in Section 8 is a major negative for a top-tier venue like ICLR, which expects human-authored rigor. Ambiguous statements about parameter freezing further hurt clarity.
*   **Reproducibility:** **Medium.** Implementation details (hyperparameters, base model) are provided, but the construction of the "StreamCode" dataset and the specific training schedule (how often $f_\theta$ vs $g_\phi$ update) are not fully detailed.

### Suggestions for Improvement
1.  **Revise Prose and Disclosure:** The authors must completely rewrite the manuscript to ensure professional academic English. If Section 8 ("We use LLM polish...") is accurate, it must be properly disclosed in the standard AI usage disclosure section, not buried as a section heading, as this violates ICLR's transparency expectations.
2.  **Clarify Training Dynamics:** Explicitly define the training schedule for $f_\theta$. Is it pre-trained and then frozen? Or is it fine-tuned alongside $g_\phi$ during inference? The current description ("pre-training phase... deploy on-line" vs "Gradients flow through $f_\theta$") is contradictory.
3.  **Strengthen Novelty Positioning:** Provide a detailed ablation or analysis distinguishing COM from existing continuous meta-learning methods (e.g., LEO, EWC). The authors should clarify what specifically about the *contrastive* component prevents forgetting more effectively than standard weight regularization.
4.  **Detail Data Construction:** Describe the **StreamCode** benchmark construction in detail. How are tasks sequential? How is "noise" simulated in the feedback? This is crucial for the validity of the continual learning claims.
5.  **Fix References:** Several references list future dates (e.g., 2025), which undermines credibility. Ensure all citations are current or correctly marked as preprints (arXiv) with stable links.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Quantitative Results Reporting**: Provide explicit performance numbers for Adaptation Accuracy and Forgetting Rate to substantiate abstract claims. Without concrete data, the claims of "significantly higher robustness" are unverifiable.
2. **Ablation of Components**: Remove contrastive loss, memory buffer, and meta-learner individually to prove each contributes to performance. Without this, the "unified framework" claim is weak as one component might drive all gains.
3. **Feedback Noise Sensitivity**: Test adaptation quality when feedback signals are intentionally corrupted or delayed. This directly validates the core claim of robustness to "noisy feedback at the time of deployment."
4. **PEFT Baseline Comparison**: Compare against LoRA-based continual learning methods instead of static fine-tuning. Static baselines do not validate online adaptation capabilities and make efficiency claims misleading.
5. **Compute Efficiency Verification**: Report exact FLOPs and wall-clock time to verify the "3-5x fewer updates" claim. Efficiency is a core contribution and requires rigorous computational measurement, not just parameter counts.

### Deeper Analysis Needed (top 3-5 only)
1. **Frozen Base Capacity Limit**: Analyze whether adapter-only updates can sufficiently adapt a 16B model for complex code tasks. If the frozen base limits expressivity, the method cannot solve the adaptation problem as claimed.
2. **Memory Buffer Sensitivity**: Analyze performance variance as buffer size $C$ scales down. This tests practicality in memory-constrained deployment settings which is critical for "online" systems.
3. **Loss Landscape Conflict**: Visualize gradient alignment between contrastive and meta-learning objectives. Ensure the "unified" loss doesn't create optimization conflicts that destabilize training.
4. **Task Similarity Impact**: Correlate performance drop with semantic distance between sequential tasks. This validates the "task-invariant representation" claim against actual task distribution shifts.
5. **Convergence Stability**: Plot loss curves over online steps to prove the method does not diverge. Online learning systems must demonstrate stability over long streams, not just final accuracy.

### Visualizations & Case Studies
1. **Embedding Clustering**: t-SNE visualization of instruction embeddings before and after contrastive training. This proves the representation learning module actually clusters semantically similar instructions as claimed.
2. **Forgetting Heatmap**: Matrix showing performance on all past tasks at every time step. This directly visualizes catastrophic forgetting patterns better than a single aggregate metric.
3. **Code Generation Cases**: Side-by-side examples of code generated by COM vs. baselines on unseen languages. Qualitative proof is needed to show the model isn't just memorizing syntax but understanding logic.
4. **Efficiency Trade-off Curve**: Plot Accuracy vs. FLOPs to visualize the claimed efficiency advantage. This allows reviewers to instantly assess the cost-benefit ratio against baselines.

### Obvious Next Steps
1. **Insert Results Section**: Add the missing experimental results data to the paper immediately. A paper claiming experimental superiority without a results section is incomplete.
2. **Clarify Adapter Novelty**: Explicitly distinguish method from standard Meta-learned Adapters or LoRA. The current formulation risks appearing as repackaged parameter-efficient fine-tuning.
3. **Release StreamCode Details**: Publish construction logic for the custom benchmark to ensure reproducibility. ICLR standards require constructed datasets to be fully documented or public.
4. **Evaluate Sparse Feedback**: Test scenarios where feedback is unavailable for certain steps. Real-world deployment rarely has continuous feedback, so this validity gap must be closed.
5. **Implement Bias Tracking**: Add the proposed ethical auditing mechanism to track adaptation drift. Since the paper acknowledges bias risks, it should demonstrate a mitigation strategy.

# Final Consolidated Review
## Summary

The paper proposes Contrastive-Online-Meta (COM), a framework for dynamically adapting instruction-tuned CodeLLMs during deployment. COM combines contrastive pre-training for task-invariant representations with online meta-learning for lightweight adaptation, using a frozen base model with learnable adapters and a dynamic memory buffer. The claimed contribution is preserving programming knowledge while adapting to new instruction patterns without catastrophic forgetting.

## Strengths

- **Problem formulation addresses real deployment challenges**: The focus on online adaptation for CodeLLMs in dynamic environments—where instruction patterns and user feedback arrive continuously—is practically significant. Existing instruction tuning approaches lack mechanisms for continuous adaptation without retraining.

- **Modular architecture separation**: The design separates a frozen base CodeLLM from trainable components (contrastive embeddings and meta-parameters), which is theoretically sound for balancing stability (preserving programming knowledge) with plasticity (adapting to new tasks). The claim that only ~5% of parameters require updates, if validated, would meaningfully reduce deployment costs.

- **Multi-metric evaluation framework**: The proposed metrics—Adaptation Accuracy, Forgetting Rate, Generalization Gap, and Update Efficiency—address distinct aspects of continual learning that accuracy alone cannot capture.

## Weaknesses

- **No experimental results are presented**: Section 5 describes experimental setup in detail (datasets, baselines, metrics, hyperparameters) but the paper jumps directly to Discussion (Section 6) without presenting any quantitative results. The abstract and introduction make specific claims—"3-5× fewer updates than conventional meta-learning approaches," "12-18% outperforming instruction-tuned baselines on unseen programming languages," "significantly higher robustness than standard fine-tuning"—that are entirely unsubstantiated. A paper claiming empirical contributions must present empirical evidence.

- **Multiple passages contain garbled or incoherent text**: The abstract contains "coefficients to the issues of catastrophic forgetting" and "behavior-effective thing." Section 4 references "programming England's instructions." Section 6.1 contains "scope for improvementCivil War." Section 7 has "where Headquarters and reagents of statements and feedback are still pushing and changing." These semantic errors suggest inadequate proofreading of LLM-generated content and undermine confidence in the technical presentation.

- **Positive/negative pair construction for contrastive learning is unspecified**: The paper states "positive pairs might include different implementations of the same algorithm" but never specifies how these pairs are actually constructed for CodeAlpaca-20k or StreamCode. Contrastive learning quality depends critically on pair selection; omitting this prevents reproducibility.

- **StreamCode benchmark construction is undocumented**: The paper claims StreamCode is a constructed benchmark with "5 distinct task distributions" arriving in "non-stationary streams," but provides no details on task boundary definition, stream construction methodology, or data annotation. Without this information, the continual learning evaluation cannot be reproduced.

- **Meta-learner architectural capacity is questionable**: The meta-learner gϕ is specified as a "2-layer MLP" operating on 768-dimensional embeddings, transforming them before passing to a frozen 16B-parameter CodeGen model. The paper does not justify whether this bottleneck provides sufficient expressiveness for meaningful task-specific adaptation (e.g., adapting to new APIs or programming languages), nor does it provide ablations testing this design choice.

- **Citation misattribution**: The claim of "3-5× fewer updates than conventional meta-learning approaches (Nichols et al., 2024)" cites a paper titled "Performance-aligned LLMs for generating fast code"—this is not a meta-learning approach and is not an appropriate baseline for update-count comparisons.

- **Numbered placeholder references in Section 2.3**: The final paragraph cites "[1,2]" and "[3,6]" and "[7,9]" instead of using the author-year format used elsewhere, suggesting incomplete integration.

- **Training schedule ambiguity**: The abstract describes "contrastive pre-training" followed by "online meta-learning," but Section 4.3 states "Gradients flow only through gϕ and fθ," implying fθ (instruction encoder) updates during online deployment. It is unclear whether fθ is frozen after pre-training or continuously adapted.

## Nice-to-Haves

- **Ablation of individual components**: Testing COM with contrastive loss removed, memory buffer removed, and meta-learner removed would validate that each component contributes meaningfully rather than one component driving all gains.

- **PEFT baseline comparisons**: Comparing against LoRA-based or prefix-tuning continual learning methods would provide more relevant baselines than static fine-tuning for adaptation efficiency claims.

- **Compute efficiency measurements**: The "3-5× fewer updates" claim requires actual FLOPs or wall-clock time measurements, not just parameter counts.

- **Feedback noise robustness validation**: The abstract claims robustness to "noisy feedback" but no experiments test this explicitly.

## Removed Points

*These points are flagged to be removed, treat them with caution:*

- **Criticism of LLM disclosure in Section 8**: The paper properly discloses LLM use for polishing. This is transparent and follows appropriate practices.

- **Complaints about 2025 citation dates**: ArXiv preprints commonly have dates ahead of conference cycles; this is standard practice.

- **Generic "topic is important" strength**: This applies to many papers and is not a specific contribution.

- **Request for theoretical proofs**: This is an empirical systems paper; theoretical proofs would be scope creep.

- **Request for user studies**: For an algorithmic contribution focused on model adaptation, user studies are not expected.

## Novel Insights

The architectural decomposition of stability-plasticity trade-offs into three distinct mechanisms—contrastive representation learning (preserving semantic structure), meta-learned adaptation parameters (enabling rapid task-specific updates), and dynamic memory buffering (maintaining temporal coherence)—is a principled approach. However, whether this decomposition actually works as claimed cannot be evaluated without experimental results. The insight that contrastive objectives might regularize meta-learning updates by preventing representation drift is theoretically interesting but unvalidated.

## Suggestions

- **Add a complete Results section** with quantitative comparisons for all four metrics against all baselines, including standard deviations across multiple runs.

- **Specify positive/negative pair construction**: Describe the algorithm or process for creating semantically equivalent instruction pairs and selecting negative samples for each dataset.

- **Document StreamCode benchmark**: Provide task definitions, stream construction methodology, data sources, and ideally release the dataset or construction code.

- **Clarify the training protocol**: Explicitly state when fθ is trained vs. frozen, and whether online updates affect both fθ and gϕ or only gϕ.

- **Correct citation misattributions**: Verify that Nichols et al. (2024) is appropriate for the meta-learning baseline comparison, or replace with a correctly cited meta-learning approach.

- **Proofread the manuscript thoroughly**: Remove all garbled passages before resubmission.

# Actual Human Scores
Individual reviewer scores: [0.0, 0.0, 0.0, 0.0]
Average score: 0.0
Binary outcome: Reject
