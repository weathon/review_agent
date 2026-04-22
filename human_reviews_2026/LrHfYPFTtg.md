# Keep the Best, Forget the Rest: Reliable Alignment with Order-Aware Preference Optimization

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 2, 4, 6

## Abstract
Direct Preference Optimization (DPO) has emerged as a powerful framework for aligning large language models (LLMs) with human preferences via pairwise comparisons. However, its performance is highly sensitive to the quality of training samples: when the reference policy is poorly aligned with human preferences, ambiguous pairs can dominate the gradient signal and degrade generalization. To address this, we propose RAPPO($\textbf{R}$eliable $\textbf{A}$lignment for $\textbf{P}$reference $\textbf{P}$olicy $\textbf{O}$ptimization), a simple sample-aware modification of the DPO loss that mitigates reference-policy misalignment by filtering out the hardest, most ambiguous samples. We theoretically show that RAPPO yields improved generalization guarantees. RAPPO is lightweight and requires only a few lines of code to be integrated into any existing DPO-type algorithm. Surprisingly, With this simple modification, our simulations across a broad suite of alignment tasks and benchmarks show consistent gains over DPO and recent state-of-the-art baselines. On the PKU-SafeRLHF benchmark, RAPPO attains helpfulness $0.693$ ($+34.8\%$ over DPO) and harmlessness $0.357$ ($-21.0\%$ vs DPO).

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes RAPPO, a lightweight filtering mechanism built upon the DPO framework. RAPPO introduces two key ideas: (1) reference-policy awareness, where samples are first partitioned into Aligned and Unaligned subsets based on a reference-policy consistency threshold τ; and (2) in-batch ranking and pruning, where, within the Unaligned subset, the top-q samples with the highest individual DPO losses are temporarily discarded from the current update.

The authors show that RAPPO leads to a larger expected first-order risk reduction, lower gradient variance, and a tighter stability generalization bound. Empirically, RAPPO outperforms DPO, CPO, KTO, and SimPO on the PKU-SafeRLHF benchmark.

### Strengths
1. The idea of incorporating reference-policy awareness into DPO is simple yet intuitively appealing, offering a principled way to mitigate the influence of noisy or misaligned preference data.
2. The method demonstrates clear and consistent gains across multiple metrics on a competitive benchmark.
3. The paper provides meaningful theoretical analyses on gradient variance, stability, and risk reduction, giving insight into why the filtering helps.
4. RAPPO’s design is lightweight and can be readily integrated into existing DPO-style training pipelines with minimal overhead.
5. The algorithmic formulation and ablation structure are clearly described, making the contribution easy to follow.

### Weaknesses
1.  Since the gate relies on the reference policy’s relative probabilities, its robustness depends on the reference model’s reliability. If the reference policy is miscalibrated, important training signals might be filtered out. Including robustness comparisons would strengthen the claim.
2.  Given the substantial empirical improvements, it would be valuable to release code and the corresponding commit hash.
3. Using GPT-4o as a judgment model could introduce systematic bias. Multi-rater evaluation or human calibration would make the conclusions more convincing.
4. Although the related-work section discusses Selective DPO, ORPO, and R-DPO, the large-scale comparisons include only DPO, CPO, KTO, and SimPO. Adding results for IPO, ORPO, R-DPO, or RRHF would provide a fairer empirical landscape.
5. While results are reported for q = 1, 2, 4, the paper lacks a systematic sensitivity analysis for both q and τ. Understanding how performance depends on these thresholds would make the method more interpretable and reproducible.

### Questions
see the weakness

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes RAPPO, a variant of DPO where some misaligned samples are filtered on a batch-by-batch basis. The specific algorithm hinges on two hyper-parameters. The threshold of the misalignment score in which to categorize the batch samples and q, the number of misaligned samples to toss out. Empirical validation on multiple, but not extensively thorough, demonstrate effectivenss over DPO

### Strengths
- The idea is straightforward and easy to implement
- The empirical gains (at least the ones presented) are nice
- Some theoretical analysis exist, which is always nice

### Weaknesses
- As the idea itself (filtering out misaligned samples) is quite straightforward, I believe a lot of the paper's contributino comes down to the execution and how well it can generalize. In this aspect, I don't think the empirical evidence shown in the submission is extensive enough. On the other hand, there is little analysis on the hyper-parameter sensitivity of tau and q. I feel like a lot of the algorithm's performance will depend on the exact value of those hyper-parameters and until I see some sensitivity analysis on them, I don't think I can say this method will generalize well.

### Questions
- Hyperparameter q (the number of bad samples to throw out), seem to be a flat number. This seems to have unfavorable interaction as batch sizes are not constant over different training runs and research and saying to remove a flat number of samples when the batch size can differ in multiple orders of magnitude may not be helpful. Shouldn't this hyperparameter be a percentage of the batch size? If so, this makes the interplay between tau and q even more complex, as the number of Bad Samples in which to exclude the samples from is not pre-determined either.

### Soundness
2

### Presentation
2

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
The paper presents RAPPO, an order-aware variant of DPO that filters high-loss “untrusted” pairs per mini-batch while always keeping “trusted” items. The method is simple, easy to implement, and accompanied by a stability analysis that yields a tighter generalization bound. Empirically, RAPPO outperforms strong DPO-style baselines on multiple LLM tasks. Overall: clear motivation, lean algorithm, solid theory; still room to strengthen rigor and reporting.

### Strengths
S1. Clear problem framing and insightful diagnostics (Fig. 1 and 2) quickly convey why reference-aware filtering helps.

S2. Method is intuitive, minimally invasive to DPO, code is provided, and the analysis connects the selection rule to lower variance and tighter stability.

### Weaknesses
W1. Lines 291–311 (and Theorem 4.7’s surrounding prose) use $q$ as the **kept** count per step, while Algorithm 1, Proposition 4.8, and the experimental setup (Line 424) treat $q$ as the **number removed**. This clash, along with redundant symbols, impairs readability. Please rewrite this block with a single convention, and verify whether the statement of this work still aligns after the change.

W2. In the theory part, only (11)–(12) show how $q$ affects stability via a term proportional to $\frac{1}{q}$ and $\max_{i\in\text{Kept}} w$. This reveals a trade-off (keeping more reduces the $\frac{1}{q}$ factor; removing more shrinks $\max_{i\in\text{Kept}} w$), but there is no result suggesting how $q$ should scale for optimal decrease, convergence rate, or even convergence. It would strengthen the paper to (i) formalize the trade-off and relate the choice of $q$ to Thm 4.7’s decrease/variance terms; and (ii) add sensitivity studies across $q$ and batch sizes. I expect the optimal range to be task- and model-dependent.
 
W3. Many results rely on reward models and GPT-4 judging. This may introduce systematic bias not captured in the theory. For the experiment, I suggest adding a small-scale human evaluation with inter-rater agreement (e.g., Kendall's $\tau$), but I understand it may be challenging within the limited time. Some surrogate way, like (i) ablation, is to compare a Top-K with a “random K” to see whether mild randomness mitigates judge-specific bias without harming stability, and/or (ii) add judge/reward-model robustness checks.

### Questions
See above.

### Soundness
3

### Presentation
2

### Contribution
3
