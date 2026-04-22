# Scaling Laws and Symmetry, Evidence from Neural Force Fields

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 8, 2

## Abstract
We present an empirical study in the geometric task of learning interatomic potentials, which shows equivariance matters even more at larger scales; we show a
clear power-law scaling behaviour with respect to data, parameters and compute
with “architecture-dependent exponents”. In particular, we observe that equivariant
architectures, which leverage task symmetry, scale better than non-equivariant
models. Moreover, among equivariant architectures, higher-order representations
translate to better scaling exponents. Our analysis also suggests that for computeoptimal
training, the data and model sizes should scale in tandem regardless of the
architecture. At a high level, these results suggest that, contrary to common belief,
we should not leave it to the model to discover fundamental inductive biases such
as symmetry, especially as we scale, because they change the inherent difficulty
of the task and its scaling laws.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents an empirical investigation of scaling laws for neural force fields used in learning interatomic potentials. The authors systematically study how equivariant and non-equivariant architectures scale with respect to data size, model parameters, and compute budget. The key finding is that equivariant architectures exhibit superior scaling exponents compared to non-equivariant models, with higher-order equivariant representations showing the best scaling behavior. The study provides power-law fits demonstrating "architecture-dependent exponents" and suggests that compute-optimal training requires scaling data and model size together. The authors argue that fundamental inductive biases like symmetry should be built into architectures rather than left for models to discover, especially at scale.

### Strengths
- Rigorous empirical analysis: The paper provides tight power-law fits across the tested regime, demonstrating clear architecture-dependent scaling exponents with convincing statistical evidence.
- Clear presentation of core results: The main findings about equivariant architectures exhibiting better scaling behavior than non-equivariant models are presented clearly.
- Practical insights: The finding that data and model size should scale in tandem for compute-optimal training is actionable regardless of architecture choice.
- Important research question: Investigating whether fundamental inductive biases matter at scale is highly relevant to the broader machine learning community.

### Weaknesses
- Severely limited training regime: Training on only a single epoch is a poor experimental choice that prevents data augmentation and fails to test the most interesting hypothesis from Brehmer 2025—that equivariance benefits may disappear with sufficient augmentation over multiple epochs. While the authors may draw inspiration from language model training, molecular datasets are orders of magnitude smaller, making this analogy weak.
- Insufficient scale: The maximum compute budget of ~30 GPU hours is quite small for claiming insights about "large-scale" behavior on molecular datasets. True scaling studies typically involve orders of magnitude more compute.
- Limited symmetry analysis: What the paper calls "unconstrained MLP" is actually translation-invariant (operating on relative positions). The study only assesses rotational equivariance on top of translational invariance, not the full contribution of geometric symmetries. This should be explicitly clarified as it affects the interpretation of results.
- Unexplained contradictions with recent work: The conclusions don't adequately address the success of models like Orb, which achieve near state-of-the-art performance on MatBench Discovery despite being only translational equivariant (like the unconstraint MPNN in this paper). This weakens the paper's strong claims about the necessity of equivariance at scale. Could it be that the non-rotationally-equivariant model chosen by the authors is a worse choice than Orb?

### Questions
- How do the resulting models compare to state-of-the-art models on the OpenMol dataset? That would be good to know how well the results represent real-life results on such datasets.
- Have the authors considered taking a model like eSEN and manipulating just its equivariant convolution layers to be unconstrained? That would remove some of the impact of architectural design on the comparison.

### Soundness
2

### Presentation
3

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
The paper presents an empirical study on the scalability of equivariant models vs non-equivariant models (w.r.t.  SO(3) symmetry group) for the molecular force fields task (predicting energy and forces of molecules). More specifically, the authors show how models like GemNet, eSEN, and EGNN tend to scale better than unconstrained GNN architectures on the OpenMol dataset. They study the scalability of these models across multiple access, including: model parameters, PFLOPs, and GPU hours, against validation loss of the molecular force fields task.

### Strengths
* I think the problem being studied is interesting and quite relevant to the current ongoing discussion on equivariant vs non-equivariant models design, in the area of geometric deep learning.
* Interesting to see that different equivariant models have different scalability behavior, for example, eSEN has lower validation loss with more GPU hours compared to the GemNet architecture.

### Weaknesses
* I think the evidence is not sufficient to support the claims; some rewrites might be required to not make it a general claim or show more results on other tasks. 
* The study is limited to a single-epoch training regime, and studying the GPU hours in the range of less than 100 hours is limited. It would be beneficial to see how this could be extended for longer training times, and if the same trend holds or not. 
* Also, comparing recent models like eSEN to vanilla GNN might limit the claims of the paper, as GNN/ MPNN is an old baseline now. More unconstrianed architectures should be included (e.g., how this is applied to Graph Transformers).

### Questions
See above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work introduces large-scale experimental results illustrating scaling laws for validation loss across equivariant message passing architectures and for NNIP tasks. As a result, several practical observations are made, including optimal architecture choices based on available compute.

### Strengths
The paper tackles an important and timely problem, studying the effect of higher-order equivariant features on accuracy/compute tradeoffs. It presents large-scale experiments and reports empirical scaling behavior with clear ablations. The approaches used in the study are well motivated and clearly written.

### Weaknesses
Some claims appear stronger than what the experiments directly support.

### Questions
**Q1. Generalization.** The claim on line 80

> While our study is limited in scope to Special Euclidean symmetry of neural interatomic potentials and force fields, as well as a few representative architectures, there is no reason to believe the results should remain confined to this particular domain and the choice of efficient equivariant models.

appears at odds with the line 843 on related works stating

>Despite using the eSEN same backbone, Wood et al. (2025) report that, for dense models 5, the compute-optimal strategy scales model size N faster than data size D, whereas in our setting we observe nearly equal scaling between N and D; though the tasks are different

I suggest that the authors reconsider the claims of generalization of these scaling curves beyond the scope of the present work.

**Q2. Undefined acronyms.** Please be sure to introduce acronyms like NNIPs that may be familiar to those working on these specific tasks, but not to a more general audience.

**Q3. Energy vs. direct-force training.** Can the authors please substantiate the claim in line 138

> While it is sufficient to learn the energy for predicting conservative forces, direct force prediction is significantly more scalable.

**Q4. Transformer instability.** Can the authors provide evidence for the line 149 where they observed

> instability issues when scaling vanilla transformers for this task, we focused on message-passing architectures.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors study how machine learning interatomic potentials (MLIPs) scale with data, parameters, and compute, comparing architectures with increasing degrees of equivariance. They train on the OMol neutral-molecule subset, treat atoms as “tokens,” and adopt a single-epoch training regime to mirror LLM practice. They evaluate scaling against both FLOPs and GPU-hours, arguing this better reflects practical costs for (often less GPU-friendly) equivariant models. Empirically, they report clear power-law behavior and that scaling exponents grow with architectural equivariance, so performance gaps widen at larger scales; they also observe that compute-optimal training should scale model size and dataset size in tandem. A symmetry loss regularizer improves sample efficiency but does not match the benefits of fully equivariant architectures.

### Strengths
1. The paper applies the modern neural-scaling methodology to neural interatomic potentials, an important area in AI-driven chemistry.  This fills a gap for the field.
2. The paper is well written and easy to follow.

### Weaknesses
As the first paper (that I am aware of) to investigate the neural scaling law in the MLIP domain, I think the standard / stakes are high, and incorrect claims could lead to consequences in following model design. In this paper, the authors draw a lot of methodology directly from NLP, but I think these two fields have some fundamental differences. I found the following critical flaws in the three aspect of scaling law: Compute (1, 2, 3), Data (3, 4), and Parameters (5), and additional issues (6,7)
1. Insufficient training / compute to claim asymptotic scaling: The scaling-law fits are based on small training budgets.  From Figure 1 and Section 4, the largest models run correspond to only \~10^1 GPU-hours (per run) or \~10^3 PFLOPs of compute.  This is tiny compared to typical scaling-law studies, which span orders-of-magnitude more compute (e.g. billions of tokens or thousands of TPU/GPU years). Even for MLIP training, this is too small: the eSEN small direct model is trained for around 10^5 PFLOP (80 Epochs) in OMol 4M, where the authors is training 10^3 PFLOPs on OMol 34M. That is **100x less FLOPs on 8x more data**. Moreover, each model is trained for only a single epoch through 34M samples (no repeated passes).  For neural potentials, one epoch training is extremely light: standard MLIP practice often trains tens of epochs to convergence.  The plots (e.g. Figure 1) do not show curves flattening -- the losses keep decreasing.  This suggests models are not converged even at the largest scale tried.  Fitting power laws in such a sub-convergent regime is problematic: the “effective” exponent can vary dramatically in early vs late training. In short, scaling laws are **asymptotic** statements, and here the training scale is too small to robustly infer asymptotic behavior.  This undermines the claim that explicit symmetry consistently improves scaling: with more compute, any performance gap might shrink or change.
2. Single-epoch training regime. In Section 3.1, the authors intentionally use **only one epoch** (each sample seen once) to mirror LLM practice. However, this contradicts common MLIP training, where models typically see the data many times (e.g. \~**80 epochs** in the OMol 4M baselines). Single-epoch training likely means none of the models fully fit the data distributions; indeed the validation losses are still dropping at the end of training. This choice can distort scaling-law estimates. For instance, early in training, increasing model size might seem more beneficial (higher \alpha) simply because larger models learn faster per data pass, but with multi-epoch training a smaller model could catch up. By never allowing converged fits, the authors effectively inflate the “scaling advantage” of larger/equivariant models. They justify this by wanting to “avoid confounding effects”, but do not analyze how one vs multiple epochs alters the conclusions. Without at least one multi-epoch comparison, we cannot be sure the reported power-law behaviors would hold under standard MLIP training. This is a **major methodological gap**: the conclusions about scaling hinge on a very nonstandard (for this domain) training regime.
3. Global target vs. token-wise analogy. The paper frequently draws analogies to language scaling laws, but **the supervised targets here are fundamentally different**. In language models, the loss (cross-entropy) is accumulated over **each token prediction**. Here, the primary target is the global energy of a molecule (plus per-atom forces, calculated by the gradient of the energy, thus is heavily correlated with the total energy). Each system yields **only one energy scalar despite many atoms**. The authors do treat atoms as “tokens” in counting dataset size, but this is a loose analogy: the model must capture a global property that depends on all atoms jointly. This could be a much harder task than NLP cross entropy target. Thus, as we discussed in weakness 2, seeing each atom once may not be analogous with LLM training, and the model requires much more training time to converge. As a result, statements like “we follow LLM scaling methodology” may be overreaching without addressing that molecular systems are global-structure tasks.
4. Limited dataset regime (neutral split only). The OMol dataset is a high quality and diverse dataset, but all experiments use only the **neutral-molecule subset** of OMol (\~34M samples), despite the full dataset being ~100M with diverse splits. This choice (claims to be taken due to memory limits, which if that's because of the system size, the authors could always decrease the batch size, since the batch size they chose was quite large (64) ) of using the neutral split, rather than **random sample**, could bias the domain. As described in the OMol paper, **non-equivariant models, such as GemNet, performs much better than equivariant models in EF error** when trained on all splits, such as biomolecules and electrolytes. This dataset selection clearly favors the equivariant baselines, and leads to a misleading or even incorrect results. I.e. the conclusions about “symmetry matters more at scale” could be dataset-specific. The authors should justify why neutral-only results would hold in the full OMol or other datasets (none of which is provided).
5. Unfair comparison by raw parameter count. In NLP, comparing models by parameter count is largely meaningful because mainstream LLMs use a homogeneous Transformer architecture: blocks are architecturally identical, most weights live in token-wise MLPs and attention projections, and the compute per parameter and weight reuse pattern are effectively uniform across models. As a result, plotting loss vs. parameters is reasonably apples-to-apples. In contrast, **for GNN-based MLIPs the notion of “one parameter” does not carry across architectures**: (i) parameters are shared and reused across all nodes/edges; (ii) different designs incur very different work per parameter (e.g., dihedral terms in GemNet-OC, high-order tensor ops in eSEN); (iii) body order and tensor order change effective capacity and FLOP intensity; and (iv) throughput at the same N can differ drastically. This is largely why the kappa number varies between architectures. Consequently, the Fig. 5 style plots of loss vs. N conflate capacity, compute, and inductive bias, and can be misleading. A fair comparison should be iso-compute (FLOPs or GPU-hours), or at least normalized by an architecture-dependent cost.
6. Exponent reliability and fitting issues. The fitted power-law exponents (α, β, γ) are central to the paper’s claims, but their reliability is questionable. First, as noted, the data range is relatively narrow (datasets from 10%-100%, models \~10^6-10^7 parameters): typically one requires several orders of magnitude variation to robustly determine an exponent.  Here both N and D spans are \~1-2 orders of magnitude at best. Second, the assumption $L_\inf \approx 0$ (irreducible loss) is ad hoc and may bias exponents high.  In real molecular data there should be some finite noise/error floor (analogous to the entropy in NLP); assuming zero means the model is “expected” to achieve perfect prediction eventually.  Small positive L_\inf can drastically change a fitted β or γ in a power law fit (as studied in Hoffmann et al. 2022).  Third, Figure captions reveal instability: GemNet-OC needed smoothed loss, and the authors exclude the first 1–10% of training from fits.  These steps suggest the raw curves were not clean power laws. With such variance, the numeric exponents conflict across FLOPs vs wall-time (Fig.1).  Without error bars or fit-statistics on α,β (only γ had CIs in Table 1), it’s hard to trust these values.  In summary, the power-law behavior is claimed too strongly: given the limited and noisy regime, the reported exponents may not reflect true asymptotic trends. The results could be artifacts of the experimental choices (one-shot training, hyperparams, smoothing).
7. Additional concerns: (i) The study considers only four architectures; it is not clear if these are representative. For example, EGNN and GemNet are relatively simple and old models -- how would a more modern one (PaiNN, DPA-1/2, EScAIP, even NequIP) behave? (ii) The paper also mixes different body-order and tensor-order notions without clarity. (iii) Hyperparameters (depth, widths) are tuned only at ~1M params and then scaled; it’s possible that later models were not fully optimized. (iv) The authors do not report if they repeated experiments to measure variability (confidence intervals are absent for N,D scaling). (v) Finally, the broad claim that one should “not leave symmetry to be discovered by scaling” may overstate the results. The bitter-lesson cited in the intro suggests that even biased models can eventually be outperformed by larger unconstrained ones; these experiments do not go far enough to test that (no extremely large unconstrained model was trained). In its current form, the paper’s strong conclusion about avoiding bitter lesson is not fully justified by the limited empirical data.

### Questions
1. Why restrict training to a single epoch?  Have you tried multi-epoch training (e.g. 10 or 80 epochs) on a smaller scale to see if exponents change?  How do you justify that one pass through data captures the scaling behavior of fully trained models?
2. How do you expect inclusion of the charged/large molecules (omitted in the neutral split) to affect your conclusions?  Could the symmetry advantages reverse or diminish in chemically diverse subsets?
3. How sensitive are your results to the learning-rate and batch-size choices? You tuned these at 1M parameters and then scaled. Is it possible that some models (e.g. the largest eSEN) were suboptimally trained? 
4. The introduction cites Sutton’s “bitter lesson” about scale overtaking bias. Given your modest scaling regime, how confident are you that “we should not leave symmetry to be discovered by the model”? Could larger-scale experiments eventually reverse the trend you observe?
5. Nit: where is Figure 6?

### Soundness
1

### Presentation
3

### Contribution
2
