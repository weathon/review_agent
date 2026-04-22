# Why RL Updates Look Sparse: An Implicit Compass Drives Optimization Bias

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 4, 8

## Abstract
Reinforcement learning (RL) reliably improves LLM reasoning while appearing to change only a small fraction of parameters. 
We revisit this paradox and argue that the visible sparsity is not the phenomenon itself but the trace of a persistent optimization bias, where RLVR stubbornly commits updates to preferred regions that remain invariant across datasets and RL variants, as if guided by an implicit compass.
We propose a Three‑Gate Theory to formalize this mechanism, where the Anchor Gate I shows RL induces a one‑step policy‑KL leash that keeps updates proximal to the base policy;
This constrained update is then steered by Gate II (Model Geometry) towards lower-curvature, spectra-preserving directions, a data-invariant feature;
and finally, it is filtered by Gate III (Precision), where the bfloat16 format acts as a lens that amplifies the bias by hiding micro-updates, making the underlying pattern visible as apparent sparsity.
Empirically, we validate this theory with a comprehensive suite of experiments. 
We show that RL preserves the model’s spectral structure and avoids its principal weights, in sharp contrast to SFT, which alters spectra and mainly targets those weights. 
Causal interventions confirm that this bias is destroyed when the model's geometry is disrupted, proving that the geometry is the steering core of the "compass."
By providing the first parameter-level account of RLVR's training dynamics, our work not only demystifies its optimization bias but also provides a new perspective of understanding RLVR.
Crucially, we show that RL operates in a distinct optimization regime from SFT, directly adapting SFT-era parameter-efficient fine-tuning (PEFT) methods can be flawed, as evidenced by our case studies on advanced sparse fine-tuning and LoRA variants, motivating the design of efficient geometry-aware, RLVR-native learning algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper investigates why reinforcement learning (RL) fine-tuning of large language models appears to modify only a small fraction of parameters while producing large changes to the end-to-end map.
It proposes that this apparent sparsity reflects an underlying optimization bias, the implicit compass, that steers updates toward certain regions of the model parameter space.
To explain this, the authors introduce a Three-Gate Theory: (1) a KL-regularization “leash”, (2) the pretrained model’s geometry directs those constrained updates, and (3) limited bfloat16 precision amplifies the effect by hiding some parts of updates.
Empirical analyses across several RL variants and models show stable, structured update patterns, and minimal overlap with principal weights, supporting the theory.
The paper concludes that RL’s distinctive optimization dynamics differ from supervised fine-tuning and that understanding this bias could inform more efficient RL fine-tuning methods.

### Strengths
To explain the success of RL in LLM finetuning is a timely topic. The main theoretical claims are well substantiated for the first two gates. The work discusses an important design principle for finetuning in general.

### Weaknesses
1) The first two gates are related so they are not actually two distinct phenomena. The KL "leash" is causing the minimal change in parameters. 

2) Furthermore, the fact that the KL divergence update leads to minimal updates is not surprising as it is a well known result understood in optimization literature as the optimization geometry [5]. For example in RL the GRPO can be written as an exponential update as in Theorem 1 [6] which leads to sparse updates.

3) The third gate is not well motivated by experimental evidence as there are only BF16 results in Table 1. First of all would this not hold for all methods i.e. less precision would act as rounding therefore less updates.

4) While the work acknowledges [1], it seems to provide similar insights than it. The subnetwork discovery in [1] is a stronger claim than the implicit regularization caused by the KL.

5) The work seems to use to much LLM generated text, evidence: The use of "Optimization Compass" (Line 351) instead of  Optimization induced regularization or implicit regularization this happens at many instances in the manuscript, sub- and super-random (Line 087) instead of positive or negative correlated.

6) There is large body of work on implicit regularization by optimization which seems to be quite important as the work is about an "implicit compass" that guides the optimization [5].

7) As the work finds thar RL leads affecting less parameters, more relevant finetuning methods in literature on subgroup finetuning are [2,3,4]. How would RL type of methods compare to these type of methods?


[1] Mukherjee, Sagnik et al. “Reinforcement Learning Finetunes Small Subnetworks in Large Language Models.” ArXiv abs/2505.11711 (2025): n. pag.

[2] Modoranu, Ionut-Vlad et al. “MicroAdam: Accurate Adaptive Optimization with Low Space Overhead and Provable Convergence.” ArXiv abs/2405.15593 (2024): n. pag.
[3] Rios, Jesus et al. “Sparsity May Be All You Need: Sparse Random Parameter Adaptation.” ArXiv abs/2502.15975 (2025): n. pag.
[4] Zhou, Chao et al. “Pay Attention to Small Weights.” ArXiv abs/2506.21374 (2025): n. pag.

[5] Gunasekar, Suriya et al. “Characterizing Implicit Bias in Terms of Optimization Geometry.” ArXiv abs/1802.08246 (2018): n. pag.
[6] Mroueh, Youssef. “Reinforcement Learning with Verifiable Rewards: GRPO's Effective Loss, Dynamics, and Success Amplification.” ArXiv abs/2503.06639 (2025): n. pag.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper attributes the sparse appearance of RL updates to three factors: (i) an on-policy KL anchor, (ii) the model’s pretrained geometry, and (iii) the use of bf16, which renders many small parameter updates numerically invisible. The analysis shows that RL maintains relatively stable spectral properties, induces less subspace rotation than SFT, and tends to avoid modifying principal weights—whereas SFT more frequently affects principal directions in the parameter space.

### Strengths
- Clear narrative supported by strong evidence.
The paper presents a coherent story backed by quantitative evidence. The results show strong cross-run overlap (Jaccard/consensus metrics), indicating structured consistency rather than random speckle patterns.

- Insightful PEFT takeaway.
The analysis demonstrates that “principal-only” masks underperform, while non-principal or low-magnitude masks better align with dense KL trajectories and maintain accuracy closer to the dense baseline.

- Precision as an interpretive lens.
The work highlights how numerical precision affects interpretability: with fp32 storage or larger learning rates, the observed “sparsity” largely disappears. This suggests that bf16 precision mainly influences the visibility rather than the existence of updates.

- Cross-family generalization.
Although most evidence is centered on Qwen, the paper includes additional results on LLaMA and Mistral, indicating that the observed phenomena are not model-specific.

### Weaknesses
- Missing precision sweep.
The paper does not include results for fp16 or bf8. A small precision ablation would help validate the claim that the observed sparsity is primarily an artifact of numerical representation.

- Incomplete SFT comparison.
The authors claim that SFT targets principal weights, yet they do not present SFT update-mask overlap plots at the same granularity as those for RL. Including these would substantiate the contrast more rigorously.

- Limited cross-model evaluation.
While the paper includes some non-Qwen results, these experiments are lighter and less comprehensive. More like-for-like evaluations on LLaMA and Mistral would strengthen claims of generality.

- Reproducibility concerns.
The analysis code—particularly for computing masks, rotations, and subspace probes—does not appear to be released. Open-sourcing these components would enhance reproducibility and transparency.

### Questions
- If SFT is trained with a KL-to-base regularization term, does its update-mask overlap with principal weights decrease toward RL’s pattern? Could the authors include SFT-mask overlap visualizations for direct comparison?

- Could the authors include fp16 and bf8 storage or accumulation settings to further evaluate the “precision as lens” hypothesis?

- Could the authors share code or detailed settings for the right-multiplication block-orthogonal rotations (e.g., per-head grouping, GQA configurations), as well as your exact procedure for computing subspace angles? Clarifying these would help reconcile potential discrepancies in replication.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates why Reinforcement Learning with Verifiable Rewards (RLVR), achieves major reasoning improvements in large language models while modifying only a small portion of parameters. The authors propose that this apparent sparsity reflects an implicit optimization bias—an “implicit compass” guiding updates.

They introduce a Three-Gate Theory:
- Anchoring (Gate I): KL regularization constrains updates near the base policy.
- Geometry (Gate II): The pretrained model’s structure steers updates toward low-curvature regions, avoiding principal weights.
- Precision (Gate III): bfloat16 precision amplifies apparent sparsity by hiding small updates.

Extensive experiments across models and RL variants show that RL preserves spectral structure, avoids principal weights, and exhibits stable, geometry-aligned update patterns. Causal interventions confirm that model geometry drives this bias. The work reframes sparsity as a byproduct of structured optimization.

### Strengths
1. Fresh perspective on RL updates:

The paper gives a genuinely new way to think about why RL updates look sparse. The “implicit compass” idea connecting KL constraints, model geometry, and precision is original and thought-provoking.

2. Strong empirical evidence:

The experiments are thorough and consistent across models. I especially liked the causal intervention tests — they make a convincing case that the observed patterns aren’t just artifacts.

### Weaknesses
1. General claim but limited evaluation:

The paper makes a rather broad theoretical claim about the nature of RL updates under KL constraints, arguing that the observed sparsity pattern and directional bias are universal properties of such optimization dynamics. However, the experimental validation is restricted to RLVR applied to large language models. This raises questions about how general the findings really are.

To strengthen the claim, it would be valuable to test whether the same “implicit compass” behavior appears in other RL formulations or domains. For instance, examining smaller-scale robotic control tasks, or an imitation learning setup where a pretrained policy is fine-tuned either through RL or supervised learning, could reveal whether this bias is a general feature of KL-regularized RL or specific to language models.

2. Unclear practical takeaway:

While the proposed “implicit compass” theory is conceptually appealing and provides a new lens to interpret the geometry of RL updates, its practical implications remain underexplored. The paper convincingly shows that RL updates behave differently from supervised fine-tuning, but it does not clearly explain how this understanding can be used to improve algorithm design.

For example, could this theory inform new geometry-aware optimization schemes, selective parameter updates, or adaptive KL regularization strategies? Are there concrete steps to make RL training more stable or sample-efficient based on this insight? Clarifying how the theory translates into tangible algorithmic guidance, or even including a small proof-of-concept experiment would significantly enhance the practical impact of the paper.

### Questions
1. Relation to recent sparsity work:

Recent papers in the RL literature [1,2] have shown that introducing explicit sparsity into model weights can improve scaling performance. According to your Three-Gate Theory, sparsity already emerges naturally as a byproduct of the RL update process constrained by KL regularization. How should we reconcile these findings? Why does imposing stronger, explicit sparsity still seem to help, if sparsity is already an implicit consequence of RL optimization? Does your theory suggest that RL inherently requires stricter or more structured forms of sparsity beyond what naturally emerges?

[1] Network Sparsity Unlocks the Scaling Potential of Deep Reinforcement Learning, ICML 2025 (Oral)

[2] Rethinking the Role of Dynamic Sparse Training for Scalable Deep Reinforcement Learning, arXiv 2025


I lean toward a weak reject at this stage. The paper presents an interesting and original theoretical perspective on why RL updates appear sparse, but the practical significance are not yet convincing enough. That said, I am open to improving my score if the authors can clearly demonstrate why this theory matters; for example, how the proposed “implicit compass” insight could meaningfully guide the design of more efficient or better RL algorithms.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper asks why RLVR seems to improve reasoning models while touching surprisingly few parameters. The authors argue that observed “sparsity” is mostly an artifact of an implicit optimization bias (“implicit compass”) rather than true selectivity. Their Three-Gate Theory combines: (1) an on-policy KL leash (Anchor Gate), (2) pretrained geometry with easy vs. principal directions (Geometry Gate), and (3) bf16 precision masking tiny updates (Precision Gate). They show RLVR preserves spectral structure and avoids principal weights compared to SFT, and a causal intervention that scrambles layer geometry disrupts the effect. Finally, masks derived from this perspective approach dense training, pointing toward RL-specific PEFT.

### Strengths
- Mechanistic clarity: The Three-Gate framing connects training constraints to parameter movement in a way practitioners can reason about.
- Causal test: Scrambling/rotating geometry to break the bias goes beyond correlation.
- Better measurement: The bf16-aware sparsity metric corrects prior over-interpretations.
- Practical value: The mask results suggest RL-specific PEFT is both necessary and feasible.

### Weaknesses
- Curvature proxy: “Principal weights” stand in for high curvature. Even a coarse Fisher-diagonal estimate would help validate this mapping.
- Precision disentanglement: It’s not fully shown whether bf16 is merely a lens vs. a contributing cause. A strict fp32-only run (weights + optimizer + no casts) would clarify Gate III.
- Generalization breadth: Deep analysis centers on Qwen/DeepSeek. Quantitative replication on Llama/Mistral would support universality claims.

### Questions
- Beyond bf16: If you train and store everything in fp32, do the consensus stripes and spectral preservation persist?
- Curvature validation: What is the correlation between your principal-weight masks and per-parameter Fisher diagonals (even on a small batch)?
- Gate coupling: How do the stripes change as you sweep KL β (explicit) or clipping ε (implicit leash)? A monotone response would directly tie Gate I to Gate II.

### Soundness
3

### Presentation
4

### Contribution
4
