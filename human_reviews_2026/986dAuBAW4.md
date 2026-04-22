# Identity Bridge: Enabling Implicit Reasoning via Shared Latent Memory

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
Despite remarkable advances, large language models often fail at compositional reasoning tasks, a phenomenon exemplified by the ``curse of two-hop reasoning''. The inability of large models to perform implicit reasoning as expected remains an intriguing mystery. In this work, we unexpectedly discover that even a simple, single-layer Emb-MLP model can effectively learn to generalize out-of-distribution (OOD) multi-hop reasoning. Specifically, we demonstrate that transformers can successfully address multi-hop tasks through simple shortcuts rather than implicit reasoning with the help of Identity bridge. Our experiments and theoretical analysis established the learning mechanisms of the Emb-MLP model, which is the excellent approximation of GPT-2 on multi-hop task. We further compare the performance of the Emb-MLP model with GPT-2 across various complexities of two-hop reasoning and identify that smaller initializations or larger weight decay parameters enhance the model’s ability to harness implicit reasoning. Finally, we generalize our observations to real-world LLMs, presenting evidence that they implicitly acquire this necessary alignment during pretraining, which underpins their compositional abilities.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper explores why LLMs cannot solve what is known as two-hop reasoning ($A → B, B → C$ implies $A → C$). They propose to add during training an identity mapping between intermediate "bridge" entities. The authors argue it improves the model's ability to compose two relations together. They demonstrate this effect on a simple synthetic setup and on real world tasks, demonstrating this phenomenon at model scales up to 8B.

### Strengths
The proposed solution is simple and demonstrates significant improvement on two-hop reasoning tasks. The authors relate several interesting ideas, such as connecting grokking to the identity bridge.

### Weaknesses
* The notation and presentation of the synthetic setup are difficult to follow. For example, the authors introduce multiple entity and relation sets $\varepsilon_1, \varepsilon_2, \varepsilon_3$ to denote the subject, bridge, and object sets, respectively, and there is to distinguish between the three. The index notation $i$, $j$ is heavily overloaded (it is used to index the entity set $\varepsilon_i$, the elements within the bridge entities $\varepsilon_{2,i}$, the elements within the relation set $r_i$, and so on). 

* In Figure 2, the results primarily serve to show that the choice of weight initialization matters for generalization on harder complexity tasks. But I find this setup to be not well motivated and could benefit from further clarification. This finding seems to suggest that two-hop capabiiites can only be unlocked given certain initialization settings. I can see that section 4.4 further explains this, but then I am confused as to why section 4.3 on the Emb-MLP is necessary. In short, I am confused as to what is the natural question that motivates the study of different initializations in Figure 2.

### Questions
Two-hop reasoning still feels relatively contrived. How well does this analysis/experiment translate to multi-hop reasoning, beyond just two steps?

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
3

### Summary
The paper addresses a core limitation of LLMs, failure at compositional reasoning, particularly two-hop reasoning.
It introduces Identity Bridge, a simple zero-hop auxiliary task that enforces latent-space alignment between bridge entities and improves out-of-distribution (OOD) two-hop generalization.

### Strengths
- Conceptual clarity. The “identity bridge” idea is strikingly simple yet yields strong intuition about latent geometry and shared memory.
- Theoretical-empirical coherence. The nuclear-norm analysis provides a plausible mechanistic link between auxiliary supervision and OOD reasoning.
- Good connection to literature. Relates to grokking, implicit bias, and compositional generalization.
- Readable presentation. Figures 3–8 visualize reasoning patterns and hidden-state alignment.

### Weaknesses
1. Novelty scope. The “identity bridge” can be viewed as a trivial self-supervised auxiliary loss; the work should better distinguish it from prior identity-based pretraining (e.g., autoencoding, copy tasks).
2. Theoretical depth. The Emb-MLP uniform-attention analysis is elegant but may be too idealized; discussion should address limitations of extending results to real transformers.
3. Empirical diversity. All tasks are synthetic or limited to the TWOHOPFACT dataset; a real-world compositional benchmark would greatly strengthen claims.
4. Interpretation of LLM results. The “implicit identity bridge during pretraining” claim (Sec. 4.5) is speculative; more ablations or probing experiments are needed to support this inference.

### Questions
See above

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper studies a phenomenon in LLMs reasoning where models often fail to learn "A to C" from "A to B" and "B to C", which is termed as the curse of two-hop reasoning. One understanding is that the latent representation of B from the first hop does not necessarily aligns with the representation from the second hop. The paper proposes a simple fix, called identity bridge, by adding zero-hop identity "B to B" to bridge this gap. Empirical results on synthetic data seem promising. Theoretical understanding is also provided using a simplified linear model, showing that the identity bridge induces nuclear norm regularization.

### Strengths
The paper presents a clear idea that tries to address the curse of two-hop reasoning. The proposed mechanism is simple and interpretable.  An intuitive theoretical link is provided to better understand how this simple fix helps with the two-hop reasoning phenomenon. Empirical results on synthetic data match with theoretical insights.

### Weaknesses
1. I am still confused why a new mechanism is required given that CoT already works well. The paper discussed some shortcomings of CoT, but all of them are not verified in this paper's setting. I can imagine that CoT would work perfectly on both the synthetic data and real-world data. Given that the accuracy improvement brought by the identity mapping is not significant on real-world dataset, I would still do CoT compared with the identity bridge. The motivation of this paper to study the identity bridge needs further explanation.

2. The claim on implicit nuclear norm regularization is not mathematically rigorous. Eq. (1) and Eq. (2) are two different optimization problems, and they have different solutions. It is not correct to just study Eq. (2) as a reformulation of Eq. (1). There is no formal and rigorous proof on the equivalence between the two but only some empirical observations on synthetic data. This is not enough to just claim identity bridge induces implicit nuclear norm regularization.

3. The statements of Theorem 1 and 2 are not precise. In particular, important assumptions such as Assumptions 1-4 are not even mentioned and discussed in the statements. This is misleading and gives a sense that the theoretical results are very general. In fact, they only work for very limited cases. For example, why can we just assume $1^\top W=0$ in Assumption 2? This is never verified and particularly important for the theorems to be true. Therefore, I think the theoretical results are a bit overclaiming.

4. More baselines and ablation studies could be added. Are the empirical results stable against randomness, e.g., change a different random seed?

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper considers the task of 2-hop reasoning: Given an entity e1, and relations r1 and r2, the objective is to find tail entity e_answer by following the reasoning (e1, r, e_bridge) and (e_bridge, r2, e_answer). This task is commonly hard for LLMs without chain-of-thought reasoning, especially when the pair (r1, r2) is not observed during training time. To improve such OOD 2-hop reasoning, the paper introduces “identity bridge” tasks where the model has to simply implement the identity function over bridge entities. 

In addition to experimenting with transformers the paper introduces a simpler neural family, “Emb-MLP” which is a simple BoW based predictive model. Initial results show that both transformers and Emb-MLP have similar performance on the 2-hop OOD reasoning task, and thus the paper analyzes Emb-MLP further.

### Strengths
- Results show that this additional auxililary training objective dramatically improves OOD generalization on 2-hop tasks. By training on the bridge identity task, geometry of the embeddings for Emb-MLP task shifts in a way that supports OOD generalization.
- Identifies a practical intervention (identity supervision) that addresses a known failure mode and provides both empirical and theoretical characterization of the phenomenon
- The theoretical analysis (Theorems 1-2) for the Emb-MLP model is rigorous, with detailed proofs
- Experimental validation across multiple complexity levels shows consistent patterns

### Weaknesses
- The theory relies heavily on Assumptions 1-4, particularly the regularity and non-degeneracy assumptions (3-4), which are not empirically validated
- The gap between the uniform-attention Emb-MLP theory and actual transformer behavior is acknowledged but not bridged
- The real-world LLM experiments (Section 4.5) are more correlational than causal—they show probability shifts consistent with the theory but don't definitively prove the mechanism

### Questions
- Do learned solutions empirically exhibit the non-degeneracy properties assumed?
- How sensitive are the results to the specific identity task formulation? What if identity mapping is noisy or only partially supervised?
- The paper claims identity bridges emerge during pretraining—can you provide more direct evidence (e.g., probing experiments on models at different pretraining checkpoints)?

### Soundness
3

### Presentation
3

### Contribution
3
