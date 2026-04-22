# Self-Consistency Improves the Trustworthiness of Self-Interpretable GNNs

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 4, 4, 8

## Abstract
Graph Neural Networks (GNNs) achieve strong predictive performance but offer limited transparency in their decision-making. Self-Interpretable GNNs (SI-GNNs) address this by generating built-in explanations, yet their training objectives are misaligned with evaluation criteria such as faithfulness. This raises two key questions: (i) can faithfulness be explicitly optimized during training, and (ii) does such optimization truly improve explanation quality? We show that faithfulness is intrinsically tied to explanation self-consistency and can therefore be optimized directly. Empirical analysis further reveals that self-inconsistency predominantly occurs on unimportant features, linking it to redundancy-driven explanation inconsistency observed in recent work and suggesting untapped potential for improving explanation quality. Building on these insights, we introduce a simple, model-agnostic self-consistency (SC) fine-tuning strategy. Without changing model architectures, SC consistently improves explanation quality across multiple dimensions and benchmarks, offering an effective and scalable pathway to more trustworthy GNN explanations. Our code is publicly available at \url{https://github.com/ICDM-UESTC/SelfConsistencyXGNN}.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduce a simple, model-agnostic self-consistency (SC) post-training strategy for pretrained Self-Interpretable GNNs (SI-GNNs). 
Rigorous analysis demonstrates that faithfulness can be explicitly optimized through the strategy, and such optimization genuinely improves explanation quality.
Experiments show that SC consistently improves explanation quality across multiple dimensions and benchmarks, offering an effective and scalable pathway to more trustworthy GNN explanations.

### Strengths
- The paper provides a fresh perspective (Self-Consistency) to improves the trustworthiness of self-interpretable GNNs, , the theoretical and empirical verification of the key assumptions is sufficient, there are no obvious flaws.
- The method demonstrates strong explanation performance across multiple tasks, improving various types of self-Interpretable GNNs.
- The paper is well-organized and easy to follow.

### Weaknesses
1. During the fine-tuning phase of SC (Section 3.1), this work freezes the GNN encoder and assumes that "the encoder representation is already optimal, and only the explainer needs optimization". However, this work fails to validate this design and could further compare it with a control group where "the encoder is not frozen".  
2. Has the author conducted statistical significance tests? What percentage of the experimental results significantly outperform each baseline? This is crucial for understanding the overall performance of SC.

### Questions
Please refer to the weaknesses for suggestions. This article is highly accomplished, and the weaknesses mentioned above will not change my positive evaluation.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper studies a common mismatch in self-interpretable GNNs (SI-GNNs): models are trained with classification plus conciseness regularization, but are evaluated by faithfulness (whether the highlighted subgraph alone reproduces the prediction). The authors argue that faithfulness is intrinsically linked to self-consistency (SC)—the explanation produced on a graph should be reproduced when the model is run on its own explanation. They propose a simple, model-agnostic training strategy that adds a self-consistency loss during a short fine-tuning stage where the GNN encoder is frozen and only the explainer and classifier are updated, yielding a final objective.
They analyze how enforcing self-consistency drives importance scores toward a few “near fixed levels,” and how this interacts with conciseness regularization (CR) such as sparsity (GISST) or MI constraints (GSAT) to suppress unimportant edges while preserving important ones.
On four benchmarks (BA-2MOTIFS, 3MR, BENZENE, MUTAGENICITY) and four SI-GNN families (GISST, GSAT, GAT, CAL), SC improves consistency, faithfulness, explanation accuracy, and informativeness; it also complements explanation ensembling (EE) and is substantially more efficient at inference. SC alone may hurt when CR is absent (GAT/CAL), but adding CR stabilizes and recovers gains. The method requires no architectural change and adds one low-sensitivity hyperparameter ($\eta$) .

### Strengths
1. Reframes faithfulness optimization as enforcing *self-consistency* during training and connects it to redundancy on unimportant edges; provides a model-agnostic, plug-in loss with a two-stage schedule that freezes the encoder  .
2. Extensive experiments on four datasets and four SI-GNN families. Clear reporting that SC alone can fail without CR, plus the complementary effect with EE. Solid ablations on ($\beta$) and ($\eta$) sensitivity and two-stage training to rule out confounds  .
3. The training procedure and metrics are well defined; the figures make the stability effect tangible, and the paper carefully explains when SC helps and when it needs CR support .
4. Practical because it needs no architecture changes, adds a single hyperparameter of low sensitivity, and improves multiple explanation dimensions while being more efficient than EE at inference time .

### Weaknesses
1. The two-stage setup assumes that Step 1 already produces explanations covering the ground-truth rationale. Step 2 then removes redundancy. If Step 1 misses key parts of the rationale, SC may stabilize an incomplete subset rather than recover it. The paper should explicitly state this assumption and test its robustness.

2. Adding more initial explanations (e.g., multiple seeds or ensembling in Step 1) could alleviate this limitation. The gains seen in Table 1 when combining with EE support this interpretation.

3. All datasets are small; results on large-scale or node-level tasks would clarify generality.

4. Limited theory: The fixed-level analysis is illustrative but not formal; it would be valuable to characterize conditions guaranteeing convergence toward faithful explanations.

5. Incomplete metric coverage: Improvements on FID− are strong, while FID+ results remain mixed; additional discussion would improve completeness.

### Questions
1. What happens if the Step 1 SI-GNN misses part of the ground-truth rationale? Please evaluate robustness by ablating a portion of important edges before Step 2 and reporting whether SC can recover faithfulness or simply stabilize incomplete masks.

2. How many models are produced overall, and which one generates explanations? My understanding is that only a single model—the Step 2 checkpoint—is used at inference. If multiple initial explanations are generated in Step 1, are they ensembled before Step 2 or during inference?

3. Consider generating multiple Step 1 explanations and comparing unions or averaged masks before Step 2. This could test whether broader initial coverage yields better results.

4. Since SC alone can hurt for GAT/CAL, what is the weakest form of conciseness regularization ((L_1) or entropy penalty) that stabilizes training?

5. Could encoder-side regularization, such as partial Mixup on masked graphs, reduce encoder distribution shift and improve FID+?

6. Would allowing the last encoder layer to update during Step 2 better adapt to explainer changes without destabilizing learned representations?

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
4

### Summary
The authors address a relevant and widespread problem in GNN (and SE-GNN in particular), which is the consistency and reliability of the explanations being extracted, and propose an approach to encourage self-consistency of explanations at training time.

### Strengths
Trying to directly enforce consistency at training time seems like a sensible direction.

Experimental results confirm improvements and provide insights into explanations quality and robustness.

### Weaknesses
The problem of consistency of explanations was already addressed in the work by Tai et al, where they propose a simple esamble strategy (EE). The novelty of the work is thus not dramatic. The authors show that their approach improves over EE (apart from being clearly much faster), and their combination further improves results. The rationale for these results is unclear to me. Is there any substantial difference in what the approaches tackle justifying this? Is this a matter of hyperparameter choice? This is important to correctly evaluate the relevance of the contribution.


There are plenty of notions of faithfulness of explanations, but a key aspect is that one should measure both sufficiency (e.g. with FID-) and precision (e.g. with FID+). I thus think the authors should include FID+ (or similar metrics) to get the full picture. Given that the proposed SC component is combined with conciseness regularization (CR), I do not expect this to undermine the utility of the approach, but it would allow to get a better picture of its contribution, especially when seeing the FID+ results without CR (table 3).

### Questions
Can you clearly motivate the performance difference between SC and EE? aren't they basically optimizing for the same objective?

Can you add FID+ results and comment them to better understand the interplay between SC and CR?

### Soundness
3

### Presentation
3

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
The authors propose a simple but sensible model-agnostic strategy for improving the faithfulness of self-explainable GNNs. The idea is to penalize the model for "changing its mind" whenever fed with a local explanation of a certain prediction during training. The technique is validated using four approaches and compared
against an alternative (explanation ensembling) on four datasets, with encouraging
results.

### Strengths
- **Originality**: The idea behind the penalty is aligned with existing results,
  but otherwise original -- to the best of my knowledge.

- **Quality**: The proposed idea is sensible. The empirical setup is also good
  -- the choice of datasets, metrics and competitors all look good. I appreciate
  how the authors clearly distinguish between faithfulness (measured with Fid-)
  and plausibility/explanation accuracy. This is surprisingly rare in the
  literature.

- **Clarity**: The text is very clear and well structured. The visualizations
  are helpful.

- **Significance**: The contribution is welcome and I think it bridges a
  serious gap in the literature, by making theoretical insights practical.
  The fact that the approach is model agnostic also helps.

TL;DR: good paper, I like the idea and the execution.

### Weaknesses
- **Originality**: As I mentioned, I believe the proposed technique follows
  naturally from existing results (eg Azzolin et al, who the authors mention).
  This is however not a major issue for me - the penalty, as I mentioned,
  is novel.

- **Clarity**: My only real complaint is that Table 1 has too many colors, making
  it diffult to focus on what's really relevant. I'd suggest to tone it down.

### Questions
None.

### Soundness
3

### Presentation
4

### Contribution
3
