# EPiC: Towards Lossless Speedup for Reasoning Training through Edge-Preserving CoT Condensation

- Avg Score: 3.33
- Decision: Reject
- Scores: 4, 4, 2

## Abstract
Large language models (LLMs) have shown remarkable reasoning capabilities when trained with chain-of-thought (CoT) supervision. However, the long and verbose CoT traces, especially those distilled from large reasoning models (LRMs) such as DeepSeek-R1, significantly increase training costs during the distillation process, where a non-reasoning base model is taught to replicate the reasoning behavior of an LRM. In this work, we study the problem of CoT condensation for resource-efficient reasoning training, aimed at pruning intermediate reasoning steps (i.e., thoughts) in CoT traces, enabling supervised model training on length-reduced CoT data while preserving both answer accuracy and the model’s ability to generate coherent reasoning. Our rationale is that CoT traces typically follow a three-stage structure: problem understanding, exploration, and solution convergence. Through empirical analysis, we find that retaining the structure of the reasoning trace, especially the early stage of problem understanding (rich in reflective cues) and the final stage of solution convergence (which closely relates to the final answer), is sufficient to achieve lossless reasoning supervision. To this end, we propose an Edge-Preserving Condensation method, EPiC, which selectively retains only the initial and final segments of each CoT trace while discarding the middle portion. This design draws an analogy to preserving the “edge” of a reasoning trajectory, capturing both the initial problem framing and the final answer synthesis, to maintain logical continuity. Our analyses leveraging the CoT landscape and measuring the mutual information between CoT steps provide further validation for this design. Experiments across multiple model families (Qwen and LLaMA) and benchmarks show that EPiC reduces training time by over 34% while achieving lossless reasoning accuracy (e.g., on MATH500), comparable to full CoT supervision. Additionally, we show that EPiC outperforms other condensation methods, including teacher-guided regeneration of condensed CoTs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
A framework for thought-level condensation is introduced, enabling efficient knowledge distillation by capturing reasoning thoughts with a smaller model. While the effectiveness is demonstrated, uncertainties remain regarding the proposed techniques.

### Strengths
* This paper introduced a framework for thought-level condensation that enables efficient knowledge distillation. It effectively captures reasoning thoughts using a smaller non-reasoning model.

* The authors conducted analysis through visual illustrations and quantitative analysis to demonstrate that reasoning information is preserved.

* Experiments were conducted to demonstrate the effectiveness of the proposed framework.

### Weaknesses
* Figure 1 uses bar charts for both performance and efficiency, which is not appropriate since they belong to different categories and should be represented on separate axes.

* Section 4 is crucial for the proposed method, but not all details are clearly presented. While all the key points are logical, they need to be clearly articulated and analyzed to make the techniques convincing. 

* In Figure 3, the example of CoT trace is interesting, where the authors pointed out the concerns of token based CoT condensation that drops self-reflective words and transition words, resulting in fragmented input. Is it the intrinsic issue of the token based CoT condensation, or just TokenShip may use an imperfect scoring mechanism for tokens? Anyway, there is no result if thought-level pruning would be possibly mapped back to the token level to make the comparison tangible. 

* While the visualization in Figure 4 can be useful for delivering the message that dropping the middle thought is reasonable, it also trigger a concern if the visualization is really based on real trajectory of reasoning, or just a make-up illustration. It is a bit questionable why the tail though can jump to the correct answer region significantly from the yellow dot that has been far away from even comparing to the start position. If this observation is true, it seems the author can even skip the head of thoughts. 

* In Table 1, it seems the proposed EPiC performs relatively even more stronger when the condensation ratio \tau is low. The authors should provide some insight. Besides, the gap seems a bit too huge. Is the baseline method used here strong enough?

### Questions
Please check the weaknesses section.

### Soundness
2

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
The paper studies the problem of Chain-of-Thought (CoT) compression in reasoning training, which enhances the reasoning capability of smaller models by training them on distilled CoT data. The paper proposes an Edge-Preserving Condensation method called EPiC, which involves pruning the intermediate steps of a reasoning trajectory. The authors further analyze the rationale of EPiC through a comparison of mutual information. Finally, the authors conduct experiments across different benchmarks and models to verify that EPiC is utility-preserving and highly efficient.

### Strengths
S1: The research problem of this paper, the efficient training of reasoning models, is both timely and highly relevant.

S2: The paper is well-written and easy to follow. Additionally, the proposed method is simple yet effective.

S3: The authors conduct extensive experiments to verify the effectiveness of EPiC, and the reported results show that it achieves a strong utility-efficiency trade-off.

### Weaknesses
W1: The experimental setup presented in Figure 2 seems questionable, which weakens the paper's motivation. The S1 and LIMO datasets are significantly smaller in the number of examples than OpenR1Math. Therefore, it is expected that models trained on S1 and LIMO would underperform those trained on OpenR1Math. A fairer comparison would involve ensuring that the total number of tokens in the condensed dataset (e.g., after 50% randomized thought-level condensation) is comparable to the token counts of the S1 and LIMO datasets.

W2: The proposed method is presented as a heuristic and appears to be motivated by a single example visualization in Figure 4. The argument would be more persuasive if it were supported by a statistical analysis of the spatial distances between each step thought and the correct answer across an entire dataset. While the visualization in Figure 4 is well-presented, the example shown might be a corner case or a specific, non-representative instance, which reduces the method's persuasiveness.

W3: There appears to be a contradiction regarding whether the middle portion of a CoT trace is less informative when comparing Figure 4 and Table 5. Table 5 reports a higher MI for the MoC compared to the ToC, which suggests that the middle portion is actually more informative than the tail.

W4: While the paper includes several experiments, the section could be strengthened by including more baselines and more in-depth analysis.

-	In Table 2, adding MoC as a baseline would help to directly test the hypothesis that the middle portion of a CoT trace is less informative.

-	In Table 2, EPiC significantly underperforms training on the full dataset. The paper lacks an in-depth analysis of this performance gap, which could help in understanding the failure cases of EPiC.

W5: Line 279 seems to assume that the length of the CoT is the same for the “understanding and convergence” phases. This assumption seems unreasonable, as one would expect this length to vary across different samples and datasets.

Minor points:

-	In line 325, “xWe” -> “We”

### Questions
Q1: What method is used to partition CoT into three distinct sections?

### Soundness
3

### Presentation
3

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
The paper proposes EPiC, an edge-preserving condensation method that prunes the middle portion of chain-of-thought (CoT) traces while retaining the initial and final reasoning segments to reduce training cost.
Based on empirical analyses (e.g., mutual-information metrics), the authors show that these “edge” segments preserve logical coherence and supervision quality.
Experiments across multiple model families demonstrate notable training-time reduction with only mild loss in reasoning accuracy.

### Strengths
- The topic and motivation, resource-efficient reasoning training are important and timely.

- The paper is well-written and easy to follow, with a clear presentation of the problem and method.

- The authors conduct comprehensive empirical analyses, offering useful insights that intermediate reasoning steps may be less important than early or final stages.

### Weaknesses
**[W1] Inconsistent and limited performance improvements.** In Table 2, the proposed method outperforms the baseline on MATH, AIME-25, and GPQA, but shows a significant drop on AIME-24. Moreover, efficiency metrics such as the number of tokens do not show clear advantages, following trends similar to competing methods.

**[W2] Insufficient baselines and related work.** Recent studies have analyzed the importance of reasoning steps or proposed strategies (e.g., [1, 2]) that appear directly applicable to this setup. Including these as baselines and discussing them in the related-work section would provide a more holistic comparison.

[1] Choi et al., Think Clearly: Improving Reasoning via Redundant Token Pruning, EMNLP 2025 (Findings)

[2] Cui et al., Stepwise Perplexity-Guided Refinement for Efficient Chain-of-Thought Reasoning in Large Language Models, ACL 2025 (Findings)

### Questions
**[Q1]** Why is the maximum sequence length in GPQA experiments restricted to 4k tokens? Reasoning performance is often sensitive to the length limit; reporting results with longer contexts would strengthen the claim.

**[Q2]** As I understand, EPiC preserves reasoning properties such as reflection markers (e.g., “Wait”). Is there a specific reason for this? Intuitively, intermediate reasoning steps might contain more such reflective cues.

**[Q3]** Intermediate steps can affect test-time scaling, as they enhance reasoning diversity. Can EPiC maintain or recover this advantage under test-time scaling scenarios?

### Soundness
2

### Presentation
3

### Contribution
2
