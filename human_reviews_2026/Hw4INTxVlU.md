# Sample-Efficient Alignment for LLMs

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
We study methods for efficiently aligning large language models (LLMs) with human preferences given budgeted online feedback. We first formulate the LLM alignment problem in the frame of contextual dueling bandits. This formulation, subsuming recent paradigms such as online RLHF and online DPO, inherently quests for sample-efficient algorithms that incorporate online active exploration. Leveraging insights from bandit theory, we introduce a unified algorithm based on Thompson sampling and highlight its applications in two distinct LLM alignment scenarios. The practical agent that efficiently implements this algorithm, named SEA (Sample-Efficient Alignment), is empirically validated through extensive experiments across three model scales (1B, 2.8B, 6.9B) and three preference learning algorithms (DPO, IPO, SLiC). The results demonstrate that SEA achieves highly sample-efficient alignment with oracle's preferences, outperforming recent active exploration methods for LLMs.
We will release our codebase to hopefully accelerate future research in this field.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces SEA (Sample-Efficient Alignment), a new framework for aligning large language models (LLMs) with human preferences using minimal online feedback. Existing alignment paradigms are unified as a contextual dueling bandit (CDB) problem. By leveraging Thompson Sampling principles from bandit theory, SEA actively balances exploration and exploitation during online alignment. The experiments across model scales and preference learning algorithms show the high sample efficiency of the proposed algorithm.

### Strengths
1. The perspecitve of CDB is interesting and novel.
2. The implementation is introduced in details. 
3. The relevant LLM alignment methods are covered, which is helpful for the readers to understand the new algorithm.

### Weaknesses
1. The base models in the experiments are all from the Pythia family. The results of open-source model from other family is needed to demonstrate the algorithm is generally effective.
2. The pseudocode of Algorithm 1 is confusing. Only one colored box, which refers to E&E or BAI, will be enforced. But the current presentation will make the readers understand as $y_t'$ is first sampled by E&E then sample by BAI, which means E&E is actually disabled.
3.  The datasets and tasks in the experiments are not diversified enough. The performance of the proposed algorithm on various task domain is an important message but missed here.
4. The author list two key properties for sample-efficient alignment algorithm, "online interaction" and "active exploration". And in Section 3,  a new term is introduced as "online exploration". However, the correlation between "online interaction" and "active exploration" is not fully illustrated, making the reader confused with different terms.

### Questions
1. With the formulation of CDB, could you provide any theoretical analysis for the algorithm to validate its sample efficiency (such as regret)?
2. For the results shown in the ablation analysis in Section 6.2, is there any takeaway messages?

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
3

### Summary
The manuscript addresses the problem of aligning LLMs by formulating it as a contextual dueling bandit problem, with the goal of improving the sample efficiency of fine-tuning procedures. The authors propose a practical algorithm along with a system solution that effectively operationalizes this framework. Experiments are conducted on both TLDR summarization and alignment tasks. The results demonstrate the efficacy of the proposed method, as evidenced by superior win-rate-versus-sample-size curves compared to the baselines.

### Strengths
* It introduces a novel and interesting perspective on framing the LLM alignment problem: contextual dueling bandits. This may provide a fresh theoretical angle for future research.
* The experiments are comprehensive.

### Weaknesses
* The overall writing quality needs to be greatly improved! I really think there are a lot of redundant parts that reduce readability and clarity.
    - For example, section 2 and 3 introduces an overwhelming amount of preliminary knowledge without a clear narrative that connects it to the motivation for viewing LLM alignment as a contextual dueling bandit problem.
    - Another example is that the actual algorithm is in the appendix while a not practical one is displayed in the main text. 

    I feel like the writing should focus on presenting only the most essential content, rather than extensively detailing how the question is formalized. Readers are not obliged to follow the authors' entire reasoning trajectory!
* The paper should include at least a few example responses to avoid the concern that outputs from both the baseline and the proposed method may be of low quality.
* Although this may be somewhat subjective, I am not fully convinced that the theoretical framing introduced in the paper is very suitable: one natural question is that if every step in Algorithm 1 is not practical, why do I want to squeeze the LLM alignment problem into an inappropriate shoe. To achieve Algorithm 1, many sub-optimal solutions are adopted. For example, in section 4.2.2, a local maximum is applied to approximate the global maximum.

### Questions
1. For step 6 in Algorithm 2, is the termination condition based on selecting a different $\phi_k^t$? Since by definition, $y_t$ seems deterministic.
2. I find it a bit counterintuitive that it helps when both responses $y_t$ and $y'_t$ are the optimal response under some rewards (for E&E). It seems challenging for contrastive methods like DPO to effectively learn from close-quality responses.
3. Just out of curiosity: I kind of get the impression that the magic of the method comes from the epistemic reward model and its online update. Is this interpretation right or am I missing something? It would be helpful if the authors could provide additional discussion or intuition to clarify the underlying mechanisms driving the method's success. For now, it seems everything comes from the literature with limited explanation.

### Soundness
2

### Presentation
2

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
This paper studies the topic of LLM alignment. The authors provide a framework based on contextual dueling bandit to model the existing alignment algorithms. They point out that online interaction and active exploration are two key properties which are critical for the sample efficiency. Their framework also considers both settings of explore&exploit and best arm identification which applies to a wide range of scenarios. Based on their framework, they propose a principled alignment algorithm inspired by Thompson sampling to improve the sample efficiency. Built on top of the principled algorithm, they deploy several practical techniques to achieve a tractable algorithm. Finally, the authors empirically show the effectiveness of their method through experiments.

### Strengths
1. The paper is clearly presented and easy to follow. The introduction of the related works are comprehensive and thorough.

2. It is interesting to apply Thompson sampling to the area of LLM alignment as a way to balance exploration and exploitation.

### Weaknesses
1. This paper focuses on the sample efficiency of the alignment algorithms and try to model it through contextual dueling bandit framework. However, through sec 2 and 3, it is not clear to me how the framework helps to show theoretically the deficiencies of the existing alignment algorithms. For example, the authors claim online interaction and active exploration are essential for sample efficiency. How is this supported by the framework?

2. The framework did not provide a conclusive result that shows why the proposed algorithm outperforms the existing methods in terms of sample efficiency. Besides, the proposed algorithm is intractable practically, hence lots of practical designs are needed. If ones considers the factors of simplicity, implementation, stability, and performance together, does the algorithm really bring significant advantages? The experiments seem to be implemented on well-controlled and limited tasks. 

3. A considerable portion of this paper is doing systematical review of the existing alignment methods through the lens of contextual dueling bandits. But the reformulation is mostly some re-definitions of the concepts. I didn't see how the reformulation brings new insights or guidance. This makes me doubt the real contributions of this work.

### Questions
Please refer to the weaknesses part.

### Soundness
2

### Presentation
3

### Contribution
2
