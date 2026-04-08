## Human Reviewer 1

### Summary
This paper investigates contamination detection methods for benchmarks in base models when they are being conditioned as LRMs via SFT and RL, and in advanced Large Reasoning Models via SFT with CoT contaminated data. They find that GRPO conceals contamination, and minimal detectable traces are left. 10 different detection methods and 6 mathematical reasoning benchmarks are used. A theoretical analysis is provided to show that PPO-style clipping and importance sampling are the root cause of the concealment. The paper makes the key claim that memorization based detection are not optimal for LRMs, and that these can evade it easily.

### Strengths
- It is highlighted that almost all the contamination detection methods (tested) consistently perform near random guesses in all the benchmarks after performing SFT on the benchmark samples with Co. This is a very significant insight.
- The paper is thorough in exploring its claim within the mathematical reasoning domain, using 6 common benchmarks in the field. And in using a wide range of detection methods (10).
- It is comprehensive in both providing results to support its claim and a thorough theoretical analysis of the premise

### Weaknesses
- The authors mention benchmark performance after SFT but do not mention results after SFT and GRPO.
- While enough benchmarks from the mathematical reasoning domain are used, no other domain is surveyed to make the points the paper raises applicable globally.

### Questions
- Do you have some figures on the length of outputs across the paper? Have you found any tested detection methods sensitive to output length?
- How do you generate your CoT RL data for each benchmark?
- In the first proposed direction in your conclusion (lines 477-478), how would the release of intermediate training checkpoints help with the issues the paper raises?
- Do you generate your own contaminated CoT and SFT data? If so, can you share it and how it was generated?
- Embedding based methods are mentioned but not used, why?

Suggestions:
- In line 72,can you clarify what you mean by “in the later stage”.
- In line 81, what kind of contamination is this? Is it verbatim contamination, paraphrased, etc..
- In the theoretical analysis it would be good to explicitly define E in formula 1 (and subsequent mentions).
- Given that this paper could aid malicious actors, can you provide more concrete recommendations for evaluators in the conclusion?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper investigates the fair evaluation of Large Reasoning Models on public benchmarks, showing that contamination that can be detected from SFT can be effectively concealed using GRPO, pushing many contamination detection methods to near-random performance, while the contaminated LRMs retain improved performance over their uncontaminated counterparts.

### Strengths
- Benchmark contamination is a critical and timely problem to address, and the interplay between GRPO and contamination detection is interesting to investigate.
- Experiments are well-designed: There is a clear concealment effect of GRPO on almost all detection methods, and current contamination detection approaches are clearly inadequate for ensuring the integrity of public leaderboards.

### Weaknesses
The main weakness comes from the underlying motivation for this work. Primarily, it seems the authors are highlighting that contamination detection methods are ineffective, yet it is unclear whether these methods are being used in practice. Recent work has highlighted pitfalls in data contamination approaches [1,2] and prohibiting training on test tasks [3]. A further discussion on how this work fits into prior work on fair evaluation and data contamination would strengthen the work.

In addition, do the authors believe that tailoring contamination detection methods will be the way forward for ensuring faithful evaluation when developers are incentivized to game benchmarks?

Minor
- Line 156 "Olypaid"

[1] Liu, Ken Ziyu, et al. "Language models may verbatim complete text they were not explicitly trained on." arXiv preprint arXiv:2503.17514 (2025).

[2] Fu, Yujuan, et al. "Does data contamination detection work (well) for llms? a survey and evaluation on detection assumptions." arXiv preprint arXiv:2410.18966 (2024).

[3] Dominguez-Olmedo, Ricardo, Florian E. Dorner, and Moritz Hardt. "Training on the test task confounds evaluation and emergence." arXiv preprint arXiv:2407.07890 (2024).

### Questions
- Could the authors clarify their claims in the Discussion and Conclusion? Specifically the claim "This fundamentally challenges the assumption that all the detection approaches rely on, which is that benchmark contamination is more about memorizing the benchmark samples."
- Could the authors provide additional details about why GRPO does not improve concealment on the Verbatim/Neighbor attacks?

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 3

### Summary
The paper studies benchmark contamination for LRMs in two practical scenarios: 1) SFT contamination on a base model is initially detectable, but a small amount of RL greatly conceals contamination signals across many existing detection methods. 2) When advanced LRMs are SFT-contaminated with CoT on benchmark samples, pass@1 improves significantly while the contamination detection becomes near random. The paper provides theoretical reasoning for this phenomenon by showing the gap in the log-likelihoods of benchmark members' and non-members' contracts after RL training. Such contraction is attributed to the common importance-sampling + clipping trick in RL.

### Strengths
1. This paper looks into benchmark contamination, which is a timely and important issue in modern LLM evaluation. Showing that light RL can significantly mask contamination signals is surprising and consequential.

2. The paper clearly splits the study into two realistic scenarios, including 1) base-model SFT contamination followed by RL, and 2) CoT contamination of advanced LRMs, making the threat model concrete and realistic. It also clarifies where current detectors fail and why mitigation is nontrivial.

3. Attributing the contamination concealment to the importance weighting + clipping is interesting. This claim is supported by theory and empirical evidence in the paper.

4. The paper includes extensive experimental evaluation across a wide range of representative detectors and reasoning benchmarks.

### Weaknesses
1. Table 1 suggests RL brings little or no gain on clean data. Is the RL correctly applied/tuned in the experiments? Why does RL not further help with reasoning even when there is no contamination?

2. Some prior works have shown that contamination detection can be evaded in LLMs. A clearer positioning relative to these closely related papers could be added to the main text (currently largely in the appendix/references).

### Questions
1. The main results are on math tasks with long CoT. Can the post-LRM “near-random detection” result replicate on coding, QA, and other tasks where solutions are relatively less CoT-heavy?

2. Do larger/smaller models strengthen or weaken contraction? Is there any non-monotonicity regarding model size (e.g., medium models most vulnerable)?

### Soundness
4

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
3