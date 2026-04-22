# SELF-HARMONY: LEARNING TO HARMONIZE SELF-SUPERVISION AND SELF-PLAY IN TEST-TIME REINFORCEMENT LEARNING

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 6, 4, 8

## Abstract
Test-time reinforcement learning (TTRL) offers a label-free paradigm for adapting models using only synthetic signals at inference, but its success hinges on constructing reliable learning signals. Standard approaches such as majority voting often collapse to spurious yet popular answers.
We introduce Self-Harmony, a framework built on a simple intuition: the correct answer should remain stable across both an original question and its paraphrase. Self-Harmony operationalizes this by employing a single model in two complementary roles: a Solver to produce answers and a Reframer to rephrase the input.  Based on this, we further propose a pseudo-label method: instead of majority voting, it aggregates answer frequencies across these original and reframed views using the harmonic mean. This is a process that naturally selects for solutions stable under reframing, thereby avoiding the common trap of favoring view-dependent, spurious answers.
Crucially, this requires no human supervision or auxiliary models. Across diverse reasoning benchmarks, Self-Harmony achieves state-of-the-art results at the label-free test-time setting, ranking first in 28 of 30 settings across multiple methods. Beyond accuracy, it demonstrates unprecedented robustness, with zero training failures in all experiments, underscoring its stability and reliability.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents "self-harmony" an alternative to final-response consistency like self-consistency often used for test time scaling and more importantly for unsupervised supervision at test time. While self-consistency involves temperature sampling multiple solutions and selecting the one corresponding to the majority answer, the authors propose looking at the same problem from "multiple views" done by creating multiple rephrasings of the input query and selecting the answer consistent across these views by using the harmonic mean. The model is then trained on this unsupervised signal and the performance is reported. The authors argue that self-harmony is useful even in cases when self-consistency fails, i.e., model consistently arriving at the wrong answer.

### Strengths
- Obtaining unsupervised signals for improving reasoning performance is an important area of work with potential applications in scaling reasoning performance during pretraining or in the wild, where annotations are not available. This paper makes a contribution towards the broader goal.
- The paper is well-structured and easy-to-follow with decent empirical gains in the main resutls as well as some theoretical justifications for the view-invariant info-max objective.

### Weaknesses
- Missing connections to previous work in "unsupervised" training signals for reasoning: 
    - https://arxiv.org/abs/2411.04109
    - https://arxiv.org/abs/2210.11610
    - other papers on RL with external reward models for reasoning: https://arxiv.org/abs/2406.16838, https://arxiv.org/abs/2403.13787, etc.
- While the main motivation hinges on self-consistency being ineffective when the model is consistently wrong, the citations and experiments to back this up are lacking:
    - To the best of my understanding, Liu et al 2025b uses self-consistency to compare across different prompting strategies and the shortcoming of self-consistency at large was not clear to me, the authors need to be more explicit when making this connection (L157-160).
    - Since the work requires actual model training on this unsupervised signal, simply the accuracy of the pseudo label does not suffice (as in Fig 3b). Unclear if the original test set (ground truth labels) is biased towards getting correctness right, or are we measuring the how often the consistency indicator lines up with correct or incorrect prediction. It would be prudent to report F1, or correlation between the metrics and the binary ground truth by Spearman or Somer's D correlation as done in https://arxiv.org/abs/2411.04109.
    - As shown in previous works, https://arxiv.org/abs/2411.04109 and https://arxiv.org/abs/2509.06870 (concurrent), as the size of the majority answer increases, self-consistency becomes a better correctness indicator. Therefore, the authors should show that self-harmony outperforms self-consistency when setting consistency threshold to filter out smaller answer sizes. 
- Also see clarifications on experimental setup underspecified in the paper below.

### Questions
In addition to weaknesses, I have the following questions about the experimental setup:
- What is the size of the training dataset, are we to assume that the test set (without labels) was used as the unsupervised signal? Is the same model trained for multiple tasks or different ones, please specify and compare against the other case? If the training is done on test queries (which are rather small -- at least for Math), how does the performance scale as the amount of training data increases?
- Why do the authors use base model for Qwen but instruct models for Lllama?
- Why not have GRPO with a trained reward model (since the test set is mainstream evaluation for reasoning tasks) aimed at reasoning as a baseline too and show self-harmony outperforms it?
- Regarding 3(a), how often does the collapse happen and is it just on this specific dataset and/or model type? What are possible explanations or is it the entropy collapse seen in RLHF and how does the step where it happens relate to the size of the training set and the batch size?
- Previous works show that self-consistency is useful even as the model size scales https://arxiv.org/abs/2509.06870, does the same hold for self-harmony. Can self-harmony be used without training to get more gains on the testsets?
- The main paper should also contain examples of the rephrasings and any evaluation done to ensure the content and original intent were preserved.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Self-Harmony, a framework where a single model plays two roles: a Solver that answers questions and a Reframer that rephrases them. The method aggregates answer frequencies across the original and reframed questions using the harmonic mean, which is shown to outperform baselines such as majority voting on the original questions.

### Strengths
**Soundness**: The proposed method is intuitive and straightforward to implement. It builds on a sound intuition — that a correct answer should remain consistent across semantically equivalent but stylistically distinct formulations of the same question. The paper also provides a comparative analysis of pseudo-label quality between Self-Harmony, TTRL, and Co-Reward, demonstrating that Self-Harmony produces the highest-quality pseudo labels.

**Substance**: The experiments are comprehensive, covering multiple model families including Qwen3 and Llama 3.1/3.2. The proposed approach shows consistent performance improvements across these models. Moreover, the ablation studies indicate that the training process is robust to different hyperparameter settings.

### Weaknesses
**Missing analysis**: It would strengthen the paper to include a deeper analysis of why Self-Harmony improves pseudo-label quality. For instance, one could perform a simple inference-based categorization: for each question, sample multiple responses and classify questions into four types based on model confidence and correctness:

- Confident (majority answer ≥ N/2 votes) and correct

- Not confident (majority answer < N/2 votes) but correct

- Confident and incorrect

- Not confident and incorrect

Then, apply Self-Harmony method for generating pseudo-label to analyze where the improvements primarily arise — e.g., whether it mainly helps on low-confidence or high-confidence cases, or corrects certain systematic errors.

**Minor issue**: In Table 2, the caption should appear above the table.

### Questions
As mentioned in Weakness 1, I’m curious to see a breakdown showing where the improvements primarily originate compared to methods like majority voting.

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
This paper proposes Self-Harmony, a label-free test-time reinforcement learning (TTRL) framework that improves reasoning stability without external supervision. The key idea is that a correct answer should remain consistent across both an original question and its paraphrase. A single model plays two cooperative roles: a Solver that answers and a Reframer that rephrases the question to provide a second view. Pseudo-labels are then chosen using the harmonic mean of answer frequencies from both views, emphasizing solutions robust to rewording while down-weighting view-specific errors. The authors provide theoretical justification for this criterion and integrate it into a self-play reinforcement loop. Across five reasoning benchmarks (MATH500, GSM8K, AIME24, GPQA, MMLU-Pro) and five open-source models, Self-Harmony achieves new state-of-the-art results—ranking first in 28 of 30 settings—and demonstrates superior training stability and pseudo-label accuracy.

### Strengths
This paper stands out for its clarity and strong empirical results. The idea of enforcing consistency across paraphrased versions of a question is simple yet powerful—it directly targets a core weakness of existing test-time RL methods that overfit to their own biases. Using a single model in dual roles (Solver and Reframer) is elegant and avoids the need for extra models or supervision, making the approach practical and scalable. The harmonic mean pseudo-label rule is theoretically grounded and nicely bridges information-theoretic reasoning with an intuitive motivation. 

The experiments are extensive and convincing: strong gains across five benchmarks, stable training curves, and robustness across model sizes all strengthen the paper’s credibility. Overall, Self-Harmony feels like a solid, well-executed contribution that advances test-time adaptation in a meaningful and reproducible way.

### Weaknesses
The harmonic mean rule, while supported by a theoretical derivation, still feels somewhat heuristic in how it’s applied. In practice, the paper does not deeply analyze why the harmonic mean performs better than simpler alternatives like averaging or weighted voting, or under what conditions it might fail. For instance, if the paraphrasing step inadvertently shifts the semantics of the question—introducing small wording biases or contextual cues—the harmonic mean could incorrectly down-weight valid answers. A sensitivity analysis or qualitative study of such “semantic drift” cases would make the claim of robustness more convincing.

Finally, the “self-contained” assumption might break for smaller or less instruction-tuned models, since paraphrasing quality heavily depends on the model’s own linguistic competence—an issue not fully explored in the experiments.

### Questions
Could you provide more intuition or empirical evidence for why the harmonic mean is superior to arithmetic or geometric averaging in this setting?

How sensitive is the performance to small perturbations in the paraphrased questions or to imperfect semantic equivalence?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors propose **Self-Harmony**, a new TTRL framework inspired by the Infomax principle. The core intuition is that the correct answer should appear consistently (robustly) across different versions of the question. The authors demonstrate that the argmax of the harmonic mean of the two probabilities under two different versions of the question approximately yields a good pseudo-label, and this is algorithmically achieved via appropriate prompt-engineered self-play. Its empirical efficacy is shown across various reasoning benchmarks and Qwen & Llama models, showing its superiority over prior TTRL approaches.

### Strengths
- Well-written
- Clear and solid intuition
- Theoretically well-grounded
- Extensive experiments and benchmarks, as well as publicly released anonymized codes. Ablations provide a complete understanding of each component of the proposed algorithm and clearly show its superiority.

### Weaknesses
- The authors mention that the reward design plays a role in obtaninig SOTA performance (Section 4.3), but the only mention of how the weights involved with the rewards are set ($w_f, w_d$) is in Appendix E.4, mentioned in the passing only "weights set to match the proportional influence ...". This design choice should sufficiently be elaborated as well.
- Although there isn't anything (significantly) wrong with the theoretical discussions, I believe there are some parts that warrant further discussion:
  - I think it is beneficial to explicitly state the "second-order approximation" in the main text for the camera-ready version.
  - The assumption that the view random variable $V$ is uniform is implicit and made explicitly only in the proof. This should be made clear in the main text.
  - The theory is only derived for two views and $\lambda = 2$ and uniform prior. I believe that more discussions on this would make the paper better; see Questions.

### Questions
1. Is there any connection between the core intuition here and group-invariant learning [1] or invariant risk minimization [2]? I don't recall any discussions relating to this line of works in the paper.

2. (Minor) Are there any literature from human psychology or education for the claim "common human robustness heuristic: when confronted with uncertainty, people often check solutions ..." (line 161)?

3. What happens if $\lambda$ is also tunable? In other words, under two views, would one still be able to derive Eqn. (1) as a function of $\lambda$? If that is the case, then what is the impact of tuning $\lambda$ across various values?

4. Similarly, with fixed $\lambda$, what happens if one considers more than two views? For instance, when $\lambda = 2$, then does multiple views lead to the general form of the harmonic mean? What might be the impact of having more views? At least from theoretical viewpoint, more (diverse) views should give better and better performance, although there is a possibility of plateauing over certain threshold.

5. Follow-up question: If it is possible, I wonder what the impact of the second-order approximation is. Yes, computationally, having a closed-form solution is hugely beneficial, but the second-order approximation of the KL is tight only when the two probabilities are quite similar. Thus, if one puts in enough budget, I wonder if maximizing the infomax objective directly (with an appropriate KL estimator) gives better pseudo-label, and thus better performance. Or, whether the performance gain is minimal and the approximation objective is good as it is.

6. Also a follow-up: the authors consider uniform prior over the views. I wonder if one could be a bit more clever here, e.g., depending on the "persona" or the "strengths" of the current LLM, one could take that into account to make the prior non-uniform?

7. (Minor) As in many infomax papers (and as a theoretician myself), I wonder if this formulation of infomax (and its approximation) has been theoretically studied before, i.e., whether prior generalization bounds are applicable here (e.g., [3]). Or does this warrant further investigation for future work? If yes, which part of the new formulation yields novelty compared to prior literature?

8. (Minor) Very recently, there have been some issues regarding floating-point operations (https://x.com/QPHutu/status/1984258808332550245). Can the authors comment on whether this issue(?) applies to their experiments?



[1] https://www.jmlr.org/papers/v21/20-163.html

[2] https://arxiv.org/abs/1907.02893

[3] https://www.sciencedirect.com/science/article/abs/pii/S1566253525008383 (arXiv: https://arxiv.org/abs/2501.16768)

### Soundness
3

### Presentation
3

### Contribution
3
