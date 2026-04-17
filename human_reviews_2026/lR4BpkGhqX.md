# All by Large Language Model Itself

- Decision: Reject
- Scores: 4, 4, 2, 8

## Abstract
The scaling laws constitute one of the fundamental principles of large language models (LLMs), which reveal that the model performance constantly improves as the training data increase. In this paper, we propose dynamic reinforcement learning (RL), which takes a step to achieve the scalability of RL for training the LLM by itself. Dynamic RL operates by sampling data from the dynamically changed LLM itself, estimating golden answers based on the model’s own outputs, and then using this self-generated data to optimize the model. Its dynamic characteristic allows the data distribution to continuously adapt to the evolving model, leading to better alignment between training data and model capabilities. Unlike conventional approaches, dynamic RL requires neither static, pre-collected datasets nor external verifiers for correctness. All is done by the large language model itself. Experimental results demonstrate that dynamic RL can continually improve model performance over a thousand of training steps and achieve results comparable to models trained on large-scale external datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes dynamic reinforcement learning (dynamic RL), a self-contained training framework for large language models (LLMs) in which both questions and solutions are generated and refined by the model itself, without reliance on external datasets or human-labeled answers. The method builds upon semi-dynamic RL (e.g., DeepSeek-R1), but eliminates the need for static human-authored questions. To address the absence of ground-truth answers, the authors estimate “golden answers” via majority voting over multiple sampled solutions per question. The paper introduce four reward components:Solution reward, Question reward, Question diversity reward and Answer diversity reward. Results show that dynamic RL achieves performance comparable to semi-dynamic RL trained on large-scale human-curated datasets, despite using only self-generated data.

### Strengths
- The paper is well-structured and easy to follow.  The framework is described with mathematical rigor and algorithmic transparency.
- This paper proposes a fully autonomous RL loop for LLMs, removing dependence on any external data—a bold step toward “self-improving” systems.

### Weaknesses
1. **Unverified estimation quality**: No analysis of how often majority voting recovers the *true* answer. Table 3 shows cases where it fails, but no aggregate statistics (e.g., % of questions with correct $l(q)$) are given.

2. **Inadequate validation of diversity mechanisms**: The paper claims to mitigate mode collapse but provides no direct evidence. There is no comparison of question/answer entropy with vs. without $R_{dq}, R_{da}$, or visualization of question embedding clusters over time, or measurement of repetition rates (e.g., % of duplicate or near-duplicate questions).

3. **Flawed question similarity metric**: Token overlap (Eq. 11) is **not semantically meaningful**. This undermines the validity of $R_{dq}$. e.g.: Consider two questions: "Solve for \(x\): \(2x + 3 = 7\)." and "Find the value of \(x\) that satisfies \(2x + 3 = 7\)." These are semantically identical, but their token sequences differ significantly. The overlap-based similarity in Eq. (11) would deem them dissimilar, causing the diversity reward \(R_{dq}\) to incorrectly encourage redundancy. Conversely, questions with high token overlap but different meanings (e.g., same variable name but different equations) may be wrongly penalized. Thus, token overlap is not a reliable measure of semantic similarity. A more robust metric (e.g., embedding similarity) would strengthen claims.

4. **Incomplete ablation study**: The ablation (Table 2) removes $R_q$, $R_{dq}$, $R_{da}$, but don't remove $R_s$. It needs to be tested whether learning is truly driven by self-consistency or just diversity/exploitation heuristics.

5. **Lack of training dynamics analysis**: The paper does not show how the four reward signals co-evolve during training, for example, whether they are balanced or a specific one dominates. This is essential to understand the optimization landscape.

### Questions
1. **Estimation accuracy**: What is the empirical accuracy of the majority-voted golden answer $l(q)$ against ground truth (e.g., on a subset of generated questions manually verified)? 

2. **Question reward efficacy**: Does the average $p(q)$ increase over training steps? If not, how do you reconcile this with the claim that $R_q$ encourages “estimable” questions?

3. **Mode collapse evidence**: Can you provide quantitative metrics of diversity (e.g., number of unique answers, question embedding variance) with and without $R_{dq}$ and $R_{da}$? Without this, the mitigation claim remains unsubstantiated.

4. **Ablation of $R_s$**: Why was the solution reward $R_s$ not ablated? Is learning possible without it?

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
This paper proposes RL, in which the LLM autonomously generates and solves its own problems. Compared with the traditional RL framework, the proposed dynamic RL also provides a learning signal for the LLM in its role as a problem setter, guiding it to produce questions of moderate difficulty. Meanwhile, the LLM’s problem-solving ability continues to improve throughout training as well.

### Strengths
- The proposed method is clearly present and easy to follow.
- The topic of enabling LLMs to automatically generate high-quality question-answer pairs is important, especially as model capabilities continue to grow while high-quality, valuable question-answer pairs become increasingly scarce.

### Weaknesses
- The proposed method essentially leverages the LLM’s own **self-consistency** to provide learning signals for both its solver and question-setter roles. Unfortunately, such methods primarily reinforce the determinism of the model’s outputs rather than genuinely enhancing its reasoning ability, especially when it comes to solving challenging reasoning problems. For example, when the base model already possesses a reasonable level of reasoning capability, substantial further improvement typically requires RL or SFT training on competition-level problems (e.g., AIME). However, for such difficult problems, the LLM’s responses are mostly incorrect, and even majority voting is likely to yield wrong answers. Consequently, the proposed method cannot provide reliable learning signals for these cases, nor can it effectively encourage the generation of high-quality, high-difficulty problems.
- The experimental results partly confirm my above opinion: on AIME24 and AIME25, the proposed method achieves only limited gains. Under a well-designed prompt template, Qwen2.5-Math-1.5B base can already reach a score of 16.7 on AIME 2024 [1], which is higher than that achieved by Dynamic RL. On simpler benchmarks, Dynamic RL performs better, likely because it strengthens confidence in answers that are already mostly correct.
- The method resembles multi-objective RL, and as with such frameworks, balancing multiple objectives is inherently difficult. In particular, the coefficients $\lambda$ in Eq. (13) likely require careful tuning, which limits the scalability and robustness of the proposed method.
- The experiments are relatively limited. The authors should:
    1. Validate the effectiveness of the method on larger and more diverse base models, even without direct RL comparisons, simply demonstrating consistent performance gains would help.
    2. Provide more experimental details, such as the prompt templates, Avg@32 results, and the evaluation framework, as these directly affect the fairness and reproducibility of the evaluation [2].

Overall, I believe the paper does not yet meet the acceptance standard now. However, I also acknowledge that Dynamic RL, even without external prompts and answers, achieves significant improvements on simpler benchmarks and performs comparably to RL. Therefore, I recommend a borderline reject, while looking forward to the authors' response.

[1] Liu, Zichen, et al. "Understanding r1-zero-like training: A critical perspective." arXiv preprint arXiv:2503.20783 (2025).

[2] Hochlehnert, Andreas, et al. "A sober look at progress in language model reasoning: Pitfalls and paths to reproducibility." arXiv preprint arXiv:2504.07086 (2025).

### Questions
- I suspect that allowing the LLM to generate its own problems may largely exploit overlap with data the model has already seen during pretraining. If this is the case, Dynamic RL is effectively performing RL on existing problems rather than on genuinely novel ones. I would like to see the authors’ perspective on this point.

### Soundness
2

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
4

### Summary
- This paper introduces Dynamic RL, a novel framework designed to enable an LLM to continually improve its performance without relying on any static datasets or verifiers.
- The proposed method uses the model to generating its own training data by sampling both questions and corresponding solutions from its current policy. For the reward signal, the proposed method uses the majority voting as the groud truth labels for answer generation and the function of pass rate for question generation.

### Strengths
- The motivation of this paper is clear: using LLM itself for iterative self-optimization without external datasets and verifiers.
- The paper is easy to follow.

### Weaknesses
- The paper is highly similar to the existing work [1][2] for proposer/generator co-evolving.
- The framework's foundation depends on the assumption that majority voting over the model's own outputs can serve as a reliable proxy for ground truth. This is a very strong assumption. Also, the first question that needs to be clarified for self-improvement is: where does the training motivation come from?
- This paper lacks the comparison of baseline approaches: (1) self-rewarding methods[3][4], which leverage LLM-as-a-Judge for self-improvement with iterative SFT/DPO/PPO; (2) Unsupervised RL methods[5][6], which use various entropy-based reward estimation for RL.
- This paper lacks the analysis of the co-evolution of question generation and answer generation (e.g., dynamics of training/evaluation metrics, special reasoning behaviors), and the reason why they are beneficial to each other.
- The proposed method introduces four different reward terms balanced by coefficients, in addition to the threshold. The paper notes that the balance between these is important, but does not provide a sensitivity analysis.

[1] Absolute Zero: Reinforced Self-play Reasoning with Zero Data

[2] Co-evolving llm coder and unit tester via reinforcement learning

[3] Self-Rewarding Language Models

[4] Self-Improving Alignment with LLM-as-a-Meta-Judge

[5] EMPO: Fully Unsupervised LLM Reasoning Incentivization

[6] Self-Rewarding Reinforcement Learning for LLM Reasoning

### Questions
Same as the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors design a synthetic data generation algorithm that generates both prompts and responses using a self-reinforcement technique. They deploy the algorithm on a math question-answering task. They assume access to a function that can evaluate whether two answers are equivalent, which presumably can be well-approximated for math answering tasks. 

The reward function for responses samples several solutions, and rewards agreement by counting the number of solutions equivalent to the majority vote. The question reward function attempts to avoid collapse by guiding the answers toward questions of intermediate difficulty. Given the sample of solutions, they measure the fraction of unique solutions. The question reward function is parameterized by a target fraction tau, where specifies where reward is maximized. It tapers linearly for smaller or larger fractions. Intuitively, this avoids collapse by avoiding questions that the model is good at answering (or, more accurately, confident at answering), as well as overly-difficult questions. 

They also introduce a diversity reward. One can measure the syntactic similarity between two questions by measuring token similarity. One can also measure whether the answer produced by the questions are equivalent. These can be used to encode a diversity score for a given parameterization of the model. 

They employ a few additional practical tricks: 1) filtering questions that are likely to produce degenerate answers, and 2) standardizing reward functions. 

Finally, they balance the various rewards with hyperpameters, and optimize the model using standard policy gradient. 

Their empirical results evaluate their technique on a number of math datasets using Qwen2.5-Math-1.5B and 7B. Their results compete with related work that use external datasets to guide prompting. In contrast, their approach uses no external data. They provide ablation studies for each of the reward functions. They also provide observational results from the view of exploration/exploitation.

### Strengths
Strength #1: The paper is very clear, and combines a collection of conceptually clean ideas into a practical algorithm. With some minor exceptions, all the implementation details are provided, and the approach seems readily implementable. 

Strength #2: I found the main empirical results to be impressive, demonstrating the effectiveness of the method against relevant work that makes use of external data.

### Weaknesses
Weakness #1: The exploration/exploitation section is by far the weakest part of the paper, and seems tacked-on at the end. I think it needs heavy editing. What are the colors in Figure 3? I thought r(q) is the answer support size? I also thought r(q) was the dependent variable in Figure 3? If so, what does r(q) = m = 16 mean? Is A_s introduced elsewhere in the paper? 

Weakness #2: I find the paper to be very complete in its description of the algorithm. However one missing detail is the matching function S_a. While I believe there are candidates for S_a in a math-answering task, can the authors explain how this was implemented in their experiments? The existence of such a function seems critical to making this approach work in this (and other) settings.

### Questions
Question: Since dynamic RL does not use any data at all, is there a way to use the datasets (MATH-7.5K, DAPO, DeepScaleR) to warm-start the models somehow? Maybe by finetuning the base model? I suggest this only to make a more favorable comparison between the fully and semi-dynamic approach. 

Grammar nits: 
“training LLMs by itself” is grammatically broken. So is “All is done by large language model itself.” I can’t tell whether these are intentional. Consider “training the LLM by itself” and “all is done by the LLM itself.” 

"However, the policy model πθ constantly changed during training, while the data is sampled from a static distribution, which can not be adapted to the evolving policy model πθ, inherently limits the learning scalability."

“we first samples questions from the policy”

“Since generating questions may also lead the model to produce solution implicitly”

More major: I’m having trouble parsing the final paragraph in section 3.4. I don’t understand what either sentence means. Please re-write.

### Soundness
4

### Presentation
3

### Contribution
3
