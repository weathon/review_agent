# Stabilizing Reinforcement Learning for Honesty Alignment in Language Models on Deductive Reasoning

- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
Deductive reasoning is the process of deriving conclusions strictly from the given premises, without relying on external knowledge. We define honesty in this setting as a model's ability to respond only when the conclusion is logically entailed by the premises, and to abstain otherwise. However, current language models often fail to reason honestly, producing unwarranted answers when the input is insufficient. To study this challenge, we formulate honest deductive reasoning as multi-step tasks where models must either derive the correct conclusion or abstain. We curate two datasets from graph structures, one for linear algebra and one for logical inference, and introduce unanswerable cases by randomly perturbing an edge in half of the instances. We find that prompting and existing training methods, including GRPO with or without supervised fine-tuning initialization, struggle on these tasks. In particular, GRPO optimize only for final task outcomes, leaving models vulnerable to collapse when negative rewards dominate early training. To address this, we propose Anchor, a reinforcement learning method that injects ground truth trajectories into rollouts, preventing early training collapse. Our results demonstrate that this method stabilizes learning and significantly improves the overall reasoning performance, underscoring the importance of training dynamics for enabling honest deductive reasoning in language models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper addresses the problem of abstaining giving an answer in deductive reasoning when the premises arent sufficient to answer a question. The authors show that untrained LLMs (albeit small LMs) are unable to do the task well as task complexity increases. The authors introduce ANCHOR; that adds a complementary SFT objective to GRPO to train models that can abstain.

### Strengths
- The motivation is clearly presented and the problem is relevant: Models should know when a question is unanswerable and rather than producing a confident incorrect answer
- I like the set of tasks used in the paper to test the methods. Being able to systematically vary difficulty is a neat setup. Correspondingly, I like figs 1,2
- The paper does a good job of evaluating existing models on their proposed tasks.

### Weaknesses
- While the synthetic tasks provide a good knob to tune difficulty / complexity, I dont know how realistic the task setup is. The queries seem overly complicated, unnatural, and distant from how llms are used.  The unanswerable cases are created through edge removal etc. that may not reflect how unanswerable questions arise. Adding a more realistic / conventional task might help the paper with ecological validity.
- My main concern are the counterintuitive GRPO+SFT results. Why does RL actively make the SFT’ed model worse? I also have concerns around the implementation of GRPO:
    - Small group size of 5
    - Very Off-policy: (global batch 1024, mini-batch 64, micro-batch 2)
- The paper views honesty pretty narrowly as abstention.
- ANCHOR requires access to ground truth trajectories during the RL phase which might be an unrealistic requirement.

**Minor:**
- place related work in the main draft
- add examples of the task in the main, to improve clarity

### Questions
- How do stronger models perform on these tasks? Either open source frontier models or GPT-5 / Claude
- How is the data for sft generated?
- Could SFT+GRPO with a stronger KL penalty (e.g., 0.01 instead of 0.001) recover ANCHOR-like behavior? Since ANCHOR forces the model to be close to the SFT data.

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
This paper addresses the challenge of honesty alignment in LLMs—that is, ensuring models accurately indicate what they do and do not know. To overcome the scarcity of relevant data, the authors developed two new datasets. They evaluated three Qwen models on these benchmarks and observed that performance deteriorates as task complexity increases. Finally, the authors introduced *ANCHOR*, an approach that incorporates ground-truth trajectories into reinforcement learning algorithms.

### Strengths
1. The paper is well-written and easy to follow.
2. The study on the reasoning tasks of variying deductive difficulty is well-designed and provides good insights.
3. The proposed datasets for honesty alignment are a good contribution to the community.

### Weaknesses
1. The statement that "when GRPO has all incorrect rollouts for one question, the so-called 'gradient vanish' problem will prevent any learning progress" is somewhat debatable:
    - The zero-advantage issue holds for a single question. However, in actual optimization, a mini-batch usually covers several questions, so having all zero advantages in a mini-batch is rare, and optimization can still proceed.
    - The Adam/AdamW optimizer retains momentum, which allows optimization to continue even if a zero-gradient occurs rarely in a mini-batch.
    - If the curriculum learning hypothesis—discussed in the paper—holds, GRPO will not completely prevent learning progress unless all questions are unanswerable. However, this may come at the cost of reduced training efficiency. The result of GRPO+Easy-to-Hard can provide some insight here.
    - This issue pertains only to GRPO, not to all RL post-training techniques (e.g., PPO, ReMax). Discussion of this distinction is necessary.

2. When forcing the ground-truth trajectory, the advantage of RL over SFT—namely, going beyond the dataset distribution—is likely diminished, as the LLM is being compelled to follow specific reasoning paths.
   - There is a lack of comparison with SFT + Ground-Truth trajectories. SFT might be more efficient while sharing the same limitation.
   - For larger LLMs (i.e., Qwen-3B), GRPO + Easy-to-Hard already achieves better performance, which may hint at the issue raised here.


3.  The paper only conducts experiments on the small LLMs (0.6B, 1.7B, 3B). Considering the issue mentioned in W2, the results may not generalize to larger models (7B, 32B, 70B).
 
4. In the evaluation, the authors claim that the tasks are deductive; i.e., the prompts are self-contained and provide all necessary information to answer the questions. However, this ignores the internal knowledge of the LLMs. What if the LLMs rely on their own knowledge to solve the task? How can we determine the behavior of the LLMs?

5. The paper presents a variety of design choices, but each is not discussed in sufficient depth. In particular, the curriculum learning technique (Easy-to-Hard) appears to address most of the issues raised in the motivation—provided the base model is not too weak relative to the tasks. A more thorough discussion of the necessity and contributions of each individual design would strengthen the paper.

> I personally think this work makes a valuable contribution and has notable strengths, although it also retains some unignorable weaknesses.

### Questions
1. Need to clarify the LLM usage, as it is used as the target in research.
2. Wrong results on Table 1? GRPO gets 1 on the unanswerable set and 0 on the answerable set.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes ANCHOR (Augmented with Necessary Correct and HOnest Reasoning), a method that injects ground-truth reasoning trajectories into GRPO rollouts to stabilize reinforcement learning training for mathematical reasoning tasks. The authors introduce GRAPHLA, a dataset of multi-step linear algebra reasoning problems with answerable and unanswerable instances, and demonstrate that ANCHOR outperforms standard SFT and GRPO baselines.

### Strengths
1. Well-motivated problem: The paper clearly articulates the instability issues in GRPO when negative rewards dominate, and the importance of honesty alignment in mathematical reasoning is well-established.
2. Controlled experimental setup: The GRAPHLA dataset provides a clean testbed for studying deductive reasoning with formally verifiable answerability, avoiding confounds from factual knowledge.
3. Clear presentation: The paper is generally well-written with clear motivation, methodology description, and experimental results.

### Weaknesses
1. Critical Issue: Lack of Originality and Missing Related Work. The most significant problem with this submission is that the core contribution—injecting ground-truth/reference trajectories into GRPO rollouts to stabilize training—has been previously proposed and thoroughly investigated. Most notably: LUFFY (Learning to Reason Under Off-Policy Guidance) [Yan et al., 2025, arXiv:2504.14945] proposes essentially the same approach. Also, there is no comparison with related works in experiments.
2. Limited Technical Novelty. This paper propose a straightforward approach: deterministically injecting one ground-truth trajectory per group is a natural and obvious extension of GRPO—much simpler than LUFFY's policy shaping mechanism.
3. Limited scope: Evaluation is restricted to a single task domain (linear algebra) and relatively small models (≤3B parameters)

### Questions
See Limitation part.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work investigates the LLM's ability to not only solve answerable problems but also to reliably identify and abstain from answering unanswerable ones through the lens of deductive reasoning. The authors introduce two novel datasets, GRAPHLA (linear algebra-based) and GRAPHLI (logical inference-based), featuring balanced instances of answerable/unanswerable questions. They propose ANCHOR (Augmented with Necessary Correct and HOnest Reasoning), which injects ground-truth trajectories into GRPO rollouts to stabilize reinforcement learning when all sampled responses are incorrect. The authors formally show that ANCHOR's gradient effectively adds a clipped, SFT-like term to the GRPO update, unifying supervised and reinforcement learning. Empirically, they demonstrate that ANCHOR successfully stabilizes training and outperforms baselines (SFT, GRPO, SFT+GRPO, and curriculum learning) on their new datasets.

### Strengths
1. The diagnosis of the "gradient collapse" failure mode of GRPO in the context of honesty alignment is a novel and valuable contribution. And the ANCHOR method is simple, well-motivated.
2. The heatmaps in Figure 1 demonstrate that baseline models fail at both reasoning and abstention as complexity increases.

### Weaknesses
1. Both datasets are synthetic and highly structured. Real-world honesty alignment involves messier scenarios where unanswerability is not binary or deterministically verifiable. The clean graph structure may not capture the complexities of knowledge boundaries in practice.
2. No comparisons to related works on stabilizing gradients, like Melo et al, 2025 [1] 
3. In many real-world scenarios, a single, verifiable ground-truth reasoning path is often unavailable, expensive to annotate. This is feasible for the synthetic datasets (GRAPHLA, GRAPHLI) created by the authors.
4. Only Qwen models are tested. Experiments on other model families (Llama, Mistral, etc.) would strengthen claims.
5. No convergence guarantees are provided. Under what conditions does ANCHOR converge?

[1] Stabilizing Policy Gradients for Sample-Efficient Reinforcement Learning in LLM Reasoning https://arxiv.org/abs/2510.00819

### Questions
1. The tasks are extremely difficult even for humans (e.g., solving systems with 10+ variables, tracking 15-step logical chains). Models might fail not due to dishonesty, but simply because of insufficient capacity. Can this be possible?
2. Process reward models that provide step-by-step supervision could be a strong baseline.
3. Recent work on difficulty-aware RL (GRPO-LEAD) shows that careful curriculum design can significantly improve GRPO training. The learning results from the curriculum are presented as a limitation. Will techniques like this help? 

[1] GRPO-LEAD: A Difficulty-Aware Reinforcement Learning Approach for Concise Mathematical Reasoning in Language Models https://arxiv.org/abs/2504.09696

### Soundness
3

### Presentation
3

### Contribution
3
