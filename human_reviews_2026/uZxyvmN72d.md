# Extending RLVR to Open-Ended Tasks via Verifiable Multiple-Choice Reformulation

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 6

## Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) has demonstrated great potential in enhancing the reasoning capabilities of large language models (LLMs), achieving remarkable progress in domains such as mathematics and programming where standard answers are available. However, for open-ended tasks lacking ground-truth solutions (e.g., creative writing and instruction following), existing studies typically regard them as “non-reasoning” scenarios, thereby overlooking the latent value of reasoning capabilities. This raises a key question: Can strengthening reasoning improve performance in open-ended tasks? To address this, we explore the transfer of the RLVR paradigm to the open domain. Yet, since RLVR fundamentally relies on verifiers that presuppose the existence of standard answers, it cannot be directly applied to open-ended tasks. To overcome this challenge, we introduce Verifiable Multiple-Choice Reformulation (VMR), a novel training strategy that restructures open-ended data into verifiable multiple-choice formats, enabling effective training even in the absence of explicit ground truth. Experimental results on multiple benchmarks validate the effectiveness of our method in improving LLM performance on open-ended tasks. Notably, across eight open-ended benchmarks, our VMR-based training delivers an average gain of 5.99 points over the baseline. Code will be released upon acceptance to facilitate reproducibility.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper extends Reinforcement Learning with Verifiable Rewards (RLVR) to open-ended tasks (e.g., creative writing, instruction following) via Verifiable Multiple-Choice Reformulation (VMR). RLVR originally excels in STEM tasks with clear ground truths but fails in open-ended ones. VMR restructures open-ended data into multiple-choice formats (one chosen, one rejected response) with random order to ensure verifiability. Experiments on 8 benchmarks show VMR-based RLVR outperforms baselines, with a 5.99-point average gain, enhancing LLM reasoning and performance, even surpassing some 32B-scale models.

### Strengths
1. Unlike RLVR’s reliance on explicit ground truths, VMR transforms free-form data into verifiable multiple-choice pairs. It enables rule-based rewards without ambiguous evaluations, solving RLVR’s inapplicability to open-ended scenarios.

2. Across 8 benchmarks, it achieves a 5.99-point average gain over the base model, with standout gains in creative writing. It even outperforms larger 32B-scale models, proving its efficiency in enhancing LLM capabilities.

3. Random response ordering avoids positional bias. Compared to Baseline II (same data without VMR), it still gains, showing improvements stem from VMR’s design, not just data scale, ensuring reliable training.

### Weaknesses
1. The method proposed in this paper solves the verification problem in open domains to a certain extent, but it faces significant issues in practical application. It seems that promoting this method to mathematical reasoning would require extremely high costs. The entire method relies on two candidate answers, and the verifier matches answers A and B with the ground truth (GT) option. How can this method be applied to mathematical reasoning where there is a unique GT? Is it necessary to forcibly construct an incorrect answer and then have the verifier make a judgment? This seems unreasonable and redundant.

2. An ideal experimental setup should involve training separately using RM-based datasets and VMR-based datasets with the same queries. In the current experimental setup, the method "combines RM-based and VMR-based datasets in equal proportion. RM-based queries are scored by the reward model, while VMR triples are verified using rule-based reward functions." This makes it impossible to decouple the roles of the reward model and the verifier, and I have doubts about the reliability of the experimental results.

3. The paper does not disclose the training equipment and time. Training a 14B-scale model with the training parameters described in Section 3 will incur extremely high costs. It remains questionable whether the cost-benefit ratio is sufficient to justify the promotion of this technology.

4. In the bar chart of Figure 1, some numbers overlap with each other.

### Questions
Please refer to the Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Verifiable Multiple-Choice Reformulation (VMR), which converts preference pairs (chosen vs. rejected responses) into verifiable binary-choice questions. A rule-based verifier then provides exact rewards (1/0) based on whether the model selects the better response.

### Strengths
It is important and anticipated to extend RLVR to open-ended tasks.

### Weaknesses
- **The papers lacks methodological novelty and the method is largely a straightforward combination of existing techniques** (e.g., RLVR, GRPO, and preference-based data formatting) without introducing significant conceptual or architectural innovation. While the VMR is a useful engineering trick, it does not constitute a fundamental advance in reinforcement learning or reasoning modeling. The method section is also overly verbose, repeating well-known formulations without sufficient focus on what truly differentiates the proposed pipeline.

- The writing—particularly in the abstract and introduction—lacks focus and fails to clearly articulate the core problem, contribution, and significance. Key claims are buried in lengthy paragraphs, and the narrative does not effectively motivate why extending RLVR to open-ended tasks is non-trivial or why VMR is a principled solution.

- Inadequate formatting and scholarly presentation. The paper suffers from inconsistent or incorrect formatting, including improper citation styles, titles position of tables and position of **REPRODUCIBILITY STATEMENT**. 

- Critical implementation details are missing. For instance: The number and type of GPUs used for training are not disclosed; Training time, memory consumption, or computational cost are omitted; Hyperparameter sensitivity or ablation studies (e.g., impact of the 1:1 RM/VMR data mix) are not provided.

- The paper over-relies on LLM-as-a-judge metrics. Most benchmarks (e.g., MT-Bench, AlpacaEval) use automated LLM-based evaluators, which are known to exhibit biases. 

- The method is only validated on a single base model (DeepSeek-R1-Distill-Qwen-14B). It remains uncertain whether VMR’s benefits transfer to other architectures, scales, or instruction-tuned models.

### Questions
Please see weaknesses.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper tackles challenges of applying RLVR to open-ended tasks like creative writing. The authors propose a training strategy called Verifiable Multiple-Choice Reformulation (VMR). This method restructures data from open-ended tasks into a multiple-choice format, which makes it possible to verify the answer. The experiments find that the proposed VMR improves the performance or LLMs on open-ended tasks, shoing an average gain of 5.99 points over the baseline.

### Strengths
* This paper tries to tackle the LLM training problem that it is not easy to do RL training with open-ended questions. This is an important question that the community tries to solve.
* The empirical results from the proposed method seems good, with a noticeable gain comparing with the baselines. 
* The paper is clear that readers can understand most of the concepts introduced easily.

### Weaknesses
*  The reward verifies only whether the model selected the pre-labeled preferred response, not that the response is objectively better. For the RM-based subset, line 257, the labels themselves are produced by an automated reward model (URM-LLaMA-3.1-8B). Therefore, the pipeline still inherits RM bias/noise even though the training reward is rule-based. This undercuts the claim that they avoid RM issues (line 063, in figure 2).
* Most reported wins depend on LLM-as-judge (e.g., MTBench, AlpacaEval-2, WildBench, CreativeWriting V3, ArenaHard 2.0, CreativeWriting), which can share stylistic biases with the training signal. There’s no human eval to validate the improvement, making over-optimization to judge preferences a real risk.
* In the experiment, the authors compare with reward model-scored RL baselines. There’s no DPO/KTO (or other RLHF methods) baseline trained on the same pairwise triples, despite those being the most obvious alternatives. Some gains could stem from the extra signal in pairwise data rather than the on-policy RL objective or the VMR prompt itself. 
* The A/B candidates come from existing datasets, not from the current policy model, which makes it skeptical if the proposed method can really improve the LLM generation quality
* The proposed method is a form of RLHF, it's just like the actor-critic/PPO loop. The “RM-based dataset” uses open-ended queries whose rewards are assigned by a reward model (URM-LLaMA-3.1-8B). For VMR, each item has a human-labeled chosen vs rejected answer. They convert it to A/B and give a binary reward (1/0) if the policy picks the chosen one (see Figure 3 and Eq (9)). Functionally, that’s RLHF with a degenerate reward model that returns 1 for the preferred option and 0 otherwise. The policy still maximizes expected reward from human preferences via policy gradient.

### Questions
* Though the motivation of this paper is to transform open-ended questions into verfiable ones, I wonder what is the necessity of doing so. For training LLMs with RL, is it a good and general enough solution to convert the open-ended questions?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper extends Reinforcement Learning with Verifiable Rewards (RLVR) from STEM domains (mathematics, programming) to open-ended tasks lacking ground-truth solutions (creative writing, instruction following). The key innovation is Verifiable Multiple-Choice Reformulation (VMR), which restructures preference data (chosen/rejected response pairs) into multiple-choice questions that can be verified using rule-based functions. For each query, the model is asked to choose between two randomly-ordered responses, and receives binary reward based on selecting the better one.

### Strengths
- Novel extension of RLVR to open-ended domains where standard answers don't exist
- Sound mathematical formulation connecting VMR to standard RLVR framework
- Clear problem motivation explaining RLVR's limitation in open-ended domains
- Addresses important gap: extending RLVR beyond STEM domains

### Weaknesses
- The connection between multiple-choice discrimination and open-ended generation is assumed but not justified
- Only one base model tested (DeepSeek-R1-Distill-14B); crucial to test on other models
- Dependency on high-quality preference data limits applicability
- Heavy reliance on LLM-as-judge metrics which have known biases

### Questions
- How does VMR perform on models without built-in reasoning capabilities ?
- Can you provide error bars or significance tests for the improvements?
- The reasoning density improvement is quite small. Is this statistically significant?
- How does the method perform when preference annotations disagree or are noisy?

### Soundness
3

### Presentation
3

### Contribution
3
