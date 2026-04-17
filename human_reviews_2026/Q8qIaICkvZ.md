# First return, entropy-eliciting explore

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Reinforcement Learning from Verifiable Rewards (RLVR) improves the reasoning abilities of Large Language Models (LLMs) but it struggles with unstable exploration. We propose FR3E (First Return, Entropy-Eliciting Explore), a structured exploration framework that identifies high-uncertainty decision points in reasoning trajectories and performs targeted rollouts to construct semantically grounded intermediate feedback. Our method provides targeted guidance without relying on dense supervision. Empirical results on mathematical reasoning benchmarks(AIME24) show that FR3E promotes more stable training, produces longer and more coherent responses, and increases the proportion of fully correct trajectories. These results highlight the framework's effectiveness in improving LLM reasoning through more robust and structured exploration.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes FR3E, a novel RL fine-tuning algorithm that identifies highly uncertain tokens and performs rollouts from these tokens to estimate values for intermediate tokens. These values are then utilized to adjust advantages for stable learning. FR3E demonstrates superior performance to GRPO++ on various math reasoning benchmarks.

### Strengths
- The idea of structured rollout based on token entropy, is both novel and timely.
- The proposed method consistently improves performance against GRPO across various benchmarks and models.

### Weaknesses
- Section 4.3 lacks a clear motivation. Specifically, it does not adequately explain how advantage modulation promotes exploration.
- The paper does not provide values for key hyperparameters, such as the number of forking tokens and the number of responses generated for each forking token.
- The paper compares FR3E against GRPO++, but the latter is mentioned without any introduction.
- It is questionable whether the paper did a fair comparison with GRPO++.

### Questions
- For a given prompt, do you first generate one base response, then identify K forking tokens, and from each of those tokens, generate M new responses? And to confirm, is the total number of generated responses per prompt, K * M, equal to 16?
- Following up on my previous question, why was the number of responses per prompt for GRPO++ set to 4? This value seems a bit low to me. Wouldn't it be more effective to use 16, consistent with FR3E? For a fairer comparison, FR3E should be evaluated against GRPO++ with a group size of 16. In FR3E, if the forking tokens are concentrated at the beginning of the sequence, the computational cost becomes nearly identical to that of parallel sampling. In fact, your method is more expensive, as it also incurs the additional cost of sampling the base response.

- Could you elaborate on the advantage computation process? Specifically, for the M responses generated from the forking token, are advantages calculated consistently with GRPO, including standard normalization? Also, what is the procedure for calculating the advantage for the base response?
- What is the role of advantage modulation? Given a forking token, the sum of advantages for the generated M responses should be zero, so what does multiplying them by a coefficient actually change?
- Why was the PPO clip ratio set to [0.22, 0.28], a choice that differs from the conventional settings in GRPO ([0.2, 0.2]) and DAPO ([0.2, 0.28])?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work proposes FR3W, which is a structured RL exploration framework that (i) discovers top high-entropy tokens in the reasoning trajectories, (ii) conducts structured rollouts at different high-entropy states, and (iii) learns via an adaptive advantage modulation factor. Extensive evaluations based on Qwen series show that the proposed method outperforms GRPO++ on math tasks.

### Strengths
1)	The proposed idea is simple and sound (targeted exploration at high-entropy states), achieving good overall performance on math tasks.
2)	The authors have conducted further model analyses on the training dynamics, sources of gains for a better understanding.
3)	The writing is clear and the method is easy to follow.

### Weaknesses
1)	This work mainly concentrates on the math tasks. Is this work still effective in other tasks, such as more challenging agent-related scenarios with sparser reward signals?
2)	There have been quite a few entropy-aware RL methods recently, which can be mentioned in related works (the differences should be discussed to highlight the contribution proposed by this work).
3)	The base reasoning trajectory is essential in FR3E. It is strongly suggested that the authors could give an in-depth discussion on how to select satisfactory base reasoning trajectories that potentially lead to success.
4)	In Page 5, the figure could be replaced by high-entropy samples in this work.
5)	The effectiveness of the advantage modulation factor \alpha should be evaluated. For example, FR3E w/o $\alpha$, and FR3E w/o $\alpha$ when $\alpha<1$, are two promising ablation versions that should be compared (indicating whether downscaling the positive signal is beneficial).
6)	FR3W should be evaluated on other base LLM series besides Qwen2.5.
7)	Why did the authors select GRPO++ as the only baseline? There are some methods that adopt similar ideas (e.g., DAPO [1], which also adopts clip-higher), which should be compared as baselines.
8)	The detailed training costs (e.g., the overall costs of rollout) should be given (the explanation in Appendix D.4 is obscure). Does the model improvement mainly come from more rollouts?

[1] Yu Q, Zhang Z, Zhu R, et al. Dapo: An open-source llm reinforcement learning system at scale[J]. arXiv preprint arXiv:2503.14476, 2025.

### Questions
Please refer to Weaknesses.

### Soundness
3

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
The manuscript proposes a novel framework FR3E (First Return, Entropy-Eliciting Explore) to make exploration for Reinforcement Learning with Verifiable Rewards (RLVR) more structured. The framework consists of two phases, which are the namesakes for the framework: First Return and Entropy-Eliciting Explore. In the first phase reasoning steps are identified by using token-level entropy to identify tokens with a high uncertainty. From these tokens the top-K tokens, with K being a hyperparameter, are selected, which are then used to divide the trajectory into reasoning steps. For each of these intermediate reasoning steps, generation is restarted and based on the partial trajectory (till this reasoning step) rollouts are generated to provide reward signal for the reasoning step. Additionally the advantages of the reasoning steps are adapted based on the rewards of the reasoning step as well as its predecessor to encourage exploration and to stabilize training.

Disclosure: I accidently learned the author names by checking the references of another manuscript, that I reviewed for ICLR.

### Strengths
* topic is relevant and timely
* clearly written
* reasonable evaluation:
  * models of three different sizes are tested
  * quite a few benchmarks tested

### Weaknesses
* limited novelty: Besta et al. (Reasoning Language Models: A Blueprint, arXiv:2501.11223, Jan. 2025) already propose the use of entropy as a metric to identify decisions point as well as outcome-driven process based rewards, albeit to be fair they only present their ideas without actually implementing and evaluating them.
* some aspects of evaluation I expected are missing:
  * GRPO++ details are missing, inclusive a discussion why GRPO++ is a competitive baseline
  * 5.1: hyperparameter such as K are missing
  * cost analysis, for example:
    * additional compute cost compared to GRPO++? (D.4 is not that convincing)
* reproducibility statement: number of GPUs are not discussed in 5.1

minor issues:
* abstract: "benchmarks(AIME24)" - missing whitespace
* related work: missing brackets around the references
* preliminary: already discuss phases before they are introduced in 3.4
* Figure 1, caption: Start sentences with a capital letter.
* Fig. 3 d) to f) - missing y axis label
* 4.3: "in the appendix C" -> "in Appendix C"
* 5.1: no references for DeepScaler/SimpleRL/VeRL and for the benchmarks
* Figure 3: consider removing the titles within the subfigures for better readability (also Figures 4 and 5)
* 5.2: no reference for GRPO++
* 5.3.2, Higher Entropy Enables Healthier Exploration: "a similar pattern appears at a different scale (Figure 3b) and on Qwen2.5-32B (Figure 3c)" - Shouldn't this be switched, i.e. "different scale (Figure 3c) and on Qwen2.5-Math-7B (Figure 3b)"? Or should the second part completely be omitted, since the Qwen2.5-Math-7B results are discussed in the subsequent sentence?
* 5.3.2, line 431: "achieve s" - typo: unnecessary whitespace
* Appendix B: no references for DAPO dataset
* Appendix C, equation 16: $O_j$ might not have been defined
* references:
  * consider the proper capitalization of the titles, at least for proper names and abbreviations to improve readability
  * place of publication of arXiv references can only surmised from the URL
  * [Brown et al. 2020], [Wei et al. 2022] - properly capitalize the journal/booktitle: "Advances in neural information processing systems" to be consistent with [Ouyang et al. 2022]
  * [Cui et al. 2025b], [Ecoffet et al. 2019], [Forootani 2025], [Guo et al. 2025], [Pignatelli et al. 2023], [Wu et al. 2024], [Zhou et al. 2022]  - cited differently than the other arXiv references
  * [Lightman et al. 2023] - was published at ICLR '24
  * [Ouyang et al. 2022] - missing volume number and pages numbers to be consistent with [Brown et al. 2020]
  * [Ranzato et al. 2016] - doesn't look like a proper bibtex entry
  * [Zhou et al. 2022] - was published at ICLR '23

### Questions
* introduction: CoT [Wei et al. 2022] is a prompting scheme, so why it is cited in regards to RL?
* Did you conduct an analysis into how many tokens with high uncertainty are found, i.e. how reliable the top-K mechanism works?
* Maybe I did not read the manuscript properly, but what value of K did you use in your evaluation study. It seems not to be mention in 5.1.
* How to ensure that words are not split at the token level?
* Did you test other model families?
* How can Qwen2.5-Math-7B can reach the maximum sequence length if it plateaus in Figure 4 at around 2.5k tokens, when the maximum token length is 16k?
* FR3E produces longer chains: How did you verify that they are consistent and do not result in over-thinking, which is one of the aspect that you want to address with the framework?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes FR3E (First Return, Entropy-Eliciting Explore), a structured exploration framework that identifies high-entropy points in
reasoning trajectories and performs targeted rollouts to construct semantically grounded intermediate feedback. This method provides targeted guidance without relying on dense supervision, solving granular credit assignment.

### Strengths
1. Performance gains: FR3E demonstrates either superior or at least competitive performance compared to GRPO++, notably for general-purpose LLMs (Qwen2.5-7B, Qwen2.5-32B), with more modest gains for domain-specific models (Qwen2.5-Math-7B).

2. Improved training dynamics: FR3E shows notably higher and more stable entropy throughout training, visible in Figure 3, suggesting healthier exploration and avoidance of entropy collapse.

3. Fine-grained credit assignment: The adaptive advantage modulation component keeps advantage estimates well centered and tightly distributed around zero, which theoretically reduces gradient estimator bias and allows for more stable optimization.

### Weaknesses
### Method
1. The process of advantage calculation is insufficiently descriptive in the main text. I have seen Appendix C, but still a little confused. For trajectories that share the same prefix (*e.g.*, $P_{j, m], P_{j,0}$ ) but different rewards, do they have different advantages on the shared tokens? For one trajectory, are the advantages over all tokens in FR3E the same, or do they differ depending on the divided state?

### Experiments
2. Missing relevant hyperparameters: In Section 4, the authors divide the original trajectory into $K$ state blocks and generate $M$ targeted rollouts for each state. What's $K$ and $M$?

3. Unclear computational costs: As the authors present in Appendix D.1, GRPO++ employs default rollout numbers of 4 per prompt. In FR3E, it seems to require $K \\cdot M$ rollouts per prompt, which is much more than GRPO++baseline. It raises concerns on whether the performance gain stems from a larger number of rollouts. I suggest providing detailed training time, token usage, and inference costs comparison with baselines.

4. The model is limited to the Qwen2.5 series. While the Qwen2.5 series is well-pretrained to provide a solid foundation in post-training, it also raises concerns on data contamination in widely used benchmarks [1]. Consequently, breakthroughs are predominantly observed for the mathematically strong Qwen2.5 series on benchmarks such as MATH-500, AMC, and AIME, and seldom transfer to models like Llama. I believe a more in-depth investigation on other model families (*e.g.*, Llama) is needed to validate the effectiveness of FR3E.

### Missing References
5. The dataset and benchmarks used in this paper are not cited (Section 5.1), *i.e.*, GSM8K, Math500, Minerva Math, Gaokao2023en, OlympiadBench, which is inappropriate.

6. There are several works that enhance the exploration capability in RL training [2][3][4]. I suggest discussing them in the related work.

---

[1]  Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination. arXiv preprint arXiv:2507.10532

[2] Reasoning with Exploration: An Entropy Perspective on Reinforcement Learning for LLMs. arXiv preprint:2506.14758

[3] TreeRL: LLM Reinforcement Learning with On-Policy Tree Search. ACL 2025

[4] Reasoning with Reinforced Functional Token Tuning. arXiv preprint:2502.13389

### Questions
1. What is GRPO++? What's the difference with vanilla GRPO? I don't see any description or reference.

2. Why not use DAPO as a baseline? Is DAPO better than so called GRPO++?

3. I notice that FR3E achieves a longer response length than GRPO++ in Figure 4. Does this mean FR3E encourages overthinking? The authors claim "FR3E enables longer and more consistent reasoning chains compared to GRPO++", but I don't see any supported evidence.

### Soundness
3

### Presentation
2

### Contribution
3
