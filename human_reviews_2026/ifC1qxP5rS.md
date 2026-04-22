# The Road Less Traveled: Enhancing Exploration in LLMs via Sequential Sampling

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Reinforcement learning (RL) has been pivotal in enhancing the reasoning capabilities of large language models (LLMs), but it often suffers from limited exploration and entropy collapse, where models exploit a narrow set of solutions, leading to a loss of sampling diversity and subsequently preventing RL from further improving performance. This issue is exacerbated in parallel sampling methods, where multiple outputs are drawn from the same distribution, potentially causing the model to converge to similar solutions. We propose \textbf{SESA}, a novel \textbf{SE}quential \textbf{SA}mpling framework that mitigates this challenge by generating diverse solution sketches sequentially before expanding them into full reasoning paths. This approach ensures broader exploration by conditioning each new output on previous ones, promoting diversity throughout the process and preventing policy collapse. Our experiments on a synthetic task show that sequential sampling consistently outperforms traditional RL methods in terms of path diversity and recovery from collapse. Further evaluations on real-world tasks demonstrate that SESA improves both the exploration of valid strategies and the overall performance of LLMs. On three agent benchmarks, SESA lifts success rates by $+0.25$, $+0.42$, and $+0.07$ absolute over the base model (up to an additional $211\%$ relative improvement over baseline RL), underscoring its exploration advantage. This work introduces a structured approach to exploration, paving the way for more effective and diverse reasoning in RL-trained LLMs. Code can be found at https://anonymous.4open.science/r/SESA-5E63.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper identifies a practical failure mode for RL fine-tuning of LLMs: when multiple rollouts are drawn i.i.d. from the same autoregressive policy (parallel sampling), the policy tends to collapse to a small set of high-reward outputs and exploration vanishes. To counter this, the authors propose SESA, a two-stage sequential sampling framework: (1) sequentially generate short method sketches conditioned on previous sketches to force diversity, and (2) expand each sketch in parallel into full solutions. During training SESA uses sequential sketching to diversify rollouts but computes policy updates with likelihoods conditioned only on the original input. Empirically SESA is evaluated on several reasoning/agent benchmarks. Reported results show large improvements in Pass@k and success rates versus parallel-sampling RL baselines.

### Strengths
1. The topic of enhancing exploration in RL finetuning for LLM is important
2. The proposed method appears well-founded and effective, supported by extensive empirical evidence.

### Weaknesses
1. **Efficiency concern:** The proposed sequential sketching approach inevitably increases computational cost during rollouts. However, the paper does not provide an analysis of computational efficiency, such as wall-clock time or resource overhead, which would help quantify this trade-off.
2. **Limited model and baseline coverage:** Experiments are conducted only on Qwen-7B, making it unclear whether the observed improvements generalize to other major LLM families. In addition, the set of baselines is limited. It would be valuable to include comparisons with other exploration-oriented methods, such as entropy-based methods[1].
3. **Mismatch between training sampling and evaluation distribution:** The method generates candidate responses conditioned on prior sketches, yet policy updates are computed using likelihoods conditioned solely on the input. Although the paper claims this design maintains training–evaluation consistency, the statistical implications, such as importance weighting, off-policy bias, and variance, are not thoroughly analyzed.
4. **Ablation and sensitivity analysis:** The criteria for selecting sketch lengths and templates are not well justified. Further analysis is needed to understand how sensitive performance is to sketch length, template wording, and other design choices.
5. **Implementation clarity:** The paper lacks sufficient details on the implementation of both the proposed method and baselines, including prompt sketch formats, hyperparameters, and component configurations. Providing these details would improve reproducibility.
6. **Missing metrics:** The paper does not report pass@1 results, which are standard for evaluating reasoning performance. Additionally, other metrics reflecting training dynamics, such as policy entropy, are also important to report.

[1] The entropy mechanism of reinforcement learning for reasoning language models

### Questions
see weaknesses

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces SESA (Sequential Sampling), a two-stage reinforcement learning framework aimed at enhancing exploration in large language models (LLMs). The method replaces traditional parallel sampling with sequential sampling—each output is conditioned on previously generated samples to promote diversity and prevent policy collapse. Experiments across synthetic, reasoning, and agentic tasks show modest gains in diversity and success rates, suggesting potential benefits for exploration in RL-based LLM training.

### Strengths
•	Clarity and readability: The paper is well written and easy to follow, with clear descriptions of motivation, method, and results.
•	Relevance: Exploration and diversity in RL training for LLMs are timely and important topics, especially as reinforcement fine-tuning becomes central to reasoning model development.
•	Conceptual simplicity: The idea of conditioning subsequent samples on earlier ones is intuitively appealing and could be a useful heuristic to improve diversity during training.

### Weaknesses
•	Misleading framing and terminology: Despite being presented as a new “sampling” method, the approach does not involve actual sampling from the model’s stochastic policy but instead relies on modified prompting. This makes the title and framing potentially misleading.
•	Lack of cost and efficiency analysis: The sequential approach introduces longer input contexts (due to inclusion of sketches), yet the computational or latency overhead is not quantified or discussed. This omission is critical for evaluating practicality.
•	Synthetic evaluation questionable: The “Path Exploration” task does not convincingly demonstrate that sequential sampling uncovers meaningful “strategies.” It is unclear what constitutes a “strategy” in this simplified guessing setup, and whether the gains would persist in more realistic scenarios.
•	Unclear experimental setup: Important details such as which LLMs are used in each experiment (e.g., in Table 1) are missing, making reproducibility and cross-model generalization claims difficult to assess.
•	Inflated performance reporting: Percentage improvements are reported relative to the base model rather than the relevant baseline (e.g., RAGEN), leading to overstated claims. Standard deviations and number of runs are not reported, limiting statistical reliability.

### Questions
1.	Why is sequential conditioning referred to as “sampling”? Could you clarify how this differs from prompting with additional context rather than modifying the model’s sampling distribution?
2.	Why is no sketch used during evaluation? Would incorporating sketch conditioning at test time further improve results?
3.	The authors mention deliberately focusing on weaker base models. How does this choice affect the generality and relevance of the proposed method for stronger LLMs?
4.	Can you quantify the additional computational cost (context length, inference time, memory) introduced by sequential sampling compared to parallel sampling?
5.	How is the number of sequential samples (G) chosen, and what happens if it becomes too large or too small?

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
3

### Summary
+ Summary & Contributions
	- The authors focus their attention on the well-known issue of entropy collapse and loss of response diversity during the RL-based fine-tuning of LLMs
	- Rather than generating multiple potential outputs independently and in parallel (leaving high risk of duplicate/redundant samples), the authors advocate for a sequential sampling scheme whereby individual responses are obtained one after another, conditioned on the previously-generated responses. Intuitively, such a sequential generation process makes the model privy to early responses and therefore more likely to generate further diverse candidates.
	- Recognizing the complexity of generating responses conditioned on longer and longer histories of earlier responses, the authors propose a more computationally-sensible two-stage approach. The first pass generates a high-level response strategy/method sketch sequentially to ensure diversity before a second pass to generate full responses based on the diverse sketches in parallel. 
	- Empirical results are presented demonstrating improved coverage/diversity of responses as well as better pass@k accuracy than two parallel sampling baselines (DAPO & RAGEN).

### Strengths
+ Quality
	- Strengths
		- The idea of conditioning on the history of previous responses to preserve diversity is sensible, albeit obvious, way to ensure diversity. Importantly, the authors recognize the computational impracticalities of such conditioning and offer a more practical, scalable alternative through a two-stage, hierarchical RL procedure.
	- Weaknesses
		* Major
			- If I'm understanding correctly, the quantity labeled by the authors as "Coverage@k" is really just accuracy, right? Of exactly $U = 20$ correct answers, how many were successfully recovered by the model. Why not just call this accuracy? More importantly, the design of the PathExploration task doesn't really seem logical as a barometer for good exploration. It seems like the authors synthetically generate 20 needles ("valid string locations") in a massive, China-sized haystack (all possible Province-City-District locations in China). If these 20 special "treasure points" are chosen independently, then there really is no signal to facilitate good exploration. A LLM just needs to get lucky and stumble into these 20 points by guessing through all the others. This seems analogous to a binary tree where one (instead of 20) special leaf node is chosen to have a reward of 1 while all others are zero. If no other node provides signal for where the rewarding leaf node is, then an agent has no recourse but to stumble around and touch everything. Indeed, these kinds of MDPs can be used to prove exponential gaps in sample complexity between RL and imitation learning [2]. Bearing that in mind and after reading through Appendix A.1, I'm surprised and skeptical of the good coverage reported in Figure 3. It seems plausible that perhaps there is some latent LLM prior skewing towards certain points that result in them being generated with such high frequency after so few evaluation steps. More broadly, while this is still positive for sequential vs. parallel sampling, it doesn't seem all that resemblant of the exploration problem faced by LLMs in general, where there is likely latent structure which characterizes more favorable responses.
			- I'm confused about how the authors are able to obtain empirical support for "Conclusion III" (L208) that sequential sampling following the entropy collapse due to traditional parallel sampling is able to recover response diversity. RL fine-tuning occurs in an on-policy manner, whereby policy-gradient updates are performed on prompt-response pairs generated by the current LLM (policy) in order to try and improve response quality. If the collapse has already occurred, this implies that the LLM is producing tokens (and, therefore, responses) with probability 1 or close to it, making sampling of any other response impossible. Could the authors provide more details on how exactly sequential sampling is able to induce policy-gradient updates that revitalize the diversity of the response distribution?
		* Minor
			- To the best of my knowledge, there is no such fundamental principle as "exploration drives continual improvement" in RL. Standard RL problems are often instantiated such that there are well-defined optimal policies for each sequential decision-making problem; once any of these optimal policies have been attained, no further improvement is possible. This is in contrast to the continual/lifelong RL setting, where a lack of convergence does imply there is always more to learn, but I suspect that is not the point the authors were trying to make here. They would likely be better served by simply calling out the classic exploration-exploitation trade-off.
			- It doesn't seem correct to characterize sequential sampling as inducing off-policy learning (475). It would be interesting to hear why the authors believe that to be the case.

+ Clarity
	- Strengths
		- Overall the paper is well-written.
	- Weaknesses
		* Major
			- The authors make a rather odd choice to introduce their method in Section 3 after having already presented preliminary experiments and conclusions. This is strange and doesn't seem to help the structure of the paper. In fact, it would be clearer for the authors to introduce their method and then present all empirical findings.
		* Minor
			- N/A
		

+ Originality
	- Strengths
		- The authors' instantiation of diversity-preserving response distributions via sequential sampling is, to the best of my knowlege, novel.
	- Weaknesses
		* Major
			- N/A
		* Minor
			- The idea of generating so-called "sketches" which encapsulate the high-level, abstract structure of some behavior is already established in the hierarchical RL literature [1]. One reasonable interpretation of this paper is that the authors have resurrected this idea from deep RL and brought it to bear on RL fine-tuning of LLMs. One novelty

+ Significance
	- Strengths
		- The positive results demonstrated across the experiments provide a sufficiently compelling picture that would likely incentivize readers to experiment with sequential sampling as a mitigation strategy for LLM response diversity collapse.
	- Weaknesses
		* Major
			- Perhaps the biggest issue facing this paper is that the authors are by no means the first to observe that RL fine-tuning can lead to a collapse and drastic loss of response diversity over time. While their proposed sequential sampling scheme seems like one plausible way to address the issue and improve over traditional, collapsing parallel sampling, the authors have failed to evaluate against other proposed, existing approaches for preserving response diversity (for example, the work of [3] although there are likely many other alternatives). Why not compare to any of the other approaches, for instance, cited in the related work section (L457-462)? By failing to evaluate against a stronger set of baselines that encapsulate alternative methodologies for handling the well-known collapse issue, it's unclear how significant sequential sampling is as a resolution.
		* Minor
			- N/A
			
		
+ Final Remarks
	- Overall, there are issues on the axes of quality and significance for this paper that aren't negligible. The lack of comparison to alternative methodologies for mitigating response diversity collapse is the main issue that prevents this paper from being ready for publication at this time. 



+ References
	1.  Andreas, Jacob, Dan Klein, and Sergey Levine. "Modular multitask reinforcement learning with policy sketches." In International conference on machine learning, pp. 166-175. PMLR, 2017.
	2. Sun, Wen, Arun Venkatraman, Geoffrey J. Gordon, Byron Boots, and J. Andrew Bagnell. "Deeply aggrevated: Differentiable imitation learning for sequential prediction." In International conference on machine learning, pp. 3309-3318. PMLR, 2017.
	3. Veselovsky, Veniamin, Benedikt Stroebl, Gianluca Bencomo, Dilip Arumugam, Lisa Schut, Arvind Narayanan, and Thomas L. Griffiths. "Hindsight Merging: Diverse Data Generation with Language Models." In The 41st Conference on Uncertainty in Artificial Intelligence.

### Weaknesses
Please see above.

### Questions
Please see above.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a framework to encourage exploration in reinforcement learning with large language models. Instead of generating samples in parallel, the authors propose to produce a sequence of sketches with a single prompt. The sketches are then expanded into full reasoning paths. The approach aims to maintain diversity and mitigate policy collapse during post-training. Experiments on reasoning datasets (Sudoku, AIME24), and agentic benchmarks (Sokoban, Countdown, FrozenLake) show  performance improvements over parallel-sampling baselines.

### Strengths
- The paper tackles a timely problem in reinforcement learning for large language models.
- The paper has a clear motivation, is well written and easy to follow. The idea of self-generated strategic sketches as explorative guidance is novel, interesting and intuitively appealing.
- The proposed framework is lightweight and seems easy to integrate into existing pipelines.
- Empirical results show consistent improvements over RAGEN in toy environments.

### Weaknesses
- During rollout, samples are generated conditioned on previously produced sketches, yet during training, each sample’s likelihood is evaluated only under the unconditioned input. This effectively treats off-policy samples as if they were on-policy, introducing bias without importance correction.
- The method is only a prompt-level adaptation. The performance boost comes from sampling with different prompts. It remains unclear whether the approach genuinely maintains exploration entropy or converges to a set of strategies that just yield broader initial coverage due to the prompt design.
- The paper provides no details about the strategies discovered or how they evolve over training.
- There is no ablation study on the effect of the number of sketches, sketch length, or prompt structure on performance.
- Critical hyperparameters (learning rates, clipping thresholds, batch sizes, etc.) are omitted in the manuscript.
- The chosen benchmarks are relatively simple; validation on more complex, real-world environments (e.g., WebArena or WebShop) would strengthen the empirical claims.

### Questions
- Would offline training methods (e.g., DPO) not be better suited in this setting?
- Could you report Pass@1, Pass@8, Pass@16, and Pass@64 metrics for AIME24 to better understand diversity scaling?
- Please justify the Pass@16/Pass@1 ratio as a diversity indicator. When two models have equal Pass@16 but different Pass@1, this metric penalizes stronger models.
- Could you provide examples or statistics of the strategies found and how their diversity evolves over training?
- How does the number of sketches affect performance, and how is the optimal number chosen?
- How does model capacity (e.g., smaller vs. larger base LLMs) influence the effectiveness of sequential sampling?

### Soundness
2

### Presentation
3

### Contribution
2
