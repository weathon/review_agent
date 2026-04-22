# Better LLM Reasoning via Dual-Play

- Avg Score: 2.67
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2

## Abstract
Large Language Models (LLMs) have achieved remarkable progress through Reinforcement Learning with Verifiable Rewards (RLVR), yet still rely heavily on external supervision (e.g., curated labels). Adversarial learning, particularly through self-play, offers a promising alternative that enables models to iteratively learn from themselves—thus reducing reliance on external supervision. Dual-play extends adversarial learning by assigning specialized roles to two models and training them against each other, fostering sustained competition and mutual evolution. Despite its promise, adapting dual-play training to LLMs remains limited, largely due to their susceptibility to reward hacking and training instability. In this paper, we introduce PasoDoble, a novel LLM dual-play framework. PasoDoble adversarially trains two models initialized from the same base model: a Proposer, which generates challenging questions with ground-truth answers, and a Solver, which attempts to solve them. We enrich the Proposer with knowledge from a pre-training dataset to ensure the questions' quality and diversity. To avoid reward hacking, the Proposer is rewarded for producing only valid questions that push the Solver's limit, while the Solver is rewarded for solving them correctly, and both are updated jointly. To further enhance training stability, we introduce an optional offline paradigm that decouples Proposer and Solver updates, alternately updating each for several steps while holding the other fixed. Notably, PasoDoble operates without supervision during training. Experimental results show that PasoDoble substantially improves the reasoning performance of LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper aims to improve the reasoning capabilities of a large language model (LLM) on tasks grounded in a specific knowledge base. The approach trains two identical LLMs: the Proposer, which has access to this knowledge base and generates the most challenging questions; and the Solver, which does not have access to it and attempts to answer them. Training is performed using Reinforcement Learning with Verifiable Rewards (RLVR), where the objective function encourages the Proposer to produce difficult, knowledge-based questions that expose the Solver’s weaknesses. The authors study both online (joint updates) and offline (question-buffer) training variants and report gains on six math benchmarks across several Qwen base models, with the largest improvements on Qwen3-1.7B.

### Strengths
The paper introduces a well-defined training setup that leverages reinforcement learning with verifiable rewards to improve reasoning performance. It achieves strong empirical results: Qwen3-1.7B-Base improves by about 20 points in pass@1 accuracy, despite using limited supervision. The presentation is clear, with consistent terminology and a straightforward description of the training process. The method is evaluated on multiple math benchmarks, demonstrating solid improvements over strong baselines.

### Weaknesses
- Several average scores reported in Table 1 are incorrect — at least six appear miscalculated (e.g., Qwen3‑1.7B Coldstart: 29.55 → 24.63; PasoDoble Offline: 47.51 → 39.59). These are not minor rounding errors, but significant numerical inconsistencies that affect the paper’s main claims. This undermines trust in the evaluation and should be corrected. 

- After correcting the scores, Coldstart consistently underperforms the corresponding Base models across all configurations, despite being fine-tuned on the same domain-specific knowledge. This is unexpected and suggests that the supervised finetuning stage may be ineffective or even detrimental.

- The paper omits discussion and comparison to closely related approaches, particularly Agentic Adversarial QA for Improving Domain-Specific LLMs [1]. That work also uses a two-agent setup to expose model weaknesses through adversarial question generation, but follows a different methodology: an offline framework that selects challenging questions using text-based gradient feedback rather than reinforcement learning. A direct comparison—either conceptual or empirical—would help clarify the novelty of PasoDoble and better position it within the broader landscape of dual-agent self-training methods.

- The experimental setup focuses exclusively on mathematical reasoning tasks (e.g., GSM8K, MATH, OlympiadBench), which limits the generalizability of the method. While math is a well-established domain for evaluating structured reasoning, it's unclear whether the proposed approach would transfer to other domains such as programming, science QA, or commonsense reasoning. Including evaluations on a more diverse set of benchmarks would strengthen the claims of improving general reasoning capabilities.

[1] Grari, V., Tomoiaga, C., Lamprier, S., Hashimoto, T., & Detyniecki, M. (2025). Agentic Adversarial QA for Improving Domain-Specific LLMs. In Second Workshop on Test-Time Adaptation: Putting Updates to the Test! at ICML 2025.

### Questions
- After correcting the reported averages in Table 1, Coldstart consistently underperforms the Base model across all model sizes. Could the authors clarify why fine-tuning on a domain-specific knowledge base results in worse performance? Does this point to issues in the training setup, data quality?

- Could you clarify how PasoDoble differs conceptually from Agentic Adversarial QA for Improving Domain-Specific LLMs [1]? Both use a two-agent setup for adversarial question generation — is there a specific reason it was not discussed or compared in the paper?

- Have you considered applying PasoDoble to non-math domains (e.g., code generation, scientific QA)? If not, what are the key limitations or challenges?

- How sensitive is the performance to the thresholds for clipping?

### Soundness
3

### Presentation
3

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
This paper proposes PasoDoble, a Dual-Play training framework where two LLMs compete and co-evolve to improve reasoning ability.
The Proposer generates challenging questions using a knowledge base, and the Solver learns by solving them.
Without any labeled data, the method achieves gains of over 20 points on math reasoning benchmarks.

### Strengths
- The paper proposes a Dual-Play learning framework that enhances reasoning ability by having two LLMs compete with each other.
- It stabilizes the Proposer’s question generation using a knowledge base and ensures stable adversarial training through a reward design based on correctness and diversity.

### Weaknesses
- The proposed method appears unfair because it uses a knowledge base, while the baselines do not. I am particularly concerned about how much knowledge or formatting from the evaluation data may have leaked into the knowledge base.
- The paper does not quantitatively show how valid the generated problems were, nor how invalid the discarded problems actually were.
- Training both the Solver and the Proposer roughly doubles the computational cost compared to standard training.
- The idea of improving performance through competition is not particularly novel.
    - https://arxiv.org/abs/2404.10642
    - https://arxiv.org/abs/2311.08107
    - https://www.arxiv.org/abs/2510.18407
    - https://arxiv.org/abs/2504.19162

### Questions
- Why does performance degrade when the Proposer is frozen? This setting essentially corresponds to standard self-learning with an added knowledge base, so a performance drop seems counterintuitive.

### Soundness
2

### Presentation
2

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
This paper proposes a reinforcement learning technique, PasoDoble, to iteratively train two instances of the same base model in an Adversarial Learning manner, where reward values are inversely designed. The two instances are given respective roles of Proposer and Solver, trained jointly or alternatively. The Proposer is trained to generate diverse and challenging problems with ground-truth answers leveraging pre-training Knowledge Base, while the Solver is trained to solve the problems accurately. Through empirical experiments, the authors show that this technique improves mathematical reasoning capacity by approximately 20 points on average on larger models (1.5B-1.7B) of Qwen family. Yet, the effectiveness of this design is not found on smaller models. In addition, this technique evidently sustains mathematical reasoning capability improvement for hundreds of training steps, exceeding R-Zero’s 3-iteration plateau. However, this technique fails to transfer to out-of-domain tasks. As a main contribution, this paper highlights adversarial dual-play training that can reduce LLM’s dependence on high quality supervised data, where mathematical reasoning improvement can be achieved through pre-training knowledge.

### Strengths
- The methodology is explained relatively clearly.
- The in-domain results seem promising, despite lack of out-of-domain generalization.

### Weaknesses
- This paper explains the main methodology, reward function design and findings clearly. However, the conclusion is on weaker grounds due to insufficient baselines, inadequate methodology validation and result interpretations.
- Missing important baseline: SFT model using Knowledge Base should be a critical baseline to highlight the advantages of this technique. If SFT can achieve a similar level of mathematical reasoning capacity, the value of this technique remains unclear.
- Insufficient validation of reward hacking prevention: The paper claims to guarantee question quality and prevent reward hacking of Proposer by removing questions with low Solver accuracy. However, this does not guarantee the question quality. Conversely, high Solver accuracy doesn’t necessarily imply high quality questions. It could easily be common hallucination by both Proposer and Solver, given they are initialized from the same base model. Although the authors sampled 100 questions to study the question quality, this is done by LLM, not human. There is insufficient validation to claim Proposer always generates high quality questions. 
- Title overstatement: The title “Better LLM Reasoning” seems like an overstatement. The experiment results only show improvements of larger models on mathematical reasoning domain with no transfer to out-of-domain tasks.
- Ambiguous statistical demonstration: The graphs in the paper show no error bar or confidence intervals. It is unclear if the result is concluded with multiple runs of different seeds.
- Lack of explanation of result: The paper doesn’t provide clear explanations for this technique’s failure on smaller models. Is this because Proposer could not interpret the Knowledge Base given its capacity? This was mentioned in the ablation study section, but there is no discussion directly addressing the experiment results.

### Questions
- Diversity reward design: In the diversity reward, the similarity of questions is calculated with token occurrence. However, this does not guarantee high semantic distance. What other options are considered?

### Soundness
2

### Presentation
3

### Contribution
2
