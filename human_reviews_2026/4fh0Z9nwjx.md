# Exploring Expert Failures Improves LLM Agent Tuning

- Decision: Reject
- Scores: 6, 6, 4, 2

## Abstract
Large Language Models (LLMs) have tremendous potential as agents, excelling in tasks requiring multiple rounds of decision-making. 
For large-scale deployment, a smaller LLM is commonly fine-tuned by learning from teacher-model trajectories and subsequently improving itself via interaction with the environment.
A key challenge is that many complex training tasks never yield a successful trajectory (zero reward): the teacher's trajectories fail to solve them, and the student’s limited exploration cannot discover one despite many attempts. 
Without reward signals during training, the student is unlikely to solve similarly difficult test tasks.
Applying Rejection Sampling Fine-Tuning (RFT) to WebShop highlights the issue: GPT-4 (the teacher) may succeed on only 36\% of the training tasks, and RFT inherently favors actions drawn from those successes. 
As a result, the student cannot complete most complex tasks for which the teacher does not provide a direct solution because these tasks require more advanced action sequences. 
To discover reward signals in these complex tasks, we examined the failed teacher trajectories on these challenging tasks, and found that teacher's trajectories often contain valuable guidance—such as plans and key actions—that student seldom used during its exploration. 
Motivated by this insight, we introduce Exploring Expert Failures (EEF), which uses expert actions to improve the exploration during training and carefully incorporates them into the training by masking out potentially harmful actions to prevent contamination of the learning process.
This further allows us to let our student model utilize additional weaker yet more cost-efficient teachers, such as GPT-3.5 Turbo, without inheriting the weaker teacher's suboptimal behaviors.
Consequently, EEF successfully resolves many previously unsolvable tasks and significantly enhances agent performance on test tasks.
Notably, our approach achieved a remarkable 62\% win rate in WebShop, surpassing both RFT (53.6\%) and GPT-4 (35.6\%). 
To the best of our knowledge, this establishes a new state-of-the-art, achieving a score of 0.81 on WebShop and 81/100 on SciWorld, two widely used and challenging tasks for evaluating LLM agents.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
LLM agent training relies on successful expert trajectories. When an expert fails, there is no learning signal, which limits generalisation. However, there may still be useful sub-sequences even within failed expert trajectories, which are currently discarded. The paper proposes a method to reuse these segments to improve learning, identifying beneficial vs harmful actions, and then using selected positive trajectories for fine-tuning. With their new method, they achieve SOTA on a few benchmarks.

### Strengths
It's a good idea, learning from failure, or rather extracting value even from failed trajectories. I always wondered why approaches like q-learning which only calculate their score at the very end are so wasteful. This new method is not exactly very complex, but I would count that as an advantage, rather than a disadvantage. The prove is in the results, which seem to confirm the efficacy of their idea, outperforming GPT-4, and showing robustness for different model bases.

### Weaknesses
I am not quite sure whether the WebShop and ScienceWorld benchmarks are really enough to show generalisation to open-domain or real-world agent tasks. If the authors truly believe in their paradigm, I would like them to add more benchmarks. Is that feasible as a revision? Would lead me to probably increase my recommendation.

Also, the masking and identification of the beneficial actions are more of a heuristic than a deeper analysis. A bit more reflection would likewise upgrade the paper. This would also help against the possible critique of the missing more thorough hyperparameter sensitivity analysis, especially given the rather small set of benchmarks.

### Questions
See weaknesses: Both listed weaknesses could be addressed, questions are written there.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper addresses learning from failed expert trajectories when using rejection fine-tuning (RFT). The authors propose EEF (Exploring Expert Failures): run the student from intermediate expert states, identify important states, find trace segments that lead to success when started from those states, mask harmful earlier steps, and add the successful segments to supervised fine-tuning. EEF is simple to add to RFT and yields substantial gains on two agentic benchmarks.

### Strengths
- Tackles a practically important problem: exploration in long-horizon, sparse-reward tasks.

- The method is simple and practical to implement on top of existing RFT pipelines.

- The paper includes ablations and diagnostics that help connect the method to the observed performance gains.

- Writing and running examples are clear and help explain the idea.

### Weaknesses
Here are some weaknesses that if addressed, can prove EEF’s effectiveness, robustness, and practicality.

- Missing success-rate statistics: The paper does not report how often simulations started from intermediate expert states actually find successful continuations (vs. starting from $s_0$). Without these frequencies it’s unclear whether EEF’s core mechanism is generally effective or only works in selected cases.

- There’s no controlled ablation comparing fine-tuning on $s_0$-only, $s_r$-only, and both. Training only on recovery segments could probably reduces performance from initial states. It could be interesting to understand how much of the performance improvements comes from finetuning on $D_{s_0}$ vs $D_r$.

- Since the paper shows gains using cheaper GPT-3.5 traces, a clearer discussion or an explicit cost breakdown (compute / API dollars / token counts) would help readers weigh spending on higher-quality demonstrations versus more simulation budget. A recommendation for budget allocation under a fixed cost constraint might be especially useful.

- Evaluation covers only two benchmarks; adding another long-horizon domain would help determine whether EEF addresses a general failure mode or a domain-specific phenomenon.

### Questions
- Could you provide approximate compute / costs (e.g., API dollars or token counts) for the EEF GPT-4 runs, the RFT×6 baseline, and the mixed GPT-3.5+GPT-4 variant in Table 3?

- Have you observed cases where EEF reduces performance (for example by overfitting to brittle recovery actions)? If so, how common are those cases?

- On “simplicity bias”: is there prior empirical work you can cite? If not, would it be possible to quantify how often failed trajectories in your datasets seem explained by simplicity bias?

- How does RFT perform worse than SFT POS in table 3? Isn’t RFT with one iteration equivalent to SFT POS? Since the paper reports the best model across RFT iterations, RFT should in principle perform at least as well as SFT POS. Could you clarify this discrepancy?

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
4

### Summary
The paper introduces Exploring Expert Failures (EEF), a fine-tuning method that enhances the performance of LLM-based agents on complex, multi-step tasks where even expert models (e.g., GPT-4) frequently fail. EEF can expand exploration by reusing expert actions, acknowledging that failed trajectories frequently encode useful signals. The method is evaluated on the WebShop and ScienceWorld benchmarks, demonstrating its effectiveness.

### Strengths
The paper is written well and easy to follow. 

The idea of reusing valuable actions from failure trajectories to expand exploration sounds promising.

The experiments on the WebShop and ScienceWorld benchmarks demonstrate the effectiveness of the method.

### Weaknesses
The novelty of the proposed method appears limited, as the key concept of utilizing failure trajectories is not new. It should be noted that similar ideas have been investigated in prior works, including IPR [1], LEMA [2], and STeCa [3].

The experiments are not sufficiently comprehensive. (1) Several related works are not compared against, such as IPR [1], LEMA [2], and STeCa [3]. (2) Although LLAMA3-8B Instruct and Mistral-7B-v0.3 were used as base models, the results for Mistral-7B-v0.3 are not presented in sufficient detail. Moreover, since Mistral-7B-v0.3 was released a year ago, it is recommended to conduct experiments using Qwen3-8B as an additional base model. (3) In the experiments, a fair comparison should be made (with Iter=3) against the method that only training on solutions from the initial state. (4) It is necessary to compare EEF with different solution selection strategies, such as the randomly selecting strategy. (5) The individual effects of the two types of important states, $D_{s_0}$ and $D_r$, should be verified to understand their respective impacts on the experiment. (6) The analysis experiments currently focus only on WebShop. Similar analyses should be extended to the ScienceWorld benchmark. (7) In the Efficiency Analysis, a brief discussion on the trade-off between the simulation budget (M) and performance gains would be helpful.

Regarding the methodology, harmful states are identified through agent simulation. Could sampling issues potentially lead to inaccurate judgments in this process?

[1] Watch Every Step! LLM Agent Learning via Iterative Step-level Process Refinement, EMNLP 2024 \
[2] Learning From Mistakes Makes LLM Better Reasoner, 2023  \
[3] STeCa: Step-level Trajectory Calibration for LLM Agent Learning, 2025.2

### Questions
See the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces a method called Exploring Expert Failures (EEF) to improve the fine-tuning of LLM agents. The authors address a key limitation of standard techniques like Rejection Sampling Fine-Tuning (RFT), where agents learn only from successful expert demonstrations and thus fail to master complex tasks where the expert often fails. The core idea of EEF is to salvage useful information from the expert's failed trajectories by simulating the agent's performance from intermediate steps. By identifying segments of a failed path that can lead to success, EEF incorporates these "beneficial actions" into the training data, demonstrably improving performance on benchmarks like WebShop and SciWorld.

### Strengths
The problem studied in the paper is very interesting, especially the analysis of RFT, although it seems that this problem has not been well solved.

### Weaknesses
The methodology's innovation is arguably incremental. It builds directly upon the existing RFT paradigm, and its primary contribution is a more sophisticated data filtering and augmentation strategy rather than a fundamentally new approach to agent learning. The effectiveness of EEF is heavily dependent on extensive simulation and sampling. The process requires re-simulating trajectories from numerous states within failed expert attempts to identify useful sub-paths, raising concerns about its computational expense and scalability. The performance gains are achieved through what is essentially a more guided, brute-force exploration of the expert's failure space. In essence, the method refines the "guess-and-check" nature of sampling-based tuning by adding a more targeted "check" phase, but it does not move beyond this data-intensive framework.


1. What if the model consistently fails to sample a successful trajectory?

2. Is there an analysis of the sampling efficiency? Specifically, what proportion of the explored trajectories are ultimately found to be useful for training?

3. As I mentioned in the summary, this method relies on extensive sampling and is confined to Supervised Fine-Tuning.

4. From another perspective, the method like GRPO also samples a large number of rollouts but then uses Reinforcement Learning to optimize the model, rather than SFT. This appears to be a more logical approach.

### Questions
Stated in Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
