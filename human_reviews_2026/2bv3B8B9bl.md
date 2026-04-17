# ATPO: ADAPTIVE TREE POLICY OPTIMIZATION FOR MULTI-TURN MEDICAL DIALOGUE

- Decision: Accept (Poster)
- Scores: 8, 4, 2, 8

## Abstract
Effective information seeking in multi-turn medical dialogues is critical for accurate diagnosis, especially when dealing with incomplete information. Aligning Large Language Models (LLMs) for these interactive scenarios is challenging due to the uncertainty inherent in user-agent interactions, which we formulate as a Hierarchical Markov Decision Process (H-MDP). While conventional Reinforcement Learning (RL) methods like Group Relative Policy Optimization (GRPO) struggle with long-horizon credit assignment and Proximal Policy Optimization (PPO) suffers from unstable value estimation in this context, we propose a novel uncertainty-aware Adaptive Tree Policy Optimization (ATPO) algorithm. Our method adaptively allocates the rollout budget to states with high uncertainty, quantified by a composite metric of Bellman error and action-value variance. This strategy enables more accurate value estimation, while fostering more efficient and diverse exploration. To mitigate the high computational cost of tree-based RL, we introduce two key optimizations: an uncertainty-guided pruning mechanism to minimize the number of rollouts, and an asynchronous search architecture that leverages KV cache reuse to maximize inference throughput. Extensive experiments on three public medical dialogue benchmarks demonstrate that our algorithm significantly outperforms several strong baselines, culminating in Qwen3-8B model surpassing the much larger GPT-4o (+0.92% accuracy).

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposed an adaptive tree policy learning method for multi-turn dialogue systems. It builds a conversation tree to calculate the value of nodes and uses reinforcement learning algorithms to improve the system with feedbacks. An uncertainty-aware tree expansion component is also introduced to expand the conversation tree. Experiments show performance improve on different datasets and different large language models.

### Strengths
1. The designs of the framework seem reasonable, because conversation has different flows and can be modeled as a tree.\
2. The presentation of the paper is clear.
3. Experiments demonstrate performance improvement on different datasets and different models.

### Weaknesses
1. It will be better if other aspects of conversation quality, such as informativeness and helpfulness can be evaluated.
2. The paper claims an improvement of computation cost with KV cache techniques, so a detailed analysis on efficiency will be better.

### Questions
This methods seem more general. Can it be applied to general conversation settings other than the medical domain?

### Soundness
3

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
This paper proposes a new RL training method to improve LLMs ability at solving medical multi-turn dialogues, which involves asking many (clarification/diagnosis) questions in multiple turns before returning the final answer. The authors identify that a key challenge in optimizing multi-turn dialogue is the high uncertainty in user's responses, resulting in often insufficient exploration/data coverage for training a robust enough medical agent. The proposed method (ATPO) aims to solve this by utilizing tree search algorithms, so that during RL rollout ATPO prioritizes on expanding nodes with high uncertainty. During optimization, ATPO then treats each branch of the search tree as an independent trajectory to perform loss computation/backpropagation. The authors experimented on three medical benchmarks and showed improved performance of ATPO compared to baselines such as GRPO and TreePO.

### Strengths
- The notion to expand nodes and allocate more rollout budget to states with high uncertainty is sound. I believe the overall design (with some abstractions) could be applicable to other dialogue tasks beyond medical benchmarks.

- The authors provided evaluation on three different medical benchmarks, showing improvement of ATPO compared to other training methods such as GRPO and TreePO.

- The authors also provided interesting analysis of performing tree-based training during RL. Specifically, the authors experimented with using different uncertainty metrics during RL/the importance of down-weighing policy updates by node visit counts for stability.

### Weaknesses
Despite the overall positive experimental results, I believe there are some uncertainty/flaw in the experimental setup that could substantially undermine the results and comparisons made. If these are addressed I am willing to increase my soundness and overall score. I detail them below.


1. Tree based methods generally requires much more compute compared to methods such as PPO/GRPO, as they need to perform multiple policy and value inference per state. However, neither Table 1 nor Figure 2 reports FLOPs statistics or # of LLM calls during rollout, raising concerns that the observed gains may stem from simply using more compute rather than true algorithmic improvements. I suggest the author to (1) report FLOPs used during rollout stage across different methods (especially comparing search based method such as ATPO and non-search based method such as GRPO and PPO), and (2) provide compute-equal comparisons between ATPO and methods such as GRPO and PPO (e.g., if ATPO uses N flops during the entire rollout stage, adjust the batch size and group size of GRPO so that it also uses N flops during rollout).


2. Many results in Table 1 are actually improvements *within one standard deviation*. For example, the MedQA and MedMCQA result on Qwen3-1.7B (TreePO vs ATPO), MedMCQA result on Qwen3-4B, and MedicalExam and MedMCQA result on Qwen3-8B. Additionally, simply prompting GPT-4o already achieves performance within one standard deviation of the best results across all benchmarks. Could the authors provide a statistical significance analysis to confirm whether these improvements are indeed meaningful?


3. Since simply prompting GPT-4o achieves near best result (within one standard deviation) in Table 1 across all benchmarks, I wonder why did the author not compare against a simple baseline of directly distilling (correct) dialogue outputs obtained by prompting GPT-4o? Instead, this paper uses self-play data obtained from Gemini-2.5-Pro (L321-323), whose performance is also not reported in Table 1. This is because (1) improvements from RL tends to heavily depends on the quality of the initial SFT checkpoint, which may be undertrained in this work; (2) distilling outputs from GPT-4o is far more computationally efficient than SFT + RL, making it a more practical choice in the context of this paper if prompting GPT-4o already achieves near-best performance.

### Questions
- Have the authors measured test performance when a user simulator *different from training* is used (e.g., LLaMA-3.3 or LLaMA-4 models). It is likely that RL training with a fixed user simulator may overfit to specific tone/response formats of the user simulator used during training, and may not generalize to other unseen users/user simulators at test time.

### Soundness
2

### Presentation
3

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
This paper proposes a framework, namely ATPO, for generating multi-turn medical dialogues, which enhances the performance of LLMs in medical QA datasets.

### Strengths
(1) This paper has some theoretical derivation for the framework.

### Weaknesses
1) However, I find some parts of the theoretical derivation questionable. 1） When defining the Bellman error in equations (1) and (2), the authors use a one-step lookahead. This is questionable since with one step, there is no long-term reward (i.e., long-term exploration). In this case, equations (1) and (2) would collapse to the average reward of all the states. If the authors finally generate an answer based on the whole dialogue, it makes more sense to enable the Bellman error to look multiple steps. 2) When writing equations (1) and (2), the authors need to freeze one of the critics to calculate the Bellman error. I didn't see any mention of this.
(2) This is a fundamental question. The authors are using LLM-generated data to train LLM. This can be problematic. For example, the authors used Gemini-2.5-Pro to simulate the conversation data. There will be data leakage if the Gemini-2.5-Pro also uses the same dataset for its pretraining or fine-tuning. Thus, the improvements in Table 1 can be attributed to the data leak.
(3) The title of this paper talks about the optimization for multi-turn medical dialogue. But there is no evaluation of the quality of the generated dialogue! Instead, the authors use medical QA performance to justify the usefulness of the multi-turn dialogue. The authors need to think clearly about this. If your purpose is to have a better performance on medical QA, then you need to compare with the SOTA methods. If the purpose is to generate better multi-turn medical dialogue, you need to improve Cons (1) and (2).

### Questions
I encourage the authors to keep polishing this work. If the authors can improve or convince others about the Cons (1) and (2), and reorganize the paper, I believe this can be a good work.

### Soundness
1

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
4

### Summary
This paper addresses sampling for RL models being trained for multi-turn conversations with long-term objectives. Specifically, the background use-case involves answering medical multiple-choice questions in an interactive scenario where the LLMs must ask clarification questions proactively (seek information when missing) before providing the answer.

To this end, the authors extend recent work on Tree Policy Optimization developed for reasoning tasks for information-seeking conversational scenarios. The default TreePO would incur high computational costs to roll-out all possible trajectories from a given “state” for multi-turn conversations while estimating the policy and value functions during RL. 

To mitigate these costs, they propose measuring the uncertainty of a state and pruning mechanisms based on this uncertainly thus making the sampling process efficient while ensuring diverse exploration. Their proposed metric combines both the epistemic uncertainty (limitations of the model wrt selection among possible actions from a given state) and aleatoric uncertainty (e.g. induced due to user responses to questions) and  only those branches which meet thresholds on uncertainty are retained resulting in efficient tree expansion during sampling.

Coupled with asynchronous search architectures and KV caches that enable prefix sharing, their experiments show not only improvements in final task (MCQA) accuracy but are also compute efficient. In particular, their performance is on-par with SOTA and slightly out-performs GPT-4o model using a significantly smaller Qwen3-8B model on three different datasets from the medical domain.

### Strengths
The paper makes a good extension to recent works on using TreePO for training RL models for reasoning tasks and conversation forests on the task of answering MCQs in medical domain. This is an important task in the important domain of medical diagnosis that may specifically prefer small, in-house models than proprietary ones such as GPT.

The paper is well-written, building up the intuition for the proposed metric as well as the developing equations for actor/policy and critic/value updates.

The main claims are suitably supported via detailed experiments and ablation studies on three separate datasets from the topic. 

Overall, a valuable contribution advancing the state-of-the-art in modeling efficiency aspects of sampling in RL for multi-turn conversations.

### Weaknesses
Not weaknesses specifically but some areas for improvement where details are lacking or there are clarity issues are listed below--

- Contributions (line 84) --This aspect is not really highlighted much in the paper and as such the setup seems identical to that described in the TreePO paper (Li et al 2025b cited by the authors). If different, please highlight the differences and why this forms a core contribution. 

- In general, the performances in Table-1 seem very close to TreePO in many instances—would like to see some explanation on key differences between the two methods (if there is anything else apart from the pruning aspect which save on computation)—the lines 333-338 are not very clear 

- Overall discussion on parameters, possible application in practice, and some details on experiments need presentation improvements, some of which are listed in the Questions section.

### Questions
- Would like to see some discussion on how this may work in “real world” usage where the MCQ options are not available? How might that look like? How will the process scale when the number of possibilities become 10 instead 4? What are the consequences when the user assistant does not behave “ideally” providing the correct answer in response during inference? May be the above can be discussed using the anecdotal example in Figure 3.

- Some discussion on where the pruning might breakdown, chances of throwing out useful trajectories? Intuition for setting the threshold, how does one do this in practice and why was it set to 0.5/1.5 in your experiments? 

- Lines 255-257 seems to be missing some qualification information, as to when the critic value estimates are used -- requires rewriting for clarity.

- For the datasets, where atomic fact extraction was done with Gemini etc (Appendix A.2), include the prompts and other details or provide references, if same as those used for other datasets. What does an atomic fact look like? How do you ensure with GPT-4o that the user assistant is behaving correctly? (305-308)

### Soundness
3

### Presentation
4

### Contribution
3
