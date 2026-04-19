# DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 5

## Abstract
Lean is an advanced proof assistant designed to facilitate formal theorem proving by providing a variety of interactive feedback. In this paper, we explore methodologies to leverage proof assistant feedback to augment the capabilities of large language models in constructing formal proofs. First, we deploy online reinforcement learning using Lean verification outcomes as the reward signal to improve the proof completion policy. This straightforward approach shows great promise in enhancing the model's alignment with the formal verification system. In addition, we propose RMaxTS, a variant of Monte-Carlo tree search that employs an intrinsic-reward-driven exploration strategy to generate diverse proof paths. The tree structure is organized to represent the transitions of intermediate tactic states, extracted from the compilation messages given by Lean's tactic mode. The intrinsic reward is constructed to incentivize the discovery of novel tactic states, which helps to to mitigate the sparse-reward problem inherent in proof search. These techniques lead to a more efficient planning scheme for formal proof generation, achieving new state-of-the-art results on both miniF2F and ProofNet benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces DS-Prover-V1.5, a model integrating proof assistant feedback to enhance theorem-proving capabilities. The authors propose using Lean’s proof assistant in reinforcement learning (RL) and Monte-Carlo Tree Search (MCTS) settings, achieving competitive results on the miniF2F and ProofNet benchmarks. Two main innovations stand out: an intrinsic reward-based exploration strategy (RMaxTS) in MCTS to encourage diverse proof paths and a truncate-and-resume mechanism to handle proof verification errors. These advances lead to state-of-the-art results, highlighting the role of proof feedback in RL training.

### Strengths
1. The paper tackles a critical problem, merging theorem-proving and reinforcement learning with proof assistant feedback.
2. Strong empirical results demonstrate improvements in state-of-the-art theorem proving on established benchmarks.

### Weaknesses
1. I have various doubts listed in the question section.
2. I'd love to see a short technical details section in the main body, model sizes, and the number of tokens used in training. Btw. what is the number of tokens in the RL training?
3. I'd love to see more ablations (e.g., other models). I realize that such experiments are costly, still, perhaps feasible with smaller models, to study in more detail the influence of design choices.

### Questions
1.	Could you clarify the standard deviation formulas used in tables? Based on binomial distribution $\sigma = 1/\sqrt{npq}$, results seem to approximate $\sigma \approx 3%$ when  p = 1/2  and  n = 250 .
2.	The gains achieved with RL over Supervised Fine-Tuning (SFT) in Tables 1 and 2 are modest. What factors contribute to these limited improvements?
3.	Similar to the RL results, improvements with RMaxTS appear modest in Tables 1 and 2. Could you provide more context?
4.	What is the exact sample budget (e.g.,  16 \times 4096 )? Is it comparable between RMaxTS and other baselines? Given potential hardware utilization differences, are wall-time comparisons available?
      - It would also be interesting to see token-budget-aligned results.
5.	The improvements in Figure 3 also appear modest. Any insights?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies using LLM to perform theorem proving in Lean. They propose a framework that integrates supervised fine-tuning, reinforcement learning, and Monte-Carlo tree search for efficient exploration and exploitation of tactic space. Experiment results show that the propose method achieves new state-of-the-art results on both miniF2F and ProofNet benchmarks.

### Strengths
This paper studies theorem proving using LLM, which is an important direction towards artificial general intelligence. The paper is well-written. The experiment results are quite promising as it establishes new state-of-the-art results. The proposed tricks for resolving hard exploration problem is also novel (RMaxTS + discounted UCB) and insightful.

### Weaknesses
Lean is not very well-known to the community in my opinion, and the proposed method is highly tailored for Lean (for example, tactic-level tree abstraction). Some general introduction about Lean would be very helpful in understanding the design of the algorithm.

### Questions
1. During the training, is the context-length enough for all prompts? Do you have ablation studies on how context length affect the final performance?
2. Have you tried sliding window instead of discounted weight to handle non-stationarity? Sliding window is simpler and more commonly used in the bandit community.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a class of models named DS-Prover-V1.5 that combines the proof-step and whole-proof generation paradigms. The models come with two versions: (a) supervised fine-tuned (DS-Prover-V1.5-SFT), and (b) incorporating reinforcement learning (DS-Prover-V1.5-RL). In the SFT case, the data is augmented in two ways. First, the intermediate tactic state information extracted from Lean proof assistant (position of each tactic and the tactic states before and after its application) is added, and a corresponding prediction task is formulated. Second, the proof dataset from (Xin et al., 2024) is extended by providing a natural language solution at the beginning of the proof block and by inserting comments for Lean tactics with the help of DeepSeek-Coder V2 236B (Zhu et al., 2024). The paper considers the chain-of-thought (CoT) and non-CoT versions for proof completition. In the RL case, the model is trained on verification feedback from the Lean 4 prover using the GRPO algorithm (Shao et al., 2024), with a reward structure assigning 1 in the case of a proof verified as correct and 0 otherwise. A subset of theorem statements from the supervised fine-tuning dataset is used as training prompts. 

The paper combines both models (SFT and RL versions) with the MCTS planning algorithm. The paper follows previous work on adding the following innovations to the classical MCTS: (a) in the selection phase, a so-called virtual node option (Wang et al., 2023) which allows expanding a non-leaf node, (b) in the expansion phase, a whole-proof consisting of path of nodes, truncated at the first verification error (if any) from the Lean prover, is added to the tree during each iteration, (c) in the backpropagation stage, an RMax exploration mechanism is added (Brafman & Tennenholtz, 2002), with extrinsic and intrinsic (Jin et al., 2020) rewards, and a discounted bonus for upper confidence bound (Garivier & Moulines, 2011).

The setup is tested on two datasets: miniF2F and ProofNet. The paper claims that the DS-Prover-V1.5-RL version outperforms baselines in the single-pass and tree search categories. The following ablation studies are provided: (a) CoT vs. non-CoT versions, (b) single-pass vs. MCTS, (c) discounted vs. non-discounted UCB, and (d) presence vs. absence of tactic state information.

### Strengths
* The paper combines a proof-generating model with (a version of an) MCTS algorithm, unifying the proof-step and whole-proof generation approaches.
* The approach applies several MCTS innovations and uses the Lean prover as a verifier. 
* The paper claims that DS-Prover-V1.5-RL performs the best in single-pass and tree search categories.

### Weaknesses
* The paper claims `The model is named DS-Prover-V1.5, as it builds upon the prior work of DeepSeek-Prover-V1(Xin et al., 2024).`, but does not provide the description of DeepSeek-Prover-V1. In particular, no details on the underlying language model were provided. 
* The paper provides no pseudo-code or loss function formulas, making understanding the approach's details difficult. 
* GRPO is not described (even in the Appendix).
* Experiments:
	* For some categories, the baselines are other versions of the same model.
	* Tables 1-4 do not include confidence bounds for the best-performing methods.
	* Some important information is missing:
		* Not all model details are disclosed for use in the MCTS algorithm.
		* No comparison is provided between the number of parameters or FLOPS used by the evaluated models.
		* No wall clock time is given.
	* The results of Tables 1-4 are not discussed in great detail other than extracting numbers from the tables. What are the key reasons/hypotheses for the performance differences? What are the qualitative and quantitative findings with respect to the difficulty of samples from the datasets? Are there any features that stand out or can be used to classify theorems?
	* The potential data leaks are not discussed.
	* No number of seeds is reported.
	* What are the $\mu$'s and $\sigma$'s (present in tables and figures and defined in the caption of Figure 3) computed over?
	* The confidence intervals are asymptotic and one-$\sigma$ (as defined in the caption to Figure 3), corresponding to $68\%$ (assuming a standard normal distribution), a rather narrow confidence level.
	* The results in Figure 4 suggest that many differences are not statistically significant.

Edit 29 Nov 24: I increased the rating 5->6.

### Questions
Additionally to the questions provided above, in the MCTS expansion phase, was the rollout a single whole-proof generation, or were there many? If the answer is the latter, how many were used? Additionally, did the Authors consider bootstrapping in the AlphaZero style?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper presents a method for improving LLM performance on theorem proving tasks, using RL on feedback from a proof assistant tool Lean to finetune the model, in combination with a Monte-Carlo tree search to iterate and search the space of possible proofs.

### Strengths
This paper is fairly well written and clear. The problem being addressed (automatic theorem proving) is important and well motivated. The experimental evaluation is reasonably conclusive. Figures 1 and 2 illustrate the method well.

### Weaknesses
First off, I will say this paper is a little outside my area of expertise, which is in traditional deep RL rather than LLM finetuning, and I'm not familiar with theorem proving as a problem domain, so take the following with that qualifier.

That said, my main concern on this paper is that it doesn't do a great job of justifying the composition of multiple standard methods as non-trivial scientifically. Finetuning LLMs using RL with automated domain-specific feedback isn't a new idea, nor is MCTS on top of RL agents generating action sequences (including for proof/program synthesis- "Program Synthesis Through Reinforcement Learning Guided Tree Search" is an earlier example, but the idea has been explored a few times in different ways I believe). There are some technical details involved in applying these ideas using an LLM (such as how to get individual tactics as atomic actions from the model), but from my perspective these don't seem like significant contributions that will be applicable outside this paper or closely related work.

All of that might be acceptable if the result was a major breakthrough in an important problem domain, but the results as presented seem incremental, honestly- compared to the supervised fine tuning baseline, the reported improvement in accuracy is around 1-3% against a baseline of ~15% or ~50% depending on the benchmark. This doesn't seem like a major improvement in the domain of theorem proving that would motivate a relatively straightforward combination of existing algorithms.

All that said, perhaps there's something I'm not seeing here, or context I lack as an outsider to the field this work comes from, so I'm open to persuasion on the above. However, as is I am inclined to recommend rejection since I don't see a significant novel contribution in this work.

### Questions
Some additional suggestions/questions

-I'm a little skeptical GRPO would outperform PPO on this task since it involves actual multi-step decision making and thus the value function should provide significant benefit, at least if trained as a proper multi-step MDP. Perhaps it would be worth testing both versions?

-Why generate an entire proof and then truncate to the first error? Why not generate one tactic at a time, as is standard in multi-step RL?

-Am I correct that given an equivalent sample budget the difference between the otherwise equivalent MCTS and 1-shot generation methods is simply the way candidate solutions are generated (starting from a leaf in the search tree versus fully resampled)? Does the reduced number of new tokens generated (since some are derived from the existing leaf node) result in meaningful compute savings for the MCTS algorithm?

-Figure 4 needs to define what the shaded regions are, especially since the regions are heavily overlapping, leading me to question the statistical significance of the benefits of RMaxTS over the ablations.

-In tables 1 and 2, why do some accuracy numbers have +- intervals (standard deviation? I don't see these intervals defined in the caption or introductory text) and others don't?

### Soundness
3

### Presentation
2

### Contribution
2
