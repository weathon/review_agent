# Language Models Can Learn from Verbal Feedback Without Scalar Rewards

- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
LLMs are often trained with RL from human or AI feedback, yet such methods typically *compress nuanced feedback into scalar rewards*, discarding much of their richness and inducing scale imbalance. We propose treating verbal feedback as a conditioning signal. Inspired by language priors in text-to-image generation, which enable novel outputs from unseen prompts, we introduce the **feedback-conditional policy (FCP)**. FCP learns directly from response–feedback pairs, approximating the feedback-conditional posterior through maximum likelihood training on *offline* data. We further develop an *online bootstrapping* stage where the policy generates under positive conditions and receives fresh feedback to refine itself. This reframes feedback-driven learning as conditional generation rather than reward optimization, offering a more expressive way for LLMs to directly learn from verbal feedback.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In classical reinforcement learning approaches,  feedback from the environment are usually represented as scalar rewards. While this principle has guided the design of traditional RL algorithms, it becomes restrictive in the context of language models, where feedback is often expressed in rich verbal form. Training LLMs with such scalar feedback leads has severe limitations and is prone to a significant loss of information.

The authors introduce a feedback-conditional policy (FCP) framework. In the offline stage, a reference policy model receives an input instruction, generates a response, and subsequently obtains verbal feedback from the environment. This enables the definition of a feedback-conditional posterior distribution, the posterior distribution of the outputs given the input and the feedback. Since this distribution is generally intractable, the authors propose to approximate it by learning a feedback-conditional policy via minimization of the forward Kullback–Leibler (KL) divergence. The learned model is thereby capable of generating outputs conditioned on arbitrary user-specified feedback c.

The authors also introduce an online bootstrap procedure that conditions learning on positive feedback. The model parameters are updated through a two-step process: first, sampling an output given positive feedback, and then sampling new feedback based on that output. 

Experimental evaluations are conducted on  mathematical and general reasoning tasks, while feedback signals are produced by GPT-5-nano.  Two baselines are considered: Rejection Sampling Finetuning (RFT) and GRPO.

### Strengths
- Training LLM with non-scalar reward is a very challenging task and of particular interest to avoid losing information and correct ambiguity of LLMs outputs in human-machine interactions.

- The proposed method is well motivated and the online procedure allows to consistently improve the offline trained policy.

- The experiments show that the proposed method offer performance that align with GRPO (and may slightly outperform GRPO in some cases).

### Weaknesses
- The experiments display mean accuracy for the various datasets making it hard to asses the statistical significance of the proposed method. The variability of the results over independent runs should be given.

-  The paper sets the focus on the fact that the core of RL is the online interaction with the environment and provides an algorithm which may reach slightly better performance than scalar-based RL approaches. However, it is not clear if the proposed feedback-based online method can provide consistent performance (if I understand well Table 1 & 2 for instance). When the proposed approach provides a lower accuracy, can it be improved by processing the feedback differently, optimizing the FCP differently etc. 

I do not think the objective of such a paper is to  achieve performance that surpasses necessarily the current state of the art, but rather to gain a deeper understanding of the benefits of the proposed methodolgy of the policy optimization and to disentangle its genuine methodological contributions from implementation effects.  

The method is interesting but I am not sure what is the main message except that the online boostraping seems to consistently improve the offline method.

### Questions
- The choice of the reverse KL is a natural choice and allows a straightforward optimization. However, other choices could be used for instance to balance the dissymetry of the KL divergence. Did you investigate optimizing other divergences ? Do you have empirical insight on the impact the way the FCP is trained has on the overall performance ?

- Tables provide accuracy to compare the different methods. I believe it would be important to add std over independent runs of the experiments to assess the statisstical significance of the proposed performance between methods.

- The experiment  using real-world user and professional reviewer style shows that the approach with user-style feedback offer an interesting gain with the base model but the professional style improves only slightly the performance. Does this suggest that there is room for improvement in the way feedbacks are preprocessed to improve the performance of the model ?

- In classical LLMs usage, human interaction is iterative to improve the performance of the output. Is it possible to extend the proposed approach by incorportating several iterative rewards in the online training to mimic the classical behavior of human-LLMs interaction ?

### Soundness
3

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
This paper proposes a new feedback-based training strategy for Large Language Models (LLMs), based on using verbal feedback rather than the conventional real value, reward based, feedback. 

Their theoretical justification follows previous approaches using optimal policy behavior cloning. These approaches use following facts

- If we consider the objective of minimizing the expected reward (in this case log probability of feedback c) constrained to be close to some reference policy via KL divergence, it has a closed form solution in the form of an energy model (optimal policy)

- that optimal policy has an intractable partition function, since it requires summing over the countably infinite space of sentences. Sampling from that policy via rejection sampling or importance sampling is however simple

- rather than directly solving that objective with an amortized policy and gradient descent as PPO/GRPO do, or deriving a related optimization problem as DPO does, some other approaches (such as DPG, GDC, GDC++ and BRAIn) just sample from that optimal policy and do behavior cloning

What aspects are different in this new work

- use of verbal feedback, previous approaches use conventional rewards, which blends well with optimal policy behavior cloning
- the do not amortize, previous approaches just train a model to predict the optimal policy output given the input, but do not condition on the reward/feedback
- this brings a fundamental complication, now we need feedback at test time, which is defined as conditioned on an input and an output. There are additional changes to account for this including using a model that provides feedback and a second tuning stage

### Strengths
1. The idea of using verbal feedback is not exactly new e.g. [1] but feels like a very natural evolution of current trends. The authors also make a compelling case for it

2. The authors also manage to connect it with some well established optimization principles related to optimal policy.

3. Results seem strong against well know baselines like GRPO, although I have my concerns about the setup

[1] RLVF: Learning from Verbal Feedback without Overgeneralization, 2024

### Weaknesses
1. Unfortunately presentation is not at the same level of the rest of the paper. The authors dedicate a lot of space to make a case for the use of non-verbalized rewards, including quoting Sutton in a separate paragraph. Then declare "Due to space constraints, related work is deferred to Appendix B.". This is beyond some formatting issue. Related work is central to a paper and can not moved to an optional section such as appendix. It is great if the authors want to include an extended related work section in the appendix, but the core relevant works should be in the body.

2. The paper could also be more clear. Maybe is the way I understand the method, but the question of how do you operate at test time (since you will need feedback) is central and is only addressed implicitly and later in the paper. On has to read the algo descriptions to understand how this problem is solved. I could not follow as well the justification of why the negative feedback can be used for learning. I think the analogy to composition of concepts in image to text may make the idea harder to follow. 

3. The paper also misses a line of work that directly overlaps with this work. The claim "we instead propose to minimize the forward KL divergence [...]" is not correct. This has already been proposed in other works solving the KL-constrained reward maximization problem described in the summary (which is reverse KL) with a forward KL approximation (see my summary for the differences). I would say at least [1] who aplies DPG to LLMs for style control, [2] who brings the same point about the optimal policy made here and brings it closer to RL of LLMs and [3] which directly formulates it as Bayesian optimization, are relevant here

[1] A Distributional Approach to Controlled Text Generation, 2021
[2] On reinforcement learning and distribution match- ing for fine-tuning language models with no catastrophic forgetting, 2022
[3] BRAIN: Bayesian Reward-conditioned Amortized INference for natural language generation from feedback, 2024

I do think the authors can make a convincing case for fixing all of the above (see questions).

4. however, there are also concerns with the experimental setup. If I understood this correctly, human feedback is simulated using gpt-5-nano, which is particularly strong in math and instruction following (used in the setup). While all methods have access to this judge as a provider of either scalar reward or verbal feedback, only the proposed method here can use this feedback as input to the prompt.

That is you are using gpt-5-nano to provide some hint in the input to your model, which is trained to utilize this hint from this specific model. This is a big advantage and would require the use of an additional, powerful model at test time. If this is the case then I do not think the comparison in fair. 

Further, forward-KL optimal policy behavior cloning approaches are not used nowadays, which implies that they are not very competitive. This is just a hunch but this may further point to the fact that the hint from a stronger model is behind the improvements.

### Questions
How would you address weaknesses 1-3?

Is my understanding of the use of gpt-5 in weakness 4.  correct?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a new framework, Feedback-Conditional Policy (FCP), for training large language models (LLMs) directly from verbal feedback rather than scalar rewards. Instead of reducing nuanced feedback into numerical values, the authors treat feedback as a conditioning signal and learn the posterior π_θ (o∣x,c)

The approach includes an offline training (learning from response, feedback pairs) and online bootstrapping (conditioning on positive feedback to improve iteratively). Experiments on mathematical and general reasoning benchmarks show that FCP achieves competitive or better performance than scalar-based baselines such as RFT and GRPO, without relying on verifiers or reward models.

### Strengths
1. Learning from verbal feedback is an important direction for scaling alignment. The authors clearly articulate why scalarization leads to information loss and imbalance. With LLMs being deployed at such large scale, collecting descriptive feedback can reveal the precise nature of model issues, rather than relying on simplistic scalar rewards (such as thumbs up or thumbs down).


2.  The distinction between real-world users and professional reviewers is particularly valuable, and the corresponding ablation study effectively illustrates the interplay between these two feedback sources.

3.  The experimental setup is well designed and clearly demonstrates the effectiveness of the proposed approach, particularly through the analysis of different feedback conditions.

### Weaknesses
1. Generally, it is costly to collect verbal feedback from humans for training, and hence FCP can be used only with LLM as the environment that is capable of giving both scalar and verbal feedback.

2. In Algorithm 2, $p_{user}(c+)$ is one of the inputs that is used to sample $c+$. It is done on the basis of scalar rewards (scalar scores) obtained in the offline phase (line 300 to 303). That is, the method assumes access to an environment that provides both verbal and scalar feedback (to construct $p_{user}$). However, Table 1 claims that FCP+bootstrap doesn’t require a scalar reward – this seems inaccurate. You use the accompanying scalar reward for sampling {c+} from $p_{user}$. 

3. Typically, models are finetuned by experts and used by others in various scenarios. Often, these users are oblivion to the training details. I wonder how do you except users to provide their inputs along with the “desired feedback”. Further, the user feedback could be from a different distribution than the one used for training, whereas in all experiments, you sample a {c+} from the ones obtained during training. Maybe it's worthwhile to report numbers in a scenario where c+ is sampled from a different distribution.  

4. The assumption that users will provide explicit, well-formed positive feedback (e.g., “accurate and concise reasoning”) during inference may not be realistic. I believe it is crucial to investigate how FCP behaves when presented with no feedback or ambiguous feedback as input. 

5. While it is reasonable to use scalar rewards from GPT-5-nano for experimental comparison, to more convincingly demonstrate the effectiveness of verbal feedback, it would be valuable to include experiments using *verifiable feedback* (e.g., correctness-based signals). The goal need not be to outperform such setups, but rather to quantify how close verbal feedback learning comes to verifiable feedback performance.

### Questions
When testing on the trained task, the input prompt is not just the query x, but also feedback (wrapped in special tokens <EF>). How is the prompt created for OOD tasks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes to learn from the verbal feedback directly to avoid collapsing rich verbal feedback into a single scalar reward. It treats feedback as a conditioning signal and trains a feedback-conditional policy (FCP) that models $\pi_\theta(o \mid x, c) \propto \pi_{\mathrm{ref}}(o \mid x)\, p_{\mathrm{env}}(c \mid x, o)$. Concretely, they (1) collect an offline dataset of (instruction $x$, model response $o$, feedback $c$) triples from a reference policy plus a feedback generator, and maximize log-likelihood of the response given the feedback, and then (2) do an online bootstrapping stage where they condition the current model on positive feedback, roll out, get fresh verbal feedback, and train again on those newly labeled triples. This reframes “learning from feedback” as conditional generation rather than reward optimization. Experiments on math and general reasoning tasks show that offline FCP is  comparable to RFT, and FCP + bootstrapping reaches or slightly surpasses strong scalar baselines like GRPO/RFT+GRPO.

### Strengths
Learning directly from rich, multi-aspect LLM feedback instead of collapsing it into a single scalar reward is an interesting and important problem that is relatively new to the LLM era. This paper offers a conceptually novel solution to this problem, although the empirical gains over scalar-based baselines are not yet significant.

### Weaknesses
1. The empirical performance gains over baselines are not significant. 
2. The paper mostly compares against basic scalar/fine-tuning baselines (e.g., GRPO/CFT), but does not evaluate against recent critique-based methods such as Critique-GRPO: Advancing LLM Reasoning with Natural Language and Numerical Feedback.

### Questions
In Sec. 2.1 you make two approximations: (1) you approximate $p_{\text{off}}(o \mid x, c^+)$, and (2) you replace the reverse KL in the ideal objective with a forward KL. Could you clarify the impact of these approximations?

### Soundness
3

### Presentation
3

### Contribution
3
