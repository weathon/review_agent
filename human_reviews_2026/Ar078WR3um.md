# Learning to summarize user information for personalized reinforcement learning from human feedback

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 4, 6

## Abstract
As everyday use cases of large language model (LLM) AI assistants have expanded, it is becoming increasingly important to personalize responses to align to different users' preferences and goals. While reinforcement learning from human feedback (RLHF) is effective at improving LLMs to be generally more helpful and fluent, it does not account for variability across users, as it models the entire user population with a single reward model, meaning it assumes that everyone's preferences are the same.
We present a novel framework, **P**reference **L**earning **U**sing **S**ummarization (**PLUS**), that uses reinforcement learning (RL) to learn to produce text-based summaries of each user's preferences, characteristics, and past conversations. These summaries condition the reward model, enabling it to make personalized predictions about the types of responses valued by each user. Both the user-summarization model and reward model are trained simultaneously, creating an online co-adaptation loop. We show that in contrast to the standard Bradley–Terry model, summaries produced by PLUS capture diverse aspects of user preferences, achieving a 11–77\% improvement in reward model accuracy. Key strengths of PLUS are: (1) robust performance with new users and conversation topics, achieving a 25\% improvement over the best personalized reward model technique used for RLHF; (2) zero-shot personalization with state-of-the-art proprietary models like GPT-4 (e.g., PLUS-summary-conditioned responses achieved a 72\% win rate compared to 28\% for default GPT-4o); (3) learning from flexible user contexts beyond preference labels, and (4) interpretable representation of users, enabling greater transparency and user control in pluralistic LLM alignment.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes an approach for learning a reward model that is conditioned on a summary of a user profile. This means that the reward model can be personaliized. The approach jointly learns a summarization model with the reward model to ensure that the optimal info is included in the summary. The paper reports results showing that by doing this, the reward model accuracy improves and has a higher win rate on the PRISM dataset

### Strengths
This paper provides strong results showing that PLUS can better personalize. The authors show results on the PRISM dataset. this is a diverse, multicultural dataset and reflects what an LLM may encounter in the real world. Furthermore, PLUS shows robustness to new topics and users which is more difficult for approaches that, for example, assign a new user an id rather than a language-based summary.

### Weaknesses
The authors could add some important details related to training such as convergence criteria. Furthermore, PPO in a MARL setup is highly sensitive to learning rate and other parameters. The authors should discuss the possibilty of instability in training in the limitations section.

It would be interesting if the authors provided quantitative results to back up their claim that the personal embeddings dont capture information as well as the language summary - perhaps via mutual information measures between the user info and the summary/embedding

The paper would benefit from providing confidence intervals for results since some of the improvements are small and only tested over 3 seeds

The results in Table 10 in the Appendix are noteworthy and should be discussed in the paper.

A human evaluation of the summaries would be interesting to include to determine if better summaries are actually produced by training in this way. Its possible that the reward model learns to exploit the summarizer in ways that dont actually produce better summaries.

### Questions
Why is the oracle accuracy so low at times?

Can the authors comment on the privacy considerations of this approach compared to others? 

Could the authors add details and discussion on the actual training of the summarizer and reward model?

Could the authors include quantitative results showing how well their language-based summarization captures the user compared to embedding approach?

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
3

### Summary
This paper proposes the PLUS framework for modeling user-specific preferences. A text-generating summarizer produces a user summary $z$ from the user context $c$, and a reward model $r_\phi(· | z)$ is optimized conditioned on that summary to capture personalized preferences. The summarizer and the reward model are trained in an alternating fashion. Experiments on benchmarks such as Pets and UltraFeedback-P-2 demonstrate strong performance compared with baselines.

### Strengths
# strengths
1. Representing user preferences via natural-language summaries makes the user representation easy to understand and inherently interpretable, while avoiding issues with overly long raw contexts.
2. PLUS is evaluated across diverse datasets and settings (including OOD), providing evidence of its effectiveness.
3. PLUS achieves improvements over baselines in most settings and even surpasses strong baselines such as GPT-4o and GPT-4.1.

### Weaknesses
# weakness
1. Data sparsity per user. With only a few dialogs for each user (e.g., Pets uses 3, UltraFeedback uses 2–4, PRISM uses 3), can the reward model learn a reliable representation of that user’s preferences? Is the reward model potentially *under-converged* in such low-data regimes?
2. *Training stability.* Alternating optimization can be hard to converge. It would be important to provide *training curves for the reward model* to assess stability (e.g., accuracy,  variance across steps).
3. Cost and scalability. Building a reward model at the *user granularity* raises the question: when new users arrive, does the system need to update the reward model each time? If so, this could lead to *high computational cost*.
4. *Context-dependent preferences.* A single user’s preferences may shift across contexts, which questions whether modeling a single, static per-user profile is appropriate. For instance, a user may like cats in a general conversation about stray cats, yet under a context emphasizing ecological harm, the same user’s preference toward stray cats could diminish.

If the authors can provide additional results to address these concerns, I would be happy to raise my score.

### Questions
See above weaknesses

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
The submission proposes a method to learn personalized reward models called PLUS.
It consists of a summarization model, the output of which is used to condition a reward model.
The summarization model and reward model are trained in an iterative fashion, alternating between doing PPO updates on the summarization model and supervised crossentropy updates on the reward model conditioned on the generated summaries.
The user summaries are textual.
Experiments show that the proposed method leads to a notably higher reward modeling accuracy, using the resulting reward model for preference-tuning is not evaluated.

### Strengths
* Adapting to a user's individual preference is an important problem
 * The method is well motivated and very reasonable
 * The results show a notable improvement in RM accuracy

### Weaknesses
* The evaluation is limited to reward modeling, it was not tested whether the trained reward model can be used in a subsequent step to train a better policy. Recent research [1,2] has shown that a higher RM accuracy does not necessarily lead to a better performance after RL training. This is my main issue with this paper.

The abstract also claims a "25% improvement over the best personalized RLHF technique". This is incorrect, at most the paper can claim such an improvement over the "best personalized RM technique", not over the best RLHF technique.

* Experiments were only conducted with rather small models (<3B). Particularly for the ICL baseline, ICL is known to perform better for larger models [3]. It is thus unclear the proposed method is advantageous to ICL for larger reward models which would be used in practice.

 * Writing is unclear or incorrect at times, examples below in the minor issues

Minor issues:
 * "BTL reward model [...] is incapable of creating systems that can adapt to users' unique needs and and preference" the paper's own ICL results show the opposite. They are worse than the proposed method, but not fully incapable of adapting to the preferences.
 * GPT4 is not a state-of-the-art model
 * Figure 1: using a religious example seems prone to invite discontent, a non-religious, non-political example may be more inclusive
 * "zero-shot" is used inconsistently and incorrectly throughout the paper. In L295 it refers to generalization to the test-set, in L342 it refers to not training the summary model, which is not zero-shot and in L445 it refers to prompting GPT4 without fine-tuning it. None of these are "zero-shot" in the classical sense of generalizing to entirely unseen classes.
 *  L201: The proposed method is described as "two-stage sequential". To me this sounds like we first train the summarizer, then the reward model, and then we are done. Instead the method alternates between both steps. L210 "Once the summarizer is fixed" also sounds like we fix it only "once", contradicting the alternating training scheme. As it is key to the paper I would recommend revising the presentation.



References:

[1] Razin et al. "What Makes a Reward Model a Good Teacher? An Optimization Perspective", NeurIPS 2025 

[2] Chen et al. "The Accuracy Paradox in RLHF: When Better Reward Models Don't Yield Better Language Models", EMNLP 2024

[3] Brown et al. "Language Models are Few-Shot Learners", NeurIPS 2020

### Questions
* Why were no RL experiments conducted with the trained reward model? It would significantly strengthen the paper.
 *

### Soundness
3

### Presentation
3

### Contribution
3
