# All Roads Lead to Likelihood: The Value of Reinforcement Learning in Fine-Tuning

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8

## Abstract
From a first-principles perspective, it may seem odd that the strongest results in foundation model fine-tuning (FT) are achieved via a relatively complex, two-stage training procedure. Specifically, one first trains a reward model (RM) on some dataset (e.g., human preferences) before using it to provide *online* feedback as part of a downstream reinforcement learning (RL) procedure, rather than directly optimizing the policy parameters on said dataset via *offline* maximum likelihood estimation. In fact, from an information-theoretic perspective, we can only *lose* information via passing through a reward model and cannot create any new information via on-policy sampling. To explain this discrepancy, we scrutinize several hypotheses on the value of RL in FT through both theoretical and empirical lenses. Of the hypotheses considered, we find the most support for the explanation that on problems with a *generation-verification gap*, *(1)* it is relatively easy to learn the relatively simple RM (*verifier*) from the preference data. Then, *(2)* the downstream RL procedure only returns policies (*generators*) that are optimal for such relatively simple verifiers. Thus, end-to-end, two-stage online FT only has to search over a reduced subset of the full space of policies, requiring less data than offline FT.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper discusses several hypotheses for the benefit of online fine-turning in preference-based RL, focussing on large language models.   Starting from a unified objectives for online and offline fine-tuning, it shows that both online and offline fine-tuning can be seen as maximizing data likelihood, although for the online setting the learned policies is within a restricted policy class induced by learned reward models. Then, this paper raises a new assumption for the benefit of online fine-tuning: simpler reward models induce smaller policy classes, which turns out to have better performance. Finally, the paper offers evidence for the benefit of using simpler reward models.

### Strengths
1. Overall, the presentation of this paper is easy to follow.

2. The discussion through the lens of maximum likelihood estimation is suitable for the LLM setting and offers new insight for understanding the benefit of online fine-tuning.

3. In particular, the assumption of the connection between simper reward models and better performance is interesting and worth investigation. For example, verifiable rewards used for training reasoning models are also simple ones.

### Weaknesses
1. The presentation of H6, which is the core assumption made in this paper, is hard to understand in the first glimpse. I would say that $\Pi(R_\text{sim})\subset \Pi$ is a general statement, since $R_\text{sim}$ is unlikely to cover **all possible reward functions**. A better way for stating the benefit of using small reward models would strengthen this paper. Alternatively, you can define $R$ to be all the reward function consisting with the preferences, so that $\Pi$ will be all the policies that can generate the preferences.

2. The empirical justification is very limited. Since the theoretical justifications in this paper are straight-forward and rely on idea conditions, it is better to include more empirical results.

### Questions
1.  beside the summary task being considered, is there any other task that also has generation-verification gap? 

2. Is the assumption H6 valid for tasks where the generation content is longer than the prompt?

3. Why do you use BoN in Fig.4?

### Soundness
2

### Presentation
3

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
This paper investigates why two-stage fine-tuning (FT) procedures—training a reward model (RM) followed by reinforcement learning (RL)—often outperform direct offline optimization, despite appearing information-theoretically inefficient.  They propose the hypothesis that such a phenomenon is due to the generation-verification gap.

### Strengths
This paper appears to be the first to investigate the underlying reason why the two-stage fine-tuning procedure outperforms purely offline approaches. The proposed hypothesis is well-motivated, and the authors provide a theoretical analysis to support it. Furthermore, they conduct extensive numerical experiments that not only reinforce their hypothesis but also help rule out alternative explanations.

### Weaknesses
Overall, I do not have major concerns about this article. The core idea is clearly articulated, and the presentation is well-structured and easy to follow. My only reservation is whether the contribution is substantial enough for publication at ICLR. Although the paper spans nine pages, a significant portion is devoted to setup and related work, with the main theoretical contribution centered on Theorem 3.1.  I would suggest that some proof sketches and key arguments be included in the main text.

### Questions
See above.

### Soundness
3

### Presentation
2

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
This paper tackles the question of why do complex, two-stage online methods (like RLHF) empirically outperform simpler, direct offline methods (like DPO), even when they both optimize the same likelihood-based objective?

The paper's first key finding is theoretical: when the policy and reward model function classes are isomorphic (i.e., $\mathcal{R} = \mathcal{R}(\Pi)$), the optimal solutions for both online and offline methods are identical (Theorems 2.2 & 2.3). This theoretical equivalence contradicts robust empirical findings.

The paper then systematically conducts controlled experiments to rule out several common hypotheses and propose a more suitable hypothesis: the generation-verification gap.

### Strengths
The generation-verification gap is a nice conceptual contribution. It reframes the online vs. offline debate to a root cause a "statistical efficiency" problem.


The paper is a model of good scientific reasoning. It cleanly formalises the theory-practice gap (Theorems 2.2/2.3) and then treats various explanations (e.g., optimization, regularization, OOD) as falsifiable hypotheses.

The paper's best evidence comes from its two "gap-closing" experiments. Predicting that the online PFT advantage would disappear on a bandit-like task (Fig 5) and a ROUGE-L task (Fig 6) is not obvious. 

The authors online DPO setup -- where an RM is used to re-label on-policy data, which is then fed into the same DPO loss is a clever way to isolate the core variable. It successfully controls for confounders like the optimization algorithm (PPO vs. DPO loss), ensuring the comparison is truly about the two-stage process versus the one-stage process.

### Weaknesses
1) The "Simplicity" of Verification is a Black Box: The entire argument of H-6 hinges on the assertion that a verifier is simpler than a generator. This central concept of simplicity is never formally defined.

2) The paper frames H-6 as the sole surviving explanation. This seems unlikely to be true. It's more plausible that other factors are also true and compound the effect. For example, for H-5  (OOD Generalization), Fig 9, Fig 10 show that global RMs do generalize better (both in-distribution and OOD) than local RMs. The paper claims H-6 causes this. But it's equally plausible that better OOD generalization is a separate benefit of the global RM architecture, which then adds to the statistical efficiency benefit of H-6.

3) How does the paper conceptually differ from [1] which seem to propose a similar argument?

4) The initial theoretical question is built on the isomorphism between policies and reward functions. But in practice offline DPO uses a local RM (architecturally tied to the policy, $r_{\pi} = \sum \log \pi$) and online RLHF uses a global RM (using the final hidden state of the full sequence). Does this create a confounder?


[1] Self-Improvement in Language Models: The Sharpening Mechanism

### Questions
See weaknesses

### Soundness
3

### Presentation
4

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
This paper aims to provide an intuitive explanation for the empirically observed performance gap between two-stage (e.g., RLHF, online DPO) and direct (e.g., DPO, offline DPO) preference fine-tuning methods. Through a combination of theoretical analysis and empirical studies, the authors evaluate multiple hypotheses concerning the value of reinforcement learning in two-stage approaches. Among the tested hypotheses, only the generation–verification gap (it is easier to learn a simple reward model from preference data and optimize policies for simple verifiers) remained supported by the evidence. The paper concludes that two-stage methods effectively operate over a reduced policy space, requiring less data than offline fine-tuning.

### Strengths
The paper is clearly written, and the results are rigorous.

It makes both theoretical and empirical contributions to understanding the origins of the performance gap between online and offline preference fine-tuning.
- shows that when policy and reward classes match, online and offline methods share the same set of optima.
- falsifies several existing hypotheses about the benefits of RL in preference fine-tuning and introduces the generation-verification gap as a plausible alternative explanation.

Provides useful practical insights:
- for problems where verification is simpler than generation, two-stage methods are preferable.
- for tasks requiring long-horizon reasoning or complex planning, the gap between online and offline approaches is likely to widen.

(Note: I have not carefully checked the additional hypotheses in the appendix.)

### Weaknesses
Potential related work: Nika et al. (2024) conducted a comparative theoretical analysis of RLHF and DPO, highlighting the generation-verification gap as a key factor explaining when RLHF statistically outperforms DPO. They show that when the reward class is of lower complexity than the policy class, RLHF tends to perform better, consistent with this paper's findings.

Nika et al., 2024. Reward Model Learning vs. Direct Policy Optimization: A Comparative Analysis of Learning from Human Preferences. ICML, 2024.

### Questions
Clarification question: 

The experiments supporting the generation-verification gap hypothesis show:
- cases where verification is easier than generation, and online methods outperform offline ones;
- cases where complex reward functions diminish the benefit of online techniques.

In LLM preference fine-tuning, even when reward and policy classes have comparable complexity, do we still observe a gap favoring online methods? If that is the case, is the generation-verification gap alone sufficient to explain this phenomenon, or am I missing something?

### Soundness
3

### Presentation
3

### Contribution
3
