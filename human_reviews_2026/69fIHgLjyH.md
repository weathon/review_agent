# Learning to Answer from Correct Demonstrations

- Decision: Accept (Poster)
- Scores: 2, 6, 8, 8

## Abstract
We study the problem of learning to generate an answer (or completion) to a question (or prompt), where there could be multiple correct answers, any one of which is acceptable at test time. Learning is based on demonstrations of some correct answer to each training question, as in Supervised Fine Tuning (SFT).  We formalize the problem as imitation learning (i.e., apprenticeship learning) in contextual bandits, with offline demonstrations from some expert (optimal, or very good) policy, without explicitly observed rewards. In contrast to prior work, which assumes the demonstrator belongs to a bounded-complexity policy class, we propose relying only on the underlying reward model (i.e., specifying which answers are correct) being in a bounded-complexity class, which we argue is a strictly weaker assumption. We show that likelihood-maximization methods can fail in this setting, and instead present an approach that learns to answer nearly as well as the demonstrator, with sample complexity logarithmic in the cardinality of the reward class. Our method is similar to Syed and Schapire 2007, when adapted to a contextual bandit (i.e., single step) setup, but is a simple one-pass online approach that enjoys an ``optimistic rate'' (i.e., $1/\varepsilon$ when the demonstrator is optimal, versus $1/\varepsilon^2$ in Syed and Schapire 2007, and works even with arbitrarily adaptive demonstrations.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses learning to generate answers to questions where multiple correct responses exist. The authors frame this as an offline imitation learning problem using training demonstrations, similar to supervised fine-tuning. Unlike previous approaches that assume demonstrators follow simple policies (justifying maximum likelihood methods), the authors make a weaker assumption: only the reward model (determining answer correctness) needs to be simple. They demonstrate that standard likelihood maximization can fail under this assumption. Their key contribution is a new learning algorithm with sample complexity that scales logarithmically with the number of possible reward functions. The method works robustly even with imperfect demonstrators and in relaxed evaluation settings (pass@k). The work suggests moving beyond likelihood maximization for learning from demonstrations may be beneficial.

### Strengths
1. The fundamental motivation, that a learner needs to produce a single correct answer and does not need to exactly mimic the demonstrator's correct answer, is a good one
2. The mathematical analysis carried out in the paper is extensive and rigorous
3. The paper is well-written and clear in its ideas, methods, and analysis

### Weaknesses
1. The reliance on reward model classes being finite and reward being binary is quite restrictive and limits the applicability of the theory of the method to tasks which are fuzzier and outside of domains such as maths. Even in programming one might want more nuanced reward signals (e.g. % of unit tests passed + some reward on how well formatted / structured the code is).
2. For domains with large state spaces (e.g. long context language modelling, often found in agentic tasks), the methods may fail to be performant due to their dependence on the size of the state space
3. No empirical experiments are carried out, not even for toy tasks.
4. It's not clear algorithm 1 is feasible to implement, since it takes in as inputs distributions over binary reward functions, and involve arg-maxing over outputs with terms inside the arg-max summing over huge spaces of reward functions. Outside of problems with few contexts and responses (in the contextual bandit sense), the algorithm is likely computationally intractable. This does not support the claim that your learner is in any way "efficient".
5. Critiques of the MLE paradigm are not supported by experiments or analysis of experimental data. Given the ability of LLMs to produce reasonable outputs (even if not perfect) after SFT training, this points to the mathematical framing of the paper being flawed, since some of its conclusions directly contradict the current state of affairs.
6. The learning algorithm is claimed to be logarithmic on $|S|$, however, given $S \subseteq (2^{\mathcal{Y}})^{\mathcal{X}}$, this makes the learner's convergence time scale linearly with the product of the context and output spaces in the worst case. Given the domain of interest is LLM finetuning, where these spaces are token sequences, the learner has an exponential dependence on the lengths of the input and output sequences. It is disingenuous to classify this as "efficient".

### Questions
1. Why did you not carry out any empirical experiments?
2. Are you able to actually implement your algorithm in code?
3. How do you square your theoretical work predicting the failure of MLE, when in practice it appears to work reasonably well (if not perfectly)?

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper studies the problem of learning from demonstrations when several correct outputs may exist for the same input. The authors argue that maximum likelihood estimation (MLE) fails in such cases because it imitates demonstrator distributions rather than learning correctness. They formalize this as an imitation learning problem with a low-cardinality reward class rather than a low-capacity policy class. Under this assumption, they show that MLE can overfit and fail to generalize, and propose an alternative “mistake-unaware” weighted-majority learner that provably converges to the correct reward mapping with logarithmic sample complexity. The theoretical results extend to non-optimal demonstrators and pass@k metrics

### Strengths
1. The paper tackles a relevant and well-motivated problem. The gap it identifies in how MLE behaves when multiple valid answers exist is real and interesting.

2. The proposed solution is conceptually neat: shifting from reasoning about small policy classes to small reward classes gives a different and insightful perspective on imitation learning.

3. The theoretical development is rigorous and internally consistent, with clear proofs and sound use of classical online-learning tools such as weighted-majority and halving arguments.

4. The exposition of the core idea, which distinguishes learning correctness from imitating distributions, is well articulated.

5. The paper effectively demonstrates that MLE’s failure is not statistical noise but a structural limitation, and that an alternative learner can overcome it in theory.

### Weaknesses
1. The work is entirely theoretical, with no experiments or simulations to illustrate MLE’s failure or the proposed learner’s behavior. This limits how convincingly one can assess its practical value.

2. The learner assumes a finite discrete hypothesis class and realizability (σ* ∈ S). While the authors acknowledge this, there is little discussion on what these assumptions imply for larger or continuous models, like LLMs.

3. The computational demands of maintaining weights over all hypotheses are briefly noted but not discussed in depth; in realistic settings, this would be infeasible.

4. While the motivation from LLMs is compelling, the theoretical framework (finite S, symbolic correctness sets) cannot directly model text generation. The connection to practical SFT or reward-learning pipelines remains purely conceptual.

5. The introduction could be more focused: it introduces several formal terms early on, which could instead appear in the problem-definition section. Presenting the motivation, gap, and main contributions upfront would make it easier to follow.

6. Although the proposed approach closes a theoretical gap, it remains unclear how it might inspire a usable method in practice. A short discussion of potential approximations or practical directions would help strengthen the paper.

### Questions
1. Could the authors comment on whether variants of their learner, perhaps approximate or sampling-based, might make it computationally feasible in larger hypothesis spaces?

2. Are there simple synthetic or illustrative experiments that could highlight the key behaviors (e.g. when MLE fails versus when the proposed learner succeeds)? Even a toy example might help ground the theory.

3. The paper assumes a finite hypothesis class; how sensitive are the results to mild relaxations of this assumption?

4. How might the “low-cardinality reward class” assumption be instantiated or approximated in realistic domains like LLMs?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies the problem of learning to generate answers (or completions) to questions (or prompts) when there can be multiple correct answers, based on demonstrations of correct answers. Compared to prior work, the main novelty of this paper is that it relies on assumptions that **the reward model lies in a low-cardinality class**, rather than the assumption that **the demonstrator belongs to a low-complexity policy class**.

Specifically, this paper formulates the problem in Section 2 and discusses why MLE can fail under this problem formulation in Section 3. Novel learning algorithms are proposed and analyzed in Section 4. Various possible extensions are discussed in Section 5. Concluding remarks and possible future work are discussed in Section 6.

### Strengths
Overall, I think this is a thought-provoking and solid theoretical paper. Specifically,

- This paper is motivated by very practical problems in LLMs, i.e., many QA problems in math/coding can have many equally valid but differently written solutions. It has also developed a rigorous contextual bandit framework to analyze them. More importantly, it has identified major limitations of MLE under this framework. As a researcher in the field of RL/bandit and LLM, I find this paper very thought-provoking.

- The learning algorithms proposed in this paper are interesting.

- The proposed learning algorithms, and some of their extensions, have been extensively analyzed. To the best of my knowledge, the analyses are rigorous and correct.

Overall, I think this is a strong paper and recommend accepting it.

### Weaknesses
- One obvious limitation of this paper is that it has not provided any experimental results. Given that this paper is motivated by practical problems in LLMs, I recommend that the authors include some experimental results on LLMs in this paper. I think the experimental results will show the following:
  - how well the proposed algorithm will perform in **practical** LLM settings, compared to MLE-based algorithms;
  - the computational tractability/efficiency of the proposed algorithms in **practical** LLM settings.
These are hard to tell from mathematical analyses.

### Questions
- Please add some experimental results if possible.

- [Minor] I think the "learning from arbitrary demonstrator" setting (Section 5.2) is a more general and more interesting setting. It is not clear to me why the authors don't present that setting as the main problem formulation of this paper, and discuss the case when $\pi_*$ is optimal as a special case. I am wondering why.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies the offline imitation learning problem in contextual bandits, under the model that only correct demonstrations are provided. This problem is of practical relevance for the supervised training of LLMs. Departing from prior work, the authors do not assume access to a small policy class (to model the conditional distribution of responses given prompts), but only the weaker assumption of access to a small "reward" class, assigning a binary reward of 1 to responses that are correct and 0 to responses that are incorrect. It is shown that standard algorithms (i.e., MLE) can fail under this weaker assumption, and new algorithms are devised that learn under this relaxed reward class assumption. Several extensions are provided.

### Strengths
The shift from policy classes to reward classes is well-motivated and potentially practically interesting. The presentation is very clean. The technical results are interesting and check out (at a glance). The results establish that MLE fails and introduce new algorithmic ideas for imitation learning from reward classes, and these are tight in terms of their dependence on the log-cardinality class. The authors study several extensions of interest, such as non-optimal demonstrators and pass@k metrics.

### Weaknesses
I don't have many complaints. The main drawbacks include: 1. the analysis assumes a finite reward class, and since the algorithm is Majority/EXP-style this seems non-trivial to extend, 2. the algorithm requires enumeration over S and is thus not computationally implementable. It would be highly desirable to obtain a more computationally tractable algorithm.

### Questions
- Do you envision convenient parameterizations of the reward class S (e.g. linear classifiers) where efficient implementations are possible?

### Soundness
4

### Presentation
4

### Contribution
3
