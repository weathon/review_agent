# Learning Correlated Reward Models: Statistical Barriers and Opportunities

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Random Utility Models (RUMs) are a classical framework for modeling user preferences and play a key role in reward modeling for Reinforcement Learning from Human Feedback (RLHF). However, a crucial shortcoming of many of these techniques is the Independence of Irrelevant Alternatives (IIA) assumption, which collapses \emph{all} human preferences to a universal underlying utility function, yielding a coarse approximation of the range of human preferences. On the other hand, statistical and computational guarantees for models avoiding this assumption are scarce. In this paper, we investigate the statistical and computational challenges of learning a \emph{correlated} probit model, a fundamental RUM that avoids the IIA assumption. First, we establish that the classical data collection paradigm of pairwise preference data is \emph{fundamentally insufficient} to learn correlational information, explaining the lack of statistical and computational guarantees in this setting. Next, we demonstrate that \emph{best-of-three} preference data provably overcomes these shortcomings, and devise a statistically and computationally efficient estimator with near-optimal performance. These results highlight the benefits of higher-order preference data in learning correlated utilities, allowing for more fine-grained modeling of human preferences. Finally, we validate these theoretical guarantees on several real-world datasets, demonstrating improved personalization of human preferences.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper demonstrates that pairwise preference data is insufficient to learn correlated utility models in Random Utility Models (RUMs), specifically the correlated probit model. It proves that best-of-three comparisons are both necessary and sufficient for identifiability, proposes a near-optimal estimator, and empirically validates its advantages over traditional models in capturing human preference correlations.

### Strengths
1. The paper is clearly written and logically organized, presenting formal identifiability theorems that rigorously establish the necessity and sufficiency of triplet comparisons for learning correlated probit models.

2. The paper delivers a persuasive critique of the Independence of Irrelevant Alternatives (IIA) assumption commonly used in RLHF, and introduces a well-justified alternative that enables more nuanced and personalized preference modeling.

3. The experimental results, though modest in some cases, effectively demonstrate the advantages of triplet-based modeling on real-world datasets (e.g., Netflix, MovieLens, Sushi).

### Weaknesses
While I am not a domain expert, I would like to offer a few comments based on my understanding of the paper. One notable limitation is the absence of experiments conducted in an actual RLHF setting (whether in reinforcement learning tasks or fine-tuning large language models). Although the theoretical and empirical results (i.e., the correlations) on general preference datasets are compelling, they may not fully demonstrate the practical value of the proposed approach. Without experiments in the RLHF context, it remains unclear how well the method performs in the scenarios it is ultimately intended to support.

### Questions
N/A

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
This paper focusses on learning correlated utility models—parameterized as a correlated probit model——to avoid the IIA assumptions required by RUMs. The paper first proves that pairwise comparison data is insufficient to identify the data generating model. The paper then proves that best-of-three observations are both sufficient and necessary to learn the data generating probit model, and provides finite sample guarantees. Finally, the paper presents a series of experiments evaluating the use of best-of-three observations to learn the parameters of a probit model for 3 real-world datasets and 1 synthetic dataset.

### Strengths
This paper explores an interesting question and provides substantive theoretical analysis to support their conclusion: pairwise comparisons are not sufficient to recover the parameters of a correlated choice model, but best-of-three comparisons are. This conclusion is interesting and, to the best of my knowledge, addresses an important gap in existing literature.

### Weaknesses
My main concerns are with the experiments in Section 6. Across all datasets, the proposed best-of-three-probit model matches or underperforms the direct matrix completion method. The authors claim that the direct matrix completion method is unrealistic in some scenarios—particularly when the set of alternatives and users is large—-but do not evaluate their method on those scenarios. Therefore, as far as I can tell, how their method also performs with larger alternatives/user sets remains an open question. Given the experimental evidence the authors do provide, there is no clear empirical benefit to using the best-of-three-probit model. I will raise my score if the authors can provide empirical evidence indicating where their best-of-three-probit model outperforms all other baselines. 

Also, regarding Figure 3: the authors call this a “welfare maximizing experiment” but then note that the quantity they evaluate by “does not directly correlate with welfare as welfare is sensitive to the magnitude of utility change whereas this plot is not”. The authors should therefore change the name and how they discuss this experiment to avoid confusion.

### Questions
When does the best-of-three-probit model outperform the direct matrix completion method?

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
This paper studies RUMs and explores the statistical and computational challenges of learning a correlated probit model that avoids the IIA assumption.
The authors first prove that pairwise preference data is fundamentally insufficient to capture correlation among utilities.
They then show that best-of-three preference data is both identifiable and sufficient, and propose a statistically and computationally efficient estimator that achieves near-optimal performance.

### Strengths
I think that the contribution of this paper is significant, particularly Theorem 3.2, which rigorously proves that the classical pairwise comparison paradigm is fundamentally insufficient for recovering the parameters of a correlated probit model.
This finding challenges long-standing assumptions in choice modeling and clearly explains why existing methods fail to capture correlations in human preferences.
Moreover, the paper provides both theoretical and practical advances by establishing the first identifiability and finite-sample guarantees for correlated Random Utility Models.
Overall, the work offers a novel perspective on preference learning to the community.

### Weaknesses
Despite its strong theoretical contributions, the paper also has a few weaknesses and open questions.

- In Theorem 5.2, the sample complexity depends on $\gamma^{-24}$. Since $\gamma$ can be extremely small in practical settings, this dependence may lead to an unrealistic sample requirement. It would be important to discuss whether this exponent can be tightened or whether a refined analysis could yield a more favorable dependence on $\gamma$.

- While the paper motivates its setting through RLHF, it treats each item as a single (prompt, response) pair. In RLHF, however, the prompt space is effectively infinite and highly structured. It remains unclear how the proposed framework could be extended to handle such a large or continuous prompt and action space. 

- I think this work is also highly related to general preference modeling frameworks, such as [1] and [2].
Could the authors compare their proposed framework with these prior approaches and clarify the key similarities and differences?

---
[1] Ye, Chenlu, et al. "Online iterative reinforcement learning from human feedback with general preference model." NeurIPS, 2024.
[2] Zhang, Yuheng, et al. "Iterative Nash Policy Optimization: Aligning LLMs with General Preferences via No-Regret Learning.", ICLR 2025.

### Questions
- How strong is Assumption 3.1? Could the authors elaborate on its necessity and implications?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper considers a correlated probit model of preferences that avoids the Independence of Irrelevant alternatives (IIA) assumption; for this model, the paper shows that pairwise preference data isn't sufficient for provably learning the correlations, and suggests the use of three way preference data to provably solve for the parameters of this model.

### Strengths
1. A very clearly written paper, a concrete model / setup and clear theoretical results.
2. Interesting observations, and a particularly relevant problem in the current scheme of RLHF based pipelines for training foundation models.

### Weaknesses
1. In principle, this is a fairly stylized model and its applicability to realistic setups, particularly in RLHF is questionable in the sense of how useful it can be compared to optimizing the standard pairwise loss.
2. This specific connection to training improved reward models (in RLHF) is not explored as part of the empirical evaluations which could've helped bolster the results offered by this paper.

### Questions
One question that I am interested in thinking about (and getting the authors to weigh in on) is what does this imply for policy learning in RLHF setups -- in particular, I am thinking about situations involving intransitive preferences.

### Soundness
3

### Presentation
3

### Contribution
3
