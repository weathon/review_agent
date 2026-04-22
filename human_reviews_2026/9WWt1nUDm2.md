# On the Inherent Vulnerability of Randomized Models to Nagging Attack

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
There has been a long debate on whether inference-time randomness enhances or reduces the safety of deep learning models.
  In this paper, we formalize a realistic threat model, and provide both theoretical and empirical evidence demonstrating that randomness in inference undermines model safety.
  The threat we consider is *nagging attack*, where an adversary repeatedly queries the model with the same adversarial example, exploiting the model’s inherent randomness until a failure occurs.
  Theoretically, we show that nagging attack introduces a fundamental vulnerability for randomized models,
  which cannot be eliminated by refining their design.
  To address this challenge, we propose *seed-rotation*, a deployment strategy that de-randomizes the model within fixed intervals while retaining stochasticity across intervals.
  We prove that seed-rotation makes the randomized model safer.
  Empirically, we compared seed-rotation to the standard randomized deployment across multiple randomized defense methods.
  The results consistently showed that seed-rotation achieved higher adversarial robustness than the randomized deployment.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies how randomness at inference time can actually make models less safe. It introduces the nagging attack, where an adversary keeps sending the same input until a randomized model fails. The authors formalize this with Adversarial Nagging Risk (ANR) and prove that any nontrivial probabilistic model becomes more vulnerable once multiple queries are allowed.
To mitigate this, they propose seed-rotation—fixing a random seed for short intervals (e.g., per day) and refreshing it periodically. Both theory and experiments show that this approach makes randomized models significantly safer without changing the underlying model.

### Strengths
+ Offers a clear theoretical framework showing why randomness increases risk.
+ The seed-rotation idea is simple, practical, and effective.
+ Writing is clear and well-organized, making complex math easy to follow.

### Weaknesses
- The rotation frequency is chosen heuristically, with no clear analysis of how interval length, query rate, or system overhead affect robustness or performance.
- The defense’s resilience to adaptive attackers is unclear—an adversary could potentially infer or exploit the fixed seed during each interval.
- The paper lacks comparisons and ablations: it doesn’t evaluate against other mitigations (e.g., query tracking, adaptive noise scaling) or isolate which randomness source drives vulnerability.
- Theoretical claims are strong, but empirical validation and formal guarantees of ANR reduction remain limited.

### Questions
1.	How was the rotation frequency chosen? Is there a measurable relationship between interval length, attacker query rate, and resulting ANR?
2.	Does frequent seed-rotation introduce computational or caching overheads in real deployments, and how do these costs compare to the achieved robustness gains?
3.	Could an adaptive adversary infer or approximate the current seed or its distribution from model outputs, potentially weakening the defense?
4.	Seed-rotation makes the model deterministic within intervals—how does this trade off between reduced nagging risk and increased vulnerability to query-based black-box attacks?
5.	The theory predicts exponential degradation with query count. Is this trend empirically verified across all defenses, datasets, and perturbation bounds?
6.	Does seed-rotation offer any formal probabilistic bounds on ANR reduction, or are the observed improvements purely empirical?
7.	Why weren’t other mitigations such as query tracking, adaptive noise scaling, or deterministic ensembles included for comparison? Could these achieve similar robustness without rotation?

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
The paper analyzes the adversarial risk induced by nagging attacks against randomized models. Both theoretical and empirical justifications are given to show that adversarial risk increases in proportion with the query budget. A seed-rotation strategy is proposed to address the vulnerability caused by randomness at inference time.

### Strengths
The proposed study answers very important questions about potential vulnerability as a result of randomness at inference time. 
Under specified assumptions, the theoretical justifications are convincing and the empirical results are promising.

### Weaknesses
The definition of adversarial nagging risk is not rigorous and needs further justification. 
The experimental setup needs to be improved.

### Questions
1. The adversarial nagging risk (ANR) is deliberately defined (Equation 6) to amplify risk as the query budget increases, even though the predictions of a classifier remain the same. Consider a randomized classifier that predicts m times for each input query x and then outputs the majority vote with a probability (tampered with a small random noise). Will the query budgets of 10 and 100 make any difference?  Why should the risks be different in these two cases given that the outcome is the same? This definition of ANR needs to be carefully justified since it has cascading impact on the theorems hereafter. 

2. In the randomized deployment, a fixed adversarial sample is tested with n random instances of a classification model, while in the seed-rotation deployment, different adversarial samples crafted in n queries within a single interval are tested with a single model. The latter does not seem to be allowed to explore the vulnerability of randomness in the instances of the classification model. It would be interesting to find out what happens if, in the seed-rotation deployment, after x_0' is crafted for h_0,  the classifiers in later intervals h_1, h_2, ..., h_n also predict for  x_0' and count the number of attack successes.

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors demonstrate that randomized models are susceptible to a “nagging attack,” in which an adversary can make multiple attempts to breach the model. This vulnerability, however, can be reduced through key-protected deployment of randomized models.

### Strengths
1. The paper is clearly written and well organized.
2. The notations and definitions are presented in a straightforward and understandable manner.

### Weaknesses
Essentially, this paper demonstrates that a probabilistic classifier faces a higher nagging risk (as defined in Definition 4.1) when the number of allowed queries approaches infinity. Implementing key-guarded deployment—where the model is fixed to a specific realization from the BHS for a given time period—reduces the effectiveness of queries in identifying the worst-case model realization, thereby improving performance under the nagging attack threat model. The main issues are:

1. This result is somewhat trivial and aligns with expectations.
2. I don’t think the nagging risk is a suitable metric for evaluating model robustness. In this framework, any model with a non-zero probability of failure will exhibit the same nagging risk as a model that fails on every attempt when the query budget approaches infinity. However, it’s clear that a model with a smaller failure probability is still preferable to one that fails consistently.
3. According to the threat model definition, the adversary possesses full white-box knowledge of the model, including the defense mechanism, except for the random seed. In that case, the adversary could construct a substitute model without the key-guarded deployment, effectively eliminating the “less efficient query” effect and arriving at the same adversarial example as if the key-guarded deployment were not used.

### Questions
1. How are seeds rotated in obtaining the experiment results in the paper? How would rotating with different circles affect the performance?

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
The paper seems to suggest a very board investigation of randomness in deep learning models to state that an adversary with access to an identical target model but not the noise distribution (in the context in which it is used in defence). The paper argues that under this threat model, repeated queries can eventually lead to an adversary discovering a sample to break a defence.

### Strengths
- The paper takes a formal analysis to study a problem
- Interesting to consider randomisation methods in deep learning and other than making evaluation of methods harder, to ensure they provide a measurable advantage vs. cost.

### Weaknesses
- the idea or randomisation is not new
- the paper has a very simple idea, but the writing is over complicated 
- The first two paragraphs in the introduction are, to me, unnecessary and confusing, trying to relate the various uses of noise or randomness to a single question is a huge oversimplification, when it is much more context dependent/should be nuanced. I find the use of the phrase randomized models misleading, it seems to refer to generative models– the introduction is overly verbose for something that is simple to describe without going too board.
- The basic question seems to be very board, but study eventually seems to consider Randomised Smoothing and a denoiser for sanitizing adversarial inputs.
- Changing model parameter is expensive – consider the cost or re-training models, storage costs etc. to a model provider
- From what I can understand, the proposed rotation simply violates the threat model assumed--now the surrogate does not match the target, effectively this is the standard setting used in randomisation methods, but to me this is not shifted out of the standard threat model and instead under the proposed strong threat model shown as a weakness.

### Questions
- How many models were trained for each task to select from?
- What is the training cost and storage cost for such a scheme?
- I would be interested in seeing results for more practical task like ImageNet with a state-of-the-art model - for example RS methods scale to ImageNet tasks.
- What is the certification guarantees under random model sampling? (what is an adversarial attack under RS? how is this formalised?)
--The references Wang 2024 and Lucas 2024 do not mention “a nagging attack”, Lines 055 – these prior studies are not based on addressing the so called “nagging attack”, it is unclear how the study can generalise these two studies to prior work on nagging attack defenses. Can this be clarified?
- The literature review to me is somewhat misleading, for example Lucas 2023 does not speak about whether randomisation increase or decrease a defence, it simply compares probabilistic classifiers in comparison to deterministic ones. What am I missing?
- The assumed threat models violate what is typically assumed in practical settings by considering a surrogate that is effectively the target model that is being attacked. So an attack becomes a problem of estimating the noise, or the seed used in a pseudo-random number generator (in some contexts). 
- The proposed seed rotation simply violates the assumption of the threat model, i.e, now the surrogate has a different seed to the target, but again, a nagging attack should be able to show that this setting is also vulnerable, i.e. it depends on how many different seeds and over what time the seeds are rotated. Would like to understand what the views on this.

### Soundness
2

### Presentation
2

### Contribution
2
