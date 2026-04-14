# Estimating the Probabilities of Rare Outputs in Language Models

- Decision: Accept (Spotlight)
- Scores: 8, 5, 6, 10

## Abstract
We consider the problem of *low probability estimation*: given a machine learning model and a formally-specified input distribution, how can we estimate the probability of a binary property of the model's output, even when that probability is too small to estimate by random sampling? This problem is motivated by the need to improve worst-case performance, which distribution shift can make much more likely. We study low probability estimation in the context of argmax sampling from small transformer language models. We compare two types of methods: importance sampling, which involves searching for inputs giving rise to the rare output, and activation extrapolation, which involves extrapolating a probability distribution fit to the model's logits. We find that importance sampling outperforms activation extrapolation, but both outperform naive sampling. Finally, we explain how minimizing the probability estimate of an undesirable behavior generalizes adversarial training, and argue that new methods for low probability estimation are needed to provide stronger guarantees about worst-case performance.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper studies the problem of estimating the probabilities of rare outputs of transformer models. The authors explore two ways to estimate these probabilities: i) importance sampling and ii) activation extrapolation, which is more novel, but less effective. For each method, they propose two sub-strategies. Results show that importance sampling performs best at estimating the probability of rare outputs and both proposed methods perform better than random sampling.

### Strengths
The paper focuses on an important and, in my opinion, understudied problem of deep learning in general, which is estimating the probability of rare, but potentially catastrophic outputs in deep learning models, and by so doing also capturing the distribution of inputs that would cause them. As stated by the authors correctly, solving this problem is key in advancing adversarial training and even reinforcement learning in many settings.

The approaches tested are principled and thoroughly derived and, while limited in the current applicability (see weaknesses), provide a starting point for further studies and developments of new methods to quantify the probability of rare outputs and all its applications.

### Weaknesses
Two main weaknesses:

1) The methods provided are quite limited in their application to larger models and real-world scenarios, and not just because the authors chose to test with small models, but also because the methods themselves perform computations that restrict their use. Specific points that introduce restrictions to be clarified:
a. The proposed method can only work with target behaviours defined as single token generation, i.e., define specific tokens that are considered rare for the next generation. This is quite limiting, as in language models target/dangerous behaviours are more often associated with semantic meaning of entire sentences and not with a single next token prediction, e.g., alignment for harmfulness.
b. Both methods involve computing the gradient of the rare token generation probability wrt the input tokens/sequence. This reinforces point a. above and further necessitates to have full access to the model, as we need to be able to compute this gradient and propagate it through the model, which is not trivial if the target behaviour is not simply a single specific token or the model is black-box.

2) The technical section describing the two method could be clearer. I suggest converting the point-by-point methods descriptions of supplementary B into pseudo-codes to make it clear what are the exact procedures to implement the different proposed methods and, if possible, put some of them in the main text. In the current form, the methods are given little space and I could not follow them without carefully reading supplementary B.

### Questions
In short, I believe this paper to be a good contribution to the community, providing a starting point to address the problem of rare output probability estimation. However, referencing the weaknesses above, I advise to address the following two points:

1) State more clearly that proposed methods are restricted in their applicability to different models and more complex scenario, in particular wrt the need to compute and propagate the gradient of the output probability.

2) detail more clearly the proposed strategies, ideally with pseudo-codes.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This work focus on probability estimation of low probability outputs in LLMs with argmax decoding. The paper explores the use of importance sampling and activation extrapolation for better estimate of these rare output events. The paper also provides experiments showing the relative performance of the various proposed algorithms.

### Strengths
- The paper is well written.
- The paper provides simple methods for low probability estimation in LLMs.
- Empirical results validate the working of their proposed methods (the importance sampling works better as shown by the authors).

### Weaknesses
- One major drawback I find is the lack of clear goal of the paper. The paper seems to provide some interesting algorithm for low probability estimation and high level discussion of how they can be used in practice. But there is no clear application or experiment that shows the usefulness of the contribution. I would urge the authors to kindly provide some clear applications and experiments to show its usefulness.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces methods for estimating the probability of sampling certain outputs when conditioned on an input distribution in cases where naive sampling is infeasible. Specifically, they introduce four methods (two based on importance sampling and two based on extrapolating model activations) for the special case where the input distribution is independent per token and the output is a single token.

### Strengths
I am not familiar with prior work in this domain, but it appears as if the paper introduces novel methods for trying to estimate the probabilities of certain outputs given an input distribution.

Many intuitions are described in prose and are very clear/helpful:
- Why MHIS may be needed, why this is better for higher capacity models
- Why activation extrapolation methods could be better in certain scenarios.
- Why formally defined distribution can be relevant.

The experiments are sound and the results show that the proposed methods are clearly better than the naive sampling baseline.

Graphics are generally clear!

### Weaknesses
To my understanding, the main motivation described for this problem stems from using these methods to reduce the probability of rare but undesirable outputs. Although a reasonable motivation, there are limited results in this direction because, as the authors point out, doing this for importance sampling reduces to adversarial training, and the activation extrapolation methods are currently worse than the importance sampling methods.
- A more thoroughly description of how you would do this for activation extrapolation methods would be good.

The current problem as formulated is restricted (ie. single token output detection) and not practically useful. However the authors recognize this, and I agree that it is a good first step.

The description of the activation extrapolation methods in the main text could be made more clear, for example the motivation for why the QLD directions could be summarized in the main text (without this, it is hard to understand the method).

Nits:
- Add lines of best fit for figure 1 and 3.

### Questions
Is there any concrete benefit that the proposed methods add to the overall motivation of minimizing adversarial outputs (ex. based on the activation extrapolation)?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
This paper studies low-probability estimation - the problem of estimating the probability that model outputs will satisfy a certain criteria on inputs sampled from some distribution. The authors are primarily concerned with models/behaviors where the probability is hard to estimate empirically using unbiased samples from the input distribution, due to the event being rare.

More specifically, the authors study the task of estimating the percentage of inputs on which a transformer model predicts a certain token $t$ as being the most likely next token. Three methods are tested - two importance sampling methods, which leverage gradients with respect to the target logit, and an activation extrapolation method, which uses an empirical distribution over the final hidden states to estimate the probability that $t$ will have the highest logit.

The authors show that importance sampling methods currently dominate activation extrapolation on all tasks, but provide strong arguments for why they might be relevant in more complex models, where finding any particular input that gives rise to a behavior is computationally intractable.

### Strengths
- The paper has a novel framing of low probability estimation as a concrete approach to understanding and improving worst-case model performance.  It is the first paper to explore extrapolation-based methods that don't require finding explicit examples of rare behaviors.


- The best performing method presented in the paper, MHIS, is motivated using existing work in automated redteaming. By adapting the Greedy Coordinate Gradient technique (previously used for finding jailbreaks) into a proposal distribution for Metropolis-Hastings sampling, the paper shows how adversarial attack methods can be leveraged for low-probability estimation.  This innovative connection between adversarial attacks and probability estimation shows how existing attack methods can be repurposed for studying the probability of rare behaviors on the normal input distribution.

### Weaknesses
- The models tested are small, and it's unclear whether these methods can be scaled up to actual models of interest.  However, the decision to use smaller models is justified, as it would be otherwise impossible to compute the true probability on all 2^32 input samples, which is necessary for measuring the performance of the methods.


- Given that adversarial training is stated as a part of the motivation for developing low probability estimation methods, there is a gap in the discussion around automated red teaming and adversarial training approaches for language models.  The related works section would benefit from an extended discussion on this area.  This context would help readers better understand how this work relates to and advances existing techniques for improving model robustness.

### Questions
- Do you have preliminary thoughts or analysis on how these methods might scale to larger language models?

### Soundness
3

### Presentation
4

### Contribution
4
