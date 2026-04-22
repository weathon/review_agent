# Optimal Aggregation of LLM and PRM Signals for Efficient Test-Time Scaling

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Process reward models (PRMs) are a cornerstone of test-time scaling (TTS), designed to verify and select the best responses from large language models (LLMs). However, this promise is challenged by recent benchmarks where simple majority voting, which ignores PRM signals, occasionally outperforms standard PRM-based selection. This raises a critical question: How can we effectively utilize verification signals from PRMs for TTS? To address this, we start by developing a theoretical framework for optimally combining signals from both the LLM and the PRM. Our framework reveals that the optimal strategy is a weighted aggregation of responses, a strategy whose effectiveness hinges on estimating weights that capture the complex interplay between the models.
Based on our theoretical results, we empirically show that these optimal weighting functions differ significantly across LLM-PRM pairs and, notably, often assign substantial negative weights.
Motivated by these insights, we propose efficient pre-computation methods to calibrate these weighting functions.
Extensive experiments across 5 LLMs and 7 PRMs demonstrate that our calibration method significantly boosts the TTS efficiency, surpassing the performance of vanilla weighted majority voting while using only $\sim 21.3\\%$ of the computation.
Ultimately, our work demonstrates that investing in a more intelligent aggregation strategy can be a more convincing path to performance gains than simply scaling test-time computation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a new test-time scaling algorithm for LLMs when using PRMs (process reward models). The core of the method consists of computing a score for each answer appearing in a pool of generated answers, where the score depends on an *PRM Signal Term* and a *LLM Signal Term*. The authors propose ways of estimating these terms for a given LLM-PRM pair.

In its current form, **I lean towards rejection of the paper.** As outlined below, the theoretical foundation seems imprecise, evaluation is not very extensive (only MATH/MATH500, which means it remains unclear to what degree results such as the calibration weights would transfer across datasets), and accuracy gains are marginal without a clear way of selecting one of the proposed methods in practice for a given LLM-PRM pair. However, the paper certainly proposes some interesting ideas, and if the above points are addressed, it could warrant acceptance.

### Strengths
The results on PRM score distributions across different LLM-PRM pairs (e.g. Figure 1) are quite interesting and could be of independent interest to the research community.

Theorem 3.2, the derivation of picking the best response in a pool of responses by computing a response-level score depending on an LLM and PRM signal, is quite interesting (yet there remain questions about the derivation, see below).

The paper is mostly well-written and clear.

### Weaknesses
### Theoretical Foundations
The theory is imprecise and unclear.
For example, the derivation from Bayes' theorem in Section 3.1 does not make sense to me. First, it is unclear how this probability $P(\alpha_k | \mathcal{G},\mathcal{P},M,V)$ is even defined (is this supposed to be the probability that $\alpha_k$ is the correct answer under the given conditioning? If so, what is the randomness over?), and why maximizing it is the desirable thing to do. Next, the authors say they can decompose the probability $P(\mathcal{G},\mathcal{P}|...)$ into $P(\mathcal{G}|...)\times P(\mathcal{P}|...)$, which would only be possible if the (conditional) probabilities of $\mathcal{G}$ and $\mathcal{P}$ were independent, which clearly does not seem to be the case.

### Experiments
The experiments do not contain any confidence intervals/error bars. This seems to be quite important, as e.g. Figure 2 shows that the entire gain in compute could very well be within the margin of error.

While the authors test out many LLM-PRM pairs, they only seem to test on MATH and MATH500, the latter of which is known to be quite easy. Results on other datasets would greatly improve the quality of the experiments. Furthermore, it is often not even clear what dataset the results are reported on (e.g. Table 1 does not contain any information on this in the text or caption. From the previous paragraph, it is likely either MATH or MATH500, but it is unclear which one of these). In particular, it is completely unclear if any of the calibrated weights would transfer across datasets, which, in the current version of the paper, makes it seem like this is an entirely dataset-dependent method.

An explanation of compute tradeoffs would help evaluate the proposed method better. In particular, in order to compute the calibrated weights, a calibration set (with LLM responses) has to be generated. This seems to be quite expensive in practice.

The results show that accuracy gains can be obtained over baselines like best-of-n (BoN) or majority voting (MV), but these gains are not very consistent, neither across LLM-PRM pairs, nor across the proposed methods (KDE vs Linear vs Logit); it remains unclear how to pick the best method for each LLM-PRM pair in practice. Furthermore, the improvement gains are often not very significant, and sometimes the proposed method is even outperformed by the baselines (cf. Table 1).

To summarize, the paper proposes an interesting approach to test-time scaling that's certainly worth exploring more. However, the main drawbacks I see are a) questionable theoretical derivation (Section 3.1); b) limited experimental validation (just one dataset, no results on compute overhead of the proposed method); c) marginal accuracy gains in practice.

### Questions
- lines 121-124: Are $s_i$ and $\alpha_i$ the same thing? Or are they different? In particular, if they are the same, then assuming a uniform prior over answers doesn't really make sense; if, on the other hand, the $\alpha_i$ are supposed to constitute the set of *all* possible answers, then it doesn't seem to make sense to assume this set is finite.
- what do the red dots in Figure 1 represent? This should be explained in the figure caption.
- is the probability $q_M$ (line 165) assumed to be specific to a prompt (i.e. varies across prompts)?
- Some of the figure fonts are hard to read (e.g. Figure 2, 3).

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates into the utilization of the Process Reward Models (PRMs) in test time scalings (TTS) for LLMs. The authors argue that simple majority voting sometimes outperforms the PRM guided method (such as Best-of-N selection) which indicates the usage of PRM is sub-optimal in TTS. The authors propose a framework with Maximum a Posteriori (MAP) estimation and demonstrate the optimal aggregation method for TTS is via weighted majority vote (with PRM). Besides the theoretical framework, they also propose two practical methods based on KDE and parametric learning to learn these weighting functions from a small labeled dataset. They empirically show the proposed method can achieve similar accuracy as baselines while using only 21.3%-37.1% of the computes, on 5 LLMs and 7 PRMs.

### Strengths
1. In general the paper is well written and the presentation is pretty clear. The proposed theoretical framework (i.e., the MAP based method)  is formulated in a clean way and mathematically makes sense. 

2. The experiments are quite comprehensive. The authors verified their claims across 35 LLM-PRM pairs and demonstrate consistent results on these models and datasets. 

3. The experiments show promising results. The proposed method is able to achieve similar accuracy as other baselines with only 21.3%-37.1% of the compute, which will be somewhat useful in the real world applications.

### Weaknesses
Despite the clear presentation and the clean mathematical formulation, I have the following concerns about the paper:

1. The independence assumption (i.e., assumption 3.1) is quite strong. Assumption 3.1 assumes that PRM scores and the LLM generations are conditionally independent of each other given the true answer, the verifier and the model. I would argue that both LLM generation and the PRM scores are very unlikely independent of each other, instead, the LLM is likely to generate responses that are "correlated" with each other. This assumption is critical, however, the paper doesn't discuss/validate it empirically, neither do it discuss the potential impact on the conclusion. 

2. The ablation study is not sufficient, especially on the calibration dataset. There are a few critical questions that can impact the generalizability/applicability of the method: 1/ How does the distribution shift between calibration and test set affect the performance? How much can the calibration results be transferred to a different domain? 2/ Even within the same domain (i.e., math), how much does the calibration results transfer between difficulty of the problem, or formulation of the problem? So on and so forth. 

3. Missing statistical significance: The authors report absolute results in table 1 and the best numbers are flagged in bold. However, the score gaps with other baseline are usually very small, I would strongly recommend to include standard deviation or confidence intervals to indicate whether these improvements are statistically significant.

### Questions
1. How sensitive is the method to the violation of the Assumption 3.1? I am wondering whether you have some empirical data points to show the correlation between responses or the PRM scores and see how does this correlation impact the final performance?

2. Did you perform ablation study to see how the calibration set domain change can impact the test set accuracy? I would like to understand the domain transferability of the method. It can be truly different domain, different types of math problems or even the difficulty of the same type of math problems. 

3. I think we are missing some baselines, have we compared with other type of methods, such as in [1] and [2]? To be fair, the baselines we compared in the paper are quite simple. [2] is a bit complicated as they require to train a model with RL, however, the method in [1]
 is very straightforward and not necessarily need the calibration dataset.

[1] Confidence Improves Self-Consistency in LLMs, Taubenfeld et al., ACL 2025
[2] The Majority is not always right: RL training for solution aggregation, Zhao et al.

### Soundness
3

### Presentation
4

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
This paper focuses on how to leverage the signals from PRMs (process reward models) in test-time scaling. They show empirical results with an optimal weighting function, incl. practical calibration methods to learn these functions. They show that using these methods improves test-time-scaling efficiency.

### Strengths
The paper presents results across 5 LLMs and 7 PRMs and they show that this helps boost test-time scaling efficiency. They also present different aggregations (see Table 1) which shows better results across multiple model-PRM pairs and the MATH datasets.  They achieve that with way less computation (see line 71 and 306 for details).

They also compare with multiple baselines, such as majority voting, best of N, vanilla weighted vote. Their weighting function calibration significantly improves TTS efficiency.

They also showcase how negative weights are necessary in using PRM signals

### Weaknesses
The paper presents very simple concepts in the abstract but does it in a way by introducing nomenclature that makes it hard to understand. For example, it is not clear why PRMs are important for TTS and their applicability. I would start with that, then present the actual efforts.

### Questions
The MATH dataset is from 2021, do we think that this approach will generalize to novel unseen benchmarks?

### Soundness
3

### Presentation
2

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
In this work, the authors propose a MAP framework for using Process Reward Model signal to guide test time scaling in LLMs on mathematical reasoning tasks. Through theoretical analysis, they propose a weighting over completions based on a PRM signal and LLM signal, which better incorporates PRM signal into test time scaling than baselines. Importantly, they find that allowing for negative weights on completions with low PRM scores and also calibrating to the expected correctness of the LLM help improve the efficiency of test time scaling.

### Strengths
* The authors have an intuitive and clear theoretical framework guiding their proposed method.
* Through experiments, they demonstrate that considering this theoretical framework can help practitioners make test time scaling more efficient.

### Weaknesses
* After introducing the MAP model and proposing a way to estimate it using KDE, the empirical results find that logit weighing with grid search performs as well if not better than KDE. This gap makes me wonder why KDE is necessary in the first place, and if it is less necessary, calls into question the utility of the model. The authors do note that this drop in performance could be attributed to error in estimating $q_M$, but they do not validate this in the paper. I would be interested in seeing an ablation using ground truth $q_M$'s to validate the KDE approach and the theoretical model. On this note, I would also like to see elaboration on the sentence "In conclusion, accurately estimating either the PRM or the LLM part of the weight requires nuanced estimation for individual questions, explaining why the non-parametric KDE estimation underperforms the parametric ones."


Notation:
 * When you say "surpasses the performance of ... with approximately 37.1% and 21.3% of compute of compute", I think this needs to be tweaked gramatically. Also, I would say "Test time compute" instead of "compute" as to not confuse readers about other axes of compute such as FLOPs or training time. My immediate thought was that compute should be the same because you run the same number of forward passes through the PRM, for a fixed N but now I understand that the point is you can get the same accuracy with fewer N.

### Questions
* What is the objective used to calibrate the parametric objectives? Is it correctness on the calibration set?
* In Figure 4, I would like to see the MSE between the dataset estimated PRM score and the per-question-optimal for the whole dataset, this might be more convincing than having 5 picked questions as examples.

### Soundness
3

### Presentation
3

### Contribution
3
