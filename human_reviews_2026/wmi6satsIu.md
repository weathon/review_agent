# DGPO: Mitigating Likelihood Displacement with Bidirectional KL Divergence Gap

- Decision: Reject
- Scores: 2, 2, 2, 4

## Abstract
The current margin-based model alignment method, represented by Direct Preference Optimization (DPO), aims to expand the margin between chosen and rejected responses. However, some works point out the log-probability of chosen response always decreases, thus affecting the likelihood of its generation. This likelihood displacement caused by gradient entanglement is a failure mode for preference optimization and has not been fully resolved. In this paper, we focus on forward and reverse Kullback-Leibler (KL) divergence on the probability distribution of preference pairs to form Divergence Gap Preference Optimization (DGPO). We prove DGPO can promote the increase of the chosen log-probability. Besides, DGPO also maintains a lightweight and automatic manner in real-world alignment. The downstream experimental results demonstrate that DGPO maintains competitive performance across various mainstream benchmarks without the reference model and additional key hyperparameters.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a loss function named DGPO to mitigate the phenomenon of decreasing likelihood of positive samples observed in many preference alignment methods. The paper presents the empirical results with enough clarity, but the quality of the rest of the paper is poor (detailed in the weakness section). Although I appreciate the effort, I do not think the paper meets the acceptance bar of ICLR at its current state.

### Strengths
1. The paper presents the experimental settings and results clearly.

### Weaknesses
1. Low readability. The paper contains too many typos, grammatical errors, and notation ambiguities. The most serious one is the notation ambiguity in the proposed loss function (Equation 10). It is unclear what this loss function really means. The context seems to suggest that $\pi_w$ and $\pi_l$ represent $\pi(y_w)$ and $\pi(y_l)$. If so, then using $D_{KL}$ is an abuse of notation because $\pi_w$ and $\pi_l$ are not distributions but two likelihoods.
2. Questionable motivation. The paper situates itself as a method to alleviate the "problem" of the decreasing likelihood of positive samples. However, whether it is really a problem is still debatable in the first place. Literature arguing against this (such as [1]) is unfairly ignored by the paper. 
3. Lack of meaningful baselines. Even if we assume that the suggested problem is indeed a problem, there are already a plethora of methods available, e.g., [2] can be a strong and meaningful baseline. However, the paper only compares the proposed method with relatively weak baselines that are not intended as cures to the decreasing likelihood of positive samples.

------
[1] Rafailov et al, 2024, From r to Q∗: Your Language Model is Secretly a Q-Function

[2] Chen et al., 2024, Noise Contrastive Alignment of Language Models with Explicit Rewards

### Questions
I do not have meaningful questions to ask besides the serious problems as I have listed in the weakness section.

### Soundness
2

### Presentation
1

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
This paper proposes a new measure for preference optimization that weights \log\pi_w and \log\pi_l by the sum (\pi_w+\pi_l).

### Strengths
Gradient entanglement in DPO-like measures is an important problem.

### Weaknesses
Innovation:

All of the derivations in Section 2.1 have been presented more clearly in other papers; they do not need to be presented with this much detail in this paper.

Correctness:

Sections 3.1-3.2 use an incorrect definition of KL divergence.  KL divergence is defined to be the average log ratio between two different probability measures, averaged over all possible token sequences using weights given by the numerator measure.  Eq. (11) shows that you have misinterpreted D(pi_w||pi_l) to be the scalar \pi_w\log(\pi_w/\pi_l) --- KL divergence is defined to be the average of that quantity over all possible token sequences, not the scalar quantity computed based on a single scalar \pi_w.

The reasons given for maximizing Eq. (8) but minimizing Eq. (9) are post-hoc and incoherent.  In fact, the KL divergence from \pi_w to \pi_l is undefined, because both \pi_w and \pi_l are scalars, not distributions.

If h_w is a function of two arguments, then its scalar derivative h_w' is no longer well-defined; you must instead explicitly write dh_w/d\log\pi_w and dh_w/d\log\pi_l.  By making that substitution it is possible to recompute Eqs. (2) and (3) as attempted in Eqs. (12) and (13), but this recomputation is irrelevant, because the linearization in Eq. (4) and (5) is no longer true, so the conditions given in Eqs. (6) and (7) are no longer true.

### Questions
Presentation:

The derivations from Eq. (1) to Eq. (7) are interesting and have been covered in other papers, but they do not make obvious that Delta-log-piw < 0.  One way to make that obvious from these equations is to show that \Lambda'<0; there may be other interesting cases.

Minor presentation issues:

p. 2 both log-probability decrease -> the two log-probabilities decrease

high correlation between positive and negative feedback -> high correlation between winning and losing examples

Some works (Yuan et al., 2024a; Razin et al., 2024) have considered to be the reason why -> Some works (Yuan et al., 2024a; Razin et al., 2024) have considered the reason why

Eq. (1) is missing a close-paren.  Eq. (1) will be more readable if the symbol for loss is in calligraphic font, and if the symbol for log is in roman rather than italic font.

The presentation will be easier to follow if Eqs. (2) and (3) are moved after Eqs. (4) and (5) (as explanations of the terms d_w and d_l) rather than before.  Immediately after Eqs. (2) and (3) are presented for the first time, you should specify that prime denotes scalar derivative - this notation is common but not universal.  Immediately after Eqs. (4) and (5) you should specify that eta is the step size.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper identifies a common failure in preference-based alignment called likelihood displacement, where training with DPO-like objectives lowers the likelihood of both the preferred and rejected responses when the two are very similar, and traces it to gradient entanglement in high-similarity preference data. To address this, it proposes DGPO (Divergence Gap Preference Optimization), a reference-free objective that compares the preferred and rejected responses using a bidirectional KL-divergence gap, effectively reweighting the gradients so the chosen response is more likely to increase in probability even when the pair is similar.

### Strengths
* The paper targets a concrete and currently observed problem in preference-based alignment (likelihood displacement) and clearly links it to gradient entanglement in high-similarity preference data.

* The core idea, using a bidirectional KL-divergence gap to reweight the chosen and rejected responses, is a principled and minimally invasive way to make it more likely that the chosen response’s probability actually increases.

### Weaknesses
* Could the authors also evaluate on Arena Hard benchmark, which is an extension of MTBench?

* In Figure2, could the authors also plot the DPO's chosen and reject probability to compare? It would also be nice to see the dynamics of the margin between the chosen and reject across the training. It is hard to see whether the method is effectively mitigating the decrease of chosen probability compared to DPO or other variants like SimPO (although its in Table 3, a visualization would be better).

* As an extension to previous comment, could we Table 3 only shows comparision of chosen. How about the rejected sentence?

* The performance gain seems very minimal compared to DPO and other variants. 

* There are a few papers tackling the decrease of chosen probability during direct alignment. For example [1], [2]. The paper lacks comparison with such methods. 

[1] https://arxiv.org/abs/2506.12725

[2] https://arxiv.org/abs/2405.16436

### Questions
See weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Divergence Gap Preference Optimization (DGPO) to address the well-known issue of probability decrease in DPO, which is also termed "likelihood displacement". 

The authors introduce a new reference-free objective function that effectively weights the standard DPO log-ratio margin by the sum of the chosen and rejected probabilities. This reduces the gradient signal when both probabilities are already low, preventing further displacement. 

Theoretical motivation is provided via a simple "Bidirectional KL Divergence" theory and experiments are conducted on standard benchmarks.

### Strengths
**Timely Research Direction**

The focus on mitigating likelihood displacement via objective function reformulation is a valuable direction. I myself was recently considering a variant of this method and was glad to see the authors' exploration in this area, which produces a nice loss function addressing this.

**Practical Efficiency**

The proposed DGPO method is reference-free, reporting a ~13% GPU memory saving and 25% speedup over DPO. Though it must be said that in practice, the likelihoods can be computed offline (or asynchronously offline) and hence the memory savings of  not storing both \pi_theta and \pi_ref may be moot. Also, aren't such gains common to all reference free methods (like Simpo)? 

**Addressing a Real Problem**

Targeting likelihood displacement is well-motivated. The intuition behind weighting the gradient by the sum of probabilities $(\pi_w + \pi_l)$ is a theoretically simple way to decrease weighting on the RL optimization when the model has already drifted too far from its initial high-likelihood region for a given prompt.

**Multi-turn Robustness**

DGPO appears to maintain performance better than DPO on subsequent turns in multi-turn benchmarks like MT-Bench (Figure 4), particularly in Reasoning and STEM tasks. This suggests the objective may be better for the model's general purpose capabilities than standard contrastive losses.

### Weaknesses
**1. Inconsistent gains against older baselines**
The empirical advantage over existing reference-free baselines (SimPO) is not clear-cut. In Appendix Figure 9 (Llama-3-8B), DGPO slightly beats SimPO on Length-Controlled Win Rate but loses to SimPO on raw Win Rate (43.3% vs 44.4%). Given that SimPO is a baseline from early 2024, a new method for ICLR 2026 should ideally demonstrate decisive improvements over it.

**2.1 Missing Literature on Reference Free DPO baselines**
Likelihood displacement, reference free DPO, and the related issue of model bias post DPO training have been active areas of research recently. The paper omits some recent works that also address these issues, sometimes with superior empirical results. For example, RefA [1] explicitly tackles length bias (a symptom of displacement) via reference-free token-level regularization. Game-theoretic approaches [3] and multi-preference optimized objectives [2] also mitigate these standard DPO failure modes, sometimes achieving higher win-rates.

**2.2 Outdated Performance Ceiling**
While DGPO improves over vanilla DPO, its absolute performance (~43% WR on Llama-3-8B) is significantly below the current state-of-the-art. The aforementioned recent methods [1, 2, 3] have achieved win rates between 50% and 60% on AlpacaEval 2 using similar base models and training data.

**Suggestion:** I recommend placing DGPO in the context of these more recent, and relevant baselines. 

**3. Theoretical Derivation**
The connection between the Bidirectional KL theory and the final loss function needs more fleshing out, perhaps a rewrite to help me see it. Currently, it seems somewhat heuristic and can at best be termed a motivation rather than directly derived. The jump from minimizing forward/reverse KL to specifically weighting the margin by the scalar sum $(\pi_w + \pi_l)$ relies on simplifying assumptions that may not fully hold for complex sequence distributions. Can the authors consider a derivation in the style of the one taken up in the DPO paper. This would greatly strengthen their work.

---

Should these concerns be addressed, I would certainly consider raising my score.


**References**
[1] Gupta, T., et al. (2025). REFA: Reference Free Alignment with Fine-Grained Length Control. COLM 2025.
[2] Gupta, T., et al. (2025). AMPO: Active Multi Preference Optimization for Self-play Preference Selection. ICML 2025.
[3] Tang, X., et al. (2025). Game-Theoretic Regularized Self-Play Alignment of Large Language Models. arXiv preprint arXiv:2503.00030.

### Questions
**SimPO Comparison:** In your Llama-3-8B results (Figure 9), DGPO achieves a lower raw win rate than SimPO (43.3% vs 44.4%). Why does your method underperform the simpler SimPO baseline? Could the issue be related to noise on the benchmark, or lack of hyperparameter tuning?


### Minor Suggestion:

Please consider a language rewrite/polish. The paper could be polished to an even greater degree for readability.

### Soundness
3

### Presentation
3

### Contribution
2
