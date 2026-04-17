# Accuracy-First Rényi Differential Privacy and Post-Processing Immunity

- Decision: Reject
- Scores: 2, 6, 6, 6

## Abstract
The accuracy-first perspective of differential privacy addresses an important shortcoming by allowing a data analyst to adaptively adjust the quantitative privacy bound instead of sticking to a predetermined bound. Existing works on the accuracy-first perspective have neglected an important property of differential privacy known as post-processing immunity, which ensures that an adversary is not able to weaken the privacy guarantee by post-processing. We address this gap by determining which existing definitions in the accuracy-first perspective have post-processing immunity, and which do not. The only definition with post-processing immunity, pure ex-post privacy, lacks useful tools for practical problems, such as an ex-post analogue of the Gaussian mechanism, and an algorithm to check if accuracy on separate private validation set is high enough. To address this, we propose a new definition based on Rényi differential privacy that has post-processing immunity, and we develop basic theory and tools needed for practical applications. We demonstrate the practicality of our theory with an application to synthetic data generation, where our algorithm successfully adjusts the privacy bound until an accuracy threshold is met on a private validation dataset.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper formally proves the PPI guarantee of some existing accuracy-first RDP/DP notions and introduces a new notion, ex-post RDP, which involves the $\epsilon$ as part of the output. The new privacy notion satisfies PPI.

### Strengths
The paper identifies a concrete invariance gap in ex-post privacy guarantees and proposes a clean, easy notion by treating the realized bound as part of the output and formalize PPI. The ex-post RDP's framework is tidy, creative recombination that satisfies PPI.

They formally proves the PPI property of some important existing accuracy-first privacy notions.

### Weaknesses
# W1:

While the paper is framed as "accuracy-first", the concrete contribution is primarily a definitional/packaging fix (carrying the per-run certificate and proving PPI on (Y,$\epsilon$)). The work does not yet advance accuracy per se (no new calibration/mechanism or demonstrated utility gains at fixed privacy). In addition, the experiments are limited and do not support the "accuracy-first" narrative.


In Related Work, the ordermeter literature is merely conceptually adjacent, but the present paper's technical contributions do not build on odometers. So, the discussion here is mostly contextual but feels disconnected from the paper's position. 



# W2:

My main concern is as follow.


The paper pointing out a limitation of existing ex-post notations: after post-processing, we often cannot construct a non-vacuous $E'(Y')$ (see the example of Brownian mechanism's "$T$" issue in Page 5 of the paper). 


However, it seems that the root cause is the lack of bookkeeping the privacy parameter/certificate, rather than an intrinsic or fundamental limiation of the existing notions (except the $\delta$-approximate ex-post).

The proposed fix is to treat the realized bound as part of the output and requireing it to be copied through post-processing. This is best viewed as an interface/notation repair, not a new privacy notion with new privatization mechanism nor an impossibility result. 


In practice, the post-processing functions/algorithms operates independently of $\epsilon$. The only requirement is to carry the certificate tag $\epsilon$ with the output so that an auditor at the end of the pipeline can attest the same per-run guarantee.


While the theory is clear (though not, in my view, especially significant), the experiments are narrow illustrations (Adult synthetic data) and do not substantiate the paper’s "accuracy-first" framing or demonstrate utility gains at fixed privacy against strong baselines. Thus, the experiments do not support the broader claims.


Overall, the contribution feels illustrative rather than substantive: a neat packaging/notation fix with a small-scale demo, but not yet a result that moves practice or theory.

### Questions
Please see the weakness.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper advocates a practical shift in how differential privacy is used in ML training: instead of fixing a privacy budget and measuring performance afterward, it proposes an accuracy-first paradigm where the training process dynamically adjusts the privacy parameters until a target accuracy is achieved, then computes the final privacy level using Rényi Differential Privacy (RDP).

### Strengths
1. Experimental results on several benchmark datasets show improved accuracy while maintaining reasonable privacy levels.
2. The authors’ accuracy-first framing is intuitive and addresses a real deployment gap between theoretical guarantees and practitioner needs.

### Weaknesses
There seems to be a lack of variance or sensitivity analysis (e.g., different α values in RDP) to confirm the stability of the bounds (robustness).

### Questions
It would be better to clarify the variance or sensitivity analysis (e.g., different α values in RDP) to confirm the stability of the bounds (robustness).

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
4

### Summary
This paper addresses post-processing immunity (PPI) within the ``accuracy-first'' paradigm of DP. It first shows that existing practical ex-post privacy definitions do not satisfy PPI. To resolve this, the authors introduce a new notation that treats the privacy bound $\epsilon$ as part of the algorithm's output. Using this notation, they propose a new definition, ex-post Rényi DP, and show that it satisfies PPI. The paper also provides a new privacy analysis for the existing Brownian mechanism, showing that it satisfies ex-post RD by proving its equivalence to a simpler sequential mechanism.

### Strengths
- The extension to ex-post RDP is natural.
- I believe the results are sound (at least by intuition), though I did not check every detail in the Appendices.
- Both theoretical and experimental results are provided.

### Weaknesses
- The paper does not propose any novel algorithm specially designed for the new ex-post RDP framework, focusing instead on re-analyzing an existing one.
- The empirical evaluation lacks a comparative analysis against other established privacy notions, so it is unclear if the framework offers a superior privacy-utility tradeoff in practice.
- The paper lacks discussion on the limitations of the ex-post RDP notion.

### Questions
- From a theoretical perspective, does the ex-post RDP notion provide a better or worse privacy-utility tradeoff, while comparing with other accuracy-first notions? What is the cost of PPI here?
- The privacy guarantee is now a random variable. How should practitioners interpret, report, or compare such guarantees? For example, if we have two mechanisms, each with a different distribution of $\epsilon$, how do we decide which one is ``more private''?

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
2

### Summary
This paper investigates post-processing immunity (PPI) under accuracy-first ex-post DP.  In contrast to conventional DP, ex-post DP allows different privacy loss on different outputs through a function $\mathcal{E}:Y\to \mathbb{R}$, and post-processing would change the output space and lose meaningful privacy loss, e.g., multiple $y$ collapse to one.  This paper proposes post-processing that outputs not only the function of $Y$ but also the privacy loss $\epsilon$, which addresses the output collapse issue.  Under this new notion, they prove that the pure ex-post DP and Rényi DP satisfy PPI, but the $\delta$- probabilistic ex-post DP does not.  For Renyi DP, they also prove adaptive composition and analyze the Brownian mechanism.

### Strengths
- Making $\epsilon$ part of the output and defining post-processing is a nice way to handle post-processing.  
- Though scattered around the literature, the paper gives a clean survey on PPI on various ex-post DP.

### Weaknesses
- The motivation of outputting $\epsilon$ is limited.  It seems like theory-first and departs from the conventional usage of post-process.  In practice, do people post-process data but also output all privacy loss in the previous post-process and composition $\epsilon_{1:K}$ as Theorem 4.5?  
- The exploration of the composition theorem is partial.  There is no composition theorem for pure ex‑post privacy analogous to Theorems 4.4–4.6.

### Questions
Can you prove adaptive composition for pure and approximate ex‑post DP, or provide counterexamples/conditions?

### Soundness
3

### Presentation
3

### Contribution
3
