# Exploit Gradient Skew to Circumvent Byzantine Defenses for Federated Learning

- Decision: Reject
- Scores: 3, 6, 5, 10

## Abstract
Federated Learning (FL) is notorious for its vulnerability to Byzantine attacks. Most current Byzantine defenses share a common inductive bias: among all the gradients, the majorities are more likely to be honest. However, such bias is a poison to Byzantine robustness due to a newly discovered phenomenon -- gradient skew. We discover that the majority of honest gradients skew away from the optimal gradient (the average of honest gradients) as a result of heterogeneous data. This gradient skew phenomenon allows Byzantine gradients to hide within the skewed majority of honest gradients and thus be recognized as the majority. As a result, Byzantine defenses are deceived into perceiving Byzantine gradients as honest. Motivated by this observation, we propose a novel skew-aware attack called STRIKE: first, we search for the skewed majority of honest gradients; then, we construct Byzantine gradients within the skewed majority. Experiments on three benchmark datasets validate the effectiveness of our attack.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a new attack called STRIKE against Byzantine-robust algorithms for Federated Learning. STRIKE is designed to leverage the so-called gradient skewness to hurt the accuracy of Byzantine-robust defenses. The authors provide theoretical and empirical analyses of STRIKE. In particular, they show the effectiveness of their attack, compared to existing ones, on non-iid FL benchmarks.

### Strengths
The topic of Byzantine robustness is important and such research on stronger attacks is valuable, especially in the more realisitic non-iid setting. Moreover, the approach taken is somewhat original and the experimental results are encouraging.

### Weaknesses
I have two major concerns. The first is related to the soundness of the paper; it seems that most theoretical results do not hold currently. My second concern is that highly relevant defenses are not featured in the experiments. I also have other concerns regarding clarity. All my concerns are detailed below.

## Major Weaknesses

1. Proposition 1 is not true as stated, and the proof of Lemma 1 is incorrect:

Regarding Proposition 1, the step leading to Equation (51) is only true if $\gamma = \Omega(\lambda^2)$, which has no reason to hold. If not, even if you take the right-hand side to be (50), it is not possible to take squares to obtain (52) (and the statement of Proposition 1 is not proven). I do not see a possible fix for this issue.

Also, the proof of Lemma 1 is incorrect; the last inequality in (26) is not true, e.g., take $X=Y$. Fix: If you additionally assume independence, you can get (26) but with square root on the right-hand side. Furthermore, (28) and (29) are clearly incorrect.

2. Since Proposition 1 is incorrect, so is Proposition 2 which relies upon the former. Even if the latter was correct, the stated result is rather superfluous as the $\rho^2$ (should depend on $t$ by the way) in the lower bound is shown to vanish across iterations by Farhadkhani et al. (2022).

3. In the experiments, the defenses used are not the best when considering data heterogeneity. Only Bucketing (Karimireddy et al., 2021), among all considered defenses, was designed for this purpose. The work of Allouah et al. (2023) propose another defense called NNM and show that it improves upon Bucketing. At least this other defense should be tested to claim having a strong attack in non-iid settings.

## Other Weaknesses

4. The related work section is inaccurate. The second paragraph in Section 2 wrongly mentions that Farhadkhani et al. (2022) studied Byzantine-robustness under data heterogeneity.

5. The term "skew" is repeatedly used without a proper definition, although a formal definition is given in page 4 without intuition on what skewness means. It seems throughout the paper that it can be replaced by gradient heterogeneity/dissimilarity. In that case, the insights given are very similar to those of Karimireddy et al., (2021), Allouah et al. (2023), etc.. In fact, skewness is a confusing term; it is widely used when referring to probability distributions, but that definition is quite different from what paper suggests. But again, a proper definition would have cleared this issue.

6. "Skewed majority" is inaccurate: if $\tfrac{f}{n} \leq 1/3$ then a set of size $n-2f$ does not constitute a majority.

7. The last sentence in Section 4.2 is clearly incorrect: convexity is an additional assumption, which makes the analysis "easier" than that of non-convex functions. 

8. The choice of search direction, fundamental to the attack presented, seems quite arbitrary. A reference to "Karl's Pearson's formula" is quickly given, but without any justification whatsoever of the search direction.

9. Parameter $\nu$ seems quite redundant with $\alpha$, especially since the experiments on the role of $\nu$ suggest that setting $\nu=1$ is good enough, without giving a guidance or whatsoever on how to choose it.

### Questions
Please address the weaknesses above.

### Soundness
1 poor

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper presents a simple yet effective Byzantine attack called STRIKE. The proposed attack is based on an observation called "gradient skew" that the majority of honest gradients are away from the optimal gradient (the average of honest gradient). The authors theoretically show that Byzantine defenses are vulnerable under gradient skew. The proposed attack utilizes the vulnerability by hiding Byzantine gradients in the identified skewed majority of honest gradients. Extensive experiments verify the effectiveness of the proposed attack.

### Strengths
1. The discovered "gradient skew" phenomenon under non-IID setting is interesting and novel, should inspire future works to follow.
2. This paper theoretically analyzes the vulnerability of Byzantine defenses under gradient skew. 
3. The proposed attack outperforms baseline attacks empirically. 
4. The paper is well motivated and written.

### Weaknesses
1.	A threat model is missing, especially on the capability of adversary. Comparisons with baseline attacks are fair under same and similar adversary capability. The proposed attack requires adversary to know gradients of all honest participants. This is not required by all other attacks, and this needs more careful treatment.

2.	Lack of discussion on any potential adaptive defense methods against this attack.

### Questions
1.	Although non-IID is common in FL, the reviewer is curious about what is the performance of the proposed attack when there is no skew?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper, the authors theoretically show that the gradient skew makes distributed learning methods more vulnerable to Byzantine attacks. Furthermore, the authors propose a novel Byzantine attack called STRIKE that exploits gradient skew. Experimental results show that STRIKE outperforms existing attacks on three different datasets.

### Strengths
1. Exploiting gradient skew to conduct Byzantine attacks looks interesting and reasonable.
2. The paper is generally well-written.
3. Experimental results on three different datasets are provided.

### Weaknesses
1. The authors provide a lower bound for $\mathbb{E}||w^t-w^*||^2$ in Proposition 2 for SGD with robust aggregation (without momentum). However, existing works (Karimireddy et al., 2021) have shown that using history information such as momentum can enhance Byzantine robustness. So, what will the lower bound for robust SGD with momentum be? Moreover, although the authors claim that the gradient skew is due to non-IID data, Assumption 2 makes the theoretical analysis restricted to the IID settings, which is confusing. Although I understand that the lower bound for IID settings also holds for more general non-IID settings, the IID assumption (i.e., Assumption 2) prevents obtaining a tighter bound for non-IID settings.

2. Since the adversary clients with STRIKE attacks require more computation cost than benign clients, the authors are suggested to provide theoretical or empirical results of how much extra computation cost will the proposed STRIKE attack take.


Due to the abovementioned concerns, I currently give a rating of 5. However, I am willing to raise my rating if the authors can properly address my concerns.

### Questions
n/a

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
10: strong accept, should be highlighted at the conference

### Rating Number
10

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors reveal an important phenomenon: current Byzantine defenses in federated learning exhibit a common inductive bias that the majority of gradients are honest. They discover a novel phenomenon named "gradient skew", where the majority of honest gradients deviate from the optimal gradient due to non-IID data. The authors then provide a solid theoretical analysis of the vulnerability of Byzantine defenses under gradient skew. Based on the analysis, they propose STRIKE attack that hides Byzantine gradients within the skewed but honest gradients. The effectiveness of the attack is validated through extensive experiments.

### Strengths
1.	The theoretical analysis for the vulnerability of Byzantine defenses is solid and technically sound.
2.	The newly discovered "gradient skew" phenomenon is novel and inspiring for the community. Extensive visualization results justify that gradient skew is common when data is non-IID, which is practical in real world.
3.	The experiments are extensive. The proposed STRIKE is compared with 6 baseline attacks under 7 defenses.

### Weaknesses
1.	While the proposed STRIKE is effective, there is a lack of discussion on the efficiency of the attack. Solving eq. (19) seems to be expensive. Could you please discuss more about the computation cost of the STRIKE attack?
2.	Is there any limitation of the proposed STRIKE? Please elaborate more on it.

### Questions
Please refer to the weaknesses.

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
4 excellent
