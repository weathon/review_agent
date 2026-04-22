# SMARTER SAMPLING FOR LLM JUDGES: RELIABLE EVALUATION ON A BUDGET

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 4, 2

## Abstract
Large language models (LLMs) are increasingly employed as judges for scalable evaluation of AI systems, where an LLM is prompted to assess the outputs of another model. This approach is particularly valuable for tasks with non-verifiable answers, but its reliability ultimately depends on alignment with human judgments. Because human annotations are expensive and time-consuming, especially in domains that demand expert knowledge such as clinical text generation, it is essential to reduce annotation effort while maintaining accurate estimates of judge reliability. In this work, we study the problem of estimating the intraclass correlation coefficient (ICC) between LLM judges and humans under limited annotation budgets. We derive Chernoff bounds on the estimation error, providing theoretical guarantees on sample requirements and reducing sample size requirements by an average of 18\% compared to the baseline. Building on this, we propose and evaluate $6$ sampling strategies designed to identify the most informative examples for annotation. Experiments on $4$ diverse real-world datasets demonstrate that our methods yield narrower confidence intervals and achieve relative improvements of 5.5\%–31\% in ICC precision over random sampling baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies how to evaluate the reliability of LLM judges with minimal human annotation, a task that has broad applications when human labels are costly to obtain. The authors derive a Chernoff-style bound on the estimation error of the intra-class correlation (ICC), which provides practical guidance on the required number of human annotations. The proposed sampling strategies are demonstrated on several real-world datasets.

### Strengths
1. This paper, with a clear motivation, is well written and well organized.
2. The problem studied in this paper is of importance across fields. This paper balance the theoretical guarantee with practical considerations.
3. Extensive empirical results are presented to demonstrate the effectiveness of proposed approach.

### Weaknesses
1. Notations in Section 3.1 are not fully self-contained. For example, the variance components are not explicitly defined in this section. It would be better to add a subsection to introduce necessary notations.

2. To understand the intuition behind the definition of \(S*\), I was wondering if $\hat \rho$ satisfies the following property: for any $S\_1 \subseteq S\_2$, it holds that $\hat \rho(H_{S\_1},G_{S\_1}) \leq \hat \rho(H_{S\_2},G_{S\_2})$. If this property is not true, it is unclear why \(S*\) defined in this way would be the correct target. 

3. With sampling strategies in Table 3, the effective samples are actually obtained by searching over the entire sample space. I wonder if there are active leaning based sampling strategies that will not look at the entire sample space. It would be super helpful if authors can elaborate more on this part.

I am happy to increase my score if my questions can be addressed.

### Questions
Please the section of weakness.

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
2

### Summary
This paper focuses on estimating the Intraclass Correlation Coefficient (ICC) between LLM judges and humans, aiming to answer two core questions: how many human annotations are needed for accurate ICC estimation, and how to select informative samples to minimize annotation costs. Centered on the estimation of the Intraclass Correlation Coefficient (ICC): Theoretically, a concentration inequality based on the Chernoff bound is derived, reducing the required sample size by an average of 18% compared to the baseline method（2012）.

### Strengths
* The study brings a fresh angle by treating the selection of annotation samples for LLM judges as a core-set selection task. It also cleverly mixes classic statistical methods (Chernoff bounds) with clustering and active learning concepts to make ICC estimation more efficient with limited resources. This combination hasn't really been tried before for evaluating LLM judge reliability on a tight budget.

* The empirical design is relatively rigorous, with evaluations across 4 diverse real-world datasets (covering 15 assessment axes) and 100 rollouts to reduce random variance.

* Overall Well-Written: The paper is generally well-written.

### Weaknesses
* Unverified bivariate normality assumption:​​ The derivation requires LLM/human scores to follow a bivariate normal distribution, but this assumption remains untested across the four datasets. The study fails to address how skewed real-world ratings or alternative distributions (e.g., Poisson/uniform) affect the Chernoff-bound error probability, or whether more robust assumptions (sub-Gaussian) are needed.


* Unclear large-sample threshold：The derivation relies on "sufficiently large n for CLT" but only vaguely mentions n<30 violating assumptions, whether datasets with different ICC values (MSLR's 0.346 vs. MedVAL's 0.716) require varying sample sizes.


* Unresolved prior ICC (ρ) requirement:​​ The sample size formula requires knowing the population ICC (ρ), but ρ is the unknown target to be estimated via human annotation. Although the paper later uses LLM scores as a proxy to estimate ρ, it fails to address how the estimation error in this proxy ρ affects the sample size calculation.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
3

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
This work focuses on how to estimate the alignment of the LLM judgement with human annotations under limited budgets. To archive this goal, the authors first develop a theoretical framework based on Chernoff bound to derive a lower bound on the human annotation sample size which can be used to estimate the ICC (which measures the human-LLM judges alignment) reliably. Then the derived bound is used to empirically select most effective sampling method for ICC estimation. The experimental results shows that the cluster-based sampling consistently yields lowest estimation error.

### Strengths
1 A theoretical framework with thorough analysis for the human-LLM judges alignment problem is provided in the paper, and serves as a theoretical foundation for the proposed core-set sampling problem in the experiment.
2 The formulations and derivations are logically sound and clearly expressed.
3 The paper is well-organized and the texts are generally easy to understand.

### Weaknesses
1 The title of the paper is a bit misleading. "Smarter Sampling" seems like a novel sampling method. Instead, what the paper actually proposed is a way to select one candidate method among **existing** sampling methods.
2 The aim of this paper is to improve the reliability of LLM evaluation systems. But little detail is given on which specific LLM is used as judge model in the main paper. The appendix A.5 mentions GPT-4o-mini and Claude-3.5-Sonnet, but the judgement capabilities of different models is also an important factor in LLM evaluation systems. Maybe the paper should estimate the ICCs of more judge models with varying sizes and capabilities.
3 I notice a gap between the theoretical sections and the experiment sections. The lower bound can be estimated and approximately calculated from the inequality at line 248 and the results are given in Table 1. But in the experiment sections, the actual sampling sizes are still empirically selected from a range of candidates.  
4 Minor: The paper lacks equations reference number, making it difficult to navigate.

### Questions
please refer to weakness section.

### Soundness
3

### Presentation
2

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
This paper tackles the problem of estimating how many human annotations are needed to estimate agreement between human raters and an LLM “judge”, and which examples should be selected for annotation under a limited budget. The authors focus on the intra-class correlation coefficient (ICC) as the reliability metric and study both (i) sample size calculation and (ii) sampling strategies. The overall idea of automating annotation planning, both estimating the required sample size and selecting which items to annotate, is well-motivated and practically useful, particularly given the growing reliance on LLM judges and the high cost of expert labels.

### Strengths
- The study drives a Chernoff bound for ICC based on Fisher’s asymptotic normal approximation, which yields a lower bound on the number of human annotations required to guarantee that the empirical ICC is within ε of the population ICC with high probability.

- They compare six sampling strategies for selecting instances to annotate (including clustering-based, stratified, and random) across four datasets and multiple evaluation axes, and report that clustering-based selection can improve ICC estimation precision.

### Weaknesses
- My main concern is that the reliability and robustness of these sample-size estimates and sampling strategies are likely to be highly task- and model-dependent. The bound is based on assumptions such as approximate normality, sufficiently large n, and ICC values not too close to the boundaries, and it is not entirely clear how sensitive the resulting recommendations are when these assumptions are violated in practice (e.g., skewed rating distributions, heavy tails, or low human–LLM alignment).

- The empirical study is conducted on a limited set of datasets and judge models, which all have reasonably high ICC in several cases;

### Questions
Can you provide results in settings where the LLM judge is poorly aligned with humans? The cheap LLM scores may be a weak proxy for the human variance structure, and the proposed clustering-based selection may behave much closer to random.

### Soundness
3

### Presentation
2

### Contribution
2
