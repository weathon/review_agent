# Conformal Data Contamination Tests for In-distribution Data Acquisition

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
The amount of quality data in many machine learning tasks is limited to what is available locally to data owners. The set of quality data can be expanded through trading or sharing with external data agents. However, external data may be contaminated or introduce undesirable sample diversity which can degrade performance of personalized machine learning tasks, as in diagnosis of a rare disease or recommendation systems. Therefore, data buyers need quality guarantees prior to data acquisition. Previous works primarily rely on distributional assumptions about data from different agents, relegating quality checks to post-hoc steps involving costly data valuation procedures. We propose a distribution-free, contamination-aware data-sharing framework that, by inspecting only a small volume of data, identifies external data agents whose data is most valuable for model personalization. To achieve this, we introduce novel two-sample testing procedures, preceding full data acquisition, grounded in rigorous theoretical foundations for conformal outlier detection, to determine whether an agent’s data exceeds a contamination threshold. The proposed tests, termed *conformal data contamination tests*, remain valid under arbitrary contamination levels while enabling false discovery rate control via the Benjamini-Hochberg procedure. Empirical evaluations across diverse collaborative learning scenarios demonstrate the robustness and effectiveness of our approach. Overall, the conformal data contamination test distinguishes itself as a generic procedure for aggregating data with statistically rigorous quality guarantees.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper considers the problem that data buyers want to acquire similar personalised data and need quality guarantees prior to data acquisition. The paper proposes a distribution-free solution that selects data only from agents with less contaminated data (data from another distribution). The main contributions include new theoretical two-sample testing procedures, data sharing procedures and experiments on medical image datasets to validate the effectiveness and practicality.

### Strengths
1. The problem is generally well motivated in the introduction. It may be helpful to further explain why the data buyers can only purchase from few buyers and would not know the distribution or value of others’ data beforehand (e.g., by getting them to predict on a validation set).
2. The solution seems theoretically grounded.

### Weaknesses
1. The main paper or the appendix should provide more background for unfamiliar readers, e.g., on the BH procedure and the definition of PDRS.
2. Sec 3 and 4 should describe technical challenges involved and describe the significance/implications of the results.

### Questions
1. Can you provide some intuition on why the proposed testing procedures do not require distributional assumptions?
2. Is the method less efficient and effective on larger datasets, e.g., the full MNIST? Also, the experiments consider mislabeled data. Does it work when the data is correctly labeled but the class distribution differs?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces conformal data contamination tests. These tests are distribution-free, allowing a data buyer to check multiple outside data sources and retain only those whose data are not "too contaminated" relative to the buyer's own distribution.
For each candidate source, it builds conformal p‑values from a small preview batch, combines them into a single p‑value, and then applies Benjamini-Hochberg across sources to control FDR.
Authors demonstrate that in medical dataset/MNIST image classification experiments, the procedure can select better collaborators and improve downstream accuracy.

### Strengths
1. The paper is well written and easy to follow.

2. The idea behind the paper is simple yet effective, and I think it addresses an important problem.

3. The paper provides both rigorous theoretical results and experimental evaluation (although see weaknesses).

### Weaknesses
Overall, I like this paper. I only have a concern about the choice of datasets for experiments and the contamination procedure. The authors considered hand-crafted noise (e.g., label noise), but not a real-world type of noise. One way to address this could be by considering additional datasets that are designed for it. It could be CIFAR-10C or CIFAR-100C, but also, for example, ImageNet (clear) vs. ImageNet-R. 
In my opinion, this would be a more realistic approach.

Additionally, I have a conceptual question about the whole approach (this is not necessarily a weakness). The approach assumes that each external agent can reveal $m$ samples for demonstration in every round, which may be problematic in privacy-sensitive settings. Therefore, each of the agents may utilize watermarks / other curruptions to preserve privacy. And these corruptions may differ from one agent to another. What do you think could be done in this case?

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies data sharing in the context of collaborative learning, where data owners have samples drawn from distributions $P_k$, which are Huber-contaminated versions of a common $P_0$. The authors design a conformal multiple testing procedure for agents to select a subset of collaborators whose contamination coefficient $\pi_k$ stands below a threshold. From a theoretical point of view, the paper proposes four valid p values, and show that the Storey p values are PRDS, hence making them compatible with a BH procedure. From a pratical point of view, the proposed procedure is implemented on MNIST and RetinaMNIST/EyesPACS with 10 agents and 100 data points. The CDF of the different p values are displayed, as well as the accuracy as a funciton of the expected number of collaborators.

### Strengths
1) The idea of using hypothesis testing in collaborative learning to limit the harmful effects of data heterogeneity is interesting.
2) The statistical analysis is well conducted, with four new p values specifically tailor to test whether a distribution is contamined beyond a given threshold.
3) The experiments, even though the number of agents and data points are low, are fairly convincing.

### Weaknesses
1) Important related works have been neglicted. The idea of performing a statistical inference or test prior to collaboration in collaborative learning with agents having data of varying quality is not new. In particular, [1] already studies this problem and propose a similar solution to the one presented in this paper (estimating the discrepancy from $P_0$ and conditioning the collaboration on it). Likewise, [2] studies the selection of clients prior to collaborating as a bilevel problem. A discussion of these papers (among others) is missing. 
2) A discussion about the four p-values introduced by the paper is missing. When is it better to use one rather than the others? Is there one of them that is easier to compute than others? It seems that the statistical analysis is a bit ad-hoc, and does connect well to the rest of the paper (about data sharing). 
3) The other never discuss the complexity of their method, so it is not clear whether it is actually implementable or if its cost is probihitive in real-world setting (high dimension, a lot of data points and agents...)

[1] Capitaine, A., Boursier, E., Scheid, A., Moulines, E., Jordan, M., El-Mhamdi, E. M., & Durmus, A. (2024). Unravelling in collaborative learning. Advances in Neural Information Processing Systems, 37, 97231-97260.

[2] Hashemi, D., He, L., & Jaggi, M. (2024). Cobo: Collaborative learning via bilevel optimization. Advances in Neural Information Processing Systems, 37, 15550-15574.

### Questions
1) Can you compare your problem and method to references [1] and [2]? 
2) Can you further discuss the four p values introduced in theorem 1 and 2? In particular, is there one that should be favored from a practical point of view (I suspect "Storey" given theorem 3). In this case, what is the interest of the three others? 
3) Can you discuss the complexity of your method? Is the computational / time cost of implementation the main reason why your experiments were conducted with few agents and data points?

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
3

### Summary
In the setting where multiple data owners each have their own distinct dataset and where there is risk of data contamination, this paper proposes an approach toward providing some quality assurance on data sharing which builds on prior methods for conformal outlier detection. The proposed “conformal data contamination tests” improve on prior data data contamination tests by providing distribution-free validity (under some standard IID/exchangeability assumptions) rather than requiring parametric assumptions. To control issues with multiple-testing for the multiple data-sharing agents, the authors use the Benjamini-Hochberg procedure. They provide empirical evaluations across different collaborative learning settings to demonstrate robustness and effectiveness.

### Strengths
Overall the paper seems sound and well-motivated for the stated goals. That is, the proposed methods could have practical use in the setting of quality-assurance/contamination testing in collaborative data sharing. Relative to the prior work on conformal outlier detection by Bates et al. (2023)--where that work can be viewed as covering the special case where one wishes to test the contamination level of 0%, among other contributions--the current work provides seemingly valid hypothesis tests for any other contamination level. It is good that the authors address the multiplicity issues inherent to testing over multiple data-sharing agents, and the Benjamini-Hochberg procedure is reasonable for doing so.

### Weaknesses
**Novelty:** Although the proposed methods are well-motivated and could be useful for the setting studied, it currently does not seem to me that there is significant enough technical innovation in this paper to be of particular interest to the ICLR audience. While I appreciate that the prior conformal outlier detection methods in Bates et al (2023) do not cover the case of data contamination (to my understanding), conformal outlier detection under data contamination is studied by the ICML 2025 paper Bashari et al. (2025), “Robust conformal outlier detection under contaminated reference data,” which is not referenced in the current paper. The authors should add some discussion about how their work relates to that of Bashari et al. (2025). Beyond this reference, I’m wondering if this paper would be a better fit for a somewhat more specialized conference than ICLR, such as *Conformal and probabilistic prediction with applications* (COPA), or a journal focused on soundness, such as *TMLR*.

### Questions
Can the authors please clarify how their proposed work relates to that of Bashari et al. (2025)?

Bashari, M., Sesia, M., & Romano, Y. (2025). Robust conformal outlier detection under contaminated reference data. ICML.

### Soundness
3

### Presentation
2

### Contribution
2
