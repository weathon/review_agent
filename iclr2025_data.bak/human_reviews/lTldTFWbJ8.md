## Human Reviewer 1

### Summary
This paper introduces some metrics for synthetic data containing treatments (with the motivating downstream use case of causal inference on medical data). The authors then introduce STEAM, a synthetic data generation framework somewhat designed to optimize for these desiderata. They note that this framework admits DP components. Using the metrics they propose, they evaluate STEAM’s performance on some causal medical data scenarios.

### Strengths
S1: The authors have identified an interesting problem, in that it does seem like there could be improvements to how we think about evaluating synthetic data for medical causal inference. The metrics they present are interesting and appear to be novel.

S2: I appreciate the authors attempts to format the paper in a way that is readable and to highlight results requiring attention in different color blocks.

### Weaknesses
W1: This paper only considers DP-GAN as far as I can tell, for evaluations of DP synthetic data generation. This ignores a rich vein of work over the past decade, listing extensively here for the authors convenience as they improve upon their paper: PrivBayes [Zhang et al., SIGMOD 2017], HDMM+PGM, MWEM+PGM [McKenna et al., ICML 2019], PATECTGAN [Rosenblatt et al. 2020], FEM/GEM [Vietri et al., ICML 2020, Liu et al., NIPS 2021], RAP [Aydore et al., PMLR 2021, NIPS 2022], MST [McKenna et al., JPC 2021], AIM [McKenna et al., PVLDB 2022], Private Genetic Synthetic Data [Liu et al., ICML 2023]. Much of this work has been shown to outperform DP-GAN, and likely better encodes causal relationships in the data.

W2: Noting that DP composition is possible in the STEAM framework, and performing a naive budget allocation, does not really require a proof. Overall, the DP treatment of the framework is severely limited, as to not really be a contribution at all beyond observing that one can use existing DP methods blackbox, in conjunction. Formal reasoning about budget allocation, potential privacy gains, etc. is lacking. Under this view, it's hard to see STEAM as a sufficiently interesting or extensive contribution as an algorithmic framework.

W3: There is only an evaluation of $U_{PEHE}$ vs. the oracle metric in Table 4. What about $JSD_{\pi}$? It’s also not clear how this argument is structured - is it that the metrics in Table 3 can’t identify T-Learner as producing the best dataset? It’s true that in Table 3 this holds out (at least to some extent, although I’d say if I had to guess based on the results in Table 3 I’d select T-learner), but it’s not obvious to me why I would assume these results would generalize past this specific DGP. I think there should be some formal distinguishable results to clearly separate the quantity these metrics provide from prior quantities. Just setting up the DGP formerly and reasoning in closed form about what would distinguish things would be a step in the right direction.

### Questions
Q1: The authors use words like “standard” often, but its not clear how they’ve established that these methods are standard, particularly w.r.t. model selection for their empirical results. How have you determined what is “standard” throughout this paper? When there is not a clear standard, you should evaluate multiple methods that have been shown to perform well, which they fail to do (see W1).

Q2: Some of the methods for generating synthetic DP data listed in W1 admit general query workloads. W.r.t. Table 1 in your paper, these queries are often marginal and thus don’t necessarily differentiate between $\mathbf{X}$, $W$ and $Y$ in the way you desire. However, its not clear to me why you couldn’t use the different distributional measurements you frame in Eq. 2 3 and 4 with these methods. Can you explore this? Particularly with methods like GEM and AIM, which are arguably state-of-the-art for DP data generation.

Q3: The DGP given in 5.1 is simple enough as to admit formal, closed-form results distinguishing the different metric quantities that the authors are interested in. Can this be worked through? For example, under this DGP, we can bound the $JSD_{\pi}$ quantity as stated, and make a direct comparison with some of the other metrics in Table 3

### Soundness
1

### Presentation
2

### Contribution
1

### Rating
3

### Confidence
5

---

## Human Reviewer 2

### Summary
The paper presents **STEAM**, a method for generating synthetic medical data to support causal inference while addressing privacy limitations. STEAM is designed to replicate critical aspects of real data, including covariates, treatment assignments, and outcome distributions, to enable accurate analysis of treatment effects.

To evaluate synthetic data quality, the authors introduce new metrics that assess how well the generated data supports causal inference, addressing gaps in traditional evaluation methods. STEAM also incorporates differential privacy to enhance security. Empirical results suggest that STEAM performs well in complex, high-dimensional scenarios, making it suitable for applications in healthcare and other fields requiring secure, synthetic data for causal analysis.

### Strengths
1.**Novelty**
  
The paper presents a novel approach to synthetic data generation that prioritizes causal relationships, often overlooked in traditional methods. **STEAM** addresses this by modeling covariates, treatment assignments, and outcomes, while ensuring usual Differential Privacy concerns are easily transferable.

2.**Quality**
  
The methodology is thorough, with clear desiderata and structured evaluation metrics. Empirical results show STEAM’s effectiveness, especially in complex, high-dimensional scenarios.

3.**Clarity**

The paper is well-organized, clearly guiding the reader through the problem, methodology, and results. Each component and metric is explained with clarity.

4.**Significance**

By enabling privacy-preserving, causally accurate synthetic data, this work has broad applications, particularly in healthcare and other sensitive fields. The proposed metrics and STEAM framework make synthetic data more viable for impactful, real-world research.

### Weaknesses
1. **Limited Accessibility of Metrics**  
   The paper would benefit from including reminders of key equations, particularly for Jensen-Shannon Divergence (JSD), $P_\alpha$, and $R_\beta$. This addition would improve accessibility for non-expert readers, helping them better understand and apply the proposed evaluation metrics.

2. **Lack of Comparison with Relevant Causal Generative Models**  
   The evaluation does not include comparisons with established causal generative models that handle interventional and counterfactual data, such as DCM [1], VACA [2], and DoWhy-GCM [3]. These models account for treatment, outcome, and counterfactuals and are capable of generating data with similar causal structures. Benchmarking STEAM against these methods could provide a more comprehensive view of its relative performance and potential advantages in synthetic data generation for causal inference.

3. **Omission of Closely Related Work**  
   The paper does not sufficiently reference the reasearch area of causal generative models. Inclusing mentions of these works would strengthen the contextual background, positioning STEAM within the landscape of existing work and clarifying its contributions to the field.

References:
1. Blöbaum, P., Götz, P., Budhathoki, K., Mastakouri, A. A., & Janzing, D. (2024). *DoWhy-GCM: An extension of DoWhy for causal inference in graphical causal models*. [arXiv:2206.06821](https://arxiv.org/abs/2206.06821).
2. Sanchez-Martin, P., Rateike, M., & Valera, I. (2021). *VACA: Design of Variational Graph Autoencoders for Interventional and Counterfactual Queries*. [arXiv:2110.14690](https://arxiv.org/abs/2110.14690).
3. Chao, P., Blöbaum, P., Patel, S., & Kasiviswanathan, S. P. (2024). *Modeling Causal Mechanisms with Diffusion Models for Interventional and Counterfactual Queries*. [arXiv:2302.00860](https://arxiv.org/abs/2302.00860).

### Questions
The $U_{PEHE}$ metric is based on evaluating a family of CATE estimators. Could the authors clarify what size this family should ideally be for reliable estimation, and outline the computational cost involved? Additional context on this aspect would be helpful to assess the feasibility of $U_{PEHE}$ across different applications.

How might the STEAM approach be extended or adapted to support arbitrary causal graphs? Any insights on this would provide useful context for understanding the broader applicability of the method to more complex causal structures.

### Soundness
2

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 3

### Summary
The paper considers the problem of generating synthetic data in the medical domain, where an organization may wish to release a privacy-preserving synthetic dataset generated from a model of a "real" dataset. The paper starts by defining the problem, considering what aspects of the causal problem should be preserved and how this should be measured, and then presents STEAM, a method for generating synthetic data.

### Strengths
The authors have identified what appears to be a really interesting question, and have been absolutely clear in communicating the setting of the paper. In many parts, the clarity of what is presented is good, and the authors have made a strong effort to highlight the relevant parts of the paper, and to help the reader navigate. 

Many parts of the paper are very accessible to a broad ML audience: I appreciated that this complex topic was presented without making it unnecessarily "mathy". 

I enjoyed the exposition in Section 7.2 of the paper, where the authors illustrated what I think is the main take-way form the work: if you generate synthetic data based on the joint, without considering causal structure, you may lose the ability to model the causal structure of interest.

The authors have provided code (which I have not checked beyond a cursory glance at the repo).

### Weaknesses
**Related work**
I'm not an expert in synthetic data generation, but I'm really surprised that no one has considered using the causal graph in synthetic data generation. Is this right? If so, why? If not, why not compare your STEAM method with them?

**Section 4** rather lacks a punchline: it would be great to see an example of when these desiderata would be breached, and what the consequences would be. The paper does move on to an illustrated example in section 5, but I felt rather disappointed at the end of section 4 that the authors did not land their argument with a conclusive demonstration of their point. 

Section 5.2.1. I found the discussion of precision and recall as a metric for assessing a data distribution rathe confusing and non-intuitive, and equation (2) was completely unhelpful. I understand what precision and recall are, but I failed to grok from the work how they are used in the "widespread" application in evaluating a data density: I am not familiar with this work. 

5.2.2 Acronym JSD not defined. 

The main weakness of the paper for me was in section 6, where the thrust of the paper is presented:
"Mimicry of the real DGP acts as an inductive bias, pushing Q closer towards the P in structure, and directly targeting each distributions from our desiderata"

**Section 6**
The main thrust of the paper is summarised in section 6:

"Mimicry of the real DGP acts as an inductive bias, pushing Q closer towards the P in structure, and directly targeting each distributions from our desiderata"

I felt that the lede was rather buried here. This sentence would surely have fitted better a the end of section 1? 

I was disappointed that the differential-privacy aspect of the problem was not treated with the same thoroughness as the causal/metric aspect. Yes, we can apply DP to any aspect of our generative model, but I feel the authors have much more they could add here. What is the interplay between DP and the ability of the dataset to represent a realistic problem? What if only the covariates need be DP?   What if there is no need for DP on the treatment assignment (is this realistic?)?

 Right now Section 6 adds very little value to the paper, beyond demonstrating the authors' understanding of causal statistics and DP. 

**Section 7**
> X using T-learner PO estimators.

reference needed please. 

**7.1** This section could do with a more detailed explanation of the setup - it probably seems obvious to the authors, but I'm not able to follow how this experiment was conducted.  I think what's happening here is that there is no DP happening (epsilon=0). Please clarify.

Section 7.3 is where the paper should really pull together to illustrate the utility of what's being proposed, and I'm afraid it falls short. I cannot figure out what dataset is being synthesised in Figure 5. Is it the toy illustration from section 7.2 above? If so, why not run on the real datasets from 7.1?

### Questions
Is it possible to make a realistically differentially private dataset where causal effects are still discoverable?

To answer this, I'd recommend setting up a DP attack where you attempt to de-anonymise a row of the data (i.e. estimate one of the covariates based on some others). Think carefully about the construction: perhaps there's a realistic case in one of the datasets where you can estimate the age and location of a participant, and whether they have a particular disease. Then, can you use STEAM to construct a dataset which would both be useful to scientists _and_ protect he identity of the individual?

### Soundness
2

### Presentation
4

### Contribution
2

### Rating
5

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper addresses the problem of generating synthetic data that includes treatment and outcome variables. The goal is to enable the release of this synthetic data, potentially with differential privacy, to support various downstream tasks. In particular, causal inference tasks such as treatment effect analysis. The paper formulates this problem into three questions, then claims that existing methods based on fidelity and utility of synthetic data are not adequate to answer these questions. Additionally, when the covariate distribution has a relatively high dimensionality, these methods fail to accurately estimate the treatment assignment and outcome generation mechanisms. To support this claim, the paper includes an experiment with a DGP to show existing metrics do not change significantly by changing the outcome generation mechanism. In contrast, this change is clearly detected by a CATE-based metric. If the joint distribution of the covariates, treatments, and outcomes is factorized using the DGP, then the required inductive bias can be established which existing methods do not capitalize on. Answers for the proposed questions are provided, based on the DGP, by first establishing a set of desiderata that the synthetic data should satisfy, and then deriving a set of metrics that relate to the performance of downstream learners. The paper introduces STEAM, a data generation method which mimics the real DGP, then demonstrates its performance by comparing it to the other data generation methods using the proposed metrics.

### Strengths
* The paper addresses an important problem.
* The problem statement and contributions are clear.
* The approach is novel, intuitive, and general.
* The claims are supported by experiments.
* STEAM outperforms existing methods on the proposed metrics in both the non-DP setting and the DP setting when $\epsilon > 1$.

### Weaknesses
* The uniform distribution of the privacy budget for STEAM based on Theorem 1 is not optimal. The paper already acknowledges this in the discussion.
* In Section 7.3, the $\delta$ in the experiments is set to $10^{-3}$ which might not be ideal and DP-GAN performs better than STEAM around $\epsilon = 1$ on the first three metrics. There are related questions below.

### Questions
* In page 5, in the first paragraph at the beginning: "Particularly as X grows in size", should "size" be replaced with "dimensionality"?
* Related to section 7.3, typical values for $\delta$ are $10^{-5}$, $\frac{1}{n}$, and $\frac{1}{n^2}$. Was the specific choice of $\delta$ in the experiments based on $n = 1000$ by setting $\delta = \frac{1}{n}$? It would be safer to choose $10^{-5}$ or $10^{-6}$. 
* Also related to Section 7.3, STEAM performs worse than DP-GAN around $\epsilon = 1$ on the first three metrics. Is that caused by the choice of some hyperparameters (e.g. DP-SGD hyperparameters)? I would also suggest to see whether this holds when $\epsilon < 1$.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4