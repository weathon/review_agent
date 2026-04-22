# Modeling Interference for Treatment Effect Estimation in Network Dynamic Environment

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 2, 8, 8

## Abstract
In recent years, estimating causal effects of treatment on the outcome variable in network environments has attracted growing interest. The intrinsic interconnectedness of network and the attendant violation of the SUTVA assumption have prompted a wave of treatment effect estimation methods tailored to network settings, yielding considerable progress such as capturing hidden confounders by leveraging auxiliary network structure. Nevertheless, despite these advances, the existing methods: (i) mainly focus on the static network, overlooking the dynamic nature of many real-world networks and confounders that evolve over time; (ii) assume the absence of dynamic network interference where one unit’s treatment can affect its neighbors’ outcomes. To address these two limitations, we first define a new estimand of treatment effects accounting for interference in a dynamic network environment, i.e., CATE-ID, and establish its identifiability under such an environment. Then we accordingly propose DSPNET, a framework tailored specifically for treatment effect estimation in dynamic network environment, that leverages historical information and network structure to capture time-varying confounders and model dynamic interference. Extensive experiments demonstrate the superiority of our proposed method compared to state-of-the-art approaches.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
Overall, this paper offers a **technically sound and empirically thorough** approach to a timely and underexplored problem — causal inference in dynamic network environments with interference.
Key strengths include the introduction of a formally identifiable estimand (Eq. 2; Sec. 3), a well-structured architecture combining temporal and graph components (Fig. 2; Sec. 4), and strong empirical results (Table 1, Fig. 3).
However, the **proof of identifiability** relies on strong and untested assumptions (Assumptions 3.1–3.2), and the **causal interpretation** of learned embeddings (especially the interference term ( e_t^i )) remains opaque.
The **datasets are synthetic**, which limits the generalizability of findings.
Overall, the work is **technically competent and innovative**, but it needs clearer causal justification and stronger empirical grounding.

### Strengths
* **Novel causal estimand and theory** — The paper proposes CATE-ID (Eq. 2) to extend CATE to dynamic settings with interference and proves its identifiability under explicit assumptions (Theorem 3.3). This formalization fills a theoretical gap in existing literature (Sec. 3).
* **Architectural integration of temporal and relational modeling** — DSPNET elegantly combines GCNs (Eq. 4) and GRUs (Eq. 5) to model time-varying confounders, capturing historical and neighborhood dependencies.
* **Explicit interference modeling** — The environment exposure variable ( e_t^i = \sum_{j \in G_i^t} d_t^j r_t^j ) (Eq. 6) effectively parameterizes spillover effects rather than using static aggregation (Sec. 4.2).
* **Adversarial confounder balancing** — The GRL mechanism (Eqs. 9–11) applies domain-adversarial learning to minimize bias across treatment groups, grounded in causal representation learning theory (Shalit et al., 2017).

### Weaknesses
* **Overly strong identifiability assumptions** — The Extended Ignorability (Assumption 3.1) and Consistency (Assumption 3.2) presuppose that all confounders are captured by Φ_z(·) learned via GCNs. No empirical verification or sensitivity analysis is given.
* **Causal interpretation is weak** — The interference representation ( e_t^i ) is learned implicitly but lacks interpretability or causal diagnostics (e.g., no analysis of which neighbor interactions drive outcomes).
* **Limited realism in evaluation** — All experiments are synthetic with simulated treatments, outcomes, and interference (Appendix A). No semi-synthetic or real-world dataset (e.g., citation, epidemiological, or economic network) is used.
* **Scalability untested** — There is no complexity or runtime comparison (Sec. 5.3 omits computational benchmarks), and DSPNET’s feasibility on large-scale graphs is unclear.
* **No uncertainty estimation** — The model reports point estimates without confidence intervals or statistical tests, which is critical in causal inference.
* **Potential redundancy with prior works** — DSPNET’s components (GCN+GRU+GRL) resemble DNDC (Ma et al., 2021) and SPNET (Huang et al., 2023); the novelty mainly lies in combining them rather than introducing fundamentally new mechanisms.

### Questions
1. **Validity of the Identifiability Assumptions**
   The identifiability proof of CATE-ID (Sec. 3.2, Theorem 3.3) hinges on the Extended Ignorability and Consistency assumptions (Assumptions 3.1–3.2).
   Could you clarify **how these assumptions are justified or empirically supported** in your experimental setup?
   For example, are all relevant confounders explicitly simulated and observed in the synthetic data, and how sensitive is DSPNET to violations of these assumptions?

2. **Interpretability of the Interference Representation ( e_t^i )**
   The model learns a dynamic exposure representation ( e_t^i ) through neighborhood aggregation (Eq. 6).
   Could you provide **qualitative or quantitative analyses** showing what this embedding captures?
   For instance, does it correlate with known structural properties (e.g., degree, clustering, influence centrality), or can it be interpreted as a measurable causal quantity such as “average neighbor treatment intensity”?

3. **Comparison with Prior Dynamic Network Models**
   DSPNET combines GCN, GRU, and adversarial balancing layers.
   Could you **clarify the novelty over DNDC (Ma et al., 2021)** and **SPNET (Huang et al., 2023)** beyond architectural integration?
   A detailed ablation or theoretical justification isolating what enables DSPNET to outperform these baselines would help clarify its unique contribution.

4. **Realism and Generalizability of the Evaluation**
   All experiments are conducted on synthetic data (Flickr and BlogCatalog dynamic variants).
   Could you comment on **how realistic these synthetic dynamics are**, and whether DSPNET could be applied to real-world dynamic networks (e.g., temporal citation graphs, contact tracing, or recommendation systems)?
   Are there any known limitations or scalability barriers when moving from synthetic to real data?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose an end-to-end framework, DSPNET, which integrates GCN and RNN to represent dynamic confounders while simultaneously modeling neighborhood interference. The method is evaluated on semi-synthetic Flickr and BlogCatalog datasets.

### Strengths
The paper addresses an important problem of modeling causal effects under network interference in dynamic networks.

### Weaknesses
Weak novelty. The overall framework of this manuscript is highly similar to Ma, Jing et al., WSDM 2021 (DNDC: Deconfounding with Networked Observational Data in a Dynamic Environment), with only an additional “neighborhood interference” modeling component. However, the interference module itself is relatively simple, and the contribution is not substantial enough to meet the innovation standards expected at ICLR.

### Questions
1. The authors claim in the Reproducibility Statement that they have provided the source code to ensure reproducibility. However, the code link on the first page of the paper is invalid.

2. The authors assume the existence of a function Φ𝑧(⋅) that can capture all latent confounders (Assumption 3.1). This assumption may be too strong in realistic scenarios. It is suggested to include robustness or sensitivity analyses in the appendix to evaluate the impact of unobserved confounders that may not be fully captured.

3. The datasets used are static and semi-synthetic. It is unclear whether the equations used to generate treatment and outcome variables, as well as the way dynamics are manually introduced, are theoretically justified or empirically grounded. The authors should clarify how this data generation process may affect the validity of the results.

4. The paper lacks architectural transparency. The number of GCN and MLP layers, hidden dimensions, and epoch are not reported, and there is no analysis of how network depth or model complexity affects performance. Including these details would strengthen both reproducibility and clarity.

5. The scalability of the proposed model to large-scale dynamic networks (e.g., with millions of nodes) is unclear. The authors are encouraged to provide a quantitative time or space complexity analysis.

6. There is a typographical error: “Abalation” should be corrected to “Ablation.”

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper works on the observation study problem where all network interference, network dynamics, and hidden confounders are present in a dynamic network causal model. The work targets the conditional treatment effect CATE-ID and estimates the conditional probability densities of treatments, confounders, and potential outcomes through an RNN coupled with a GNN. This is a methodological paper with performance validated by thorough real data simulation.

### Strengths
- **Originality**: the work is novel in considering (i) dynamic network in causal inference, (ii) set of confounders that depends on the full history. 
- **Quality**: the paper is well-written with clear explanation on the methodology and have solid empirical results. 
- **Clarity**: the writing is very clear and easy to follow. 
- **Significance** : the paper provides a general framework for causal effect estimation on dynamic networks. Although no contribution to the fundamental theory, the work provides a decent insights of deal with such complicated causal inference tasks

### Weaknesses
- The major estimand of the paper is to estimate the CATE-ID, which conditions on the realized treatments and exposures. It appears to be off-topic to most causal inference tasks as the usual goal is the **average** treatment effects under a given policy or a given stationary distribution.

### Questions
1. (related to the weakness). How can the method be generalized to other causal estimands of interest? For example, the policy value (average treatment effects under a certain stationary distribution of networks/treatments..) 
2. Does the method require the network dynamics to be stationary?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper focuses on the treatment effect estimation in dynamic network environments where both time-varying hidden confounders and network interference exist. The authors propose a new treatment effect estimand designed for dynamic settings with interference and introduce a new causal inference framework that explicitly models both time-varying hidden confounders and network interference based on graph nerual networks. To mitigate confounding bias in observational data, the proposed framework uses adversarial learning to enforce balance in the learned confounder representations. The framework is designed to accurately estimate causal effects in settings where the network structure and confounding evolve over time.

### Strengths
1. The paper highlights an important and underexplored perspective, estimating causal effects in dynamic network environments, which has received relatively little attention despite its relevance.
2. The proposed framework, based on graph neural networks, is well defined and effectively demonstrates the advantages of learning representations of hidden confounders over time and estimating treatment effects in dynamic networks.
3. This paper is very well-organized, easy to follow, and clearly presents its contributions.

### Weaknesses
1. The paper lacks clarity on how the proposed method addresses the entanglement between network structures and covariates. A more detailed explanation—whether through theoretical justification or intuitive examples—would help clarify how the model disentangles these dependencies and ensures valid causal estimation.

### Questions
1. Can the proposed approach be extended to settings where the underlying network structure is partially or entirely unobserved? Exploring the applicability of the proposed framework in such scenarios may offer a valuable direction for future research.

### Soundness
3

### Presentation
2

### Contribution
3
