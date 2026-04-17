# FedTT: Cross-City Federated Traffic Knowledge Transfer with Privacy Preservation

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Traffic prediction (TP) is a core task in urban computing, aiming to forecast future traffic conditions from historical observations. To overcome the scarcity of traffic data in emerging cities, recent studies have explored Federated Traffic Knowledge Transfer (FTT), which leverages data-rich source cities to assist data-scarce target cities without raw data sharing. However, existing FTT approaches are limited by three unresolved challenges: (i) potential *privacy leakage* since gradients or parameters generated during federated computing can still be inverted, (ii) severe *cross-city distribution discrepancies* that reduce transfer effectiveness, and (iii) \textit{low data quality} caused by missing or unreliable sensor readings. To address these challenges, we propose **FedTT**, a novel federated framework for cross-city traffic knowledge transfer with privacy-preserving. FedTT introduces three innovations: (i) a lightweight **Traffic Secret Aggregation (TSA)** protocol that achieves secure knowledge aggregation without sacrificing efficiency or accuracy; (ii) a **Traffic Domain Adapter (TDA)** that explicitly aligns heterogeneous source–target distributions for more effective transfer, and (iii) a **Traffic View Imputation (TVI)** method that leverages spatio-temporal dependencies to complete missing traffic data robustly. Extensive experiments on four real-world datasets show that FedTT achieves significant improvements over 14 state-of-the-art baselines, consistently reducing prediction error while maintaining strong privacy protection.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes FedTT, a federated framework for cross-city traffic knowledge transfer (FTT) that addresses three key challenges in urban traffic prediction: (1) privacy leakage in federated computation, (2) cross-city data distribution discrepancies, and (3) low-quality or incomplete traffic data. Extensive experiments across four real-world datasets (PeMSD4, PeMSD8, FT-AED, HK-Traffic) show that FedTT outperforms 18 baselines in MAE and RMSE, achieves better privacy resistance under reconstruction attacks, and maintains communication efficiency through parallel federated optimization.

### Strengths
- Comprehensive and well-motivated problem formulation: The paper clearly identifies and formalizes the three major bottlenecks in federated traffic knowledge transfer, grounding its contributions in real-world challenges of privacy, heterogeneity, and data sparsity.

- Technically integrated framework: The design of TSA, TDA, and TVI demonstrates solid modular synergy, i.e., each component addresses a specific limitation while maintaining a coherent overall system.

- Strong empirical results: Experiments on diverse datasets and ablation studies provide convincing evidence that FedTT improves both prediction accuracy and privacy robustness compared to state-of-the-art methods.

### Weaknesses
- Lack of theoretical rigor in privacy analysis: The TSA protocol is described algorithmically but lacks a formal proof of privacy guarantees or quantifiable leakage bounds under strong adversarial models.

- Limited interpretability of domain adaptation: The TDA’s GAN-based alignment is treated as a black box; the paper does not analyze how domain shifts are reduced or whether learned representations preserve semantic traffic structures.

- Insufficient examination of generalization and robustness: The transferability of FedTT to unseen cities or dynamic traffic distributions is not validated beyond the four datasets, leaving open questions about its adaptability and stability.

- The organization of this paper needs further refinement: First, the authors have relegated related work to the appendix, which disrupts the overall flow of the paper. Furthermore, the spacing adjustment strategy used in many places to achieve a more compact presentation of the text appears to violate the ICLR 2026 submission policy.

### Questions
- Absence of formal privacy guarantee: The TSA protocol lacks a mathematical definition of privacy (e.g., ε-differential privacy or semantic security), making it unclear whether it can resist gradient inversion or collusion attacks in practice. Although the paper uses mutual information for privacy analysis in the appendix, it doesn't appear to have the same strict privacy constraints as differential privacy. For example, the authors don't seem to define what level of privacy leakage is considered safe and what level of privacy leakage is unsafe. The main text seems to show that the proposed method is privacy-preserving empirically rather than using a rigorous theoretical analysis.

- Unverified adversarial resistance: Although this paper evaluates the performance against data reconstruction attacks, it does not provide detailed implementation details of the data reconstruction attacks. The authors need to provide this detail. 

- Opaque domain adaptation process: The paper provides no visualization or interpretability analysis of how TDA aligns feature spaces between cities, weakening claims about its efficacy in reducing domain gaps. The authors need to provide more justification.

- Potential instability of GAN-based alignment: The TDA module’s adversarial training may cause instability or mode collapse, especially when source and target cities have large structural differences. The authors need to provide more justification.

- Missing theoretical justification for convergence: The paper does not provide convergence analysis or training stability proof for the combined optimization of TSA, TDA, and TVI modules, leaving open questions about scalability to larger federations.

- Scalability of FedTT needs further consideration: The inability to transfer trained models to unseen cities increases deployment costs, as FedTT must be retrained for each new target city.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes FedTT, a novel federated learning framework for cross-city traffic knowledge transfer that addresses three key challenges: (1) privacy leakage, (2) cross-city distribution discrepancies, and (3) low data quality. To tackle these, FedTT introduces three core components: (1) Traffic Secret Aggregation (TSA), a lightweight protocol for secure aggregation without heavy overhead, (2) Traffic Domain Adapter (TDA), a module that aligns heterogeneous traffic distributions between source and target cities, and (3) Traffic View Imputation (TVI), a method that leverages spatio-temporal dependencies to robustly impute missing traffic data. Experiments on four real-world datasets show that FedTT consistently outperforms 18 state-of-the-art baselines.

### Strengths
S1. The paper has a strong motivation, clearly identifying three crucial challenges in FTT, which justifies the need for the proposed FedTT framework.

S2. The paper introduces a new paradigm for FTT with three novel modules that are technically sound and specifically designed to tackle each challenge.

S3. The paper conducts comprehensive experimental validation to evaluate FedTT on four real-world datasets across diverse city settings, outperforming 18 SOTA baselines in various terms.

S4. The paper provides formal theoretical privacy guarantees to bound the privacy leakage of FedTT, ensuring that

### Weaknesses
W1. Federated Parallel Training (FPT) is a significant contribution to improve efficiency but is only described in the appendix. Given its notable impact on training speed and communication overhead in efficiency experiments, FPT deserves a brief discussion in the main paper.

W2. Since the paper uses a large number of notations and formulas, moving the notation table to the main paper would improve readability.

W3. Some minor issues should be corrected. For example, in line 945, “Eqs. 21 and Eqs. 20” should be revised to “Eqs. 21 and 20”. In line 1121, the expression \mathcal{X}^{(R_i \to S), R_i} should be corrected to \mathcal{X}^{(R_i \to S, R_i)}.

### Questions
Please see the weaknesses.

### Soundness
4

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
The paper proposes "FedTT," a federated learning framework to solve cross-city transferred based traffic prediction problem. It incorporates three key components: Traffic Secret Aggregation (TSA) protocol for secure aggregation, a Traffic Domain Adapter (TDA) for domain alignment, and a Traffic View Imputation (TVI) method for handling missing data.  The authors claim superior performance over baselines using four common traffic prediction datasets.

### Strengths
1. The paper does a great job of identifying and clearly describing the three challenges and problem definition, although the relationship between each other should be emphasized. 
2. This paper conducts comprehensive experiments with enough baseline models and datasets.
3. The paper does have originality which is inherently incorporated in the three proposed and combined modules .
4. The overall quality of this paper is good, especially the writting.

### Weaknesses
1.	Overall, the authors advocate that there are three unresolved fundatmenal Challenges in this paper. But the three key questions are not well-correlated. The papers uses fancy-named modules and attempts to mask a fundamentally patchy and incoherent structure.
2.	The authors state that no priors studies address the three advocated challenges in a unified federated setting. There is no doubt that there is no prior studies in this domain, because whether these three key challenges co-existing in reality is questionable. 
3.	As the Problem definition section states, the Problem of FTT is simply a cross-city transferred traffic prediction problem. The motivation of doing this kind of work is not well introduced. It this paper, three key modules are proposed. The core of cross-city information sharing, which could be spatial colloralation between different sensors or different cities zones was not included. This paper mainly focuses on the data side, such as data missing, data privacy problems. 
4.	The authors statues existing imputation methods fail to effectively capture the spatiotemporal dependencies of data, which is not very convincing. The traffic view imputation section still has room for improvement in term of novelty.
5.	While the paper frames its contribution within a broader, more ambitious research paradigm, its core methodology remains somewhat conventionally focused on the cross-city transfer prediction problem. The use of expansive terminology seems disproportionate to the actual scope of the work.

### Questions
1. What is the Traffic Domain Prototype? It is not very straightforward to name it as prototype from the reviewer's perspective. Maybe more straightforward descriptions of the concepts is better for this paper, considering it attempts to solve three independent problems.
2. The motivation of the TSA module is kind of week. Why these traffic data or traffic prediction related cases/scenarios will have attacker during the inference time and what is the attacker's motivation. Traffic prediction is not like a domain needs strong data privacy.  Who will be the attacker and for what? In the ablation studies, the contribution of the TSA module is also subtle.
3. There are so many traffic/spatiotemporal data imputation studies in the past ten years, the novelty of the traffic view imputation is kind of limited. The authors could emphasize more on it. Additaionlly, traffic view is also not straightforward.

### Soundness
2

### Presentation
3

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
FedTT offers a system-level contribution to federated traffic prediction by integrating privacy preservation, cross-domain adaptation, and data imputation into one coherent architecture. The paper is technically sound and experimentally thorough, yet its conceptual innovation is moderate, and privacy guarantees is fair.

### Strengths
1. Evaluated on four real-world traffic datasets (PeMSD4/8, FT-AED, HK-Traffic) with multiple baselines; FedTT consistently improves MAE and RMSE depending on the scenario.
2. TSA is presented as an efficient alternative to DP/HE, showing better trade-off between computation and utility.

### Weaknesses
1. The paper presents a complete and effective framework, with convincing experimental results demonstrating its practical value for cross-city traffic prediction. However, in terms of originality, the method mainly integrates existing components.
2. Although TSA avoids the use of heavy cryptographic schemes, it is not formally compared to differential privacy (DP) methods, nor is its attack resistance quantitatively analyzed. For fairness, it would be preferable to first align the privacy protection strength under the same threat model, like evaluating resistance to inversion attacks before comparing the resulting utility. This would provide a more objective assessment of TSA’s real advantage in the privacy–utility trade-off.
3. The writing in several parts of the paper lacks rigor and occasionally overstates the contribution. For instance, the statement after line 182 “Existing federated traffic transfer methods often overlook the challenges associated with low-quality traffic data...” is inaccurate. In reality, numerous prior studies, including recent works on federated traffic prediction and cross-domain transfer, have explicitly addressed low-quality or missing traffic data. Such phrasing may give the impression of exaggerating the novelty of this work. Similar issues appear multiple times throughout the paper. The authors are advised to use more precise language when describing related work and to clearly distinguish between problems that are truly unexplored and those that have been partially addressed in prior literature, in order to maintain scholarly rigor and credibility.

### Questions
See the issues discussed in the “Weaknesses” section above.

### Soundness
3

### Presentation
2

### Contribution
2
