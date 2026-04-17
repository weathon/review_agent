# VFedCD: Causal Discovery under Vertical Federated Scenario

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 2

## Abstract
Causal discovery seeks to identify causal relationships among attributes, typically represented as directed acyclic graphs (DAGs) where vertices denote attributes and edges denote direct causal effects. Existing methods struggle in vertically federated scenarios. In these settings, data is partitioned across parties that hold disjoint attributes, and strict privacy constraints prevent centralized aggregation, leaving vertical federated causal discovery underexplored.
We propose VFedCD, the first framework for causal discovery in vertical federated settings. VFedCD models causal mechanisms with a shallow-encoder, deep-decoder design. Each party uses a shallow encoder to transform its local attributes into privacy-preserving features for all parties, and then a deep decoder to aggregate received features and predict local attributes, implicitly capturing causal dependencies. To avoid cycles or overly dense graph structures, a Centralized Topology Validator (CTV) extracts partial causal structures from party encoders, aggregates them into a global graph and enforces structural constraints. In addition, a Secure Dispatch Protocol (SDP) is designed to enhance the security of feature exchange and gradient propagation by redesigning encoding and aggregation with semi-homomorphic encryption and secret sharing.
Experiments on synthetic and real-world datasets with artificial vertical partitioning show that VFedCD matches the accuracy of centralized methods while guaranteeing privacy.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper studies causal discovery in the VFL setting, where different parties hold disjoint feature sets for the same individuals. It proposes **VFedCD**, a split-model framework with a coordinating component that enforces a continuous acyclicity constraint and a cross-party feature/representation transmission mechanism to enable joint DAG learning without raw-data pooling. The manuscript claims three contributions: (1) a system-level VFL pipeline for causal structure learning, (2) an algorithm that optimizes a continuous DAG objective in a distributed fashion, and (3) theoretical and empirical validation of the approach on synthetic and benchmark datasets.

### Strengths
- **Timely topic.**  
  The paper studies causal discovery in VFL, which is indeed a relevant direction given increasing data privacy concerns across domains such as healthcare and finance.

- **Conceptual motivation.**  
  The authors make an effort to bridge the areas of causal discovery and federated learning, highlighting an underexplored intersection that could be of potential interest to both communities.

- **Interesting idea.**  
  The overall workflow, dividing models across parties with a coordination component, is conceptually understandable, and the manuscript provides a basic sketch of how such a system might be implemented.

### Weaknesses
- **Poor organization and unclear notations.**  
  The paper is written in a confusing manner, with inconsistent and frequently redefined notations throughout. Key symbols (e.g., **B**, **Θ**, **D**) change their meanings across sections, making it very difficult to follow the methodology or reproduce the setup. This lack of clarity in mathematical definitions and notation usage seriously undermines readability and technical credibility.

- **Unclear problem definition and data formulation.**  
  The problem statement of VFedCD is not well specified. It remains ambiguous what exact task the model is solving, how the global causal graph is defined, and how the data are partitioned among clients. The paper does not clearly describe whether nodes can overlap between parties or how information flow is constrained under the vertical FL setting. As a result, it is unclear what precise learning objective the proposed method is supposed to optimize.

- **Questionable federated learning design.**  
  The proposed *Cross-Party Feature Transmission* mechanism seems to directly share intermediate representations or feature information across clients. This violates the core privacy assumption of federated learning, where no raw or derived data should be transmitted between parties. Without a proper explanation or privacy-preserving mechanism, the proposed framework cannot be considered a valid federated solution.

- **Lack of theoretical justification and identifiability discussion.**  
  The paper claims to “establish a theoretical foundation” but provides no formal analysis or guarantees. There is no discussion of under what assumptions the proposed approach can recover the true DAG, nor what identifiability conditions are required. Without any theoretical results or guarantees, the method remains purely heuristic and its validity is questionable.

### Questions
- The authors claim that this is the first *vertical federated causal discovery* framework. However, to my knowledge, the work *“Horizontal and Vertical Federated Causal Structure Learning via Higher-order Cumulants”* already addresses similar settings and was proposed several months ago. In addition, the authors state that they “establish a theoretical foundation” for **VFedCD**, but it is unclear what specific theory is provided — where is it presented or formally proven?

- In the *Related Works* section, many citations and categorizations appear to be incorrect or inconsistent.  
  - The **PC** algorithm was originally proposed by *Peter and Clark*, not by *Kalisch and Bühlmann*.  
  - **GES** is cited under both score-based and mechanism-fitting methods, which is confusing.  
  - **NOTEARS** is incorrectly excluded from score-based methods — it searches over DAGs via continuous optimization and thus should belong to that category.  
  - The term *“mechanism-fitting branches”* is unclear. From my understanding, the community typically classifies *LiNGAM*, *ANM*, and *PNL* as **constrained functional causal models**.  
  - There seems to be a typo in “NOTEARS-ADMMTh method.”  
  - The notations are inconsistent: for example, **D** and **B** are first defined as functions for different parties, but later redefined as datasets and causal graphs.  

- The *Problem Setup* section is rather confusing and lacks clarity on the overall formulation.  
  - What exactly is the causal graph — is it assumed to be a DAG?  
  - What is the underlying data generation process, and how is it partitioned among clients?  
  - Can nodes overlap between clients?  
  - The “learning objective” claims that the ground-truth causal graph can be obtained via a loss function, which sounds more like a *solution* rather than a *problem definition*.  
  - The relationship between **B** and **Θ** is unclear.  
  - What is the formulation of the function **h**, and how does it enforce the continuous acyclicity constraint?  
  - How is the loss **l** defined, and how does it aggregate information from multiple parameters and the global observation **x**?  
  - The term **φ** is also unclear — why does *t* range from 1 to *K*? Is **φ** an encoder specific to client *k*?

- In the *Method* section:  
  - The proposed **Cross-Party Feature Transmission** mechanism appears to transmit feature information across clients, which contradicts the privacy-preserving assumption in federated learning. Please clarify how this complies with the federated setup.  
  - In the backward process, how is the prediction loss defined — mean squared error or something else?  
  - In Eq. (5), what is the role of the additional notation **ρ**? Why is it needed?

- The authors should carefully proofread the manuscript before submission. There are many inconsistent abbreviations and typographical issues.  
  - For example, the full name of **SGD** is introduced both at line 224 and again at line 324.  
  - The acronym **DAG** is not defined until line 335.  
  - It is confusing to refer to all baseline methods as *SOTA*; most of them are not designed for **VFedCD**. How are these baselines adapted to the proposed setting?  
  - How is the synthetic dataset constructed?  
  - Finally, why is subsection 6.3 titled “Generalization”? The term seems misused in this context.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a vertically federated causal discovery framework VFedCD, which aims to learn causal structures across multiple parties with vertically partitioned data. To address privacy and communication challenges, the authors adopt a shallow-encoder deep-decoder (sEdD) architecture and design two key components: a SDP that combines semi-homomorphic encryption and secret sharing to protect data during computation, and a CTV that enforces global acyclicity and sparsity constraints across parties.

### Strengths
1. The paper provides solid theoretical analysis and proofs supporting the identifiability and correctness of the proposed framework.
2. The paper successfully combines encryption-based privacy protection with acyclicity and sparsity constraints into a unified framework, resulting in a coherent system design.

### Weaknesses
1.	Are the attributes across parties allowed to overlap? The paper assumes non-overlapping features, but in real scenarios, feature overlap is also common. How would the method handle this case?
2.	The comparison of running time is unclear. 
3.	The baseline methods are not vertically federated. It would be helpful to discuss whether existing causal discovery methods can be adapted to the vertical federated setting.
4.	Some cited papers, such as Stable Differentiable Causal Discovery (ICML 2024), have already been published. Please check and update the references.
5.	The encryption and privacy mechanisms, as well as the acyclicity and sparsity constraints, are based on existing work. The paper seems to focus on integration rather than introducing new theoretical innovations.

### Questions
1. Are the attributes across parties allowed to overlap? The paper assumes non-overlapping features, but in real scenarios, feature overlap is also common. How would the method handle this case?
2. Please provide the comparison of running time.
3. It would be helpful to discuss whether existing causal discovery methods can be adapted to the vertical federated setting.
4. Please check and update the references.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The authors propose VFedCD, a vertical federated causal discovery framework that allows multiple parties holding vertically partitioned data to collaboratively infer causal relationships among their attributes while preserving privacy. VFedCD consists of two key components: (i) Causal Topology Validator (CTV), which aggregates local structural estimates and enforces global sparsity and acyclicity, and (ii) Secure Dispatch Protocol (SDP), which enables privacy-preserving feature interaction and gradient sharing. The problem setting is relevant and the research direction is of clear practical interest.

### Strengths
1. The authors study causal discovery in the setting of vertical federated learning, which is a problem with clear practical value and application significance.

2. The proposed method integrates privacy-preserving cross-party feature interaction with global acyclicity constraints and provides corresponding theoretical support for vertical federated learning.

### Weaknesses
1. Some parts of the paper are difficult to follow, especially the description of the data encryption and decryption process, which makes it hard for the reader to clearly understand the concrete mechanism.

2. The paper does not fully articulate the actual novelty of the proposed approach or clearly position it relative to existing methods.

### Questions
1. This paper does not clearly distinguish itself from prior work in federated graph structure learning, and it does not adequately highlight the novelty of its inter-client interaction protocol in federated causal discovery.

2. The acyclicity constraint enforced by the centralized topology validator (CTV) is essentially a basic requirement for causal discovery, rather than a unique technical contribution of this work.

3. In Figure 2, the encoder within each party module is presented as producing a local causal structure, but the paper does not clearly explain how this structure is derived; if it is simply obtained by treating the first-layer weights as causal edges, this lacks sufficient theoretical justification and identifiability support.

4. The paper does not clearly explain how HE2SS encrypts data or how encrypted / split information is exchanged between clients.

5. The paper does not clarify whether there are realistic application scenarios for vertical federated causal discovery in practice.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes VFedCD, a framework for causal discovery in vertical federated learning settings where data attributes are partitioned across multiple parties. The method employs a shallow-encoder deep-decoder (sEdD) architecture to model causal mechanisms, a Centralized Topology Validator (CTV) to enforce global acyclicity constraints, and a Secure Dispatch Protocol (SDP) using semi-homomorphic encryption and secret sharing for privacy preservation.

### Strengths
1. The paper addresses a relatively underexplored area—causal discovery in vertical federated settings—which has practical relevance for privacy-constrained multi-party collaborations.
2. The proposed solution integrates multiple components (sEdD architecture, CTV, SDP) to handle both the technical challenges of distributed causal discovery and privacy concerns.
3. The paper provides extensive privacy analysis in Appendix H, including discussions of various attack scenarios and mitigation strategies using differential privacy.

### Weaknesses
1. The core technical components lack novelty. The sEdD architecture is a standard design choice in VFL; semi-homomorphic encryption and secret sharing are well-established techniques; and the spectral radius constraint is directly borrowed from Nazaret et al. (2023). The paper primarily combines existing techniques rather than introducing fundamentally new ideas.
2. The CTV is a centralized server that receives graph structure information from all parties. This contradicts the core motivation of federated learning (decentralization and privacy) and creates a single point of failure and trust. If a centralized trusted server exists, why not use simpler privacy-preserving approaches?
3. The identifiability analysis (Appendix E) is informal and lacks rigorous mathematical proofs.
4. Computational complexity O(KD²) with HE operations makes the method impractical for realistic problem sizes. Communication overhead O(K²D + KD²) is prohibitively expensive.
5. Limited Experimental Validation. For example, small-scale experiments (maximum 25 attributes); an Ablation study only on one dataset (15 attributes).
6. Table 2 shows VFedCD outperforming all centralized methods on SynTReN, which is suspicious. No discussion of why VFedCD sometimes outperforms centralized methods with full data access. Appendix F.2's explanation is speculative and unconvincing.
7. The paper doesn't adequately discuss why existing horizontal federated methods cannot be adapted, or position itself relative to secure multi-party computation literature.
8. If parties trust a central server with graph structures, why not use simpler approaches like differential privacy on centralized data? How is this fundamentally different from uploading noisy statistics to a trusted curator?
9. How exactly does graph(Θ) work in Algorithm 5, line 2? The description in Appendix D.1 is vague. How do you handle the hidden dimension aggregation? How sensitive is this to the choice of norm?
10. Do you have any theoretical or empirical evidence that the alternating optimization between local model updates (Algorithms 3-4) and global constraint enforcement (Algorithms 5-6) converges to a meaningful solution?
11. According to d-separation principles in causal inference, the proposed approach of aggregating local subgraphs into a global graph is theoretically flawed, even with acyclicity constraints.

The core issue: Each party k learns a local subgraph B^k_t by modeling causal relationships from all parties' attributes to target party t's attributes, using only intermediate features rather than the full attribute set. However, valid causal discovery requires access to all potential confounders and mediators to correctly identify direct causal relationships versus spurious correlations.
Concrete example: Consider the true causal structure A→C→B where:

A ∈ Party 1
C ∈ Party 2
B ∈ Party 3

When Party 3 learns to predict B, it receives encoded features from both Party 1 and Party 2. Since A and B are statistically dependent (due to the path A→C→B), Party 3's decoder may learn a direct dependency from Party 1's features to B, leading to an incorrect edge A→B in the aggregated graph. The CTV's acyclicity constraint only prevents cycles but cannot eliminate such spurious edges introduced by incomplete observation of intermediate variables.

### Questions
Please see Weaknesses.

### Soundness
1

### Presentation
3

### Contribution
2
