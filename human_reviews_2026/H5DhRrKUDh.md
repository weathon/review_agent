# MedAlign: Clinician-Centered Federated Meta-Learning for Medical IoT with Privacy and Interpretability Guarantees

- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
We introduce MedAlign, a resource-aware federated meta-learning framework designed for medical Internet-of-Things deployments that face strong data heterogeneity, strict privacy constraints, and tight device resource budgets. MedAlign supports collaborative optimization across distributed clinical sites while enabling per-site personalization. The system couples ontology-driven feature selection with multimodal fusion and prototype-consistent representation learning to preserve stable diagnostic boundaries across non-identical client distributions. A lightweight adaptive gating controller (RL-gating) dynamically modulates module execution according to instantaneous compute, energy, and latency conditions on commodity edge hardware, allowing efficient on-device inference and iterative updates. Privacy is enforced through a formally calibrated aggregation protocol that composes sensitivity-aware noise with a multi-round Rényi-style accountant, yielding quantifiable confidentiality guarantees with minimal impact on clinical utility. We validate MedAlign on intensive-care and wearable-health benchmarks and on commodity edge platforms; the experimental suite includes ablation studies, privacy-accounting traces, and robustness tests against reconstruction and poisoning attacks. Results show that MedAlign consistently improves diagnostic performance and training efficiency while substantially lowering communication and energy costs compared to representative baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents MedAlign, a comprehensive framework that integrates federated meta-learning, adaptive resource control, and differential privacy for medical IoT environments. The system introduces multiple synergistic modules—Context-Aware Feature Weighting (CAFW), Clinical Dependency Encoder (CDE), Prototype-Consistent Representation Learning (PCRL), and RL-based adaptive gating—to jointly optimize clinical diagnostic performance under device and privacy constraints. The design claims to balance personalization, multimodal fusion, and formal privacy guarantees through a multi-stage optimization and hierarchical clustering procedure. Experiments on three large-scale datasets (ICU, geriatric, cardiac) demonstrate superior diagnostic accuracy (+2%), 63% lower communication overhead, and quantifiable privacy guarantees.

### Strengths
The paper is notable for its system-first perspective, combining realistic clinical constraints (bandwidth, latency, IRB compliance) with architectural innovations. The integration of graph-based semantic modeling and reinforcement-driven dynamic gating represents a creative attempt to bridge theory and deployment. The experimental section is unusually detailed for a systems paper, including real-world six-month deployment and ablation tables that reveal non-trivial module interdependencies. The formal derivation of the privacy guarantee (Gaussian mechanism with calibrated sensitivity) and the introduction of prototype-consistent representation alignment further strengthen the methodological depth. From a representation-learning standpoint, MedAlign pushes federated learning toward joint optimization of interpretability, privacy, and energy efficiency—a rare combination in existing work.

### Weaknesses
Despite its breadth, the framework risks being over-engineered: each module (CAFW, CDE, PCRL, RL-Gating, DP aggregation) is described as critical, yet the interactions among them are primarily demonstrated empirically rather than theoretically. The meta-learning layer lacks a clear definition of task distribution or adaptation dynamics beyond cluster-based aggregation; without formal convergence analysis under non-IID sampling, the claimed generalization robustness remains qualitative. The reinforcement gating design—though motivated by resource constraints—relies on dense state vectors and a transformer policy whose learning stability on embedded devices is uncertain. The ontology-driven feature weighting assumes access to structured clinical ontologies on-device, which may not generalize to new sensors or hospitals. Moreover, interpretability claims are supported only by saliency and prototype alignment visualizations, without clinician-validated explanations. Finally, privacy calibration (ϵ = 1.0, δ = 10⁻⁵) is presented as “formally verified,” but no sensitivity analysis is shown for alternate privacy budgets or correlated updates—issues critical to federated healthcare deployment.

### Questions
1: You describe hierarchical personalization via angular-similarity clustering (Eq. 1–2). Can you formally prove that this step yields a tighter generalization bound than standard model-agnostic meta-learning (MAML) when client tasks are correlated but non-IID? What are the assumptions on task relatedness required for this bound to hold?

2: The RL-based gating module optimizes energy–accuracy trade-offs using policy gradients. Given that device states are partially observable and delayed (e.g., intermittent connectivity), how do you ensure policy convergence without violating the Lipschitz assumption invoked in Eq. 23?

3: Equation 28 defines an SNR-based criterion for maintaining diagnostic accuracy under differential privacy noise. Can this be derived directly from an upper bound on mutual information leakage? If so, what distributional assumptions on gradient statistics are necessary?

4: CAFW depends on ontology-based feature relevance. How would the framework behave if the ontology were incomplete or misaligned with empirical feature importance? Could the weighting module be adversarially manipulated to bias diagnosis under structured ontology errors?

5: The PCRL module aligns prototypes across sites to enforce diagnostic consistency. However, alignment may obscure legitimate subpopulation differences. How do you distinguish between harmful prototype drift and clinically meaningful heterogeneity, and could the model unintentionally “erase” rare but critical disease phenotypes?

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
This paper presents MedAlign, a comprehensive federated meta-learning framework designed for medical Internet of Things (IoMT) environments. The system addresses the challenges of data heterogeneity, strict privacy requirements, and resource constraints in distributed medical edge devices. The framework integrates five main components: (1) Context-Aware Feature Weighting (CAFW) for ontology-driven feature selection, (2) Clinical Dependency Encoder (CDE) using graph attention networks, (3) Prototype-Consistent Representation Learning (PCRL) for maintaining diagnostic boundaries across institutions, (4) Reinforcement Learning-based dynamic gating for resource-aware computation, and (5) Formally calibrated differential privacy mechanisms. The authors validate their approach through extensive experiments on three medical IoT datasets and a six-month real-world deployment across twelve healthcare institutions.

### Strengths
1. The six-month deployment across twelve healthcare institutions provides valuable empirical evidence of practical viability
2. Three datasets with comprehensive ablation studies and comparison against 18 baseline methods

### Weaknesses
1. The system integrates too many components, making it difficult to understand individual contributions and potentially limiting practical adoption
2. Key details about baseline implementations are missing; it's unclear if comparisons are fair given different design objectives

### Questions
Can you provide evidence that all five components are necessary? What happens with a simpler subset of 2-3 components?

### Soundness
2

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
3

### Summary
The paper proposes MedAlign, a clinician-centered federated meta-learning framework for Medical IoT that combines (i) ontology-guided Context-Aware Feature Weighting and cross-modal fusion, (ii) a Clinical Dependency Encoder with graph attention, (iii) Prototype-Consistent Representation Learning for cross-site alignment, (iv) RL-based adaptive gating for energy/latency constraints, and (v) a “formally calibrated” DP aggregation protocol. Reported results claim very high accuracy across multiple clinical datasets, reduced energy/latency/communication, a six-month 12-site deployment, and resistance to adversarial/backdoor settings.

### Strengths
- The paper sensibly enumerates clinical constraints (non-IID data, resource limits, confidentiality) and maps them to modules; the workflow/algorithms are laid out clearly.    
- The DP aggregation is at least specified (norm clipping + Gaussian noise) with standard formulae; a semi-honest threat model is written down.    
- The RL-gating objective and state/action design are described; the system section reports latency/energy/communication reductions on edge hardware.

### Weaknesses
- Deployment claims lack verifiable detail (and contain inconsistencies). The “six-month, 12-site” deployment states local updates every 48 hours and global aggregation biannually, which would be only twice per year—at odds with FL training and with the later performance/latency claims. No audit logs, site lists, or concrete clinical endpoints (alarm reduction definitions, prospective vs. retrospective) are provided.  
- DP and privacy accounting are incomplete. The paper presents the single-round Gaussian mechanism \sigma=\sqrt{2\ln(1.25/\delta)}/\epsilon but does not specify an accountant (e.g., RDP/zCDP/moments) for multi-round training, participant subsampling, or per-tier budgets claimed later. The SNR>=18 dB heuristic is empirical and not tied to a specific privacy composition.  
- Security model vs. robustness results are misaligned. The threat model is semi-honest, yet the “extended validation” reports low backdoor success (4.1%) with 30% malicious clients, without describing any Byzantine-robust aggregation, anomaly filtering, or certified defenses—DP noise alone can degrade, not guarantee, robustness. Methodological details are absent.    
- Ablations and system metrics show internal tensions. The text claims RL-gating reduces energy by ~23% and 86 ms latency on a Pi 4, plus 128 KB peak memory, which are ambitious. The ablation table mixes accuracy/energy/latency changes in ways that are hard to reconcile with the qualitative narrative (e.g., removing representation modules sometimes reduces energy and latency substantially). Precise measurement setups (load, batch size, window length) are missing.    
- “Formally verified aggregation” is overstated. The section title suggests formal verification, but the body reproduces the standard Gaussian mechanism with a brief theorem sketch; there’s no mechanized proof, proof artifacts, or code-level verification. The algorithmic step called “secure aggregation” is simply DP-noise addition—not the cryptographic secure aggregation expected in FL deployments.    
- Dataset/task reporting is insufficient. Datasets are named and pre-processing listed, but task definitions, label provenance, prevalence, and class imbalance are under-specified. Cross-site splits, per-site cohort stats, and missing-data patterns (crucial in clinical IoT) are not detailed, limiting reproducibility.  
- Breadth over depth. The paper integrates many modules (CAFW, CDE, PCRL, RL-gating, DP), but the novelty over prior FL/FML/graph-encoding/prototype-learning work is mostly incremental; stronger head-to-head comparisons and surgical ablations isolating each module’s causal effect (beyond aggregate tables) are needed.

### Questions
1. Please clarify the training cadence (global rounds vs. “biannually”), participating institutions, endpoints evaluated prospectively, and whether clinicians were in-the-loop. Provide logs or an audit protocol.  
2. What accountant (RDP/zCDP/moments) and sampling rates were used across T rounds? Report composed \epsilon per data tier (ICU vs. wearable) matching the “hierarchical budgeting” claim.  
3. How is the 4.1% backdoor success achieved under 30% malicious clients? Which robust aggregation or anomaly detection is in place (beyond DP noise)? Show attack details and transferability tests.  
4. Specify the evaluation harness for energy/latency (window sizes, sampling, device states), the policy training regime (on-device vs. server), and ablate gating on/off under matched workloads.    
5. If cryptographic secure aggregation is used, provide the protocol details (e.g., Bonawitz-style) and failure handling; otherwise, please avoid labeling DP-only as “secure aggregation.”    
6. Provide per-site class distributions, missingness patterns, and precise task definitions (e.g., episode-level anomaly detection vs. continuous prediction), plus patient-level splits and IRB scope.  
7. The dual-path explanation framework is intriguing—please add clinician-rated studies or case-based evaluations to assess utility and failure modes.

### Soundness
2

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
The paper proposes a framework for cross-device federated learning for medical applications.

### Strengths
1. FL in the medical domain problems is interesting and important.
2. The paper has reported performance numbers on a real RPi device, which is great to see.

### Weaknesses
1.  The method includes many components without detailing any of them. The current version of the paper is insufficient to understand them and measure their quality and novelty. The paper does not justify the selected components or provide insights into why they are necessary in this context.
2. Authors should cite relevant literature that inspired different components. The paper claims to have a differentially private mechanism, but never explains it. Though there are some results on this, it is unclear what is being protected. It mentions Explainability as one of three gaps in the literature and says it addresses it (section 2.3), but I can't find where.
3. Figure 1 is confusing. For example, the black arrow on the right seems to be in the opposite direction. It uses too many acronyms, including the ones that are never used again in the main text (AFO).
4. The writing style is odd, with almost every paragraph in the introduction having a separate subsection. This hinders readability.
5. The experiments section needs to have more clarity. Is the on-device experiment run on all edge devices in a dataset, such as 1800 devices in the M-IoT-Env dataset? What are the models being used? There is no confidence interval for Tables 1 and 2.

### Questions
As above.

### Soundness
1

### Presentation
1

### Contribution
2
