# TraMEL: An Exemplar Replay-Based Continual Learning Framework for Malware Traffic Analysis

- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Most prior work on continual malware detection has focused on static code analysis. In contrast, this paper explores continual learning (CL) for malware traffic analysis (MTA), which leverages encrypted flow features to capture behavioral signals that remain observable despite obfuscation and encryption. Unlike conventional intrusion detection systems that perform coarse anomaly detection, MTA requires fine-grained family-level classification under evolving,  imbalanced, and non-stationary distributions, making it a distinct and challenging setting for CL.

We introduce TraMEL (Traffic-based Malware Exemplar Learning), a replay-based CL framework designed for MTA. TraMEL integrates (i) adaptive exemplar selection to address long-tailed family distributions and (ii) an exemplar refinement phase to mitigate task recency bias under strict memory budgets. We evaluate TraMEL under both standard class-incremental and temporally shifted scenarios. Across CICAndMal2017 and IoT23, TraMEL outperforms strong CL baselines including iCaRL, ER, and TAMiL by 10–30 percentage points, and approaches the performance of joint training, a theoretical upper bound with full access to past data. These results demonstrate that CL on malware traffic is both feasible and practical, providing a memory-efficient approach toward real-world malware detection. Code is available at \url{https://anonymous.4open.science/r/ICLR2026-code-D575/}.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a continual learning framework for malware traffic analysis, namely TraMEL.
The methodology relies on large replay buffer, and the training loss which penalizes regression of accuracy on previously-correctly-classified samples. The classifier is then chosen among different strategies, selecting transformers for this cause.
When compared with the state of the art of continual learning (not continual learning dedicated to traffic), results are mixed, with ER competing with TraMEL (even if ER is not strictly for malware detection).

### Strengths
+ this paper ships together interesting techniques to provide a continual learning framework for traffic malware detection
+ interesting choice of including a regression-free loss function into the objective function
+ it is also interesting that transformers are acting well on tabular data

### Weaknesses
The paper is interesting, but the scope is very narrow for being considered by a venue like ICLR.
This contribution is more suited to a Security conference (A or B) like EuroS&P, ECML, SAC, or workshops like AISec, DLS, WoRMA, etc.
The reasons behind these comments are the following:

**1) Incremental contribution that does not provide insights on how to use domain knowledge.** The main modification is using the regression-free loss, while all the other components seem to be already investigated and proposed by prior work, outside the malware domain. Also, while the paper clearly states at the beginning that domain knowledge is rarely used in these contexts, it falls into the same problem. If the issue was the length of the replay buffer (which the experiments show it was) then also other methods outside security can be used by tweaking that parameter. Also, the strict resource requirements are mentioned, but the paper does not provide a quantitative way to judge such claim. Thus, the contribution feels more as an application of a new loss rather then providing systemic insights in the domain.

**2) Not discussed why there is a boost in performance.** Is it because the transformer is useful? Or is the sampling? Only the buffer replay size is discussed, but no complete ablation study is provided.

**3) Tabular data = ensembles of trees.** Why the paper deploys deep networks, when tabular data is empirically proven to be the best data format for ensembles of trees? See attached references. However, this would not be a contribution, but the correct usage of prior works published on top-tier venues.

**4) No limitations are discussed.** The paper has limitations, related to the choice of architecture, of the choice of the dataset, and other many that the paper does not take into account, being a huge miss for this paper.

**5) Missing baselines.** While the paper states that there are few work on continual learning for traffic data, some are presented. It is not clear why they were not evaluated. The papers state they are just not practical, without being convincing on why. 

**Minor comments.**
1. there is an Appendix ?? reference broken at page 4
2. space is used strangely, with paragraphs with different gaps
3. RMA attacks exist? Is there a reference to those? Or is it a novel concept here?
4. Figure 1 is not really informative for the paper.
**References.**

Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data?. Advances in neural information processing systems, 35, 507-520.

Shwartz-Ziv, R., & Armon, A. (2022). Tabular data: Deep learning is not all you need. Information Fusion, 81, 84-90.

Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). Revisiting deep learning models for tabular data. Advances in neural information processing systems, 34, 18932-18943.

Fernández-Delgado, M., Cernadas, E., Barro, S., & Amorim, D. (2014). Do we need hundreds of classifiers to solve real-world classification problems?. The journal of machine learning research, 15(1), 3133-3181.

### Questions
1) Which is the reason that bring a boost i performance? The architecture of the classifier? The replay buffer? The regression-free loss? Or is it the combination?
2) Why the paper uses complex networks, where ensembles of trees are empirically proved the best on tabular data (given a good feature extractor)?
3) What is exactly the contribution w.r.t. standard continual learning methods?

### Soundness
3

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
4

### Summary
This paper presents the TraMEL framework for fine-grained malware family classification within non-stationary network traffic. The framework addresses the challenges of catastrophic forgetting and class imbalance through a memory-efficient strategy that combines diversity-aware exemplar selection and an exclusive exemplar refinement phase. 

To achieve this, it introduces a dual-loss objective during refinement to explicitly counteract task recency bias, thereby maintaining a robust balance between stability and plasticity. Empirically evaluated on two datasets under both class-incremental and temporal-shift scenarios, it demonstrates performance gains over strong replay-based baselines. This closely approaches the theoretical upper bound of joint training accuracy.

### Strengths
+ The paper formalizes malware traffic analysis as a unique and challenging Class-IL problem.

+ The introduction of an exclusive refinement phase, combined with tailored distillation losses ($\mathcal{L}_{past}$ and $\mathcal{L}_{current}$), is a well-motivated and empirically demonstrated method for countering the prevalent task recency bias.

+ The finding that transformer architectures are superior for malware traffic's tabular data structure is important.

+ The framework achieves significant and consistent performance increases (10-30 percent) over strong, established CL baselines across multiple datasets.

### Weaknesses
- Unclear use of some terms.

- The selection of datasets and baselines should be argued better.

- The refinement mechanism and fixed buffer size inherently limit scalability, as the efficacy of rehearsal degrades dramatically if the memory budget cannot be proportionally maintained as the number of tasks grows indefinitely.

- The cluster size ($N_{k}$) for exemplar selection and the distillation weights ($\alpha, \beta$) require extensive empirical tuning, which may make it complex to deploy or generalize to new, unseen datasets without a similar tuning effort.

### Questions
The following needs clarification, better arguments to make the paper stronger. 

(1) The exemplar refinement phase is great, but I felt that its computational burden and dependency on specific loss weight tuning deserve a deeper discussion regarding future scalability. Specifically, while the refinement effectively counters recency bias, it introduces a separate, mandatory optimization step after every task, increasing the wall-clock time and computational load relative to standard replay methods. The paper can consider quantifying the exact overhead (e.g., training time in minutes/hours) added by the refinement phase compared to the joint training phase alone, especially in the tightest budget scenarios. 

(2) The K-means clustering-based selection (TraMEL-K) is shown to be crucial for tight memory budgets, but its reliance on a pre-determined optimal cluster number ($N_{k}=600$ for IoT23) is unclear to me. Here, I understand that the dependence of performance on the specific, empirically found value of $N_{k}$ (Section A.2, Table 6) suggests that this hyperparameter may not generalize well. If $N_{k}$ must be tuned per dataset, it negates some of the framework's practical use cases.


(3) The paper defines the temporal-shift scenario as disjoint families grouped by the year of first appearance, which is acknowledged as a stricter than real deployments lower bound. Although the retrograde malware attack threat model is excellent, the current temporal-shift benchmark does not fully capture the recurring nature of the RMA's third phase (reintroducing legacy families or slightly altered variants). This is because the paper uses only disjoint families, and it assesses drift, not genuine recurrence.

(4) The paper's foundational arguments on buffer constraints, related work, and core assumptions need stronger substantiation to clearly frame its novelty.

- The paper repeatedly asserts that detectors must operate under strict memory budgets due to storage and scalability limits. This motivation is currently asserted without concrete, domain-specific justification. 

-  The term closed-world assumption is central to differentiating this work from traditional IDS, but it is never formally defined. 

-  The Related Work section is dismissive toward several recent, relevant CL efforts in network security, primarily by stating their limited relevance.  I believe,  instead of dismissing related CL works (\eg SPIDER, SPCIL) or other malware CL methods (\eg MalCL), the paper can briefly and precisely articulate the technical difference in feature space.

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
3

### Summary
This paper proposes TraMEL, an exemplar replay-based continual learning (CL) framework for fine-grained malware family classification from encrypted network traffic. It addresses catastrophic forgetting in class-incremental (Class-IL) and temporal shift scenarios, evaluating multiple exemplar selection strategies—random sampling, class-mean (following iCaRL), and clustering-based (K-means) to handle long-tailed distributions, along with fixed-memory buffer management and a refinement phase using distillation losses weighted by hyperparameters α and β to mitigate recency bias. Evaluated on CICAndMal2017 (Android, 42 families) and IoT23 (IoT, 9 families) datasets using disjoint class and temporal splits, TraMEL outperforms baselines like ER, iCaRL, and TAMiL by 10-30% in accuracy, approaching joint training under tight buffers (0.5% of data, 3000 samples). Additional contributions include the Retrograde Malware Attack (RMA) threat model and exploration of backbones (MLP, CNN, ViT).

### Strengths
Novel problem framing: TraMEL extends continual learning to malware-traffic analysis, addressing encrypted flow data rather than static code features an underexplored yet operationally realistic domain. 

Refinement for bias mitigation: The two-phase training scheme, joint training followed by exemplar-only refinement, substantially reduces recency bias. 

Interpretable stability–plasticity control: The analysis of the refinement loss coefficients (α, β) offers clear insight into how past-knowledge preservation and current-task retention interact, enabling tunable control over forgetting. 

The evaluation in two complementary settings: standard class-incremental learning (Class-IL) with disjoint families as a strict worst-case benchmark and a temporal split grouping families by time of first appearance to simulate natural drifts, offers a thorough assessment of the model's robustness, capturing both controlled and realistic evolutionary dynamics in malware traffic. 

Memory-efficient: Fixed-capacity buffers compared to baselines with proportional per-class quotas scale from 200–60 k exemplars, maintaining near-joint accuracy while operating within realistic storage limits

### Weaknesses
Lack of imbalance-aware evaluation: The accuracy results give a solid picture of overall
performance, showing the thorough testing done. For even better understanding of how the
model deals with rare malware types where common ones might overshadow them, adding
balanced metrics that weigh all types equally (like average F1-score, recall across classes, or
precision-recall curves per class) would be helpful.

Lack of open-world/label-scarce robustness. Testing with fully labeled and known malware
types sets a strong foundation. To make it more applicable to real-world situations where new
threats show up without labels, adding checks for handling unknown attacks or learning from
partially labeled data would be a nice step. An ablation study on completely unknown malware
types can be helpful to determine the real-world scenario

Cross-dataset tuning fairness is unclear. The tuning of key settings (α, β) on one dataset
(CICAndMal2017) is clearly described and thoughtfully done. To build more trust in results
across datasets, noting if the other dataset (IoT23) used the same settings or was tuned on its
own would ensure everything is fair and make it easier for others to repeat or expand the work.

Backbone ablation is underreported. Using a Vision Transformer as the main model looks
promising, with the initial results suggesting it beats simpler options like multi-layer perceptrons
or convolutional networks. To back this up fully and give a clearer view, adding more detailed
numbers for those alternatives in the comparison section would make the advantages stand out
better.

### Questions
The paper excludes two minority families (Torri, Trojan), reducing IoT23 from 11→9
classes. Could you clarify the rationale? This would provide valuable context for readers
replicating or extending your experiments.

When 𝐾 is full and per-class quotas shrink after task 𝑖, which exemplars are removed
from earlier classes, and do you re-encode/re-cluster old classes under the updated
model?

Were MLP, CNN, and ViT evaluated under identical settings, like the same buffer 𝐾,
refinement budget 𝑘, and distillation weights (𝛼, 𝛽) when the claim about ViT’s superior
performance was made?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes TraMEL (Traffic-based Malware Exemplar Learning), a replay-based continual learning (CL) framework tailored to malware traffic analysis (MTA) with encrypted flows. 

TraMEL combines imbalance-aware exemplar selection (random, class-mean, and diversity-oriented K-means) and a post-task exemplar refinement phase that reduces task-recency bias via a composite loss balancing cross-entropy with two distillation terms that anchor both past and current behavior. The authors define a refinement objective that explicitly trades off plasticity vs. stability.

Evaluated in Class-IL and temporal split settings on CICAndMal2017 and IoT23, TraMEL consistently outperforms strong replay baselines (ER, iCaRL, TAMiL), narrowing the gap to joint training while operating under strict memory budgets; the study also analyzes buffer size, refinement epochs, and loss weights.

### Strengths
- The authors focused the study on encrypted malware traffic at family level, a more realistic and fine-grained setting than conventional binary IDS settings.

- Two-stage design (diversity-aware exemplar selection + refinement) targeted to long-tailed, evolving families under tight buffers; the dual-distillation refinement is thoughtfully motivated.

- Comprehensive evaluation on CICAndMal2017 and IoT23 across Class-IL and temporal splits; repeated runs; reports task-wise/mean accuracy and forgetting.

- Demonstrates that memory-efficient replay with careful selection and refinement can deliver robust, fine-grained malware family classification under drift—relevant to practical SOC/defense pipelines with retention constraints.

### Weaknesses
- The paper argues that real deployments exhibit family recurrence, yet both Class-IL and temporal settings keep disjoint families across tasks (a conservative lower bound). This complicates claims about performance under true recurrence, open-set families, or re-emergence. A small synthetic recurrence experiment (e.g., re-introducing early families later) would strengthen the validity.

- Baselines are standard replay methods; however, related class-imbalance-aware CL or compressed/coreset replay variants are not included empirically. Even a small-scale comparison (or discussion) would contextualize TraMEL’s diversity-aware selection.

- Since the study emphasizes practical constraints and evolving distributions, it would be useful to discuss continual semi-supervised one-class approaches for malware detection on traffic—an adjacent but distinct framing that reduces label burden (e.g., Continual Semi-Supervised Malware Detection, MAKE 2024), and clarify how TraMEL’s supervised replay compares or could be hybridized.

### Questions
- Could you report a recurrence experiment (e.g., re-introduce a subset of Task-1/2 families at Task-5/6) to quantify retention when families return—potentially with smaller buffers? This would align the setup with the paper’s motivation (re-emergent variants). 

- You note that larger k (e.g., >100) improves coverage. How sensitive are results to k vs. per-class sample counts, especially for extreme long tails? Any heuristics you recommend? 

- Have you thought about adding automatic tuning (or an early-stopping criteria during refinement) to bound extra compute?

- Given labeling constraints in MTA, how does TraMEL interact with semi-supervised or one-class continual regimes? A discussion (or small pilot) contrasting TraMEL’s supervised replay with continual semi-supervised detection on traffic data would be valuable; for instance, see Continual Semi-Supervised Malware Detection (MAKE 2024).

- I suggest adding a compact pseudocode box for the three-phase loop. Please, also, include the seeds and data splits you used for temporal grouping for easier replication.

### Soundness
3

### Presentation
3

### Contribution
3
