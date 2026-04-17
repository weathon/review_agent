# A Descriptor-Based Multi-Cluster Memory for Test-Time Adaptation

- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Test-time adaptation (TTA) aims to preserve model robustness under distribution shifts without access to source data. However, existing memory designs, often based on single clusters or naive sample storage, struggle to capture the diversity of target distributions and adapt efficiently over time. We introduce Multi-Cluster Memory (MCM), a novel memory management framework that organizes samples into multiple clusters using lightweight statistical descriptors such as sample means and variances. The inter-cluster distance naturally expands the coverage of the sample distribution, supports on-demand cluster creation for novel patterns, and maintains bounded capacity through an Adjacent Cluster Consolidation (ACC) mechanism that merges neighbor clusters in descriptor space. To further strengthen adaptation, we propose Relevance-guided Sample Retrieval (RSR), which selects the most target domain-relevant clusters for learning and integrates them into a Mean-Teacher self-supervised paradigm. Extensive experiments across CIFAR-10/100-C, ImageNet-C, and DomainNet demonstrate that MCM consistently outperforms prior methods under Practical TTA (PTTA) and achieves sustained robustness in recurring TTA. By delivering a memory structure that is more representative, scalable, and adaptive, MCM establishes multi-cluster memory as a practical and effective foundation for real-world test-time adaptation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Multi-Cluster Memory (MCM), a structured memory mechanism for test-time adaptation (TTA). MCM maintains a dynamic set of clusters, each summarized by lightweight channel-wise mean-variance descriptors. A new sample is assigned to the closest cluster (or spawns a new one), the memory is kept bounded by merging adjacent clusters with the nearest descriptors, and model updates are performed with samples retrieved from the clusters most relevant to the current batch. Extensive experiments on CIFAR-10/100-C, ImageNet-C and DomainNet under the Practical TTA protocol show consistent error reductions when MCM is plugged into three recent memory-based TTA methods; evaluations under the Recurring TTA protocol further demonstrate improved long-term stability.

### Strengths
1. MCM yields repeatable accuracy improvements (up to 12.1 % on DomainNet) across multiple strong baselines and datasets, and remains stable over 20 adaptation cycles.
2. Multi-cluster organisation with lightweight descriptors expands coverage of the target distribution while keeping management cost sub-linear, avoiding the linear-growth bottleneck of single-bank approaches.
3. MCM can be integrated into existing TTA frameworks with minimal code changes, making it attractive for practical deployment.
4. Under Recurring TTA, MCM exhibits lower forgetting and smaller variance than its backbones, indicating that the approach mitigates catastrophic drift effectively.

### Weaknesses
1. Contrary to the implication in the paper, multi-cluster/prototype organisation has already appeared inside the TTA literature. For example, Contrastive TTA (CVPR 2022) keeps multiple per-sample prototypes for contrastive correction; SoTTA (NeurIPS 2023) maintains several class-level prototypes to combat noisy streams; TEA (CVPR 2024) employs multiple energy centres per class. None of these works. A clear comparative discussion with these TTA-specific precedents is missing.

2. The paper seems simply use the mean and variance of a feature —i.e., a parametric summary of its distribution. The assignment distance is a Euclidean norm between these statistics, which is a simplified (diagonal-covariance) Mahalanobis distance. This connection is neither acknowledged nor compared with earlier works that use similar statistics for domain matching (AdaBN, CORAL, MMD).

3. The current mini-batch may be too small (even batch-size = 1) or internally biased to yield a reliable descriptor. MCM, however, unconditionally assigns that single sample to the nearest cluster (or spawns a new one) based on its lone descriptor. This risks instantaneous mis-clustering: the outlier descriptor can drift an existing centroid or create a new cluster that does not correspond to any true mode of the target distribution.

4. A single mini-batch can itself contain multiple domains (e.g., sunny and rainy frames sampled together). MCM computes one descriptor per sample and immediately assigns each sample to the nearest existing cluster, so samples from different domains may be scattered into separate clusters within the same batch.

5. Insufficient functional validation of MCM's own components. Most of the ablations and experiments measure final classification accuracy and error, but the paper does not include sufficient functional tests that isolate why multi-clustering helps and your functional effectiveness. For example, how the average and maximum pairwise distance of cluster centroids change among your method and other methods, and how this changes through batchs.

6. It seems that there is No cluster-error detection or correction mechanism. If an early outlier drifts the centroid, subsequent samples can be chained into the wrong cluster indefinitely. The paper should clarify whether mis-clustering errors will accumulate and whether there any internal feedback (e.g., entropy rise in a cluster) can trigger some kinds of correction.

7. Many of your experiments are hyper-parameter analysis. However, besides appropriate choice of parameter, results seem to show that the effects are sensitive to hyper-parameters. Performance varies noticeably with τ, K_max and per-cluster capacity N. All values seem to be hand-tuned; I recommend to conduct more small comparison near the best choice of hyper-parameter and among different datasets.

### Questions
Q1 Please clarify the conceptual difference between MCM and earlier multi-prototype / multi-exemplar memories What specific design choices make MCM uniquely suited to TTA? Why can multi clustering better describe categories? Can multi clustering really better describe situations that cannot be described by a single cluster.

Q2 Your descriptor is the per-channel mean and variance and the distance is Euclidean. How does this relate to the Mahalanobis distance or to second-order domain-matching objectives such as CORAL or MMD? Have you tried full-covariance distances?

Q3 What is the extra runtime and memory of MCM compared with the single-bank baseline for the same total number of stored samples? Please report some comparison of time and GPU memory. It seems that multiple clusters may need more time and slower than single-bank method.

Q4 I recommend that more recent SOTA method needed to be compared, more 2024 and 2025 method can be provided.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces *Multi-Cluster Memory (MCM)*, a novel memory management framework for Test-Time Adaptation (TTA). The authors argue that existing memory-based TTA methods, which often use a single, unstructured "single-cluster" memory, suffer from poor *representativeness* (failing to capture the full target distribution) and low *adaptability* (being slow to update under continual shifts).

MCM addresses this by structuring the memory bank into multiple, dynamic clusters. The core components of MCM are:
1.  *Descriptor-Based Management:* It uses lightweight statistical descriptors (channel-wise mean and variance) to characterize samples and define cluster centroids. New samples are assigned to the nearest cluster, or a new cluster is created if the sample is novel (distance > $\tau$).
2.  *Adjacent Cluster Consolidation (ACC):* To maintain a bounded memory capacity ($K \le K_{max}$), MCM merges the *adjacent* cluster pair (in creation sequence) with the closest descriptor distance when the cluster limit is reached.
3.  *Relevance-guided Sample Retrieval (RSR):* For adaptation, the model selects the $N_S$ clusters whose descriptors are most similar to the current mini-batch, ensuring retrieved samples are relevant for the current adaptation step.

The authors demonstrate MCM as a "plug-and-play" module, showing consistent performance improvements when integrated into three existing TTA methods (ROTTA, PeTTA, and ResiTTA). Experiments are conducted on CIFAR-10/100-C, ImageNet-C, and DomainNet under both the *Practical TTA (PTTA)* and *recurring TTA* settings, showing improvement in terms of performance and long-term stability.

### Strengths
1. While individual components borrow from prior work, the integration of multi-cluster memory management specifically for TTA is an interesting approach. The combination of descriptor-based clustering, ACC, and RSR within the TTA framework is original
2. The experimental evaluation is comprehensive, covering multiple datasets, baselines, and evaluation protocols (PTTA and recurring TTA). The ablation studies systematically examine hyperparameter sensitivity. The efficiency analysis (Section 4.4) effectively isolates the contribution of memory organization from capacity increases
3. The motivation is well-articulated through Figure 1's visualization of representativeness and adaptability limitations.
4.  MCM demonstrates practical value as a plug-and-play component.

### Weaknesses
1. Proposed approach consists of MCM, ACC and RSR. An ablation analysis for the contributions from each of these components is missing.
2. ACC merging heuristics lack theoretical and empirical validation.
3. Key hyperparameter $N_S$ (line 297) for relevant cluster selection is underexplored.
4. Limited novelty, with performance gains mainly on complex datasets.
5. Incomplete comparisons with alternative memory scaling and organization methods.
7. Missing comparison with other clustering baselines.
8. Limited generalizability beyond image classification.
9. The description of ResiTTA (Zhou et al., 2025) as "ResiTTA (Zhou et al., 2025) introduces residual connections to enhance robustness in continual learning for TTA scenarios" is wrong. "Resi" in ResiTTA stands for Resilient, not residual. The method's resilience refers to its ability to maintain rapid adaptation capabilities while mitigating overfitting
10. Categorizing EcoTTA as "Memory-Based" could be misleading. A more accurate classification would be a "Memory-Efficient TTA System".

### Questions
1.  **On $K_{max}=1$ for CIFAR10-C:** The finding in Table 7 that $K_{max}=1$ is optimal for CIFAR10-C seems to undermine the paper's core thesis. How do the authors reconcile this? Does this not imply that for some standard benchmarks, the proposed multi-cluster mechanism is actually *harmful* compared to a well-tuned single-cluster baseline (which $K_{max}=1$ effectively is)? Could the authors elaborate on why they believe this happens and what it means for the generality of MCM?
2.  **Justification for Adjacent-Only ACC:** Could the authors provide a stronger justification for limiting ACC to *adjacent* clusters in the creation sequence? Was this choice made purely for computational efficiency (i.e., avoiding a $K^2$ comparison)? An ablation comparing this "adjacent-merge" strategy to a "global-best-pair-merge" strategy (in terms of both accuracy and runtime) would be highly valuable to understand the trade-offs of this design choice.
3.  **Descriptor Robustness:** Given the acknowledged weakness of the current descriptor for noise-based corruptions, have the authors explored alternatives? For example, would using descriptors from a different feature space (e.g., earlier or later in the network) or incorporating other statistics (like histograms or higher-order moments) improve robustness to these shifts without incurring significant computational overhead?
4.  **RSR Sensitivity:** The RSR mechanism selects clusters based on their relevance to the current mini-batch. How sensitive is this process to a highly noisy or anomalous mini-batch? Could a single "bad batch" cause RSR to retrieve unhelpful clusters and trigger negative adaptation?
5. **TRIBE+MCM**: In Table 1, TRIBE without MCM does best for CIFAR-10C. Is it possible to have MCM with TRIBE and have the authors conducted any such experiment?
6. **Sensitivity to hyperparameters**: How are the hyperparameters, such as $K_max$ and threshold $\tau$ for minimum distance to create a new cluster, set in practice? Is it tuned for every dataset, using the test-time adaptation performance? If so, does it mean that the generalizability of the proposed approach is questionable for real-world datasets where we do not have access to labels for the test set? Also, a key hyperparameter $N_S$ (line 297) for relevant cluster selection seems underexplored.\


**Corrections**
1. Fix the subscripts such as: Ci should be C_i in line 260-261; Bt in line 296
2. No space/subscript: Mretrieve (line 298)

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper addresses the lack of representativity and adaptability in memory-based test-time adaptation (TTA) methods. It introduces Multi-Cluster-Memory (MCM), a memory mechanism that maintains multiple clusters of previously seen samples instead of a single global buffer. Each cluster stores samples along with their statistical descriptors, and new samples are either assigned to the nearest existing cluster or used to start a new one. When the maximum number of clusters (Kmax) is reached, an Adjacent Cluster Consolidation (ACC) step merges the most similar clusters to free space. During adaptation, the model retrieves samples from the most relevant clusters through a Relevant Sample Retrieval (RSR) step, based on proximity to the current batch. Experiments on CIFAR-10-C, CIFAR-100-C, ImageNet-C, and DomainNet show that MCM improves over standard single-cluster memory-based baselines.

### Strengths
* The technical proposal of maintaining multiple evolving clusters for online adaptation is conceptually interesting.
* Results are presented on multiple datasets, including both corruption-based (CIFAR-C, ImageNet-C) and domain shift benchmarks (DomainNet).

### Weaknesses
* The paper is difficult to follow, especially in describing the TTA setups and the details of method.

* The ablation shows that gains from using multiple clusters are modest (less than 2 % improvement between Kmax = 1 and 5). The evidence is insufficient to justify the added complexity.

* The authors criticize sequential updates (L80, L304), but it’s not clear how MCM avoids updating samples sequentially, given that samples arrive sequentially.

* Large variability in error reductions (Table 1) — across methods, datasets, and even within the same method or dataset.

### Questions
* What exactly is Practical TTA? Does it refer to the case of non-IID test samples? If yes, what is the setup to realize this?

* In Eq. 1, what do the student model fₛ and teacher model fₜ represent in the context of the TTA setup?

* In what sense is MCM cost-efficient compared to the single-cluster memory approach?

* Could you confirm that when the cluster limit is reached, only two clusters are merged, freeing a single slot so that the number of clusters becomes Kmax − 1?

* Is Kmax = 1 equivalent to the single-cluster memory baseline?

* What is the retrieval mechanism for the single-cluster case, and how does MCM + RSR compare directly to Single-Cluster + RSR?

* What factors explain the high intra-dataset or intra-method variability in the performance gains reported in Table 1?

### Soundness
2

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
The paper proposes Multi-Cluster Memory (MCM) for better long-term adaptation under continual and partial test-time adaptation. Instead of keeping a single-cluster memory of confident test samples, MCM organizes them into clusters using channel-wise statistical descriptors and assigns each new sample to the nearest cluster or creates a new one. To bound memory, it introduces Adjacent Cluster Consolidation (ACC) that merges the most similar adjacent cluster pair, retains the N lowest-uncertainty samples and evolves over time with domain changing. For adaptation, Relevance-guided Sample Retrieval (RSR) selects clusters most similar to the current mini-batch and trains with the standard Mean-Teacher consistency objective. Experiments on CIFAR-10/100-C, ImageNet-C, and DomainNet under PTTA show consistent improvements over memory-based baselines (RoTTA, PeTTA, ResiTTA).

### Strengths
- **Targeted retrieval designed for PTTA.** RSR ties the adaptation batch to clusters most similar to the current stream, which is conceptually aligned with continually changing-domain  shifts and avoids mixing unrelated modes during self-training.  
- **Modular idea; simple but effective mechanics.** Using per-channel mean/variance descriptors to structure memory is compute-light and easy to integrate, which is a good fit with TTA and easy to integrate with current memory-based methods.
- **Consistent gains across backbones and baselines.** The recurring-shift evaluation also shows stability under long time adaptation.

### Weaknesses
- **Novelty is incremental relative to prior "memory-based TTA".** The main contribution is structuring the memory with simple descriptors and adding an ACC/RSR policy to replace the standard single memory. While practical, the core mechanics for memory design (distance-based assignment, merging, uncertainty-aware replacement) resemble standard online clustering and memory curation ideas; stronger positioning versus alternative memory-bank designs would help. 
- **Scope of descriptors.** The method fixes channel-stat descriptors; it’s unclear how robust MCM is to the choice of layer or to alternative descriptors (e.g., low-dimensional embeddings or BN running stats). Also I am curious if Transformer-based model can gain from MCM design.
- **Limited component ablations** It seems like there lacks a clean ablation that removes ACC or RSR individually (beyond hyperparameter sweeps). A table with “no-ACC / no-RSR / both” would clarify where the gains come from.
- **Limited visualization and interpretability.** The paper only provides one qualitative illustration (Fig. 1) to motivate representativeness vs. adaptability. More visual analyses—e.g., memory distributions or cluster evolution plots across different datasets and stages—would make the benefits of MCM over single-memory schemes more tangible and convincing.
- **Sensitivity to Hyperparameters.** Although MCM is described as a plug-and-play module, its performance depends on several hyperparameters.

### Questions
1. It would be helpful if the paper could provide quantitative results on runtime and memory usage—such as end-to-end throughput  and peak GPU memory—for MCM compared to a single-cluster memory of equivalent total capacity.
2. In Table 1, TRIBE performs surprisingly well, in some cases matching or even surpassing MCM. Could the authors elaborate on possible reasons for this—e.g., differences in adaptation schedule, update frequency, or experimental setting?

### Soundness
3

### Presentation
3

### Contribution
2
