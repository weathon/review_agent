# Machine Unlearning under Retain–Forget Entanglement

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 8

## Abstract
Forgetting a subset in machine unlearning is rarely an isolated task. Often, retained samples that are closely related to the forget set can be unintentionally affected, particularly when they share correlated features from pretraining or exhibit strong semantic similarities. To address this challenge, we propose a novel two-phase optimization framework specifically designed to handle such retain–forget entanglements. In the first phase, an augmented Lagrangian method increases the loss on the forget set while preserving accuracy on less-related retained samples. The second phase applies a gradient projection step, regularized by the Wasserstein-2 distance, to mitigate performance degradation on semantically related retained samples without compromising the unlearning objective. We validate our approach through comprehensive experiments on multiple unlearning tasks, standard benchmark datasets, and diverse neural architectures, demonstrating that it achieves effective and reliable unlearning while outperforming existing baselines in both accuracy retention and removal fidelity.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper focuses on machine unlearning, particularly focusing on the performance of the retain set samples that are highly related to the forget set. The forgetting is achieved in two steps. The first stage is very similar to the usual forgetting formulation to balance out retaining and forgetting, except that an adaptive weight is applied. The second stage uses a loss based on Wasserstein-2 to constrain the distributional change with respect to the forget set, which then forms a span for the retain set gradients to perform gradient projections.

### Strengths
The paper is written quite clearly.

Theoretical support for the paper seems to be substantial, and there is a bound on the accuracy of the forget set after PGD. Such theoretical support is good, although I did not check the derivations in too much detail.

### Weaknesses
Overall, while I get the motivation of trying to retain good performance on difficult samples, i.e., the samples in the retain set that are highly related to samples in the forget set, I am not quite convinced of the usefulness of such a setting. For instance, one needs to further define what are the “closely related” samples and pick some out for such further training, which seems somewhat arbitrary. Such picking of samples may be more straightforward in cases with clear classes (e.g., in CIFAR classification) and for class-level unlearning, but may be difficult to transfer to more general settings, e.g., when individual samples need to be forgotten.

Although the explanation for the gradient projection with Wasserstein distance regulation is mostly clear, I feel like some intuition is missing. For instance, what does it mean to project the adjacent set gradients onto the orthogonal complement of the space spanned by the forget and retain gradients? Furthermore, why must it be the span of those specific forget and retain gradients? 


There are also some weaknesses in the experiments section, which have been placed in the “Questions” section.

### Questions
In the experiments, the compared methods, while representative, seem slightly outdated. For instance, SCRUB, SalUn and SSD were all available online already in 2023, and it has been more than two years. Also, the most recent among them was officially published in AAAI 2024, which is in Feb 2024.

At the same time, there seems to be an error in the citations for SCRUB:
Towards Unbounded Machine Unlearning is a NeurIPS 2023 paper, not 2024.


The authors can consider reporting results and/or comparing with more recent works such as:

Learning to Unlearn for Robust Machine Unlearning. ECCV 2024

Adversarial Machine Unlearning. ICLR 2025

Decoupled Distillation to Erase: A General Unlearning Method for Any Class-centric Tasks. CVPR 2025

LoTUS: Large-Scale Machine Unlearning with a Taste of Uncertainty. CVPR 2025

MUNBa: Machine Unlearning via Nash Bargaining. ICCV 2025


Often, machine unlearning works report the “upper bound” performance, which is the gold model: a complete retraining of the models only on the retain set. This shows us how far we are away from that upper bound. The authors should provide this.

Many works also report the “Unlearning Time”, which is the amount of time used for unlearning. This is important, since a complete retraining of the model would be ideal, but would consume too much resources. The authors should provide this.

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
The paper proposes a two-stage optimization framework for correlation-aware unlearning. Stage 1 establishes forgetting while preserving performance on remote retain samples. Stage 2 recovers performance on adjacent retain samples while provably constraining the forget-set loss distribution via Wasserstein-distance regularization (W2). Experiments on CIFAR-100, Tiny-ImageNet, and ToxiGen across multiple model backbones demonstrate the effectiveness of the proposed method on both adjacent and remote retain subsets.

### Strengths
1. The problem is well defined, the experimental setup is sound, and the results of the proposed method are promising; the ablation studies are comprehensive.

2. For Stage 2, the analysis of why vanilla PGD fails is interesting and motivates the use of W2 regularization.

### Weaknesses
1. The theoretical components are doubtful, but Proposition 4.1—which relates the update to the learning rate and claims the modified loss remains largely unchanged—does not fully convince; stronger bounds or clearer assumptions would help establish this as evidence.

2. Notation could be clearer: prefer $L\left(\theta ; D_r^{\mathrm{adj}}\right)$ instead of $L_r^{\mathrm{adj}}(\theta)$, and likewise $L\left(\theta ; D_r^{\mathrm{rem}}\right)$ and $L\left(\theta ; D_f\right)$.

### Questions
1. Could you give more explanations about the failure of PGD?

2. What is $m$ in Proposition 4.2.

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
4

### Summary
The paper tackles retain–forget entanglement in machine unlearning: forgetting a targeted subset often harms performance on semantically adjacent retained examples. The authors formalize the retain set as two parts, adjacent (correlated with the forget set) and remote (less related), and propose a two-stage optimization framework (TMU). Stage 1 uses an augmented Lagrangian to increase loss on the forget set while constraining performance on the remote retain set. Stage 2 applies projected gradient descent on the adjacent retain set but augments the projection with a Wasserstein-2 (W2) loss-distribution regularizer so that improvements on adjacent retains do not "re-enable" the forget set (i.e., keep forget accuracy low by maintaining the post-Stage-1 loss distribution). The paper provides propositions showing that Stage 2 decreases adjacent-retain loss while keeping changes to the forget/remote objectives second-order, and derives an upper bound on forget-set accuracy under the W2 constraint. Experiments on CIFAR-100 (ResNet-18), Tiny-ImageNet (ViT), and ToxiGen (RoBERTa-base) show lower forget accuracy with competitive or better retain performance than baselines (Fine-tune, GA, SCRUB, SSD, SalUn, L1-sparse). Ablations indicate W2 regularization is key; membership-inference evaluation reports high "unlearning privacy" for the proposed method.

### Strengths
1. Explicitly decomposing the retain set into adjacent vs. remote is well-motivated and aligns with observed collateral damage patterns in unlearning. 
2. Stage 1’s augmented-Lagrangian constraint is a clean way to preserve remote-retain performance without manual trade-off tuning; Stage 2’s W2-regularized projection addresses the empirical failure mode of plain PGD (mean-loss conservation but accuracy creep on the forget set). 
3. Propositions articulate why adjacent-retain improves while forget/remote are (first-order) preserved; the accuracy bound connects the W2 constraint to forgetting fidelity. 
4. Results span vision and language with multiple architectures; tables consistently show strong forget fidelity with limited adjacent-retain degradation, plus a clear W2 ablation and a privacy (MIA) check.

### Weaknesses
1. The method presumes a way to define and extract the adjacent subset (e.g., via superclasses or group semantics). Practical pipelines may not expose such structure; guidance or automated discovery (e.g., metric-learning neighbors) is not fully developed. 
2. While datasets are standard, scenarios remain largely supervised classification with relatively clean adjacency (superclass/subclass or group semantics). It’s less clear how well the approach extends to open-world or multilabel settings, representation-level forgetting, or generative models. 
3. Two stages (with $\lambda$, $\mu$, $\alpha$, step sizes, and projection subspaces) add complexity. The ablations help, but a deeper resource/efficiency analysis and stability study across wider seeds/tasks would strengthen claims of robustness. 
4. The accuracy bound leverages conditions (e.g., sufficiently large losses post-Stage-1 and batchwise W2) that may not strictly hold in streaming or highly imbalanced forget sets; practical guidance for setting $\alpha$ and monitoring $\epsilon$ is limited. 
5. A growing body of work frames unlearning as a multi-objective / multi-task optimization between forget and retain objectives, using gradient projection or surgery ([1], [2]), robustness-oriented or meta-learning variants ([3], [8]), parameter/connection sensitivity ([4]), bargaining/game-theoretic scalarization ([6]), and post-unlearning alignment or re-learning prevention in generative models ([5], [7], [9]). These methods surface well-known issues—gradient interference, oscillatory updates, sensitivity to weights in scalarization, and Pareto inefficiency—that your two-stage constrained-then-projected approach is implicitly designed to mitigate. However, the paper does not explicitly position TMU (proposed method) against this family or analyze when/why its W2-regularized Stage-2 projection outperforms (or fails relative to) gradient-rectified updates, Nash bargaining, or meta-unlearning schedules. A tighter comparative discussion, plus side-by-side experiments under a common protocol (same forget/retain splits, identical budgets), with diagnostics such as gradient-angle statistics, Pareto AUC, and "re-enablement" rates, would clarify the contribution boundary and strengthen claims over multi-task baselines.

[1] Learn to Unlearn for Deep Neural Networks: Minimizing Unlearning Interference with Gradient Projection, WACV 2024. 

[2] GDR-GMA: Machine Unlearning via Direction-Rectified and Magnitude-Adjusted Gradients, ACMMM 2024. 

[3] Learning to Unlearn for Robust Machine Unlearning, ECCV 2024. 

[4] Scissorhands: Scrub Data Influence via Connection Sensitivity in Networks, ECCV 2024. 

[5] Boosting alignment for post-unlearning text-to-image generative models, NeurIPS 2024. 

[6] MUNBa: Machine Unlearning via Nash Bargaining, ICCV 2025. 

[7] Meta-Unlearning on Diffusion Models: Preventing Relearning Unlearned Concepts, ICCV 2025. 

[8] Learning to Unlearn while Retaining: Combating Gradient Conflicts in Machine Unlearning, ICCV 2025. 

[9] GRU: Mitigating the Trade-off between Unlearning and Retention for LLMs, ICML 2025

### Questions
1.  How robust is TMU to noisy adjacency (e.g., approximate neighbors via embeddings)? A study where adjacency is inferred (not oracle) would boost practical relevance. 
2. Provide a simple recipe for setting $\alpha$ and stopping criteria for Stage 2 (e.g., target W2 window + retain gains plateau) to avoid overfitting adjacent retains. 
3. Report wall-clock and memory overhead vs. SalUn/SSD/SCRUB on larger backbones and longer runs; clarify sorting cost for empirical W2 and any minibatch bias. 

The paper identifies an important practical failure mode (retain–forget entanglement), proposes a simple-yet-effective two-stage remedy grounded in standard optimization plus a neat W2 distributional guardrail, and backs it with solid empirical evidence and helpful theory. The main asks are around adjacent-set construction, scalability/sensitivity details, and broader realistic applicable scenarios. Addressing these would elevate the contribution further.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses retain–forget entanglement in machine unlearning, where removing a forget set unintentionally harms correlated retained data. It introduces a two-stage framework: (1) an augmented Lagrangian step to forget targeted data while preserving unrelated samples, and (2) a Wasserstein-regularized gradient projection to recover performance on correlated retained data. Experiments on CIFAR-100, TinyImageNet, and ToxiGen show that the method achieves strong forgetting while maintaining retention accuracy, outperforming prior baselines.

### Strengths
1. The paper pinpoints the practical challenge that forgetting one subset of data (e.g., a specific subclass or biased group) can harm semantically related retained samples, a scenario highly relevant for real-world fairness and bias correction.

2. Ablation studies clarify the importance of the Wasserstein-2 term.

3. The experimental design is well-defined and comprehensive, matching the problem setup.

### Weaknesses
1. The method relies on manual partitioning of the retained set into “adjacent” and “remote” subsets, whether it assumes prior knowledge of semantic relationships is unknown; If so, this decomposition may be impractical or ambiguous in large, unstructured datasets. 

2. The experimental evaluation on vision task is limited to small scale datasets.

3. I did not get the meaning of theoretical bounds.

### Questions
How is the adjacent vs. remote retain set identified in practice for complex or unlabeled datasets?

### Soundness
3

### Presentation
3

### Contribution
3
