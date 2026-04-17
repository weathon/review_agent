# SNOWFL: Efficient and Heterogeneous Federated Learning with SNIP-Owen-values

- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
Cross-device federated learning often faces heterogeneous clients. These clients carry data with very different values for training high-performance, generalized global models, calling for effective contribution estimation mechanisms. Width scaling with thinner subnetworks and depth scaling via early exits enable participation for heterogeneous clients but still suffer from (i) noisy aggregation across mismatched subnetworks, (ii) under-trained deep layers when few clients reach them, and (iii) costly, client-isolated contribution estimates. We propose SNOWFL, which pairs server-side single-shot pruning at initialization pruning (SNIP) with coalition-structured Owen valuation. SNIP uses a small public, unlabeled set to score connections by loss sensitivity and produce layer-consistent width masks per tier aligned with fixed early exits. During training, we estimate client contributions by first computing Owen values for coalitions and then allocating credit within each coalition via update alignment and diversity. These contribution estimates will be used in both weighted aggregation and drive capacity-aware reassignment. We prove nonconvex convergence to stationarity and, under strong convexity on the retained subspace, linear convergence to a neighborhood. Under matched FLOPs and parameter budgets, SNOWFL achieves state-of-the-art accuracy on vision and language benchmarks, improving strong heterogeneous baselines by up to 15%, while valuation remains data-free except for the small public samples used once for initialization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents an integrated framework that combines system heterogeneity and contribution estimation in federated learning. First, before training, the method measures channel-wise saliency using public data to pre-select which channels to prune and where to place exit points. For each tier group corresponding to a different resource capability, it then fixes the channel masks and exit positions that can fit within the given resource budget. Second, the framework measures client contributions in two stages: first, it evaluates the contribution of each tier as a whole, and then it measures the contribution of individual clients within each tier. The measured contributions are then used to compute aggregation weights during training or to reassign clients across tiers.

### Strengths
* It is novel in that it simultaneously considers both contribution estimation and system heterogeneity in federated learning.

* It leverages only the gradients on public data and the local updates, without incurring any additional privacy cost.

* The experiments demonstrate that the proposed method outperforms system heterogeneity–related baselines under the same resource budget.

### Weaknesses
**Necessity of public data.** While the reliance on public data can itself be considered a limitation, a more concerning issue is that pruning is performed before training. This requires the assumption that the public dataset used for pruning adequately represents the actual client data used during training, an assumption that is unlikely to hold in practice. Even in BN calibration, the approach relies on the assumption that the public data are well aligned with the private client data.

**The contribution of the saliency-guided pruning at initialization is not sufficiently convincing.** Fixing both the network width and depth before training makes the method overly dependent on public data, and the experimental results provided are insufficient to adequately analyze its effectiveness.

**Although many of the detailed components of the method are empirically determined, the paper does not provide sufficient discussion or experimental evidence to justify these choices.** For example, in Equation (10), the peer-diversity term and the global alignment term appear somewhat conflicting, and several hyperparameters are introduced without any accompanying ablation study. Moreover, the experiments mainly present overall results, offering only limited evidence of the proposed method’s effectiveness.

**Writing.** The paper lacks sufficient detail in describing the method and experimental settings. For example, the client reassignment section is not clearly explained, and it remains unclear how each client’s resource capability is constrained or how the public dataset is constructed and utilized.

### Questions
* Given the models and datasets used, the absolute accuracy values reported in Table 1 (the main results) seem unusually low. What could be the reason for this?
* What is the purpose of applying clipping in Equation (7)?
* How were the clients’ resource budgets constrained in the experimental setting?
* In SnowFL, client reassignment appears to ensure that each sub-model is trained not only on fixed clients’ data but also on data from a diverse set of clients. Under what conditions or environments were the baselines evaluated?

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
1

### Summary
The SNOWFL mainly combines three concepts into a single heterogeneous FL framework, 1) width-reduction (prunning), 2) depth-reduction (multi-exit networks), and 3) client valuation and weighted aggregation. Overall, although none of the three are new, a unified framework that integrates all of them can be regarded as a fairly strong contribution. The paper mostly focuses on the third component, how to prune before starting the training using minimal global data and how to aggregate the locally trained models based on their contributions. The proposed framework is compared with some recently proposed heterogeneous FL methods and theoretically analyzed in terms of grouping convergence where Owen value matches the Shapley value.

### Strengths
1.  As described in paper summary, this work provides a unified parameter-efficient heterogeneous FL framework which combines three different techniques. I appreciate the general framework and analysis.

2. The Owen-style grouping mechanism is not popularly used in FL community, and introducing the technique to FL community is a strong contribution.

3. The paper contains solid theoretical analysis (though it is not included in the main text).

### Weaknesses
1. The SNOWFL mainly combines three concepts into a single hetergeneous FL framework, 1) width-reduction (prunning), 2) depth-reduction (multi-exit networks), and 3) client valuation and weighted aggregation. Overall, although none of the three are new, a unified framework that integrates all of them can be regarded as a fairly strong contribution. The paper mostly focuses on the third component, how to 

2. The authors argue that the depth values (exit placements) are fixed at the architecture level. However, it is not convincing because the depth directly determines how much system resources are required to train the network. The prunning is applied to individual subnetworks because the depth is assumed to be determined in advance. I believe this design overly simplifies the problem. Users may need to determine the appropriate width and depth jointly, taking into account their available resources such as memory capacity or network bandwidth. In this case, Phase 1 of SNOWFL may need to be modified.

3. Based on my experiences, BN statistics quality plays a key role in achieving good model accuracy. However, the empirical study in this paper does not show its impact of BN calibration. It appears as a single subsection, and thus I assume it is a critical component of SNOWFL. How is the performance affected by this calibration? The authors should provide more empirical results regarding this feature.

4. Section 4.5 looks redundant. SNOWFL employs SNIP and it has been discussed already. Per-round valuation cost would be better to be discussed in section 4.3 to make the section self-contained. Privacy also can be discussed when introducing each phase.

5. The theoretical analysis is critical information which supports the efficacy of Owen valuation and Shapley allocation, but it only appears in the appendix. I understand that the allowed page count is insufficient always, however the authors should have included at least the summary or the key results in the main text. Currently, due to the lack of any theoretical justifications, the proposed method is not convincing enough.

6. The empirical study also has several issues. First, the comparison lacks a few critical related works in heterogeneous FL, shown as follows. FjORD is a representative width reduction-based heterogeneous FL method and EmbracingFL is a recently proposed depth reduction-based heterogeneous FL method. Comparing SNOWFL with them will make the empirical results more powerful.

[1] Horvath et al., FjORD:  fair and accurate federated learning under heterogeneous targets with Ordered Dropout, NeurIPS 2021.

[2] Lee et al., Embracing FL: Enabling Weak Client Participation via Partial Model Training, IEEE Trans. on Mobile Computing, 2024.

7. While the empirical results look promising, the range of experimental settings is too limited. There are only two tables that directly compare SNOWFL with other heterogeneous FL methods in terms of model accuracy. What about the effectiveness of the proposed Owen valuation and Sharpley allocation as compared to other parameter valuation or weighted aggregation methods? E.g., is it always better than gradient-norm based parameter importance metrics [3,4]? What about the weight-to-gradient ratio in [5]? What about just a simple uniform random sampling or loss-based aggregation? There are many interesting and critical experiments that should have been considered. I see some ablation studies are discussed in the appendix, but they do not include these key results.

[3] Li et al., Enhancing Large Language Model Performance with Gradient-Based Parameter Selection, arXiv.

[4] Zhang et al., Gradient-based Parameter Selection for Efficient Fine-Tuning, CVPR 2024.

[5] Kim et al., layer-wise update aggregation with recycling for communication-efficient federated learning, NeurIPS 2025.

8. Algorithm 1 is written too verbally. Some lines could be replaced with just pointing out equations. Its readability is seriously poor.

9. Overall, Section 2 and 3 take up too much space and it results in pushing key results to the appendix. I strongly recommend re-writing those section concisely and bring the important results back to the main text.

Due to the several limitations above, I cannot give a positive score for now. I will check the rebuttal and re-evaluate this work.

### Questions
My questions are included in the above weakness section. Please carefully address them.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents SNOWFL, a framework designed to make federated learning (FL) more efficient and robust to device heterogeneity. The method assigns clients subnetworks of different sizes using a one-time SNIP-based pruning step at the server, which generates tiered masks that define each client’s capacity level. A small public or unlabeled dataset is used for batch normalization calibration to ensure consistent activation statistics across tiers. To evaluate client contributions fairly, SNOWFL employs an Owen-value–based scheme, which first measures the collective contribution of each tier and then distributes value within the tier based on gradient alignment and diversity. The framework aims to reduce training cost and improve fairness without additional communication overhead.

### Strengths
1. The paper addresses an important practical issue in cross-device FL: heterogeneous compute and how to engage weak devices without hurting the global model. 
2. The pipeline is relatively easy to implement, making it more deployable in practice. 
4. Experiments are broad and include ablations that show each component contributes.

### Weaknesses
The paper rediscovers an important fact that consensus of masks is essential in model pruning in FL. The paper compares to several heterogeneity baselines (e.g., DepthFL) but omits many other relevant works: SparseFL [1], EmbracingFL [2], PriSM [3], etc. 

Particularly, SparseFL was the first work that demonstrates that even when data across clients is significantly non-IID, a consensus in sparsity masks for local training is essential. [1] develops a consensus strategy without requiring any public datasets. This work simply generalizes the idea to having more than one consensus mask. 

EmbracingFL proposed a new idea, where instead of doing early exit, one can instead allocate only the output side subset of layers to clients for local training. In EmbracingFL, one doesn't need additional BN harmonization as the method implicitly takes care of that. Works like these and their follow ups have been ignored in the paper. 

PriSM proposes a SVD based model principal component dropout strategy for creating sub-models for clients that are good approximations of the global model. Additionally, clients' models together provide an excellent coverage of the all the principal components of the global model, thus providing an effective way to preserve global model performance even in heavily resource (compute/memory/communication) constrained settings. 

Other weaknesses are as follows:

1. Heavy reliance on a public / unlabeled dataset at server: SNOWFL’s SNIP masks and BN “harmonization” use a public set. This is central to performance but is unrealistic in many cross-device settings, introduces clear bias risks (public set not representative), and creates an attack surface (adversarial or poisoned public set). The paper notes the choice but does not quantify robustness to different or adversarial public sets.
2. Novelty is incremental: Components (SNIP pruning at init, early exits, contribution weighting via Shapley/Owen) are all existing techniques; SNOWFL’s contribution is their combination plus some pragmatic design choices. The paper lacks a new algorithmic/principled mechanism that meaningfully advances the state of the art beyond engineering integration. The theoretical results are also boilerplate adaptations of standard FL proofs to masked exits.
3. Computation & scalability of per-round valuation. Tier-level Shapley via MC permutations and the within-tier allocation are costly (authors note this can be reduced), but the practical wall-clock cost, memory and communication overheads are not measured. For large settings, this could be prohibitive. The assumptions used in convergence theorems are too strong for practice.
4. Sensitivity and robustness analyses are limited and key hyperparameters lack sensitivity studies. The ablations show removing SNIP/Owen hurts, but they don’t show failure modes (e.g., poor public set → collapse).
5. Empirical claims are uneven across datasets. The paper should acknowledge where it doesn’t dominate with justifications.

[1] Revisiting Sparsity Hunting in Federated Learning: Why does Sparsity Consensus Matter? (TMLR 2023)
[2] Embracing federated learning: Enabling weak client participation via partial model training. (IEEE TMC 2024)
[3] Overcoming Resource Constraints in Federated Learning: Large Models Can Be Trained with only Weak Clients. (TMLR 2023)

### Questions
Please address the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
SNOWFL (SNip-OWen-values Federated Learning) addresses heterogeneous federated learning where clients have varying computational capabilities. The paper's key innovation combines two main components: a) SNIP-based pruning at initialization: Uses server-side Single-shot Network Pruning to create task-aware, layer-consistent width masks for different client tiers, aligned with fixed early exits. This is done once using a small public/unlabeled dataset, avoiding expensive iterative pruning; b) Owen value-based contribution estimation: Extends Shapley values to coalition structures (client tiers), first computing group-level contributions via quotient-game Shapley, then allocating within groups based on update alignment and diversity. These contributions drive both weighted aggregation and capacity-aware client reassignment. Under matched FLOPs and parameter budgets across vision (CIFAR-10/100, FEMNIST) and language (Shakespeare) benchmarks with non-IID data, SNOWFL achieves state-of-the-art accuracy, improving over strong baselines by up to 15% relative improvement. Ablations confirm both components contribute (Owen has larger standalone effect), and the paper provides convergence guarantees showing nonconvex convergence to stationarity and linear convergence under strong convexity.

### Strengths
1. Good empirical results - The quantitative improvements are substantial: up to 15% relative gain (9.1 absolute points on CIFAR-10 α=0.1: 45.9% vs 36.9%) represents meaningful progress over recent strong baselines. The consistency across datasets (vision and language) and heterogeneity levels strengthens the claim. 

2. Comprehensive evaluation: Authors thoroughly validate the experimental design with ablations isolating components (Table 3), sensitivity studies (M, T_reg), per-tier analysis, and reproducible code

3. Novel synthesis: While neither component is new individually, their integration is creative: (1) using SNIP server-side with public data to generate tier-compatible subnetworks is a fresh take on pruning-at-initialization in FL, avoiding the client-data dependency and iterative retraining of prior work; (2) adapting Owen values to naturally arising coalition structures (client tiers) rather than treating clients independently is conceptually elegant and computationally sensible

### Weaknesses
1. High complexity: 10+ hyperparameters (M, T_reg, T_warm, γ_t, α_t, ρ, λ_b, etc.), multi-stage pipeline (Phase I SNIP + Phase II Owen + BN calibration), coordinate-wise masked aggregation—significant implementation burden without clear tuning guidance

2. Uneven gains: FEMNIST improvement negligible (84.2% vs 84.2% ReeFL), slower early convergence (Figure 1), no statistical testing or error bars—unclear when SNOWFL helps vs simpler methods

3. Public data dependency: Requires task-relevant public/unlabeled set for SNIP and BN calibration; sensitivity to set size/quality not studied; may not be available or well-matched in practice

4. Incomplete efficiency analysis: No wall-clock runtime, communication cost, or per-round overhead comparison; Owen valuation cost (M permutations) not quantified vs baselines

### Questions
1. When does SNOWFL help? Why marginal gains on FEMNIST but strong on CIFAR? Can you characterize problem settings (data heterogeneity, model capacity, tier count) where SNOWFL outperforms simpler baselines like ReeFL?

2. Simplified variant? Can you ablate to "minimal SNOWFL" (e.g., fixed tiers + uniform Owen, or SNIP-only without contribution weighting) to isolate essential components and reduce hyperparameter burden?

3. Public data sensitivity: How does performance degrade with smaller/mismatched public sets? What happens if public data is unavailable : can synthetic data or server-side aggregates substitute?

4. Practical overhead: What are actual wall-clock training times and communication costs vs baselines? How does Owen valuation cost scale with M, G, |S_t|—is it negligible or a bottleneck?

### Soundness
3

### Presentation
3

### Contribution
2
