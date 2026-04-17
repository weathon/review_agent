---
job_id: 53c2f327-d5e6-474c-9b59-050f4244ca08
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: qkKUG56s5r.pdf
paper: Automatic Complementary-Separation Pruning for Efficient CNNs
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper proposes a pruning method for CNNs, squarely within efficient deep learning / representation learning, which fits ICLR’s scope.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, Methodology, Experiments, Results, Conclusion) are present, written in English, and the work is technically coherent with non‑trivial experiments on standard architectures and datasets; I do not see fatal methodological or statistical errors that would justify an immediate desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any hidden prompts, manipulative instructions to reviewers, or suspicious invisible content within the provided manuscript.

---

# Expected Review Outcome:

## Summary

The paper introduces Automatic Complementary Separation Pruning (ACSP), a post‑training pruning method for CNNs that combines activation‑based and structured pruning. For each layer, ACSP builds a “graph space” where each component (channel/neuron) is represented by its Jeffries–Matusita separability vector over all class pairs, then uses k‑Medoids clustering, an MSS (Mean Simplified Silhouette) index, and a knee‑finding procedure to automatically choose both how many components to keep and which components to retain (highest‑weight element per cluster). Experiments on CIFAR‑10/100 and ImageNet across several architectures show FLOP reductions of roughly 1.5–2.5× and modest wall‑clock speedups with small accuracy drops or gains compared to several pruning baselines.

## Strengths

1. **Conceptually interesting “separation graph” formulation.**  
   The construction of a per‑layer “graph space” where each component is represented by its separability vector over all class pairs (Section 3.3.1, Equations (1)–(2), Figure 1) is an appealing, interpretable idea. It goes beyond simple magnitude or activation statistics and explicitly encodes how a neuron/channel separates different classes. This gives the method a clear semantic grounding and may be useful beyond pruning (e.g., for analysis or feature selection).

2. **Automatic determination of pruning ratio per layer.**  
   ACSP removes the need to manually specify pruning ratios by scoring all subset sizes \(k \in [2, N_i]\) through MSS and then using the Kneedle algorithm to find a “knee point” (Sections 3.3.2, 3.4.1). This addresses a real pain point in many pruning pipelines, where practitioners have to sweep sparsity levels empirically. Algorithm 1 clearly codifies the full procedure.

3. **Complementary selection principle via clustering.**  
   The use of k‑Medoids on the graph space to enforce diversity, and then choosing the highest‑weight component from each cluster (Section 3.4.2, Figure 2), is a reasonable operationalization of “complementary separation capabilities.” Figure 2 is particularly helpful: the left panel visually shows how medoids and high‑weight points differ in a 2‑D projection of the component space, clarifying the rationale for selecting high‑weight points per cluster rather than medoids themselves.

4. **Broad empirical coverage with strong aggregate results.**  
   Table 1 is extensive: it covers CIFAR‑10/100 and ImageNet‑1K, and multiple architectures (VGG‑16/19, ResNet‑56/50, DenseNet‑40, MobileNet‑V2). Across this table, ACSP is often best or second best either in accuracy change or speed‑up, and frequently achieves both good accuracy and strong FLOP reduction. For instance:
   - CIFAR‑10 MobileNet‑V2: ACSP achieves the best accuracy gain (+0.50%) and highest speed‑up (1.93×).
   - CIFAR‑10 VGG‑16: ACSP nearly matches the best accuracy gain (+0.37% vs +0.46%) while achieving the best speed‑up (2.59×).
   - ImageNet ResNet‑50: ACSP reaches a 2.25× FLOP reduction with +0.59% accuracy, competitive with the best baselines.
   This breadth gives reasonable confidence that the method is not tuned to a single architecture.

5. **Real wall‑clock timing measurements.**  
   Many pruning papers stop at FLOPs; here, Table 2 reports batch and single‑sample inference latency on real hardware. The authors explicitly note the gap between FLOP speed‑up and wall‑clock speed‑up and still show consistent latency reductions (e.g., MobileNet‑V2 on CIFAR‑10: −20.39% batch time, −2.62% single‑image time; ResNet‑50 on ImageNet: −6.32% batch, −8.07% single). This strengthens the practical relevance of the claims in the abstract about focusing on inference‑time efficiency.

6. **Method largely orthogonal to training procedure.**  
   ACSP operates post‑training using a modest amount of data for activations and a short fine‑tuning phase (Section 4.1), which is commonly acceptable in deployment scenarios. It does not require modifying the training objective, adding auxiliary losses, or training specialized controllers, which is practically appealing compared to some auto‑pruning and RL‑based approaches.

7. **Clarity and structure of exposition.**  
   Despite some notation glitches, the core pipeline is relatively easy to follow: Figure 1 walks through graph construction; Algorithm 1 ties the pieces into a layer‑wise procedure; Section 3.3.2 and the MSS explanation give sufficient intuition about the selection criterion. Overall writing is clear enough to reimplement the method with some effort.

## Weaknesses

1. **Novelty and positioning versus existing automated / activation‑based pruning are underdeveloped.**  
   While the idea of using per‑class‑pair separability vectors is interesting, many elements of ACSP echo existing themes: activation‑based importance scores, clustering‑based selection, and automated sparsity determination. The introduction briefly mentions AutoPrune (Xiao et al., 2019) and RL‑based search (He et al., 2018b; Liu et al., 2019), but the paper does not carefully articulate how ACSP fundamentally differs from or improves on these and other automated pruning schemes (e.g., those optimizing gates or exploring sparsity schedules). For instance, the key claimed advantage, “selects the pruning extent automatically in a single pass per layer” (Page 2), is not rigorously contrasted with prior one‑shot or auto‑balanced pruning methods. Without such a comparison, the conceptual originality feels modest: the contribution is more of a particular choice of separability metric + clustering + knee detection, rather than a new pruning paradigm.

2. **Scalability with respect to the number of classes and spatial resolution is a serious concern, not fully addressed.**  
   The complexity of graph construction scales as \(O(N_i \cdot p^2 \cdot \binom{C}{2})\), where \(p \times p\) is the spatial map size and \(C\) is the number of classes (Section 3.3.1). For conv layers, the method computes a JM distance at each pixel for each class pair. On ImageNet with 1000 classes, this leads to \(\sim 5\times 10^5\) class pairs per layer and \(p^2\) multiplicative factor. In practice, this seems prohibitively heavy for early layers with large spatial maps, yet the paper does not quantify or mitigate this cost; the only mention is in the Conclusions, which briefly acknowledges overhead and suggests future approximations. There is no timing breakdown of the pruning phase itself, nor ablations on approximating the separability vectors (e.g., downsampling, class‑pair sampling). As written, the method may not be practically scalable to large‑C datasets beyond ImageNet‑1K or to higher‑resolution inputs.

3. **The automatic selection mechanism (MSS + Kneedle) is heavily heuristic, with minimal analysis or ablation.**  
   Section 3.4.1 evaluates all \(k\) from 2 to \(N_i\), computes the MSS index for k‑Medoids clustering, and then uses Kneedle to select the pruning point. This is the core “automatic” component of ACSP, yet the paper does not show:
   - How sensitive results are to the polynomial degree used in Kneedle (only a brief note in Section 4.1 says a second‑degree polynomial is used, without justification).
   - Whether MSS is actually better than standard Silhouette or simpler indices in this context. MSS is imported from feature selection work (Levin & Singer, 2024), but the paper gives no empirical evidence that it matters here.
   - How MSS behaves as a function of \(k\) in practice (e.g., plots of MSS vs \(k\) for a layer and the chosen knee).  
   Without such ablations, it is hard to know if ACSP’s performance stems from the separability representation or just from picking some moderate pruning ratio via any knee‑detection heuristic. This undermines the central claim that the method “fully automates” pruning in a principled way.

4. **Ambiguities and missing details in the mathematical and algorithmic description.**  
   Several important implementation details are underspecified:
   - **Data used to estimate the distributions in Equations (1)–(2).** It is not stated whether the entire training set, a subset, or a validation set is used to build the per‑class activation distributions. Nor is there discussion of how many samples per class are needed for stable estimates, particularly for large \(C\) or imbalanced datasets.
   - **Handling of zero or tiny variances in \(B_{i,j}(c,\tilde c)\).** Equation (2) divides by \(\sigma_{i,j,c}^2 + \sigma_{i,j,\tilde c}^2\) and includes \(\ln\big(\frac{\sigma_{i,j,c}^2 + \sigma_{i,j,\tilde c}^2}{2\sigma_{i,j,c}\sigma_{i,j,\tilde c}}\big)\). If activations for some class are nearly constant in a neuron or pixel, this can cause numerical instability or undefined logs. The paper does not mention any regularization (e.g., adding \(\epsilon\) to variances) or clipping.
   - **Distance metric used within k‑Medoids and MSS.** Section 3.3.2 and the MSS description rely on a generic distance \(d(\cdot,\cdot)\), but the manuscript does not specify whether they use Euclidean distance, cosine distance, or something else on the high‑dimensional separability vectors. This choice is non‑trivial because the vectors are long (especially with large \(\binom{C}{2}\)), and distances may be dominated by a few large‑variance dimensions.
   - **Complexity of k‑Medoids over large \(N_i\) and high dimensions.** Algorithm 1 loops over all \(k\in[2,N_i]\), running k‑Medoids each time. Even if \(N_i\le 256\) (as claimed for Kneedle’s overhead, Page 4), repeated clustering in very high dimension could be significant for large models; the paper provides no pruning‑time statistics.
   These gaps make it harder to reproduce the method and to assess numerical robustness.

5. **Limited analysis of key design choices (JM, MSS, clustering, weight‑based selection).**  
   The methodology section repeatedly stresses that other separability metrics could be used (JM, Hellinger, Wasserstein are tested, Page 5), but the main paper does not show any quantitative ablation of this choice. Similarly:
   - There is no comparison between using k‑Medoids vs simpler random or k‑Means selection in the graph space.
   - There is no experiment contrasting “choose medoids directly” vs “choose highest‑weight element per cluster,” even though Figure 2 is devoted to explaining this choice.
   - There is no baseline that keeps the top‑\(k\) components purely by weight magnitude (L1/L2) with the same knee‑based \(k\), to isolate the contribution of the separability‑based representation and clustering.  
   Without these ablations, it is difficult to attribute the performance gains in Table 1 to the complementary‑separation principle rather than to standard magnitude pruning with a tuned global sparsity.

6. **Empirical evaluation of inference‑time gains is positive but modest relative to FLOP reductions, and not compared to other structured methods in wall‑clock terms.**  
   Table 2 shows that wall‑clock latency reductions are typically in the 4–10% range, which is valuable but not particularly striking given FLOP reductions up to 2.59× reported in Table 1. The paper acknowledges non‑linear hardware utilization but does not explore whether ACSP produces architectures that are more or less hardware‑friendly than competing structured methods. For example, it would be informative to compare inference times of ACSP‑pruned ResNet‑50 to those of SMCP or ResRep at similar FLOP levels. Without such comparisons, the claim that ACSP is especially “tailored for inference‑time efficiency” (Abstract) is only partially supported.

7. **Comparison to baselines is uneven and sometimes omits closely related recent work.**  
   Table 1 includes a broad set of baselines but they vary per model, and some strong or very recent pruning methods are absent. There is no experiment that systematically matches ACSP against the same set of methods across multiple architectures under controlled FLOPs or accuracy drop constraints. Moreover, highly related auto‑pruning or hybrid approaches (see Missing Related Work below) are not discussed or compared, which weakens the case that ACSP is competitive with the current state of the art rather than with a subset of prior techniques.

8. **Restriction to supervised classification and assumptions on labels.**  
   ACSP heavily relies on class labels to compute separability across all class pairs. This makes the method inapplicable as‑is to unsupervised or self‑supervised settings, or to tasks where labels are weak or noisy. The paper describes ACSP as “tailored for supervised learning tasks” but does not clearly discuss this as a limitation, nor does it explore semi‑supervised variants (e.g., using pseudo‑labels). This restricts the broader impact relative to more general activation‑based pruning methods that can exploit unlabeled data.

9. **Experimental protocol lacks some fairness and sensitivity details.**  
   Section 4.1 states that all models are “trained to their base accuracy, then lightly fine‑tuned after each layer pruning,” but it is unclear whether the baselines in Table 1 are re‑implemented under the same training and fine‑tuning setups or whether numbers are taken from their original papers (which often have different training schedules). If the latter, accuracy differences of ±0.5% are within plausible variance and may not be directly comparable. In addition,:
   - No standard deviations or multiple runs are reported, so the statistical significance of small gains (e.g., +0.09% on ImageNet MobileNet‑V2) is unclear.
   - The paper does not analyze sensitivity to the amount of fine‑tuning data or epochs; given that ACSP does fine‑tuning after each layer pruning, cumulative fine‑tuning could be considerable compared to single‑stage pruning approaches.

10. **Interpretation of Figure 1 could be richer and more quantitative.**  
    Figure 1 qualitatively illustrates how activations are transformed into separability vectors but does not show any statistics about distribution of JM scores or separation quality across layers. For example, it would be very informative to show that components selected by ACSP indeed correspond to regions with diverse class‑pair separability profiles and that pruned components cluster tightly in low‑separation regions. Without such quantitative linkage, the reader must take on faith that the constructed graph space faithfully reflects functional redundancy.

Given the number and depth of these issues, especially around scalability, missing ablations, and incomplete positioning vs automated pruning literature, the work feels promising but not yet at the level of a clear ICLR acceptance.

## Potentially Missing Related Work

The following works appear directly relevant and are not cited or discussed:

1. **X. Ding et al., “Auto-Balanced Filter Pruning for Efficient Convolutional Neural Networks,” 2018.**  
   This paper proposes an automatic, balanced filter pruning strategy which aims to decide how many filters to prune from each layer without manual ratios. It is conceptually close to ACSP’s automatic pruning extent. It should be discussed in Section 2 (Structured / automated pruning) and compared in terms of automation of pruning ratio and performance; ideally, results on at least one shared architecture (e.g., VGG/ResNet) would be added to Table 1.

2. **T. Wu et al., “Evolutionary Multi-Objective One-Shot Filter Pruning for Designing Lightweight Convolutional Neural Network,” 2021.**  
   This uses evolutionary multi‑objective optimization to jointly consider accuracy and complexity in a one‑shot pruning setting. It is another form of automatic structured pruning and should be referenced in Related Work (Structured Pruning), with a discussion of how ACSP’s single‑pass, clustering‑based selection contrasts with evolutionary search.

3. **S. Wang et al., “PSE-Net: Channel pruning for Convolutional Neural Networks with parallel-subnets estimator,” 2024.**  
   PSE‑Net uses a parallel subnet estimator for channel importance, aiming to improve pruning efficiency. It is closely related to activation‑based structured pruning and should be cited in Section 2 (Activation-Based Pruning), with discussion of differences in importance estimation and whether ACSP could be viewed as an alternative estimator operating via separability vectors.

4. **T. Zheng et al., “TDP-SAR: Task-Driven Pruning Method for Synthetic Aperture Radar Target Recognition Convolutional Neural Network Model,” 2025.**  
   While domain‑specific (SAR), this is a task‑driven pruning method, relevant to ACSP’s claim of using task‑dependent separability criteria. It would be appropriate to mention in Related Work when discussing task‑aware pruning and to briefly position ACSP as a general, classification‑oriented method not tailored to a single modality.

5. **D. Lee et al., “Lossless Reconstruction of Convolutional Neural Network for Channel-Based Network Pruning,” 2023.**  
   This work focuses on reconstruction after channel pruning to maintain accuracy, which is relevant since ACSP also prunes channels and relies on fine‑tuning. It should be included in Section 2, with commentary on how ACSP’s simple fine‑tuning compares to more explicit reconstruction strategies and whether ideas from that paper might further reduce post‑pruning accuracy loss.

6. **X. Geng et al., “Complex hybrid weighted pruning method for accelerating convolutional neural networks,” 2024.**  
   Proposes a hybrid weighted pruning approach to accelerate CNNs, similar in goal to ACSP’s structured pruning. It should be referenced in the Structured Pruning subsection, and differences in the weighting/importance criteria should be discussed.

7. **C. Heidorn et al., “Hardware-Aware Evolutionary Explainable Filter Pruning for Convolutional Neural Networks,” 2024.**  
   This introduces a hardware‑aware, evolutionary filter pruning technique. Given ACSP’s focus on inference‑time efficiency and the inclusion of latency measurements (Table 2), this work is highly relevant and should be discussed in terms of hardware awareness. It would be useful to clarify that ACSP is not explicitly hardware‑aware and to frame this as complementary or a potential future extension.

Incorporating and discussing these works would strengthen the paper’s positioning and clarify where ACSP sits relative to the evolving landscape of automated and hardware‑aware pruning.

## Questions

1. **Scalability and practical overhead of graph construction.**  
   - Can the authors provide concrete wall‑clock times for constructing the separability matrix and running k‑Medoids/MSS/Kneedle per layer on ImageNet‑1K models, broken down by layer?  
   - How does this overhead scale with \(C\) and \(p\)? Have you tried approximations such as class‑pair subsampling or spatial pooling, and if so, how do they affect performance?

2. **Details of JM distance computation and numerical stability.**  
   - What exact regularization is used to avoid division by zero or log of zero in Equation (2)? Is an \(\epsilon\) added to the variances or standard deviations?  
   - Are activations standardized or normalized before computing \(\mu\) and \(\sigma^2\)? If not, how do you handle neurons/pixels whose distributions are highly skewed or heavy‑tailed?

3. **Distance metric and normalization in the graph space.**  
   - What distance metric \(d(\cdot,\cdot)\) is used in k‑Medoids and MSS on the separability vectors? Euclidean, cosine, or something else?  
   - Are separability vectors normalized (e.g., per‑dimension scaling) before clustering? If not, could a few high‑variance class‑pair entries dominate the similarity structure?

4. **Ablations on design choices.**  
   - Can you provide ablation experiments that:  
     (a) Compare JM vs Hellinger vs Wasserstein distances for separability,  
     (b) Replace MSS with standard Silhouette or simply select a fixed percentage (e.g., 50% or 30%) per layer,  
     (c) Compare selecting medoids vs highest‑weight clustered components, and  
     (d) Compare ACSP against a magnitude‑only knee‑based baseline (no separability vectors)?  
   These would clarify which parts of ACSP are actually driving the improvements in Table 1.

5. **Data usage and fine‑tuning protocol.**  
   - When building the separability matrix, do you use the full training dataset, the 25% subset, or a separate validation set? How sensitive are the results to the amount of data used for separability estimation?  
   - Fine‑tuning is done after each layer’s pruning. Roughly how many total epochs of additional training does this correspond to for, say, ResNet‑50 on ImageNet? Could a single fine‑tuning phase after all layers are pruned perform similarly?

6. **Applicability beyond supervised classification.**  
   - Have you considered applying ACSP in self‑supervised, semi‑supervised, or regression settings where labels or discrete classes are not available? If so, what proxy for class pairs would you use in Equations (1)–(2)? If not, it would be helpful to explicitly clarify the intended scope.

7. **Fairness of baseline comparisons and statistical significance.**  
   - Are the baseline results in Table 1 re‑implemented under your training/fine‑tuning pipeline, or are they copied from original papers? If the latter, could you provide at least one setting where all methods are run under the same training schedule to ensure fair comparison?  
   - Can you report variance or confidence intervals over multiple runs for at least some setups where the accuracy differences are small (≤0.3%)?

Clear answers and additional experiments along these lines could substantially increase my confidence in the method and may shift my evaluation.

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A. The paper focuses on generic pruning methods for CNNs on standard datasets (CIFAR‑10/100, ImageNet‑1K) without sensitive data, user information, or high‑risk applications.

## Soundness Rating

2: fair.  
The method is technically plausible and experiments are reasonably extensive, but key algorithmic details (distance choice, numerical stability, data usage) and ablations of the central heuristics (MSS, Kneedle, separability metric) are missing, leaving some uncertainty about robustness and what actually drives the gains.

## Presentation Rating

3: good.  
The paper is generally well written and structured, with helpful figures (Figures 1 and 2), clear high‑level explanations, and comprehensive result tables, though some important implementation details and empirical analyses are omitted.

## Contribution Rating

2: fair.  
The idea of encoding per‑class‑pair separability vectors and clustering them for complementary selection is interesting, but the overall contribution feels incremental relative to existing automated/activation‑based pruning methods, and the lack of thorough positioning and ablations limits the demonstrated impact.

## Overall Rating

4: marginally below the acceptance threshold. But would not mind if paper is accepted.  
ACSP is a thought‑provoking and relatively practical method, with solid empirical results and a nice conceptual angle via graph‑space separability and automatic pruning ratios. However, concerns about scalability, the heavy reliance on unvalidated heuristics for automatic selection, missing ablations, and incomplete comparison to related automated pruning work collectively keep it below what I would consider a clear ICLR accept at this stage.

## Reviewer Confidence

4: confident.  
I am reasonably familiar with pruning and model compression literature, have carefully examined the math and experimental sections, and feel confident in the assessment, though some missing implementation details could alter specific judgments about scalability and robustness.