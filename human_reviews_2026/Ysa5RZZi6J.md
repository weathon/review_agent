# Nonparametric Unsupervised Data Condensation for Gigapixel Histological Images

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 2, 2, 2, 6, 2

## Abstract
Histological whole-slide images (WSIs) are central to computational pathology but are extremely large, often several gigabytes, making them infeasible for direct use in standard vision pipelines. Prior approaches reduce training cost by condensing WSIs into a fixed number of representative features (prototypes), but this approach overlooks the varying complexity and diversity of WSIs, leading to loss of critical information. To this end, we propose **NICER**, a probabilistic data condensation framework that decomposes each WSI into feature patterns to capture heterogeneity and concept prototypes to ensure compactness. By reformulating prototype construction as a nonparametric condensation problem, NICER adapts the number of prototypes to slide complexity while preserving relevant information. Experiments on four histological datasets show that NICER outperforms prior methods, yielding superior efficiency trade-offs, setting a new paradigm for histological representation learning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper describes NICER, a framework for unsupervised learning in histopathology images, based on non-probabilistic data condensation for reducing the number of prototypes through the definition of meta prototypes (or concepts).
In this way, authors formalize e new paradigm for histological representation learning, with the main claim of having a framework able to create adaptive sets of prototypes according to the slide. They provide extensive assessment and benchmark against recent baselines on a variety of pathology tasks, showcasing significant improvements.

### Strengths
1. Presentation is excellent: the way the framework is framed, the narrative flow, and the highly polished visuals makes the manuscript really enjoyable. 
2. Relevance: the depth of theoretical framework, and the proposed ideas makes the manuscript particularly adequate for ICLR.
3. Novelty: as far as I could find in the literature, the idea of condensing and reducing prototypes has not yet applied in histopathology and it is particularly valuable. The theoretical framework is well conceived and gives additional strength to the story.
4. Technical soundness: the concepts are clearly explained, and most of details for replicability are provided, even if availability of source code would be preferrable.
5. Assessment and comparison: authors report on extensive comparisons against SOTA methods on a variety of tasks, and the results appear generally convincing.

### Weaknesses
The paper appears to be solid, but I have some concerns about the fairness in comparison, and the reporting of results:
1. It seems to me that $K$ is the most important parameter, since it represents the initial target of number of concepts, that is reduced during condensation. It would be important to also report on the final number of concepts after the pipeline is complete.
2. Qualitative results and visual explainability: Fig.6 and Fig.8. Why tSNE was chosen for performing projection? I think it would provide a bias and also artifacts, as I see some small weird triangular clusters after few iterations. How do you explain them? 
Also, as far as I could understand, the final set of concepts are associated to visual features and they can be represented visually (for example for generating exemplar patches from histopathology slide). I would have expected that authors provided qualitative results about prototypical assignment, and examples on how these prototypes look, like it is done in PANTHER.
3. Fairness of comparison: as far as I could understand, the main competitor is PANTHER, but I don't think that the benchmarking is completely fair. For example, it is not clear what parameters and conditions are considered for creating Table 3; and also Figure 8 is not fair, since it seems that the prototypes obtained from Panther has limited expressivity, while instead from the original manuscript the prototypical assignment map shows the opposite; I think that this abrupt clustering of prototypes is a bias effect due to tSNE projection, but I do not feel it can be interpreted as limitation in expressivity power of the prototypes found with PANTHER framework.

### Questions
1. Minor: in the introduction, I would mention about tile-based processing that is inherently parallel.
Line 209: is. $\zeta$ parameteres appear out of nowhere in line 225. A table representing and explaining all parameters would help in readability of the manuscript.
2. The way the paradigm is theoretically framed reminds about sparse coding with overcomplete dictionaries: 
Aharon, M., Elad, M., & Bruckstein, A. (2006). K-SVD: An algorithm for designing overcomplete dictionaries for sparse representation. IEEE Transactions on signal processing, 54(11), 4311-4322. I would like that authors try to analize analogies and differences.
3. Scalability performance: what do you expect as trade-off for condensation? How much does it cost to perform condensation instead of keeping the full WholeOfBag?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces NICER, a nonparametric unsupervised data condensation framework for whole-slide histology images (WSIs). NICER adaptively determines the number of prototypes per slide by formulating condensation as a hierarchical probabilistic model (features -> patterns -> concepts). The goal is to balance information preservation and computational efficiency. The method is evaluated across four datasets and on two main tasks: cancer subtyping and survival prediction. The authors report consistent improvements over prior works in the field of prototype learning, such as PANTHER and ProtoCount.

### Strengths
1. Relevant problem: WSI condensation is a well-known bottleneck in computational pathology, as slides are composed of a very large number of tiles, many of which are redundant. Tackling this problem in an unsupervised way is of practical and methodological interest. 
2. Comprehensive empirical evaluation: The experiments cover multiple datasets and tasks with consistent baselines and metrics. The reported gains over comparable prototype-based approaches are systematic.

### Weaknesses
1. Methodological Ambiguities 
Despite the heavy probabilistic framing, the method is mathematically under-specified: 
- The hierarchical model 𝑃(𝐻,𝑍∣Ω)  is introduced but not clearly derived or grounded in an actual probabilistic generative process. 
- Many equations (e.g., Eq. 2–9) appear to be reformulations of clustering or assignment heuristics, rather than principled probabilistic inference steps. 
- The “nonparametric” property arises mainly from pruning unused prototypes, not from a Bayesian nonparametric process. 
- Numerous notation and formulation issues undermine clarity: 
  * 𝒩  (Gaussian) is confused with ℕ  (natural numbers). 
  * Confusion between probability laws and densities in expressions such as log(𝒩(⋅)). 
  * Inconsistencies between Eq. 6 and Eq. 8 
  * Typos (max vs. min, line 173). 
  * Equation 4 is ill-defined. 
  * Some variables are not defined (e.g. d l173, α l229). 
  * “Since both hi and z∗ (i) are ℓ2-normalized embeddings, we approximate ∥hi∥2 ≈ ∥z∗ (i)∥2 ≈ 1”.  It is equal to 1, no approximation here. 
- Overall, the pattern exploration method relies mostly on deterministic clustering mechanisms, with the Bayesian formalism serving mainly as an interpretive framework rather than a fully realized inference model. 
 
2. Performance Claims 
- The reported “up to 90% performance gains” (abstract) are misleading. 
- Table 1 shows that NICER features still underperform the vanilla method (whole bag). Ideally, condensation should reduce redundancy while maintaining or improving performance, as demonstrated in PANTHER paper; here, the gain comes at the cost of lower performance. 
- The paper should include comparisons to other whole-slide representation models (e.g., GigaPath, GigaSSL, PRISM, HIPT). 
 
3. Biological Interpretability 
- The biological or histological meaning of prototypes is under-explored: 
  * What morphological patterns do prototypes capture? 
  * Visualization of tile-prototype associations or spatial prototype maps would significantly strengthen the claim of the paper.

### Questions
- Some mathematical expressions are difficult to follow because several variables are not defined when first used. I recommend clearly defining all variables and revising the math sections to make them more precise, accurate, and rigorous. 
- Simplify the mathematical formulations to highlight the core mechanism. 
- What morphological or histological patterns do the learned prototypes capture? 
- Could the authors provide visualizations of tile–prototype associations or spatial prototype maps to support biological interpretability?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces NICER, a probabilistic nonparametric method for unsupervised data condensation of gigapixel histology slides. Unlike prior approaches that use a fixed number of prototypes per slide, NICER adapts the number of prototypes to each slide’s morphological complexity. It models prototype construction hierarchically, preserving diverse local patterns while enforcing compact global representation. Across multiple datasets, NICER achieves strong gains in F1 and efficiency over existing condensation and MIL baselines.

### Strengths
**Clear motivation and problem relevance.**  
  The paper identifies an important challenge in computational pathology about how to compress extremely large WSIs while retaining morphological diversity.  

**Readable structure and visuals.**  
  Figures are generally clear and intuitive. The narrative follows a standard and familiar structure, making the paper easy to navigate.  

**Potential applicability.**  
  Adaptive prototype condensation could, in principle, generalize to other domains that require scalable representation learning, even though this is not empirically verified.

### Weaknesses
**Inflated novelty.**  
The core mechanism of NICER is largely a reformulation of existing prototype learning and clustering frameworks such as PANTHER, OT-based embeddings, and Gaussian mixture condensation. The “pattern–concept” hierarchy is functionally equivalent to a coarse-to-fine clustering scheme, and the probabilistic factorization adds notation without introducing new learning objectives or inference principles. Overall, the contribution is incremental rather than conceptually novel.

**Inaccurate “nonparametric” claim.**  
Although repeatedly described as nonparametric, NICER is a fully parametric neural framework with learnable embeddings and trainable parameters. Its capacity is bounded by predefined hyperparameters and adjusted via heuristic pruning. The approach does not perform genuine nonparametric inference or kernel-based adaptation. Hence, its “nonparametric” behavior is empirical and heuristic rather than statistical or data-driven.

**Unverifiable and internally inconsistent experimental comparisons.**  
The paper does not clarify which configuration underlies the reported “SOTA” results, making them empirically unverifiable and potentially biased in favor of NICER. In addition, both PANTHER (and possibly other prototype-based methods) and the survival prediction tasks are sensitive to hyperparameters and data splits, which makes the reported improvements difficult to reproduce.

**Unsupported efficiency and generalization claims.**  
The paper emphasizes a superior efficiency–performance trade-off and even a new paradigm for histological representation learning, yet provides no runtime, GPU memory, or throughput measurements. The reported metric reflects only compression ratio, not computational cost. Moreover, all experiments are confined to a bag-of-features assumption. This narrow scope and lack of computational evidence undermine both the efficiency and generalization claims.

### Questions
- Including additional baseline comparisons beyond PANTHER and OT-based methods If possible. 

- If possible, provide a runtime and preprocessing time analysis. The preprocessing stages (patch extraction, embedding, and condensation) appear to dominate total computation time.  It would be helpful to quantify how much time each stage requires compared with end-to-end MIL training or inference.  This could clarify whether NICER improves overall pipeline efficiency.

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
4

### Summary
This paper introduces NICER, a framework for unsupervised data condensation of gigapixel whole-slide images (WSIs). The method aims to address the limitation of prior works that use a fixed number of prototypes, proposing instead a two-stage probabilistic model that adapts the prototype capacity to the complexity of each slide. It first extracts a large set of feature patterns to preserve information, then condenses them into a smaller set of concept prototypes. The authors claim this nonparametric approach achieves state-of-the-art performance on four histological datasets.

### Strengths
The paper identifies an important and relevant problem in computational pathology: the inadequacy of fixed-capacity representations for WSIs of varying complexity. The proposed two-stage approach of preservation followed by condensation is, at a conceptual level, an intuitive and promising direction.

### Weaknesses
1. The paper's central conceptual pillar is its "nonparametric" nature. However, the method is simply an iterative algorithm with a heuristic pruning step. This is a severe misrepresentation of the methodology. 
2. The paper's entire premise is built on improving the trade-off between accuracy and efficiency. Yet, there is a complete and inexplicable absence of any empirical data regarding efficiency—no training times, no inference speeds, no memory usage comparisons.
tion.
3. The methodology is described at such a high level that it is impossible to reproduce. Key design choices (initialization, network architectures, reconciliation of top-k vs. top-1) are omitted.

### Questions
1. Can you provide a rigorous justification for using the term "nonparametric"? If not, are you willing to retract this claim and re-frame your contribution more accurately as an "adaptive-capacity" model?

2. Please provide a new table comparing NICER against key baselines (e.g., PANTHER, H2T) on wall-clock training and inference time per WSI, as well as peak GPU memory consumption.

3. Please provide the exact implementation details necessary for reproducibility: (a) How was the initial pattern set Z generated? (b) What are the specific architectures and hyperparameter settings? (c) Please clarify the discrepancy between the "top-k" description and the "top-1" formulation in Eq. 2 and state precisely what objective was implemented.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces NICER, a nonparametric, unsupervised condensation framework for gigapixel WSIs. Instead of fixing a per-slide prototype budget, NICER first over-preserves morphology by learning slide-specific patterns (redundant by design), then condenses them into concept prototypes, pruning unused concepts so the final count adapts to slide complexity. The formulation is probabilistic with latent pattern–concept assignments optimized via alternating updates. Across four datasets (PANDA, NSCLC, BRCA, LUAD) and two tasks (subtyping, survival), NICER outperforms prior unsupervised prototype learners (DeepSets, ProtoCount, H2T, OT, InfiniteGPFA, PANTHER) and maintains a favorable accuracy–efficiency trade-off relative to Whole-Bag features (Tables 1–2, pp. 6–7; Table 3, p. 8). Ablations show how initial pattern count M and top-κ affect capacity and saturation (Fig. 4, p. 7; Table 3, p. 8).

### Strengths
Addresses a real bottleneck: avoids one-size-fits-all prototype budgets by adapting capacity per slide via a principled pruning mechanism (Sec. 2.3–2.4, pp. 4–5).

Balanced preservation→efficiency: the two-stage “intentional redundancy then condensation” reduces early information loss and retains rare morphology (Intro & Fig. 3, pp. 2–3).

Strong, broad empirical results: consistent wins over multiple unsupervised baselines (including the latest PANTHER approach) and across three MIL heads (ABMIL/DSMIL/ILRA) for subtyping and survival (Tables 1–2, pp. 6–7).

Trade-off evidence: clear performance–compression curves and sensitivity analyses (Table 3 & Fig. 4, pp. 8–7; Figs. 5–6, p. 7).

Practical framing: operates on foundation-encoder features, with implementation details and ablations that are actionable.

### Weaknesses
High Complexity and Computational Cost: NICER’s sophistication comes at the cost of increased complexity. The method involves an iterative two-stage learning process with many latent variables, which is more complicated to implement and tune than simpler one-step clustering or standard MIL models. The paper mentions using a 46GB GPU for experiments, indicating substantial memory and computation requirements. This could hinder adoption in practice – laboratories with limited computational resources might struggle to run NICER, whereas simpler methods (e.g. k-means or fixed GMM prototypes) are more lightweight. The authors do not report runtime comparisons, so the efficiency trade-off of NICER’s richer modeling is not fully clear.

Hyperparameter Sensitivity: While NICER is nonparametric in that it determines prototype counts automatically, it still requires setting certain hyperparameters that can affect results. In particular, the initial number of patterns (M) and the patch-to-pattern association limit (top-κ) must be chosen. The authors’ ablations show that too small an M can hurt performance and too large a κ can diffuse patch information. Thus, practitioners need to choose these carefully (the paper finds e.g. M≈200 and κ≈3 work well). NICER’s performance might degrade if these are mis-specified for a new dataset. In contrast, some simpler baselines have fewer tunable parameters. This points to a potential limitation in ease-of-use: despite its adaptive nature, NICER isn’t completely “hands-off” to configure.

Model Assumptions and Theoretical Guarantees: NICER relies on a probabilistic model (e.g. assuming patch features are approximately Gaussian around pattern means). These assumptions, while reasonable, are not deeply validated. Additionally, the optimization is heuristic (alternating updates for patterns, concepts, and assignments). The paper provides no formal proof of convergence or bounds on information loss during condensation. A skeptical reader might question whether the gains come from the sophisticated model or simply from clever engineering choices. Some discussion or theoretical insight into why the two-stage approach outperforms single-stage clustering (beyond empirical observation) would strengthen the work.


Novelty scope. The main novelty is the nonparametric per-slide capacity and the explicit pattern→concept condensation; however, the broader ideas (unsupervised prototyping, mixture-like modeling, hierarchical abstraction) overlap with existing lines. The paper could sharpen what is mathematically distinct from fixed-K GMM or from PANTHER’s soft assignment.

Clinical relevance and interpretability. While the paper demonstrates strong quantitative performance on subtyping and survival prediction, histological subtyping is a task pathologists routinely perform using visual cues. To establish clinical utility, it would be important to show that the concept prototypes learned by NICER correspond to morphologic patterns recognized by human experts (e.g., tumor regions, stroma, necrosis). The absence of such interpretability analysis limits the translational impact of the work.

External validity and batch effects. The evaluation relies exclusively on public TCGA and PANDA datasets, which are known to exhibit site-specific staining and preprocessing artifacts. These batch effects can inflate in-domain performance while limiting generalization. Validation on independent institutional or multi-center cohorts would provide stronger evidence that NICER’s adaptive condensation generalizes across data sources.

### Questions
Prototype Budget Determination: Could you clarify how NICER decides the final number of prototypes per slide? Is there an implicit threshold or stopping criterion in the generative model that prunes “redundant concepts”? For example, do you fix an initial maximum (M patterns or concepts) and then drop those with negligible assigned patches? Understanding what controls the adaptive prototype count (and how variable it is across slides) would help gauge the method’s robustness.

Rationale for Two-Stage Condensation: What is the key advantage of the hierarchical patterns→concepts approach versus a single-stage nonparametric clustering of patches? In principle one might try a Dirichlet Process or adaptive K-means on the patch embeddings directly. Does the two-step process (intentional redundancy then merging) simply preserve rare features better, or does it also aid optimization stability? Any insight or experiments comparing NICER’s two-stage pipeline to a one-stage variant would clarify why the hierarchy is crucial.

Computational Efficiency: How do the training time and memory usage of NICER compare to simpler prototype methods like PANTHER or to standard MIL models? The method appears resource-intensive; for instance, did you need to process one WSI at a time, or were multiple slides optimized in parallel? Any data on runtime per slide or per epoch would be appreciated. This information would help readers assess the practicality of NICER for large-scale or real-time applications.

Feature Extraction and End-to-End Learning: Are the patch features used in NICER fixed from a pretrained model, or were they trained/finetuned as part of this work? If fixed, did you observe any failure cases attributable to feature quality? And do you anticipate gains if one jointly learned the feature encoder with the NICER framework? It would be interesting to know if integrating representation learning (perhaps via a multi-task or self-supervised loss) was attempted, or if not, why the decoupled approach was chosen.

Position & Context: NICER currently treats the WSI as an unordered bag of patch features, focusing on morphological content. Might incorporating spatial context improve the prototypes (for example, ensuring that “concepts” correspond to contiguous regions or specific structures in tissue)? Some recent works add coordinate information or model the WSI as a graph to capture architecture. Did you consider extensions of NICER to encode spatial relationships between patches or to enforce that selected prototypes are spatially diverse? This could be relevant for tasks like tumor localization, so we wonder if it’s a plausible future direction.

Comparison with Other Unsupervised Methods: You included an “Infinite GPFA” baseline (Yu et al., 2025) which, like NICER, aims to learn latent factors without fixing their number. That method underperformed significantly. Can you shed light on why NICER achieves better results than InfiniteGPFA or other nonparametric clustering approaches? For example, is it due to NICER’s alternating optimization capturing more variance, or the specific way NICER handles patch-to-prototype assignments (top-κ redundancy, etc.)? A deeper explanation would highlight what design choices are most critical for NICER’s success relative to earlier approaches.

Clinical alignment of prototypes. Given that histologic subtyping is a task pathologists perform visually, have you examined whether the discovered prototypes correspond to interpretable histopathologic structures or features that pathologists recognize? For instance, do the top-activating patches for individual prototypes map to tumor, stroma, or inflammatory regions in a way that aligns with diagnostic reasoning? Such validation would help contextualize the model’s relevance to practical pathology workflows.

Cross-institutional validation. TCGA datasets contain known batch effects and pre-analytic heterogeneity. Have you tested NICER’s generalization when trained on one institution and tested on another (e.g., PANDA KRLS → RUMC split or external private cohorts)? If not, could the authors discuss how NICER’s adaptive, nonparametric mechanism might mitigate or exacerbate domain shifts across institutions?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 6

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work introduces NICER, a data condensation method that compresses bag-of-feature inputs (modeling whole-slide images in pathology) into a compact set of slide-adaptive prototypes for downstream use. NICER works by first learning a high-capacity set of patterns that each patch selects via top-k similarity, then condensing those patterns into a smaller set of concepts while pruning unused concepts which adapts to slide complexity. Benchmark tasks include classification(TCGA-NSCLC, PANDA) and survival prediction (TCGA-LUAD, TCGA-BRCA) with comparisons against unsupervised prototyping baselines (e.g., DeepSets, ProtoCounts, H2T, PANTHER) and MIL predictors (e.g., ABMIL, DSMIL, ILRA). Additional experiments include sensitivity to the initial number of patterns and ablating $k$ in top-$k$ selection.

### Strengths
- Good presentation and figures.
- Good number of ablation studies performed (performance trade-off, prototype diversity, $k$ in top-$k$).
- All comparisons use same feature encoder, with ablations of MIL architecture.
- Related work in pathology is well-cited.

### Weaknesses
- Study design follows that of PANTHER in evaluating on challenging pathology tasks (PANDAS, survival tasks) with PANTHER being one of the primary comparisons. However, only a few survival tasks are evaluated, with missing evaluation on external datasets such as CPTAC which was one of the core strengths of PANTHER as a prototypical method. Can NICER also generalize to CPTAC for LUAD survival?
- Method is presented nicely but missing many references to data condensation methods. How much of the method comes from existing ideas  in fundamental ML/AI? Very hard to understand the technical contribution.
- From my understanding of this work, NICER does not produce a slide-level representation similar to PANTHER. Rather, it learns a more compact set of patch feature prototypes followed by applying a MIL architecture on top.
- - More fundamental baselines that this work should compare against is k-means, adaptive clustering methods, gaussian / dirchlet process mixture models and other EM clustering ideas in reducing the WSI to a fixed set of prototypes. there exist many non-parameter approaches for solving this same task of pruning redundant clusters in clustering problems.
- - Which formulation of PANTHER Is being compared?

Overall, I don't see the novelty of NICER at this time. Core idea of reducing redundant concepts in unsupervised clustering has a very straightforward extension to pathology, with many fundamental baselines missing that can also be used to find efficient prototype sets. In terms of experimental design, NICER does not evaluate on external datasets for survival prediction, so it is unclear how NICER would generalize.

### Questions
See above.

### Soundness
2

### Presentation
3

### Contribution
2
