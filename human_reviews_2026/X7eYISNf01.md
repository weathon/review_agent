# Paradigm Shift of GNN Explainer from Label Space to Prototypical Representation Space

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Post-hoc instance-level graph neural network (GNN) explainers are developed to identify a compact subgraph (i.e., explanation) that encompasses the most influential components for each input graph. A fundamental limitation of existing methods lies in the insufficient utilization of structural information during GNN explainer optimization. They typically optimize the explainer by aligning the GNN predictions of input graph and its explanation in the graph label space which inherently lacks expressiveness to describe various graph structures. Motivated by the powerful structural expression ability of vectorized graph representations, we for the first time propose to shift the GNN explainer optimization from the graph label space to the graph representation space. However, the paradigm shift is challenging due to both the entanglement between the explanatory and non-explanatory substructures, and the distributional discrepancy between the input graph and the explanation subgraph. To this end, we meticulously design IDEA, a universal dual-stage optimization framework grounded in a prototypical graph representation space, which can generalize across diverse existing GNN explainer architectures. Specifically, in the Structural Information Disentanglement stage, a graph tokenizer equipped with a structure-aware disentanglement objective is designed to disentangle the explanatory substructures and encapsulate them into explanatory prototypes. In the Explanatory Prototype Alignment stage, IDEA aligns the representational distributions of the input graph and its explanation unified in the prototypical representation space, to optimize the GNN explainer. Comprehensive experiments on real-world and synthetic datasets demonstrate the effectiveness of IDEA, with the average improvements of ROC-AUC by 4.45% and precision by 48.71%. We further integrate IDEA with diverse explainer architectures and achieve an improvement by up to 10.70%, which verifies its generalizability.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose a paradigm shift from label space to graph representation space, which enforces the preservation of encoded information when generating local explanations, consequently providing more accurate and robust explanatory subgraphs.

### Strengths
- The proposed method is model-agnostic and can be applied to other post-hoc local explanation methods on graphs. Experimental results demonstrate consistent performance improvements across multiple datasets. Moreover, the method exhibits robust performance compared to other baselines under noisy conditions.
- While existing literature has primarily focused on label space, there has been limited research on leveraging informative features from graph representations for explanations. In this regard, this paper presents novelty by introducing a new perspective on graph explainability.
- The paper is well-written and easy to understand.

### Weaknesses
- There is a lack of justification for why both shallow and deep quantizers are necessary. While the authors explain that shallow quantization handles non-explanatory substructures and deep quantization captures explanatory substructures, the transition from quantization to disentanglement (similar to MixupExplainer) feels abrupt when first reading the methodology section, which weakens the clarity of presentation. I recommend that the authors provide additional explanation and stronger justification for this design choice.
- The representation in Equations 2 and 3 is a little bit complicated. And there is insufficient explanation of the vector arithmetic operations. The authors are recommended to provide a detailed explanation about equations 2 and 3. Additionally, shouldn't L_D (rather than L_S) in lines 205~7  be the loss that enforces the prediction of the non-explanatory subgraph towards a uniform distribution? (In Figure 2, only y_S is related to the uniform distribution). 
- Lack of theoretical analysis. 

- Minor issue: in Table 2, rank information is not provided.

### Questions
- Why are both shallow and deep quantizers necessary? Is a single quantizer insufficient for your framework? Providing clearer motivation for this dual-quantizer design would significantly enhance the paper's clarity.
- The explanation of vector arithmetic operations in Equations 2 and 3 needs more detail. Could you provide a step-by-step breakdown of these operations?
- Given that you position deep quantization as the relevant source of explanation with reference quantization serving as prototype explanation, could this framework potentially be extended to model-level explanations? Are there promising directions for future work in this area?

Please refer to the weaknesses along with the questions.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes IDEA, a two-stage framework for post-hoc, instance-level GNN explanations. It learns a prototype space of “explanatory” substructures and then aligns the explanation with the original graph in that prototype space using an entropy-regularized objective. Then the paper claimed and positioned as a shift from label-preserving objectives toward representation-level alignment, which aims to better capture fine-grained molecular motifs. After reviewing this paper, I am leaning towards a **weak accept rating (6), since parts of the technical formulation and experimental results, like the transport cost and the disentanglement guarantees, still need necessary clarification.**

### Strengths
1. The core pitch, “same label” is often too coarse for chemistry-level reasoning, and aligning explanations in a learned prototype space is a clean and intuitive step forward. 

2. The framework is conceptually coherent and is advertised as attachable to different explainer backbones, which increases practical impact.

### Weaknesses
1. **Wasserstein cost matrix feels underdefined.** The prototype alignment uses an entropy-regularized Wasserstein distance with cost matrix \(S\). Since the ordering of learned prototypes is arbitrary, for $S_{ij} = (i-j)^2\$ has no obvious semantic meaning. Without a principled metric over prototypes, I am not fully convinced the transport cost is meaningful or stable. Please justify how the indices are ordered or provide an alternative cost.
   
2. **Granularity / supervision path.** The paper defines distributions $P'_G$ (for the “purified” original graph) and $P_g$ (for the candidate explanation subgraph) over explanatory prototypes, then matches them. It is not fully clear whether these are node-level, edge-level, or graph-level histograms, and how gradients from that distributional alignment are pushed back to edge scores that are later evaluated using ROC-AUC and precision against ground-truth substructures. A short dimensional walkthrough in the main text would make the method easier to trust.
   
3. **Disentanglement may leak.** The hierarchical tokenizer is trained so that the “shallow” codebook should become non-explanatory and the “deep” codebook should become explanatory, by pushing shallow predictions toward uniform and deep predictions toward the original GNN output. But the same quantized representation is also forced to reconstruct structure (adjacency and node features), which could let shallow codes still absorb predictive motifs.
   
4. **No std/var or statistical robustness report.** The tables report averages but no standard deviations across random seeds, and there is no statistical test. For some benchmarks the margins over strong baselines are a few percent. Without error bars it is difficult to judge how robust the reported ~4–5% ROC-AUC and ~49% precision improvements really are. 

5. **Scope of novelty claim.** The paper repeatedly frames IDEA as a “paradigm shift” from label-preserving explanations to representation-level alignment. I do think the formulation is interesting and practically useful. Still, prior methods like ProxyExplainer already started to address distribution shift between a full graph and an explanatory subgraph using learned proxy distributions, not just raw label matching. So personally, I think the final version should position IDEA as a principled instantiation of this direction, rather than implying absolute first. 

6. **(Minor) writing / notation issues.** Some notation is inconsistent or rushed. E.g. the paper alternates between $U^C$, $P_u$, and “uniform” when describing how shallow codes are encouraged to be non-predictive; it is not always obvious these are the same object. There are also a few typos and spots where a symbol appears before it is defined. Cleaning up these points would improve readability.

### Questions
1. How is the prototype cost matrix $S_{ij}$ justified? Do you sort prototypes in some interpretable order (e.g., via clustering in latent space) before computing transport cost, or is the index order arbitrary? If it is arbitrary, why does the Wasserstein distance remain meaningful? 

2. When defining $P'_G$ and $P_g$, are these distributions aggregated at the graph level, node level, or edge level? How do gradients from the Wasserstein loss translate back into edge selection probabilities in the explainer so that you can evaluate edge-wise ROC-AUC and precision against ground truth substructures? Please clarify the pipeline. 

3. Does the shallow codebook actually avoid encoding predictive motifs? Can you quantify how predictive shallow vs deep codebooks are (e.g., accuracy drop if you only use shallow vs only use deep)? Right now LD encourages this split, but LS might leak class-relevant structure back into the shallow part. Some empirical evidence would strengthen the causal interpretation. 

4. Can you report mean ± std over multiple seeds for Tables 1–3? Current results seems can not assess robustness of the claimed 4–5% ROC-AUC gain and 40%+ precision gain.

5. For the one negative case (IDEA hurts V-InFoR on Mutag), can you provide evidence on a corrupted-graph benchmark where V-InFoR is useful?

### Soundness
3

### Presentation
2

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
This paper identifies a limitation in existing post-hoc GNN explainers: they are typically optimized under a label-preserving framework, which aligns the GNN's label prediction for the original graph with that of the explanation subgraph. The authors argue that this label space is not expressive enough to capture the rich structural information GNNs use, especially when multiple distinct graph structures can map to the same label. The authors first demonstrate that a naive Direct Alignment in this space is ineffective due to two key challenges and provide a novel dual-stage optimization framework designed to overcome these challenges. Experiments show that IDEA outperforms state-of-the-art baselines.

### Strengths
S1. The paper clearly and convincingly articulates a fundamental, yet often-overlooked, weakness of the dominant label-preserving framework.

S2. The proposed IDEA framework, with its use of vector quantization for disentanglement and prototypical distribution alignment, provides a reasonable solution to both problems.

S3. IDEA demonstrates consistent and significant performance gains over a wide range of SOTA eplainers on five datasets

### Weaknesses
W1. The entire framework's success hinges on the Stage 1 HGTokenizer correctly separating explanatory from non-explanatory information. This separation is enforced by the proxy objective $\mathcal{L}_D$, which pushes the $Q_S$-based prediction to a uniform distribution and the $Q_D$-based prediction to the original GNN prediction. While this is a common proxy, it's not a guarantee. It's plausible that in some cases, the explanatory signal could be simple and captured by the shallow quantizer, while a complex confounder is relegated to the deep quantizer. 

W2. The theoretical justification in Appendix G for using assignment probability as a proxy for latent space location is insightful but relies on several strong, unverified assumptions. Specifically, it assumes that the (unobserved) mapping from the true latent space to the prototypical space is linear or can be approximated linearly. It is not clear how the method would perform if this mapping is highly non-linear, which is very possible in deep models. A discussion of this limitation or a more robust justification would be beneficial.

W3. The paper relies on ground-truth-based metrics (ROC-AUC and Precision). While standard, GNN explanation evaluation also benefits from model-centric metrics that measure faithfulness (i.e., how much the model's prediction actually changes when the identified subgraph is removed or masked).

### Questions
Please refer to the weaknesses above.

### Soundness
3

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
This paper proposes a novel paradigm shift for post-hoc instance-level GNN explanation: moving the optimization target from the label space to a learned prototypical representation space. The authors identify two key obstacles in directly aligning GNN-encoded representations: (1) entanglement of explanatory and non-explanatory substructures, and (2) distributional mismatch between the original input graph and its sparse explanation subgraph.

To tackle these, the authors introduce IDEA, a two-stage optimization framework. In the first stage, a Hierarchical Graph Tokenizer (HGTokenizer) equipped with a structure-aware disentanglement (SAD) objective separates explanatory from non-explanatory components and clusters the explanatory parts into prototypes. In the second stage, assignment distributions of the purified input and explanation subgraphs over these prototypes are aligned via an entropy-regularized Wasserstein distance.

Extensive experiments across both real-world and synthetic datasets demonstrate IDEA's superiority in ROC-AUC and precision, and its generalizability across various explainer backbones is also validated.

### Strengths
1. The shift from label alignment to prototype-based representation alignment is novel and addresses a fundamental limitation in current GNN explainers.

2. The two-stage IDEA framework is carefully designed. The use of shallow and deep vector quantization for disentangling, and Wasserstein distance for alignment, is both elegant and sound.

3. Comprehensive experiments across diverse datasets and explainer architectures convincingly validate the effectiveness and generality of IDEA. Ablation studies, visualization, and extended appendix analyses (e.g., noise robustness, hyperparameter analysis) are particularly thorough.

4. The paper is exceptionally well-written, and all figures and diagrams are highly informative and polished.

### Weaknesses
1. **Limited dataset diversity**: The method is only evaluated on five datasets. Testing on more diverse or larger real-world datasets (e.g., citation networks, program graphs) would strengthen the paper’s applicability.

2. **Lack of clarity on prototype initialization**: The paper does not describe how the prototype embeddings (codewords in $C_S$ and $C_D$) are initialized. This is important for reproducibility and could affect convergence behavior.

3. **Possible invalid supervision in Equation (5)**: The second term of Equation (5) uses CrossEntropy between $\hat{y}_D$ and $\hat{y}$. However, $\hat{y}_D$ is generated from the deep quantized representations $q^\*_D$, which are OOD with respect to the frozen GNN predictor. If the predictor was trained on original (non-disentangled) representations, its output for $q^\*_D$ may not be meaningful.

4. **Residual OOD inconsistency in optimization**: Although the paper claims that aligning in representation space avoids the OOD issue, Equation (5) still uses frozen GNN predictions $\hat{y}$ as supervision for deep quantized vectors $\hat{y}_D$, which remain out-of-distribution. This inconsistency needs to be addressed.

5. **Incomplete related work**: The paper largely overlooks existing prototype-based explanation approaches. In addition to Dai & Wang (2025), key missing references include:
   - ProtGNN (Zhang et al., AAAI 2021)
   - PAGE (Shin et al., TPAMI 2022)
   - Prototype-Based Interpretable GNNs (Ragno et al., IEEE TAI 2024)
   - Towards Prototype-Based Self-Explainable GNN (Dai & Wang, TKDD 2022)

### Questions
1. Regarding Equation (5): Could you justify why minimizing CrossEntropy($\hat{y}_D$, $\hat{y}$) is valid given that $\hat{y}_D$ is OOD with respect to the frozen predictor?
2. Could you evaluate IDEA on larger or structurally more complex datasets to support scalability?
3. Is it always appropriate to enforce the shallow quantized representation to produce a uniform prediction distribution (first term in Equation (5))?
4. How are the prototype codebooks $C_S$ and $C_D$ initialized? Is the method sensitive to different initialization schemes?
5. Can the authors provide qualitative or quantitative analysis of the learned prototypes? E.g., do deep prototypes align with known substructures (e.g., benzene rings)?
6. Do shallow and deep prototypes exhibit distinguishable structural roles? Can visualizations or clustering be used to support this?
7. Some baseline results seem lower than in original papers. Are all results averaged over multiple runs? Could standard deviations be reported?
8. Hyperparameter tuning varies significantly across datasets (e.g., codebook size $K$). How practical is IDEA in scenarios without access to per-dataset tuning resources?
9. In Table 3, IDEA with V-InFoR shows performance degradation on Mutagenicity. Can the authors explain why IDEA is incompatible in this case?
10. The paper mentions that node-level tasks can be reformulated as graph-level tasks using computation graphs. Could the authors validate this claim with experiments, or clarify the scope as graph-level only?
11. How does IDEA differ fundamentally from prior prototype-based GNN explainers? Is the key contribution the new optimization paradigm (label space → representation space), or the hierarchical disentanglement mechanism?

### Soundness
3

### Presentation
3

### Contribution
3
