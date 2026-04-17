# From Memorization to Reasoning in the Spectrum of Loss Curvature

- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
We characterize how memorization is represented in transformer models. The eigenbasis of the Fisher Information Matrix for MLP weight matrices encodes shared structure used across many data points at the top of the spectrum and memorized examples at the bottom, which we can disentangle in the weight space in both language models (LMs) and vision transformers (ViTs). We connect this finding to prior theoretical and empirical work on the curvature of the loss landscape for individual memorized datapoints, and use it to propose a weight editing procedure that suppresses far more recitation of untargeted memorized data more effectively than a state of the art unlearning method (BalancedSubnet; \citet{sakarvadiamitigating}), while maintaining lower perplexity. Since the basis of curvature has a natural interpretation for shared structure in model weights, we analyze the editing procedure extensively on its effect on downstream tasks in LMs, and find that fact retrieval and arithmetic are specifically and consistently negatively affected, even though open-book and general logical reasoning is conserved. We posit these tasks rely heavily on specialized directions in weight space rather than general purpose mechanisms, regardless of whether those individual datapoints are memorized. Our work enhances the understanding of memorization in neural networks with practical applications towards removing it, and provides evidence for idiomatic, heuristic-like structures that are used to solve tasks like math and fact retrieval.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates how memorization and generalization are represented in Transformers by analyzing loss curvature. Using the K-FAC approximation of the Hessian, the authors argue that high-curvature directions correspond to shared, generalizable structures, while low-curvature directions relate to memorization and brittle behaviors. Based on this finding, they propose a weight editing method that selectively removes low-curvature components to suppress memorization while preserving model performance. The authors evaluate their approach on both language models and vision transformers, comparing against the BalancedSubnet (BSN) baseline. They additionally find that different tasks exhibit varying sensitivity to the removal of low-curvature directions.

### Strengths
The paper offers a fresh perspective on the memorization-generalization trade-off through the lens of loss curvature. The investigation of how curvature relates to different cognitive abilities is interesting and raises fundamental questions about the geometry of neural loss landscapes.

### Weaknesses
While the results are promising, I believe the paper requires a significant update before being accepted. I detail the main reasons for that below.

- **Limited scope of curvature analysis.** The reliance on K-FAC applied only to MLP weight matrices, while computationally practical, undermines the broader claims. The authors argue that curvature affects generalization and reasoning, but K-FAC analysis restricted to MLP weights lacks the precision needed to support such claims. Notably, [prior work](https://proceedings.mlr.press/v162/benzing22a/benzing22a.pdf) has demonstrated that K-FAC does not necessarily capture true second-order information. Moreover, the Fisher Information Matrix (and its K-FAC approximation) represents only one component of the Hessian, providing an incomplete picture of curvature.
- **Lack of precision of the proposed editing method.** While the proposed approach effectively reduces certain forms of memorization, it appears to damage adjacent information that should be preserved. For instance, memorization of historical quotations represents valuable knowledge that the method inadvertently destroys. The performance degradation on reasoning tasks (Figure 2) is non-negligible. The BSN method appears more targeted in its editing; I suspect an analogous Figure 2 for BSN would demonstrate significantly better preservation of reasoning abilities. Additionally, the paper focuses on knowledge and reasoning tasks but leaves unclear whether other capabilities are compromised by the editing process.
- **The paper is somewhat hard to read.** The high level structure makes a lot of sense, but individual sections are difficult to follow. The logical flow of arguments and the methodological details often lack clarity, making it challenging to understand precisely what the authors are doing.

### Questions
- In Figure 1, could the average norm of the vectors affect the overall shape of the evolution? In particular, could it explain the big change in ratio in the last layer of ViTs?
- Is there a specific reason on why keeping other types of parameters (e.g. attention weight matrices) outside the scope of the analysis?

### Soundness
2

### Presentation
1

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
This paper studies memorization in Transformer models. It shows that the eigenbasis of the approximated Hessian of weight matrices uncovers distinct disentanglement of memorization and generalization, across vision and language models. In a population-level of samples, memorization directions are actually flatter than generalizing directions. They then propose a recitation-reduction technique based on ablating memorized directions in weight space.

### Strengths
- The paper is clearly written and easy to read.

- The use of K-FAC to analyze memorization and generalization is novel to me.

- The results that memorization aggregated over a population level is actually flatter than generalization direction are interesting.

### Weaknesses
1. The limitation is not well discussed. For example, in Table 5, BSN is able to preserve higher accuracy than K-FAC edit. Also, the method requires a sweep to find which layers to edit, which is not well acknowledged as a limitation I believe. Are there other relevant limitations? For example, how does the computational cost of the proposed method compare with BSN? 

2. While Line 143-149 provides a discussion, there could have been more analysis on per-example curvature analyses and population-level analyses. For example, showing how per-sample memorization directions get canceled out when averaging across samples in an experimental way would be nice.

### Questions
1. Can you clarify why you (only) choose BSN for comparison? Is BSN the only suitable baseline?

2. Can you clarify the relation between C and Z? Also, at Line 157, C is a rank-one component, but at Line 272, C seems not?

3. Is it a fair comparison when you need to sweep to get the layers to be editted and use different energy threshold in different cases? And how are the energy threshold selected?

I would be happy to increase score if the author can address some of these questions and weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a method to edit model weights in Fisher curvature basis in order to tackle issues such as memorisation. Its central claim is that the Fisher Information Matrix of the model weights ($F_W$), when approximated layer-wise using the Kronecker-Factored Approximation (K-FAC), approximates the _curvature structure_ of the loss landscape. Each layer’s Fisher matrix is decomposed into Kronecker factors of input-activation and output-gradient covariances ($F_W \approx G \otimes A$), yielding an interpretable eigenstructure in which the curvature along a weight direction is given by $\lambda_i(G)\mu_j(A)$. High-curvature directions correspond to shared, generalisable mechanisms, while low-curvature modes that can be identified and pruned in the curvature basis, encode underconstrained, memorised artifacts whose removal reduces memorisation without degrading reasoning performance. Experiments are run on OLMO-2 7B with Balanced SubNet (BSN) as a baseline and on ViT-Base, on a variety of tasks, to confirm that pruning low-curvature Fisher modes reduces memorisation while preserving general reasoning and language quality.

### Strengths
- The work connects K-FAC to the semantic decomposition of model weights, being a novel application from its usage in optimisation. The semantic decomposition itself is shown to capture similar curvature-memorisation patterns in both language and vision, suggesting a a domain-agnostic property of deep networks.
- Reframes memorization vs generalization as spectral structure in curvature, bridging sharpness-aware optimization and localization studies.
- Unlike existing approaches like BSN which require labeled forget/retain sets, the K-FAC-based method requires no supervision beyond unlabeled data to compute the Fisher Information Matrix, in addition to memorisation labels across layers. 
- The paper also makes some interesting connections with potential for subsequent research:
     - can yield an explanation for why low-curvature subspaces harbour privacy leakage and memorised facts,
     - by being tractable, the curvature basis can lend to specific weight editing or interpretability of model weights for concepts to be controlled for

### Weaknesses
- Since the layer-wise K-FAC blocks ignore cross-layer effects, can you comment on how relevant such cross-layer correlations and coupling between attention weights and the weights of a layer's MLP would be? Is this a limitation of the proposed approach, or it is not relevant? 
 - How do we know that the Fisher curvature will be approximately the same as the loss curvature? 
 - I am unclear about the strength of the intriguing link between low curvature directions and memorisation, what is the theoretical justification behind this? How do we know these directions not correspond to just noise?
 - Further, the explanation for why K-FAC specifically captures the memorisation/generalisation distinction better than other decompositions (e.g., SVD) could be stronger. The explanation about averaging canceling idiosyncratic directions is intuitive but lacks formal analysis.


I am happy to reconsider my score if my issues and questions can be addressed.

### Questions
- How are the energy threshold for eigenvector selection determined from the range of 60-90%? Are there observable gaps in the eigenvalues separating high and low curvature directions? 
- Why is editing the specific layers selected the most effective, as are possible hypotheses for the observations from the sweeps across layers? Are there any memorisation specific eigenvectors that are consistent across layers or show some pattern?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper investigates how memorization and reasoning are represented within the weight space of LMs and ViTs by the curvature spectrum of the Fisher Information Matrix (K-FAC). The authors argue that high-curvature directions encode generalizable structure while low-curvature directions capture memorized or brittle behavior. They use this insight to design a curvature-based weight-editing method that suppresses memorized data more effectively than BalancedSubnet (BSN) while preserving general reasoning and validation accuracy.

The work is moderately novel, and the framing of reasoning vs. memorization in curvature space is conceptually elegant. However, it remains primarily descriptive and offers limited mechanistic insight for the claimed conceptual link between curvature and reasoning. Highly encourage the authors to deepen the causal analysis and broaden the evaluation during the discussion.

### Strengths
1. The paper raises an important question about how memorization and reasoning differ in weight-space geometry.
2. The use of curvature information (via K-FAC) for weight editing is technically sound and computationally efficient.
3. The idea of connecting curvature spectra with reasoning robustness is conceptually appealing.
4. The study is comprehensive, covering both vision and language domains, and connects curvature geometry to downstream behavioral changes.
5. The analysis of task brittleness (memorization → math → reasoning) is an inspiring conceptual framing.

### Weaknesses
1. The central claim is correlational: curvature and memorization are shown to co-vary. No causal analysis shows that curvature drives memorization.
2. The interpretation of eigenvectors as memorization vs. generalization directions is speculative. No visualization or qualitative evidence showing low-curvature directions actually encodes memorized samples.
3. Only one LM and one ViT are tested, and the memorization benchmarks are simplistic (mainly factual recall). Broader datasets or real-world leakage metrics (e.g., red-teaming) would strengthen the claim.
4. The paper alternates between “FIM curvature,” “loss curvature,” and “eigenbasis of weights” without always clarifying the mathematical distinctions.
5. The “reasoning continuum” is described qualitatively but not quantified; reasoning differences could reflect dataset or token-frequency effects rather than curvature structure.

### Questions
1. How robust are the results to the energy threshold (ρ) used for selecting eigenvectors—does performance vary smoothly or abruptly with ρ?
2. For the reasoning tasks, did you control for differences in perplexity or token entropy when interpreting drops in performance?
3. Is the curvature direction stable across checkpoints or random seeds, or do eigenbases drift substantially?
4. Can the method identify memorized content without labeled “memorized” examples?
5. Are curvature patterns stable across checkpoints or do they drift with training dynamics?

### Soundness
2

### Presentation
2

### Contribution
3
