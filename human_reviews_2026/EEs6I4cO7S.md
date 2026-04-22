# AbsTopK: Rethinking Sparse Autoencoders For Bidirectional Features

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 6, 2, 4

## Abstract
Sparse autoencoders (SAEs) have emerged as powerful techniques for interpretability of large language models (LLMs), aiming to decompose hidden states into meaningful semantic features. While several SAE variants have been proposed, there remains no principled framework to derive SAEs from the original dictionary learning formulation. In this work, we introduce such a framework by unrolling the proximal gradient method for sparse coding. We show that a single-step update naturally recovers common SAE variants, including ReLU, JumpReLU, and TopK. Through this lens, we reveal a fundamental limitation of existing SAEs: their sparsity-inducing regularizers enforce non-negativity, preventing a single feature from representing bidirectional concepts (e.g., male vs. female). This structural constraint fragments semantic axes into separate, redundant features, limiting representational completeness. To address this issue, we propose AbsTopK SAE, a new variant derived from the $\ell_0$ sparsity constraint that applies hard thresholding over the largest-magnitude activations. By preserving both positive and negative activations, AbsTopK uncovers richer, bidirectional conceptual representations. Comprehensive experiments across multiple LLMs and seven probing and steering tasks show that AbsTopK improves reconstruction fidelity, enhances interpretability, and enables single features to encode contrasting concepts. Remarkably, AbsTopK matches or even surpasses the Difference-in-Mean method—a supervised approach that requires labeled data for each concept and has been shown in prior work to outperform SAEs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors mathematically prove an equivalence between commonly used SAE architectures and sparse regularizers. They then use the insights gained from this connection to propose a new type of SAE which is able to encode bidirectional features as a single dictionary element. They empirically demonstrate that this new SAE architecture outperforms pre-existing SAEs on many commonly used evaluations.

### Strengths
1. The authors make a nontrivial connection between SAEs and sparse regularizers using proximal operators, which gives a new perspective on the differences between SAE architectures. This seems to be well-executed (though I didn’t go through the math and proof in too much detail).  
2. The evaluations are quite comprehensive in terms of covering all the relevant axes of SAE quality. They demonstrate that AbsTopK are a clear improvement over JumpReLU and TopK SAEs (even though the size of the improvement is relatively small on most evaluations).  
3. The SAE variant they propose is clean and elegant in how it addresses the problem of bidirectional features.

### Weaknesses
1. The evaluations should include more baselines. They currently include only 2: TopK and JumpReLU. They should include a handful of others (even if they don’t neatly fit into the framework introduced in section 2). ReLU, Matryoshka, and gated SAEs would all be welcome additions.  
2. I want to see more empirical results showing how bidirectional features are encoded in AbsTopK SAEs. The single example in Figure 1 is nice, but it’s just a single example which isn’t sufficient to convince me that this is representative.
3. I’d like to see results from more layers. Currently as far as I can tell they only test on 2 layers in each model, and they are always somewhere roughly in the middle of the model. Include early layers and late layers would give me more confidence that the observed improvements generalize.  
4. For the experiments Table 1, it would be good to see 4ish different models rather than just 2\. I agree with the reasoning for excluding the very small models here, but it would be nice to add e.g. a Llama model. I’d also weakly recommend turning table 1 into a series of plots (e.g. with MMLU on the x-axis and HarmBench on the y-axis); it would make it more easily digestible.  
5. More generally in all experiments, it would be nice to include one or two larger models (e.g. 10B+ parameters).  
6. Typo: the legend inside of each subfigure in Figures 2, 3, and 4 refers to “AbsoluteK”. I’m assuming this refers to AbsTopK, perhaps the authors switched from one name to the other at some point during the writing process but forgot to update the figures.

### Questions
See the Weaknesses section. I’d particularly recommend focusing on points 1 and 2, I see those as the main weaknesses of the paper, if you address them then I’m likely to raise my score.

### Soundness
3

### Presentation
4

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
This paper introduces a principled framework that derives sparse autoencoders (SAEs) from the proximal gradient method for sparse coding, showing that common variants like ReLU, JumpReLU, and TopK naturally emerge as single-step updates. The analysis reveals a key limitation of existing SAEs—their non-negativity constraints prevent features from capturing bidirectional semantics (e.g., male vs. female), leading to fragmented representations. To address this, the authors propose AbsTopK SAE, which applies magnitude-based hard thresholding to preserve both positive and negative activations. Experiments across multiple LLMs and interpretability tasks show that AbsTopK improves reconstruction, enhances interpretability, and enables single features to represent contrasting concepts, matching or surpassing supervised baselines.

### Strengths
1.	Although sparse autoencoders have emerged as a promising approach to improving the interpretability of large language models, their theoretical understanding remains limited. This paper proposes a novel framework based on a proximal perspective to analyze SAEs. The theoretical analysis is natural, rigorous, and insightful, providing meaningful guidance for future research in this area.
2.	The paper convincingly demonstrates that ignoring negative components in the latent space degrades the performance of sparse autoencoders. Replacing traditional activation functions with AbsTopK is a reasonable and well-motivated solution, supported by both solid theoretical reasoning and comprehensive empirical validation.
3.	The writing and presentation are clear, coherent, and well-structured. The theoretical and empirical sections complement each other effectively, and the overall logic flow is easy to follow. The theoretical part offers sufficient intuition without being overly mathematical. I appreciate the clarity and accessibility of the presentation.
4.	The authors provide thorough experimental verification showing that AbsTopK SAEs outperform existing variants in both reconstruction and interpretability-related tasks.

### Weaknesses
1.	My main concern is whether AbsTopK may compromise the monosemanticity property of sparse autoencoders. One of the most desirable characteristics of SAEs is that each dimension is activated primarily by a single concept. However, since AbsTopK activates both positive and negative top-K components, this property might be weakened. For instance, if a single feature dimension responds to both male and female concepts, it becomes difficult to isolate them semantically. In addition, the paper does not provide an evaluation of monosemanticity metrics, such as auto-interpretability scores. I would be happy to raise my evaluation if the authors can clarify or empirically address this concern.
2.	As the effectiveness of AbsTopK SAEs remains somewhat uncertain, I consider the main contribution of this paper to be its theoretical framework for understanding SAEs. Therefore, in the related work section, it would strengthen the paper to include a more explicit comparison with existing theoretical analyses of SAEs, such as [1], [2], and related studies.


[1] Chen, Siyu, et al. "Taming Polysemanticity in LLMs: Provable Feature Recovery via Sparse Autoencoders." arXiv preprint arXiv:2506.14002 (2025).
[2] Cui, Jingyi, et al. "On the Theoretical Understanding of Identifiable Sparse Autoencoders and Beyond." arXiv preprint arXiv:2506.15963 (2025).

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes AbsTopK SAE, a new variant of sparse autoencoder derived from the proximal gradient perspective on dictionary learning. The authors revisit the mathematical foundation of SAEs, showing that existing variants (ReLU, JumpReLU, TopK) can be unified under the proximal operator framework. From this analysis, they identify a structural limitation: current SAEs enforce non-negativity in activations, preventing single features from representing bidirectional concepts (e.g., male–female, positive–negative sentiment).
To overcome this, they introduce the AbsTopK operator, which selects the top-|k| activations by absolute magnitude, thus allowing both positive and negative activations. Experiments across several LLMs and benchmarks (probing, steering, MMLU, HarmBench) show that AbsTopK achieves better reconstruction fidelity, preserves model utility, and can encode contrasting semantics within a single feature dimension.

### Strengths
- Principled theoretical framing – The proximal-operator derivation provides a unifying mathematical lens for understanding existing SAE variants and justifies design differences between ReLU, JumpReLU, and TopK in a coherent way.
- Simple and reproducible modification – The AbsTopK operator is straightforward to implement and can be directly integrated into existing SAE pipelines.

### Weaknesses
- Lack of feature interpretability evaluation – The paper’s central claim concerns semantic bidirectionality, yet no rigorous analysis or quantitative metric is provided to assess whether positive and negative activations of a single feature correspond to semantically opposite concepts.
- Many linguistic or conceptual axes are not naturally symmetric or have no meaningful “opposite” (e.g., tree, city).
- Without systematic qualitative or quantitative validation (e.g., feature visualization, activation clustering, or concept alignment), the interpretability claim remains speculative.
- Interpretability vs. utility conflation – Improved reconstruction or downstream task performance does not necessarily imply better interpretability; this distinction should be explicitly discussed.

### Questions
Could you quantitatively evaluate whether positive/negative activations of a single AbsTopK feature correspond to semantically opposite text examples?
It would be useful to add qualitative visualizations (e.g., top positive vs. negative activating examples) for multiple features, not just one or two, to support the bidirectionality claim.

### Soundness
1

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
This paper proposes AbsTopK SAE, a novel sparse autoencoder variant that removes the non-negativity constraint in conventional SAEs.
While existing SAEs such as ReLU, JumpReLU, and TopK enforce non-negative activations, they cannot represent bidirectional semantic axes (e.g., male–female, positive–negative sentiment), leading to feature fragmentation.

The authors rederive SAEs from a proximal gradient framework of sparse coding, revealing that the non-negativity arises from implicit regularizers.
They introduce AbsTopK, which performs hard thresholding over the largest-magnitude activations (i.e., L0 sparsity without sign restriction), enabling both positive and negative activations to coexist in a single feature.
Experiments across four LLMs (GPT2-Small, Pythia-70M, Gemma-2B, Qwen-4B) and seven probing/steering benchmarks show that AbsTopK improves reconstruction fidelity, interpretability, and bidirectional concept encoding, matching or even surpassing the supervised Difference-in-Mean baseline.

### Strengths
- AbsTopK consistently outperforms TopK and JumpReLU across reconstruction, probing, and steering tasks.

- Derivation from the proximal gradient method grounds the design of SAEs in dictionary-learning theory, showing why prior variants enforce non-negativity.

- Both theoretical and empirical validations are presented coherently.

### Weaknesses
- The paper assumes that features should be bidirectional (e.g., male-female, positive-negative). However, not all concepts have clearly defined opposites. For such unipolar or abstract concepts (e.g., syntax awareness, topic consistency), the interpretation of negative activations remains ambiguous. It is unclear whether these activations correspond to an absence of a feature, an opposing property, or noise.

- While the experiments use standardized benchmarks such as SAEBench, the paper would benefit from direct analyses showing how AbsTopK changes the feature space compared to the standard SAEs.

- The evaluation focuses mainly on a single activation type (residual stream) and a limited set of layers. Since SAE behavior often depends strongly on layer position and activation type (Attention, MLP, Residual Stream) [1], extending experiments across multiple layers and activation sources would make the evidence more comprehensive and convincing.

- In addition, the results are promising, but the paper does not report variability across random seeds or runs. Providing standard errors or confidence intervals would strengthen the reliability of empirical conclusions.

[1] Rethinking evaluation of sparse autoencoders through the representation of polysemous words, ICLR2025

### Questions
- Several recent studies challenge the Linear Representation Hypothesis [1,2]. How does the proposed framework relate to or address these counterarguments, or do you have any discussion?

- Could the AbsTopK approach be extended or adapted to Transcoder[3] or CrossCoder[4] architectures?

[1] Interpreting Neural Networks through the Polytope Lens.  
[2] Not All Language Model Features Are One-Dimensionally Linear, ICLR2025.    
[3] Transcoders Find Interpretable LLM Feature Circuits, Neurips2024.    
[4] Sparse Crosscoders for Cross-Layer Features and Model Diffing.

### Soundness
3

### Presentation
3

### Contribution
2
