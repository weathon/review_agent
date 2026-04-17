# PatchDNA: A Flexible and Biologically-Informed Alternative to Tokenization for DNA

- Decision: Accept (Poster)
- Scores: 2, 8, 4, 6

## Abstract
DNA language models are emerging as powerful tools for representing genomic sequences, with recent progress driven by self-supervised learning. However, performance on downstream tasks is sensitive to tokenization strategies reflecting the complex encodings in DNA, where both regulatory elements and single-nucleotide changes can be functionally significant. Yet existing models are fixed to their initial tokenization strategy; single-nucleotide encodings result in long sequences that challenge transformer architectures, while fixed multi-nucleotide schemes like byte pair encoding struggle with character level modeling. Drawing inspiration from the Byte Latent Transformer's combining of bytes into patches, we propose that 'patching' provides a competitive and more efficient alternative to tokenization for DNA sequences. Furthermore, patching eliminates the need for a fixed vocabulary, which offers unique advantages to DNA. Leveraging this, we propose a biologically informed strategy, using evolutionary conservation scores as a guide for 'patch' boundaries. By prioritizing conserved regions, our approach directs computational resources to the most functionally relevant parts of the DNA sequence. We show that models up to an order of magnitude smaller surpass current state-of-the-art performance in existing DNA benchmarks. Importantly, our approach provides the flexibility to change patching without retraining, overcoming a fundamental limitation of current tokenization methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes PatchDNA, a patch‐based alternative to tokenization for DNA sequence modeling. PatchDNA segments the nucleotide sequence into variable-length patches determined by a scoring function.

### Strengths
1. The local–global–local stack is well matched to genomics: local attention preserves base-level detail and global attention over patches amortizes long-range reasoning.

2. The paper conducts a broad set of ablation studies, which makes the empirical story more credible and helps clarify what actually drives the gains.

### Weaknesses
1. An advantage is retaining single-base resolution, yet there is limited coverage of base-level benchmarks where this matters most, variant effect prediction (VEP) .

2. The method is positioned as long-context friendly, but long-range evaluation remains thin and not fully controlled. Missing or underdeveloped: Bend or LRB benchmark.


3. Ablations need to show how average patch length affects long-range performance.


4. Claiming “conservation as boundary is better than conservation as feature” lacks causal evidence and visualization. The paper asserts a strong conclusion without systematic attribution. It is unclear why boundaries confer benefits beyond adding conservation as an auxiliary input.

5. Need to visualize how boundary placement changes long-range motif cooperation.

6. Despite the elegant design, the method often trails or only matches strong baselines on core tasks, suggesting modest practical effectiveness until stronger results are shown under matched settings.

### Questions
Please see Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes PatchDNA, a DNA language modeling framework that replaces traditional tokenization with a dynamic, vocabulary-free patching mechanism inspired by the Byte Latent Transformer. Patches are determined by biologically informed scoring functions such as evolutionary conservation, allowing the model to focus computational resources on functionally relevant regions. The framework further introduces re-patching, enabling modification of patch boundaries after pretraining for task or tissue specific adaptation without retraining. Experiments across multiple genomic benchmarks demonstrate improved efficiency, and competitive or superior performance compared to much reported baselines.

### Strengths
- Innovative representation: Replacing fixed tokenization with dynamic, vocabulary-free patching tailored to DNA sequences is innovative and is a domain (biology) inspired approach.

- Biological grounding: The proposed approach and biological prior grounded. Conservation-guided patch boundaries align model focus with functionally relevant genomic regions.

- Re-patching flexibility: The approach enables changing segmentation (in downstream/post-training) for new cell type or tissue contexts without the need of retraining. This makes is a practically useful one.

- Efficiency: The proposed method has reported to be cable of handles very large context (>100kb) contexts with small models. THis would reduce FLOPs compared to single-nucleotide models.

- Empirical breadth: The authors evaluated PatchDNA on four major genomic benchmarks where the PatchDNA achieve SoTA or near-SoTA results.

- Interpretability: Patch boundaries and conservation scores offer intuitive biological interpretability of learned representations.

- Technical rigor: Clear formulation, hyperparameters (essential for reproducibility), details of the baselines are provided.

### Weaknesses
- Limited testing: I would suggest the author conduct in-silico mutagenesis or perturbation analysis to verify that patches capture causal biological signals. Conducting in-silico mutagenesis is often not that expensive to run with limited computing resources.

- Statistical significance testing: Significance testing or variance reporting is missing. Adding this will clarify the performance gains.

- [Minor] Evaluation paradigms: All the benchmarks are supervised only. It would be interesting to see the PatchDNA's performance on unsupervised or generative task validation.

### Questions
- Can the authors conduct in-silico mutagenesis to confirm that patch boundaries capture causal biological signals?

- Can the authors report statistical significance for benchmark results and explore (or provide discussion on the potential of) PatchDNA’s performance on unsupervised tasks?

### Soundness
3

### Presentation
4

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
This paper introduces PatchDNA, a new framework for modeling DNA sequences that replaces traditional fixed tokenization methods with a flexible, dynamic "patching" mechanism. The core idea is inspired by the Byte Latent Transformer (BLT) and involves segmenting a DNA sequence into variable-length patches, where the patching strategy is biologically-informed.

### Strengths
1. The design of re-patching is biologically logical, which provides alternative tokenization to DNA modeling.
2. The paper provides extensive experimental results to support its claim.

### Weaknesses
1. Though the application in DNA domain seems to be successful, the methodological contributions are incremental to Byte Latent Transformer (BLT).
2. The additional experiments are needed to validate the efficiency of the proposed method.   
2.1 The method's performance depends on a predefined threshold $θ_p$ to create patches. The paper adopts a fixed percentile, but a deeper analysis of how this choice impacts different tasks is missing. Although the appendix (Table 22) shows that performance varies with the threshold, indicating this is a non-trivial parameter to tune, a more comprehensive discussion is needed.  
2.2 The current models are pre-trained exclusively on the human reference genome. While this is common practice, it limits the model's utility for genomics research on other species. Although Appendix A.5 shows promising zero-shot transfer to mouse, a truly foundational model for genomics should ideally be trained on a multi-species corpus to learn more universal biological principles.   
2.3 There are missing baselines such as Evo and Evo2.
3. The model's performance is contributed by the inductive bias that evolutionary conservation is a universal proxy for functional importance. This reliance might become a weakness for tasks where functionally important regions are not conserved. When re-patching with a better signal is not an option (as in the case of a truly novel task), the model might be fundamentally disadvantaged by its pre-training, as the patching strategy itself could be misaligned with the task's biology.

### Questions
1. The choice of the patching threshold $θ_p$ seems critical. Could the authors provide more intuition or analysis on the trade-off it governs? For instance, how does the average number of patches and downstream performance change across a wider range of thresholds? Is there a risk of "over-patching" (losing too much resolution) or "under-patching" (losing efficiency gains) on certain tasks?
2. In Table 3, the cCRE-aware re-patching improves Gene and Cell Pearson scores but significantly degrades the Full Pearson score (from 0.471 to 0.408),  which is counter-intuitive.  Could the authors provide an explanation for this situation?  
3. This work demonstrates remarkable parameter efficiency, which is a key goal in model design. Another critical property of successful architectures like the Transformer is their adherence to scaling laws. Have the authors investigated whether the PatchDNA framework exhibits predictable scaling behavior? Specifically, if the authors increase model parameters and training data, how does the method behave in contrast to standard tokenization methods?
4. In the context of building a general biological foundation model, we might encounter diverse or completely unknown input types, rendering task-specific re-patching impossible. In such a scenario, the choice of the scoring function $g_p$ becomes crucial. How would PatchDNA handle a task where the important regions are anti-correlated with evolutionary conservation? In such a scenario, does the framework offer a way to dynamically adapt the scoring function if a mismatch with the downstream task is detected?
5. The recently proposed Evo model is state-of-the-art on many genomic benchmarks. Although it is a much larger model, a comparison would be valuable. Have the authors considered benchmarking PatchDNA against Evo on any tasks?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces PatchDNA, which draws on the ideas of the Byte Latent Transformer (BLT) to shift DNA sequence modeling from traditional tokenization to "patching." The authors further introduce patch boundary selection based on conservation signals and a "re-patching" mechanism that allows for the redefinition of patch strategies after pre-training, thereby enhancing flexibility and efficiency.

Core Contribution: By leveraging concepts from computer vision (CV) and BLT, the approach transforms the tokenization problem into a "patching" problem, completely eliminating the need for a fixed vocabulary. It utilizes external biological signals, such as conservation scores (PhyloP), to guide the delineation of patch boundaries.

### Strengths
- The idea is novel, completely eliminating the limitations of a fixed vocabulary and circumventing the rigidity of k-mers and the statistical constraints of BPE. 
- By guiding patch boundaries with conservation scores, it provides a biological inductive bias. 
- Experimental results show performance that is comparable to or even surpasses that of large-scale models, while being more computationally efficient.

### Weaknesses
- Non-End-to-End: The patch boundaries rely entirely on external signals (conservation, entropy), and the model itself does not learn the importance of the regions.

- Insufficient Empirical Support: The performance improvements from re-patching are not adequately quantified, remaining largely at the conceptual level.

- Limited Interpretability: There is a lack of in-depth analysis of the relationship between the representations learned by the model and biological functions.

- High Implementation Complexity: While being "vocabulary-free" offers flexibility, it results in poor standardization and reproducibility.

**Improvement Suggestions:**

- Introduce a trainable boundary scoring network to jointly optimize patch segmentation and representation learning.
- Conduct a quantitative analysis of the actual impact of re-patching on performance and efficiency.
- Increase the analysis of the overlap between patches and functional regions (conserved regions, regulatory elements).
- Publicly disclose memory usage and computational load comparisons to ensure the claimed efficiency is credible.


Besides, this paper seems to have been previously presented at a NeurIPS Workshop (https://openreview.net/group?id=NeurIPS.cc/2025/Workshop/AI4D3&referrer=%5BHomepage%5D(%2F)#tab-accept-oral); however, my comments do not take this into consideration. I leave it to the AC to make a judgment on this matter.

### Questions
please refer to the weaknesses part

### Soundness
3

### Presentation
3

### Contribution
3
