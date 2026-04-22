# MSAFlow: a Unified Approach for MSA Representation, Augmentation, and Family-based Protein Design

- Avg Score: 5.33
- Decision: Reject
- Scores: 8, 4, 4

## Abstract
Multiple Sequence Alignments (MSAs) provide fundamental information about protein evolution and play crucial roles in downstream applications, such as structure prediction and family-based design. However, constructing high-quality MSAs requires significant computational resources to query natural protein databases, and traditional techniques fail to retrieve sufficient data for proteins with limited homology. While recent generative models have been proposed for MSA augmentation, they often struggle to capture complex, high-order dependencies in sequence distributions while maintaining permutation invariance. To address these challenges, we introduce MSAFlow, a framework built on two key innovations. First, its core is a novel generative autoencoder that pairs a compressed AlphaFold3 (AF3) MSA representation with a conditional Statistical Flow Matching (SFM) decoder to faithfully model a family's sequence distribution that preserves permutation invariance. Second, we introduce a latent flow-matching model that performs zero-shot generation of MSA embeddings from a single sequence, enabling powerful augmentation for orphan proteins. By integrating these components, MSAFlow operates as a unified framework for MSA representation, augmentation, and family-based design. Our experiments demonstrate that MSAFlow significantly outperforms existing models on family-based protein design and MSA augmentation tasks, especially for low-homology proteins. MSAFlow is lightweight, fast, and memory-efficient, offering a single, versatile solution for diverse protein engineering tasks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces a lightweight generative framework MSAFlow that unifies multiple sequence alignment (MSA) representation, augmentation, and protein family–based design within a single model. MSAFlow integrates a compressed AlphaFold3-derived MSA encoder with a conditional Statistical Flow-Matching decoder to model protein family sequence distributions while maintaining permutation invariance. It further employs a latent flow-matching model to generate synthetic MSA embeddings directly from a single sequence, enabling zero-shot augmentation for orphan or low-homology proteins. Empirically, MSAFlow achieves state-of-the-art results on MSA reconstruction, shallow/zero-shot augmentation, and enzyme family design tasks—surpassing larger models like MSAGPT and EvoDiff while being more efficient (130 M parameters). The framework advances protein generative modeling by providing a unified, efficient, and biologically consistent approach for encoding, augmenting, and designing protein sequence families.

### Strengths
Originality
- The paper introduces a highly novel formulation of MSA generative modeling through the integration of AlphaFold3-derived MSA embeddings with Statistical Flow Matching (SFM). Unlike prior MSA generation approaches (e.g., MSAGPT, EvoDiff, ProfileBFN), MSAFlow unifies representation, augmentation, and family-based design within a single permutation-invariant framework. The incorporation of a latent flow-matching module for zero-shot MSA embedding generation from single sequences represents a creative and impactful extension beyond existing paradigms.

Quality
- The methodology is rigorous, grounded in well-established theoretical principles, and executed with technical precision. The model design—combining compressed AF3 embeddings, conditional DiT architecture, and spherical-geodesic flow formulation—is mathematically coherent and experimentally validated. The experiments span multiple benchmarks (reconstruction, zero-shot augmentation, enzyme design), include comprehensive ablations, and demonstrate consistent superiority over baselines.

Clarity
- The paper is well-structured and clearly written, making complex ideas accessible through precise explanations and well-designed figures. Mathematical formulations are clearly presented and logically connected to the overall framework. The authors contextualize their approach within prior work effectively, highlighting conceptual distinctions and technical improvements.

Significance
- The contributions are significant for both protein informatics and machine learning. By enabling biologically plausible MSA generation even for low-homology or orphan proteins, MSAFlow directly advances the frontier of data-efficient protein design and structure prediction. Its efficiency and versatility (130M parameters, scalable to variable sequence lengths) make it highly practical for large-scale biological applications. The framework opens new research directions in unified latent modeling of evolutionary sequence spaces and conditional generative protein design.

### Weaknesses
- Although the paper reports runtime and memory advantages, it does not analyze scaling behavior with sequence length or MSA depth. Including profiling results for longer sequences or very deep MSAs would strengthen claims of scalability and efficiency.
- The paper does not explore what biological or evolutionary features are captured in the learned latent space. Since MSAFlow’s novelty lies partly in compressing and manipulating evolutionary distributions, analyses of latent representations (e.g., clustering by protein family, correlation with conservation or coevolution metrics) would help interpretability and justify the model’s generalization claims.
- The model’s reliance on AlphaFold3-derived embeddings (via Protenix) and ESM2 representations raises questions about its independence from large pretrained models. It is not entirely clear how much of MSAFlow’s success is attributable to the novel flow-matching formulation versus the quality of these pretrained representations. An analysis using simpler or alternative encoders could clarify the framework’s intrinsic capability.

### Questions
- Could the authors provide a more detailed analysis of what evolutionary or functional information is captured in the learned MSA latent representations? For instance, do embeddings cluster by protein family, correlate with conservation or coevolution scores, or reflect known functional motifs? Such an analysis would help clarify the biological interpretability of MSAFlow’s latent space.
- Since MSAFlow relies on AlphaFold3-derived and ESM2 embeddings, how much of the observed performance gains stem from these pretrained representations rather than the flow-matching framework itself? Could the model maintain strong results when trained with simpler or independently trained encoders?
- While the paper demonstrates strong efficiency at moderate scales, can MSAFlow handle extremely deep MSAs (e.g., >10,000 sequences) or very long sequences (L > 1000)? Providing empirical or theoretical scaling analyses would clarify the limits of the approach for large-scale applications.
- How diverse are the generated sequences compared to ground-truth MSAs? Have the authors measured sequence identity distributions across generated sets to confirm that MSAFlow produces evolutionarily diverse yet plausible variants, rather than near-duplicates?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents MSAFlow, a unified autoencoding framework for multiple sequence alignment (MSA) representation, augmentation, and family-based protein design. MSAFlow leverages compressed AlphaFold3-based MSA embeddings and a conditional Statistical Flow Matching (SFM) decoder to reconstruct, generate, or augment MSAs in a permutation-invariant manner. The model further introduces a latent flow-matching component enabling zero-shot generation of synthetic MSAs from single-sequence embeddings (e.g., ESM2), thereby expanding the range of proteins for which high-quality MSAs can be operated on or synthesized. Empirical results suggest MSAFlow significantly outperforms existing approaches on challenging protein design and MSA augmentation tasks, particularly for low-homology or orphan proteins, while maintaining notable efficiency in resource usage.

### Strengths
+ Unified Framework: Integrates representation, augmentation, and family-based design into a single modular encoder–latent–decoder architecture (Figure 1).

+ Mathematically Rigorous & Efficient: Uses permutation-invariant encoding and Statistical Flow Matching on categorical manifolds, achieving strong accuracy with only 130 M parameters.

+ Comprehensive Validation: Demonstrates consistent SOTA or competitive results across reconstruction, few-shot augmentation, and protein design benchmarks (Tables 2–5, Figures 4–5).

### Weaknesses
+ Limited Theoretical Justification: The paper lacks clear reasoning for using Fisher-Rao geodesics and sphere mapping over alternatives (e.g., Wasserstein flow) and does not justify mean pooling or the chosen manifold’s suitability for protein sequences.

+ Insufficient Ablation Clarity: Ablation studies do not isolate contributions of key components (e.g., AdaLN vs. SFM decoder), making it unclear which design choices drive performance gains.

+ Benchmark and Discussion Gaps: Metrics and baselines are unevenly compared, missing confidence intervals and diversity analyses, while limitations and potential failure cases are only briefly addressed.

### Questions
1. Could the authors clarify why the Fisher-Rao geodesic interpolation and unit sphere mapping is specifically advantageous for protein sequence distributions, as opposed to other statistical manifold metrics or flow-matching paths (such as Wasserstein, or purely categorical diffusion)? Are there concrete empirical or theoretical performance gains for this choice?
2. Can the authors provide a direct ablation between global and position-wise AdaLN conditioning in the decoder? Specifically, can any performance gain be attributed to resilience to sequence permutation, or is it rather due to more fine-grained representation? Details in Figure 3 would suggest the latter, but controlled experiments should make this explicit.
3. Will full code and pretrained models be released for all baselines? Are all test/validation splits, configurations, and hyperparameters for main and ablation experiments provided in the main text (not only appendix)? Can the authors supply statistical confidence intervals (bootstrapping or other) for main table results (e.g., Table 2, Table 3)?

### Soundness
2

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
4

### Summary
The authors propose a flow matching framework that can generate MSA embeddings for both full MSAs as well as single sequences, thereby performing MSA augmentation. They compare it to other frameworks performing similar tasks for zero-/few-shot MSA generation in the context of structure prediction as well as family-based enzyme design.

### Strengths
[S1] Combnation of learning a latent MSA representation with SFM for decoding to categorical amino acid distributions is elegant and well motivated. The position-specific adaln is also a nice use of the compressed information in SFM.

[S2] Show improvement on certain subclasses of sequences for prediction accuracy

[S3] Compared to previous autoregressive baselines, MSAFlow is truly permutation invariant wrt the ordering of the MSA, which is desirable

### Weaknesses
[W1] The bitwise information content comparison to the deep MSA is a bit unfair since many of these sequences are probably highly similar; previous work has shown that with clustered MSAs one can reduce the MSA depth quite a bit without performance loss. For a fair information content comparison the authors should cluster the ground truth MSA down to the same depth as their reconstructed MSAs and compare performance.

[W2] The authors use the results in Table 2 to argue that they significantly outperform prior baselines. However, it is not really clear that this is the case; in the few shot setting MSA all methods are worse or similar to no/shallow MSA, and the few-shot settings seems the practically relevant one.

[W3] In Figure 5 the authors compare their method to MSAGPT and show that they outperform it on three examples from a scarce MSA dataset. However, as shown in Table 2 the real proper baseline is No/Shallow MSA, this should be added to the Figure to make the point in a convincing manner.

### Questions
[Q1] In Figure 4, one can see that the zero shot MSA helps for certain sequences but not for others. Did the authors perform any analysis to understand which sequences fail to benefit from their approach?

[Q2] How meaningful are the results in Table 1 given the relatively high standard deviation of the different methods as well as the GT? Also is it desirable to have drastically lower standard deviation than the GT on this benchmark?

[Q3] Why are there no results on variable length for Q15I65 in Table 3? And why is the 84% performance on Q15BH7 bolded despite underperforming compared to ProfileBFN?

[Q4] The authors describe different latent diffusion models in their background section, but do not mention the current SOTA method La-Proteina in that section. Is there a specific reason for that?

### Soundness
2

### Presentation
3

### Contribution
2
