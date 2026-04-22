# FragFM: Hierarchical Framework for Efficient Molecule Generation via Fragment-Level Discrete Flow Matching

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
We introduce FragFM, a novel hierarchical framework via fragment-level discrete flow matching for efficient molecular graph generation. FragFM generates molecules at the fragment level, leveraging a coarse-to-fine autoencoder to reconstruct details at the atom level. Together with a stochastic fragment bag strategy to effectively handle a large fragment space, our framework enables more efficient, scalable molecular generation. We demonstrate that our fragment-based approach achieves better property control than the atom-based method and additional flexibility through conditioning the fragment bag. We also propose a Natural Product Generation benchmark (NPGen) to evaluate the ability of modern molecular graph generative models to generate natural product-like molecules. Since natural products are biologically prevalidated and differ from typical drug-like molecules, our benchmark provides a more challenging yet meaningful evaluation relevant to drug discovery. We conduct a comparative study of FragFM against various models on diverse molecular generation benchmarks, including NPGen, demonstrating superior performance. The results highlight the potential of fragment-based generative modeling for large-scale, property-aware molecular design, paving the way for more efficient exploration of chemical space.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces FragFM, a hierarchical framework for molecular graph generation that operates at the fragment level by combining a discrete and continuous flow matching approach with a coarse-to-fine autoencoder. The design enables efficient and scalable molecular generation by modeling chemically meaningful fragments, rather than individual atoms, while ensuring all-atom validity during molecular reconstruction. The authors also propose NPGen, a new benchmark focusing on natural products, which challenges existing approaches to generate larger, more complex, and more diverse biologically relevant compounds. Experimentally, FragFM outperforms atom- and fragment-based models on both standard molecule benchmarks and NPGen, and also demonstrates the ability to conditionally generate molecules based on desirable properties while maintaining validity.

### Strengths
1. The fixed-length graph embedding for fragments and fragment bag strategy allows scalable training for large fragment libraries without sacrificing chemical expressivity. By introducing the conditional fragment bag sampling term $\lambda_B$, the fragment bag restriction does not hinder conditional generation. 
2. For both conditional and unconditional molecule generation, FragFM maintains high RDKit validity, demonstrating that the hierarchical approach and Blossom algorithm are effective methods to ensure chemical reasonableness.
3. FragFM outperforms atom-based and fragment-based methods in most metrics while also demonstrating a high sampling efficiency, even compared to an existing flow matching model DeFoG.

### Weaknesses
1. Despite the inclusion of chemical validity as a benchmark, the authors omit retrosynthetic analyses (i.e. Syntheseus) for generated molecules. This is especially important considering the reliance of the model on connecting building blocks via junction atoms, which does not necessarily guarantee a feasible experimental synthesis route.
2. Although many relevant natural products are macrocycles, the ground-truth graphs are derived directly from the results of BRICS decomposition, which results in acyclic graphs. As the Blossom algorithm employed only considers pairs of junction atoms with edges decided by this fragmentation, the model is likely unable to generate macrocycles.
3. While the hierarchical combination of fragment-level flow matching and coarse-to-fine reconstruction is well-executed, the scope of the work is somewhat incremental. Flow matching and autoencoder-based 2D molecular generation at the fragment level are individually present in the literature, as is the architecture used during training. 

Minor:
1. The work omits training details for the conditional property regressor.
2. Though it may exceed the scope of the work, the model does not explicitly consider information relevant to some conditional generation tasks, such as the local geometry of the protein pocket for binding affinity.

### Questions
1. It is somewhat unclear how the regressor, which decides conditional fragment bag sampling weights, is trained. Does the model predict the property values of each fragment directly? If not, is it instead trained to predict the average property value of all molecules in the training dataset containing the fragment? 
2. Does the NPGen dataset contain macrocycles? 
3. Does conditional generation with higher $\lambda_B$ result in lower diversity? It would be interesting to see the trade-off between the strength of conditioning signals like $\lambda_B$ and $\lambda_X$ and the diversity of the results.
4. Is there any merit to changing the relative schedules of the latent vector and molecular graph during sampling? For instance, would it be possible to construct an analog design experiment by conditioning on a fully denoised latent vector of a difficult-to-synthesize molecule, then allowing the discrete graph flow matching module to run and generate analogs with a chosen fragment library?

### Soundness
3

### Presentation
3

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
This paper studies the problem of molecular graph generation. The authors propose FragFM, a novel hierarchical framework via fragment-level discrete flow matching. The authors also propose the Natural Product Generation benchmark for comprehensive evaluation. FragFM enables effective distribution learning and property-guided molecular generation.

### Strengths
- FragFM includes a coarse-to-fine autoencoder to deal with the ambiguity in reconstructing atomic connections.
- FragFM introduces a dynamic fragment vocabulary.

### Weaknesses
- In related work, hierarchical generation models such as MolGrow[1], MolHF [2], and Coarse-to-fine [3] should be included. And their comparison with fragment-based methods should be discussed.
- The coarse-to-fine autoencoder is not fully data-driven, which depends on predefined fragmentation rule and Blossom algorithm.
- Several important baselines are missing in the proposed benchmark NPGen. For example, SAFE-GPT, a fragment-based method that performs well in MOSES.
- Lacking ablation. The effectiveness of the autoencoder and the dynamic fragment vocabulary should be discussed.

[1] MolGrow: A graph normalizing flow for hierarchical molecular generation

[2] Molhf: A hierarchical normalizing flow for molecular graph generation

[3] Coarse-to-fine: a hierarchical diffusion model for molecule generation in 3d

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces FragFM, a hierarchical framework for molecular graph generation that combines fragment-level discrete flow matching with a coarse-to-fine autoencoder for efficient and scalable drug-like molecule generation. The authors also propose NPGen, a new benchmark designed to evaluate the generation of complex, natural product–like molecules.

### Strengths
1. Interesting ideas that construct flows on latent variables and coarse graphs. 
2. The “in-bag” InfoNCE formulation is computationally efficient, and provides a surrogate for different computation budget, with the bag size increasing, $x_1$ posterior becomes unconditional.
3. Strong performance on both standard benchmarks and NPGen.

### Weaknesses
1. The paper only evaluates 2D molecular graphs. It would strengthen the contribution to include 3D molecular generation. 
2. While fragment-level generation is inherently interpretable, the paper does not analyze fragment–property correlations or latent space structure.

### Questions
1. How sensitive is FragFM to the choice of fragment decomposition rules (e.g., BRICS vs. other methods)? 
2. Coarse graphs are generated by fixed chemical tools, can this process be learned by another graph autoencoder instead?
3. How is the fragment property predictor trained?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a fragment-based approach for molecular generation, with a particular focus on scalability to larger molecules (natural products; NPs).

The focus on NPs makes it difficult for me to judge the impact (of both the method and the dataset); I suspect it's somehow niche compared to the general small molecule generation, but at the same time it offers a comprehensive contribution (with both a benchmark and a new method).

### Strengths
- The paper is clearly written and easy to follow.
- The approach focuses on scaling to the generation of large molecules (namely natural products), which is an interesting application area.
- The approach seems novel, and has domain-specific contributions (e.g. coarse encoding)
- A new benchmark dataset is introduced, which is an additional contribution.
- Besides the comment below, the evaluation is in my opinion sound and fairly typical for 3D generation papers; it supports the claims made.

### Weaknesses
- The authors rely on fairly old baselines in experiments past unconditional generation. This raises questions about relative performance of the method compared to SotA. It’s also not clear why only an atom-based baseline is used in conditional generation.

### Questions
- The authors provide a figure for sampling times; how does the model compare in terms of training time?

### Soundness
3

### Presentation
3

### Contribution
3
