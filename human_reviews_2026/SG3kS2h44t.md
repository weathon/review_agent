# ReLaSH: Reconstructing Joint Latent Spaces for Efficient Generation of Synthetic Hypergraphs with Hyperlink Attributes

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Hypergraph network data, which capture multi-way interactions among entities, have become increasingly prevalent in the big data era, spanning fields such as social science, medical research, and biology. Generating synthetic hyperlinks with attributes from an observed hypergraph has broad applications in data augmentation, simulation, and advancing the understanding of real-world complex systems. This task, however, poses unique challenges due to special properties of hypergraphs, including discreteness, hyperlink sparsity, and the mixed data types of hyperlinks and their attributes, rendering many existing generative models unsuitable. In this paper, we introduce ReLaSH (REconstructing joint LAtent Spaces for Hypergraphs with attributes), a general generative framework for producing realistic synthetic hypergraph data with hyperlink attributes via training a likelihood-based joint embedding model and reconstructing the joint latent space. Given a hypergraph dataset, ReLaSH first embeds the hyperlinks and their attributes into a joint latent space by training a likelihood-based model, and then reconstructs this joint latent space using a distribution-free generator. The generation task is completed by first sampling embeddings from the distribution-free generator and then decoding them into hyperlinks and attributes through the trained likelihood-based model. Compared with existing generative models, ReLaSH explicitly accounts for the unique structure of hypergraphs and jointly models hyperlinks and their attributes. Moreover, the likelihood-based embedding model provides efficiency and interpretability relative to deep black-box architectures, while the distribution-free generator in the joint latent space ensures flexibility. We theoretically demonstrate consistency and generalizability of ReLaSH. Empirical results on a range of real-world datasets from diverse domains demonstrate the strong performance of ReLaSH, underscoring its broad utility and effectiveness in practical applications.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose **ReLaSH**, a hypergraph generative model that produces random hypergraphs together with _hyperedge-level features_. The key idea is to construct a **joint latent space** of hyperlinks and their attributes and learn it in a probabilistic, likelihood-based manner.

- **Data-driven latent space learning:** Build a joint latent representation that maximizes the likelihood of observed hyperedges and their associated features.

- **Latent reconstruction and decoding:** Reconstruct this latent distribution via a generator, and decode new samples from it to synthesize realistic hyperedges and their attributes.

### Strengths
- **\[S1]** The problem of _hypergraph generation with features_ is timely and important.

- **\[S2]** ReLaSH introduces, to my knowledge, the first _probabilistic and likelihood-based_ joint generative framework for hypergraphs with hyperlink attributes.

**\[S3]** The experiments span datasets with diverse feature modalities (textual, numerical, categorical), showing the model’s potential generality.

### Weaknesses
See “Questions” below.

### Questions
- **\[Q1]** _Missing references._ It seems that several recent works also tackle hypergraph generation with attributes. Could the authors clarify the novelty of ReLaSH relative to these and, if feasible, include them as baselines?

  - \[r1] Gailhard et al. _Feature-aware Hypergraph Generation via Next-Scale Prediction_, arXiv:2506.01467 (2025).

  - \[r2] Chun et al. _Attributed Hypergraph Generation with Realistic Interplay Between Structure and Attributes_, arXiv:2509.21838 (2025).

  - \[r3] Badalyan et al. _Structure and Inference in Hypergraphs with Node Attributes_, _Nature Communications_ 15 (2024): 7073.

- **\[Q2]** The proposed framework currently handles _hyperedge features_ only, while _node features_ are at least equally important and more common in real applications. Could the authors discuss whether and how ReLaSH could be extended to incorporate node features?

- **\[Q3]** I have concerns about the _evaluation metrics_, which do not appear to be standard. Could the authors justify why these particular choices are reasonable compared with alternatives, and clarify what properties of the generative distribution are expected to be captured if these losses are ideally minimized?

- **\[Q4]** Relatedly, is it conceptually possible to design a model that _directly optimizes_ these metrics (or differentiable surrogates) instead of using them only post hoc? What practical or theoretical challenges would such a design entail?

- **\[Q5]** Several configurations (k = 6, 12, 24, 48; with/without calibration) are reported, but the paper does not discuss how to _choose_ an optimal configuration for new datasets. It would be useful if the authors could provide practical guidelines or an automatic criterion (e.g., based on validation FED or structural RMSE) for selecting the latent dimension and deciding whether to apply calibration.

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
The authors present a framework for hyperedge-attributed hypergraph generation.
They combine various generation techniques that were successful in diverse domains to achive this goal, and demonstrated its effectiveness in certain hypergraph datasets, compared to several baseline methods.

### Strengths
S1. The paper is grounded on several theoretical results.

S2. The authors leverage diverse datasets and put the results in the Appendix.

### Weaknesses
**W1. [Key novelty]** While the authors present various theoretical results, I cannot understand what is the key novelty of the work. Is this the first work that generates hyperedge attributes? If so, in hypergraphs, changing node to hyperedge, and hyperedge to node, is trivial in many cases (e.g., if node: author and hyperedge: a paper's author list, then node: paper and hyperedge: a set of publications published by an author). There are certain works on hypergraph generative models with features [1], and authors should compare their work with such methods.

**W2. [Implication of theoretical results]** Could the authors present more intuitive results regarding their theories? For instance, Lines 354 - 355 state: "Theorem 3 implies that if m ≍ n ≍ p, the error introduced by estimating (Z, B, α, γ) is asymptotically negligible ..." Does this mean that the model can capture the ground-truth data distributions under certain assumptions?

**W3. [Comparison with tabular generative models]** To my knowledge, the MIMIC 3 dataset is often processed as tabular data. In my opinion, the effectiveness of the proposed method should be compared with the SOTA tabular data generative methods.

**W4. [Dataset construction]** I think the description regarding how the datasets are transformed into hypergraphs is not sufficient. Could the authors elaborate on it?

[1] Attributed Hypergraph Generation with Realistic Interplay Between Structure and Attributes

### Questions
See Weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces ReLaSH, a three-stage framework for hypergraph generation. (1) A likelihood-based joint embedding maps the observed hypergraph into a low-dimensional latent space. (2) A distribution-free generator is trained in this space to model and reconstruct the latent distribution. (3) The trained likelihood model decodes sampled latents into hyperlinks and associated attributes to synthesize new hypergraphs. On the theory side, the authors argue, and provide supporting analysis, that operating in the latent space mitigates the curse of ambient dimensionality. Empirically, experiments on co-citation, recipe, and symptom co-occurrence hypergraphs show consistent performance across domains.

### Strengths
1. The paper targets hypergraph generation with hyperlink attributes, going beyond structure-only models. The likelihood-based joint formulation explicitly links structure and attributes, which is a well-scoped and meaningful contribution.

2.  Decoupling (i) likelihood-based joint embedding, (ii) distribution-free latent modeling, and (iii) likelihood decoding is clean and practical. Each component is swappable (e.g., alternative generators or attribute heads), which favors reproducibility and future extensions.

3.  The analysis provides an error decomposition and argues why operating in a low-dimensional latent space can mitigate the curse of ambient dimensionality. This gives readers an interpretable lever for understanding where performance gains originate.

4. Experiments on co-citation, recipe, and symptom co-occurrence hypergraphs indicate consistent improvements across heterogeneous domains, suggesting the approach is not tailored to a single benchmark and generalizes reasonably well.

### Weaknesses
1. Your FED metric relies on an embedding machine $E^T$ trained on the full population: you first embed the entire population hypergraph to obtain reference parameters, then perform an 80/20 split to choose $k$ by minimizing train/test FID, and finally reuse this $E^T$ to embed real and generated samples for FED. This makes the evaluator and evaluated models non-independent and is a threat to validity. Could you add a fully independent evaluation, for example training $E^T$ on a disjoint subset of the population or using a fixed external encoder such as a frozen autoencoder or a random orthogonal projection to compute FED, and compare the results and sensitivity with your current setup?

2. In the real-data experiments, ReLaSH/ReLaSHc use Forest Diffusion, while Gau-Diff uses a standard diffusion. This inconsistency in score learners may confound the comparison. Could you provide a small matched-architecture ablation?

3. In B.6.1, you select the latent dimension $k$ by minimizing the train/test embedding FID. Is there a clear theoretical connection (e.g., consistency or error bounds) between this selection criterion and your generative likelihood? If the data deviate from the model assumptions, is this criterion still reliable? Could you provide a sensitivity curve (FED, reconstruction error, and downstream metrics vs. $k$) to show that the selected $k$ aligns with overall performance?

### Questions
See Weaknesses

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
This paper introduces ReLaSH, a generative framework for hypergraphs with hyperlink attributes. Its core contribution is a novel integration of a likelihood-based joint embedding model with a distribution-free latent space generator. The method first embeds the high-dimensional, sparse hypergraph and its attributes into a structured, low-dimensional latent space, which is then reconstructed and sampled from. This approach effectively circumvents the challenges of high-dimensional generative modeling, is supported by theoretical analysis, and demonstrates superior performance across multiple real-world tasks.

### Strengths
1. The proposed joint latent space reconstruction framework is highly innovative. By unifying the embeddings of both hypergraph structure (hyperedges) and attributes into a cohesive low-dimensional space, it effectively circumvents the challenges of direct high-dimensional generation. 

2. The method is inherently domain-agnostic, as demonstrated by its seamless and successful application across highly heterogeneous scenarios—from synthetic medical records and recipes to co-citation graphs. 

3.ReLaSH is fortified by solid theoretical contributions, including identifiability proofs and a decomposition of the generation error.

### Weaknesses
1.The paper proposes a novel latent space partitioning strategy but provides no principled or systematic method for selecting the dimensions k1, k2, and k3 for a given dataset. The experimental setup relies on an ad-hoc equal partitioning scheme or setting k2 = k3 = 0 for specific cases, which lacks justification.

2.The baseline methods compared in the experiments are not up-to-date, with the most recent being from 2020.

3.While the method introduces several key hyperparameters (e.g., k1/k2/k3, $\lambda$) and the appendix includes some empirical studies, a systematic sensitivity analysis is still missing. It remains unclear how the model performance varies with changes in these hyperparameters.

### Questions
1.Regarding the selection of k1, k2, and k3:
1.1. Could you provide ablations to demonstrate the effectiveness of the proposed tri-partite partitioning strategy compared to a simpler, unified latent space?
1.2. Please supplement experiments comparing the equal partitioning method with non-equal alternatives to show its (non-)optimality.
1.3. Could you offer practical guidance on how to determine these values for a new, unseen dataset?

2. To strengthen the experimental validation, it is crucial to include more recent and stronger baseline methods to ensure the timeliness and competitiveness of the reported results.

3.The paper would significantly benefit from a systematic sensitivity analysis of its key hyperparameters (e.g., k1/k2/k3, λ, diffusion steps). Please show how the performance metrics trend with variations in these parameters to better understand the model's robustness and tuning requirements.

### Soundness
2

### Presentation
2

### Contribution
3
