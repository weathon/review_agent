# Differentially Private Synthetic Data via APIs 3: Using Simulators Instead of Foundation Models

- Decision: Reject
- Scores: 2, 6, 6, 2

## Abstract
Differentially private (DP) synthetic data, which closely resembles the original private data while maintaining strong privacy guarantees, has become a key tool for unlocking the value of private data without compromising privacy. Recently, Private Evolution (PE) has emerged as a promising method for generating DP synthetic data. Unlike other training-based approaches, PE only requires access to inference APIs from foundation models, enabling it to harness the power of state-of-the-art (SoTA) models. However, a suitable foundation model for a specific private data domain is not always available. In this paper, we discover that the PE framework is sufficiently general to allow APIs beyond foundation models. In particular, we demonstrate that many SoTA data synthesizers that do not rely on neural networks—such as computer graphics-based image generators, which we refer to as simulators—can be effectively integrated into PE. This insight significantly broadens PE’s applicability and unlocks the potential of powerful simulators for DP data synthesis. We explore this approach, named Sim-PE, in the context of image synthesis. Across four diverse simulators, Sim-PE performs well, improving the downstream classification accuracy of PE by up to 3X and reducing FID by up to 80%. We also show that simulators and foundation models can be easily leveraged together within PE to achieve further improvements.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes Sim-PE, an extension of Private Evolution (PE) that replaces foundation models with simulators as the generative backend. Sim-PE supports two scenarios: when simulator APIs are accessible, the RANDOM and VARIATION APIs are implemented by sampling from the feasible parameter space; and when only a large pool of simulator-generated data is available, the VARIATION API is realized through nearest-neighbor sampling. Experiments on image datasets show that Sim-PE outperforms PE under strong distribution shifts and further benefits from combining simulators with foundation models.

### Strengths
- A simple but effective approach that 1) improves over PE; 2) demonstrates the generality of the core mechanism of PE

- Comprehensive experiments on image synthesis, covering diverse simulator types, access settings, and combinations with foundation models

### Weaknesses
- The technical novelty is limited. The PE framework remains essentially unchanged, and adapting its RANDOM and VARIATION APIs to simulators is straightforward. Key limitations of PE persist, including but not limited to mode collapse, sampling bias (e.g., top-K), and non-monotonic performance w.r.t. iterations.

- The main contribution lies in identifying simulators as a new data source rather than introducing a new algorithmic insight. The empirical study is confined to image data, where the use of simulators is somewhat obvious and less compelling. More interesting domains such as robotics or physical simulation are left unexplored.

- The strong results (Sec 4.2) largely stem from cases where the simulator distribution is already well aligned with the private data (rendering digits with Python PIL). It remains unclear how to choose or adapt a simulator for arbitrary private datasets, which limits the general applicability of the proposed approach. As a concrete question, what simulator would you consider if the goal is to generate clinical or biometric data?

Overall, this work identifies an interesting direction but lacks sufficient technical depth for ICLR.

### Questions
Sim-PE heavily depends on the choice of the similarity metric or embedding space used for the DP-NN voting and for the VARIATION API (nearest-neighbor sampling). In your experiments, this embedding comes from pretrained vision models like CLIP, which themselves are trained on massive web data. Doesn’t this reintroduce reliance on large foundation models, the very dependency Sim-PE claims to eliminate? How would Sim-PE operate in a domain where no strong pretrained embedding exists?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses a key limitation of Private Evolution (PE), a state-of-the-art framework for generating differentially private (DP) synthetic data. The standard PE framework relies on inference APIs from foundation models (FMs), but its performance degrades significantly when a suitable FM for the private data domain is unavailable (e.g., using an ImageNet-trained FM for MNIST data).

The authors' key insight is that the PE framework is general and only requires two abstract APIs: a `RANDOM_API` to generate initial samples and a `VARIATION_API` to produce variations of a given sample. The paper proposes **Sim-PE**, a novel approach that implements these APIs using non-neural-network data synthesizers, which the authors refer to as "simulators".

The paper introduces two practical methods for integrating simulators:
1.  **Simulator Access:** When the simulator's code is available, `RANDOM_API` renders images with random parameters, and `VARIATION_API` renders images with slightly perturbed parameters.
2.  **Data Access:** When only a large, simulator-generated dataset is available (e.g., due to proprietary assets), `RANDOM_API` samples a random image from the dataset, and `VARIATION_API` selects a random nearest neighbor of a given image.

The authors also demonstrate that simulators and FMs can be used together in a hybrid approach, for instance, by using a simulator for diverse initialization and an FM for refinement.

Experiments show that Sim-PE is highly effective. 
- On MNIST, where standard PE fails (27.9% accuracy), Sim-PE with a text-rendering simulator achieves 89.1% accuracy, a 3x improvement. 
- On CelebA, Sim-PE successfully selects high-utility samples from a public dataset and a hybrid approach with a weak simulator outperforms using either the simulator or the FM alone. 
- Finally, Sim-PE is shown to be up to 80x more computationally efficient than standard PE.

### Strengths
The core idea is simple and elegant. While not groundbreaking, this paper opens up PE to a new class of generative tools beyond foundation models. 
The presentation is very clear. I especially appreciate the underlying motivation for the choice of experiments as well as the ablation studies.

### Weaknesses
- While the authors show that it is not necessary to have a foundation model that is aligned with the private data, it doesn't directly solve the issue of cold starts, but instead shifts the dependency on a good foundation model to a good simulator.

### Questions
-  The hybrid model is very promising. Have the authors considered a "mixed" or "parallel" strategy? For example, within a single iteration, could the variation API somehow combine a simulator and FM? This seems possible given the framework's modularity.
- How sensitive is the performance of SIM-PE to the choice of simulator? While ImageNet FM -> private MNIST has low utility due to a mismatch with the foundation model, how does the utility of SIM-PE degrade with mismatched simulators?

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
4

### Summary
This work extends Private Evolution framework from foundation model APIs to simulators and simulator-generated datasets. Key insight: PE's privacy guarantee depends only on DP Nearest Neighbors Histogram, so generation backend can be abstracted via RANDOM and VARIATION APIs. For accessible simulators, APIs implemented through parameter perturbation. For data-only simulators, nearest-neighbor variation avoids wasting privacy budget on distant samples. On MNIST with ε=1, Sim-PE achieves 89.1% classification accuracy versus PE's 27.9%, reducing FID by 80%. Framework enables combining simulators with foundation models.

### Strengths
- Backend-agnostic design demonstrates real impact. MNIST shows foundation model pretrained on ImageNet achieves 27.9% on digits while text renderer achieves 89.1% (Table 1). When distribution mismatch exists, appropriate backend selection matters.
- Handles realistic deployment scenarios: accessible simulator with direct parameter control, and data-only simulator common when proprietary assets involved.
- Nearest-neighbor iterative refinement (Section 3.3) avoids privacy budget waste on irrelevant samples. Table 3 shows clear improvement over naïve approaches.
- Combination strategy (Table 2) demonstrates framework flexibility: simulator + foundation model outperforms either alone on CelebA (FID 11.9 vs 22.0 for PE alone, 99.5 for weak simulator alone).

### Weaknesses
- My main concern is with the privacy accounting for the data-based Sim-PE (Sec 3.3). The method uses an external embedding model (Inception, Appendix F.1) for its nearest-neighbor search, but the privacy properties of this embedding are never analyzed. If this embedding was trained on related private data, it could leak information, and this seems to be an unaddressed gap in the privacy proof.

### Questions
- Could the authors clarify the privacy accounting for the external embedding model? How can we be sure it doesn't leak information, and shouldn't it be consistent across all baselines for a fair comparison?
- How sensitive is the data-based Sim-PE to the choice of the embedding model? Does performance change significantly if using CLIP, DINOv2, or other task-specific embeddings?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces simPE, an extension of PE. In traditional PE, the work relies on foundational models. In simPE, the authors show that PE framework can be extended to other simulators as well and not rely on only foundational models.

### Strengths
The paper is well written and easy to understand. The work proposed is simple to use but powerful and extends PE to work with simulators. The work also provides clear, actionable guidelines for both simulator access and data-only scenarios.

### Weaknesses
1. The paper's contribution is minimal. It extends the same framework of PE to work with simulators. Since PE itself needs only APIs, the main contribution of the paper seems to be replacing foundational models behind the APIs with simulators.
2. The work focuses mainly on image generation because of dependencies on simulators. The quality of the images also depend on the availability of the simulator's capability to generate data similar to the original dataset.

### Questions
1. How sensitive is Sim-PE to simulator quality?
2. How do biases in simulators transfer to the synthetic data? Are there ways to detect or mitigate such transfers?

### Soundness
3

### Presentation
3

### Contribution
2
