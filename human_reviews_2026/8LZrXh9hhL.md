# A Single Architecture for Representing Invariance Under Any Space Group

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 8, 4

## Abstract
Incorporating known symmetries in data into machine learning models has consistently improved predictive accuracy, robustness, and generalization. However, achieving exact invariance to specific symmetries typically requires designing bespoke architectures for each group of symmetries, limiting scalability and preventing knowledge transfer across related symmetries.  In the case of the space groups—symmetries critical to modeling crystalline solids in materials science and condensed matter physics—this challenge is particularly salient as there are 230 such groups in three dimensions. In this work we present a new approach to crystallographic symmetries by developing a single machine learning architecture that is capable of adapting its weights automatically to enforce invariance to any input space group. Our approach is based on constructing symmetry-adapted Fourier bases through an explicit characterization of constraints that group operations impose on Fourier coefficients. Encoding these constraints into a neural network layer enables weight sharing across different space groups, allowing the model to leverage structural similarities between groups and overcome data sparsity when limited measurements are available for specific groups. We demonstrate the effectiveness of this approach in achieving competitive performance on material property prediction tasks and performing zero-shot learning to generalize to unseen groups.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents the Crystal Fourier Transformer (CFT), a single neural network architecture capable of enforcing invariance to any of the 230 crystallographic space groups without requiring separate architectures for each group. The key innovation is an analytical method for constructing symmetry-adapted Fourier bases by deriving explicit constraints that group operations impose on Fourier coefficients. These constraints are encoded in precomputed, group-specific adjacency matrices MG\textbf{M}_G
MG​ that act as routing mechanisms, allowing a single Transformer model to share weights across all space groups while adapting to different symmetries. The authors demonstrate that CFT learns geometrically meaningful orbit distance representations, achieves competitive performance on Materials Project benchmarks for predicting material properties (total energy, band gap, bulk/shear moduli), and successfully generalizes zero-shot to space groups unseen during training.

### Strengths
Mathematical rigor and elegance: The analytical approach to deriving Fourier constraints (Proposition 3.1, Theorem 3.2) is theoretically sound and provides exact rather than approximate invariance. The connection to graph theory through dual representations is intellectually satisfying.

Architecture innovation: Creating a single model that adapts to 230 different space groups through precomputed routing matrices $M_{G}$ is a clever solution to weight sharing across related symmetries. This is fundamentally different from prior work requiring separate architectures.

Comprehensive experimental validation: The paper validates three distinct aspects: (1) geometric correctness (orbit distance learning with MAE=0.102Å), (2) competitive performance on established benchmarks, and (3) zero-shot generalization capability. This multi-faceted evaluation is thorough.

Practical efficiency: The 6-7× speedup over graph-based methods (Table 2) combined with competitive accuracy makes this practically attractive. The simple matrix-vector product for symmetry adaptation is elegant.

Strong zero-shot results: Figure 3 shows minimal performance degradation on held-out groups, validating the weight-sharing hypothesis and demonstrating genuine transfer learning across symmetries.

### Weaknesses
Limited performance gains: While CFT is competitive, it doesn't consistently outperform baselines. It beats ALIGNN/Matformer on only 2/4 properties (Table 1), and the improvements are often within error bars. For Band Gap, Matformer achieves 0.213±0.003 vs CFT's 0.306±0.006. The practical advantage over existing methods is unclear.

Abelian group restriction: The limitation to commutative Lie groups is significant. The shared eigenvector parametrization (Appendix C.2) that ensures commutativity is a fundamental architectural constraint. The paper doesn't discuss whether this could be relaxed or what barriers exist.

Theory:
What fraction of reciprocal lattice points satisfy phase-consistency?
How does approximation error scale with frequency cutoff radius?
Under what conditions does the Gaussian approximation for joint entropy fail?
Why specifically do Gaussian-distributed datasets cause problems (acknowledged but not investigated)?

Limited baseline comparisons: The paper doesn't compare against other symmetry discovery methods even on small problems they can handle. Direct comparison on 2D wallpaper groups would strengthen claims.

Evaluation scope:
Zero-shot experiment uses only 6 groups (~10% of data); testing on more held-out groups would be convincing.
Only Materials Project dataset evaluated; other crystal property databases would validate generalization.
No comparison of learned vs. ground-truth basis functions beyond the orbit distance task.

### Questions
Theoretical completeness: In Theorem 3.2, you filter out frequencies with inconsistent self-loops. Can you characterize which groups/frequencies are affected? What proportion of the reciprocal lattice typically survives this filtering for common space groups?

Gaussian distribution failure: You mention that "datasets with Gaussian distributions are challenging for the current implementation" and attribute this to covariance-based entropy approximation. Can you:
Provide concrete failure modes or examples?
Explain why this is problematic (Gaussians should be easy to model with covariance)?

Comparison with prior symmetry methods: Can you provide direct quantitative comparisons with on problems they can handle (low-dimensional representations)?

Architecture details:
During training, how is the space group G provided to the model? As a one-hot encoding, or implicitly through the routing matrix?
For the orbit distance pretraining, why use ResNets rather than simpler MLPs? Were alternatives tested?
How was the frequency cutoff radius chosen?

Generalization breadth: The zero-shot experiment holds out 6 groups. Can you:
Test on more aggressive holdouts (e.g., entire crystal systems)?
Show how performance degrades as held-out groups become less similar to training groups?
Analyze which group properties enable transfer?

Basis function validation: Beyond orbit distance, have you visualized or analyzed whether the learned basis functions match the theoretical predictions from Theorem 3.2? Can you show examples of learned vs. theoretical basis functions?

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
The authors introduce a novel algorithm for constructing a symmetry adapted Fourier basis that generalizes invariance across the point groups. This basis provides an alternative to the more commonly used graph-based representation of crystal structures.

### Strengths
The paper tackles an important problem of finding invariant neural representations of crystallographic observations that generalize across all point groups. The work is thoroughly motivated and theoretically sound.

The paper presentation is exceptional, aside from several grammatical errors and points that need further clarification.

The numerical results are promising, particularly those on the improved computational performance.

### Weaknesses
The paper experiments are limited in the following aspects
1) the orbit distance results are only reported for the CFT architecture, making it unclear if the technique has improved the learning of the representations.
2) The material property prediction results demonstrate marginally equivalent or worse results in comparison with SOTA techniques.
3) It's unclear the effect of the choice of the basis dimension and resulting encoding dimensions and the role/benefits of using the pre-trained encoder.

### Questions
**Q1. Orbit-distance evaluation is architecture-limited.** Clarify how inconsistent self-loops in the graph representation (line 212) are detected. My understanding from Figure 1 is that these correspond to single-node connected components by virtue of being self-looping nodes; confirm whether this is the intended general definition and provide the formal criterion used.

**Q2. Property prediction is not yet compelling.** Clarify the role of the pre-trained positional encoder. In particular, is there a measurable performance benefit to pre-training the embedding rather than learning entirely end-to-end?

**Q3. Basis choice and encoder role are under-specified.** Provide ablations on the dimensionality of the chosen basis, especially for the encoder training used in later experiments. My understanding is that the observed error in the orbit distance task is due to basis truncation, but it should be clarified. Also, please report the exact basis dimensions used (main text and Appendix)

**Q4. Zero-shot comparisons.** Please compare to the zero-shot capabilities of other SOTA architectures.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper "A Single Architecture for Representing Invariance Under Any Space Group" presents a novel approach to producing space group invariant representations of crystal structures. The main innovation is Crystal Fourier Transformer (CFT) which uses a Fourier basis conversion to make a space-group invariant representation of the atomic coordinates. They showcase that CFT has comparable performance to other state of the art models while having considerably less training and inference time.

### Strengths
- The setup for the use of Fourier basis and the explanation of how they are computed and the associated images make it very clear what the process is and were very helpful.
- The inclusion of training and inference performance is very important as one of the key use cases of these models is a replacement for DFT calculations.
- The comparison against many state of the art models in terms of both performance and computation time showcases the key benefits of the model and how other state of the art methods have become quite slow in comparison due to the costly crystal graph representation. This clearly shows how their method can be used to create more efficient models and representation techniques that could be built upon in future work.

### Weaknesses
- You make a claim that the performance is only marginally higher on the held out groups, yet groups 71 and 140 in the Bulk Modulus test have a MAE of almost double the seen groups, I don't think that counts as marginally and would warrant further explanation on why their performance is so low. 
- The model is not state of the art in any specific task showcasing that while their method does have benefits in computational performance, it does not significantly boost performance on any tasks. Seeing as how modern machine learning models are already much faster than DFT even with longer training and inference times compared to CFT, it would be better to see how their method could improve performance.
- Seeing as how one of the main claims is that computation time is significantly reduced, I would like to see more clarity in terms of number of parameters of each model and how that may affect runtime as well. It would be interesting if the models were all similarly sized but had vastly different runtime or if the performance of CFT is comparable to other models which may be much larger (or smaller).

### Questions
1. Are the times listed on Table 2 inclusive of the pretraining time required for CFT positional encodings?
2. I am not sure I fully understand the setup for the synthetic dataset in section 5.2. To my understanding, you can't select random positions for atoms in a crystal as they are locked to certain regions by the space group of the crystal. If my understanding is wrong, please let me know, but how did you place the atoms when creating the synthetic dataset so that it followed the space group rules?
3. What is the explanation for why training on Shear Modulus sometimes increases MAE?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces the Crystal Fourier Transformer (CFT), an architecture for crystal property prediction. The core idea is to use a pre-computed, group-specific matrix, $M_G$, to explicitly project atomic coordinates into a G-invariant feature space. This design enables a single, weight-shared Transformer backbone to process all 230 space groups, aiming to address data sparsity and enable generalization to unseen groups.

### Strengths
1. The paper's primary strength is its rigorous theoretical derivation. It provides a clear analytical treatment of how space-group operations constrain Fourier coefficients.
2. The method avoids the complex graph construction required by GNNs. Its Transformer architecture is highly parallelizable, aligns with mainstream models.
3. The use of the $M_G$ matrix as a "group-conditional router" to explicitly inject invariance is novel and provides an elegant solution for weight sharing across all groups.

### Weaknesses
1. In the primary property prediction task (Table 1), the model only achieves performance on par with, or marginally better than, outdated baselines. The lack of comparison to current SOTA (2024-2025) models makes the "competitive" claim unconvincing.
2. The "zero-shot" task compares CFT only against itself and completely omits baselines under the same zero-shot conditions. This leaves the central claim of generalization entirely unvalidated.
3. The paper's core selling point—generalization to unseen groups —is of limited practical value. 

Missing related work on similar projection into invariant features:
1. A new perspective on building efficient and expressive 3D equivariant graph neural networks， Neurips 2024

### Questions
1. Why was it assumed that GNNs cannot generalize in this zero-shot setting? Baselines must be evaluated under the identical zero-shot setup (e.g., holding out the same 6 groups). It is plausible that neural networks can learn this generalization implicitly, and the authors have not proven their explicit $M_G$ method is superior or necessary.
2. Why focus on the saturated task of property prediction rather than the current SOTA challenge of crystal structure generation? Has this G-invariant encoding scheme ($M_G$) been tested for generative tasks, and does it offer any advantage?

### Soundness
3

### Presentation
3

### Contribution
2
