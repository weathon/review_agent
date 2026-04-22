# Protein generation with embedding learning for motif diversification

- Avg Score: 3.33
- Decision: Reject
- Scores: 6, 2, 2

## Abstract
A fundamental challenge in protein design is the trade-off between generating structural diversity while preserving motif biological function. Current state-of-the-art methods, such as partial diffusion in RFdiffusion, often fail to resolve this trade-off: small perturbations yield motifs nearly identical to the native structure, whereas larger perturbations violate the geometric constraints necessary for biological function. We introduce Protein Generation with Embedding Learning (PGEL), a general framework that learns high-dimensional embeddings encoding sequence and structural features of a target motif in the representation space of a diffusion model's frozen denoiser, and then enhances motif diversity by introducing controlled perturbations in the embedding space. PGEL is thus able to loosen geometric constraints while satisfying typical design metrics, leading to more diverse yet viable structures. We demonstrate PGEL on five representative cases: a monomer, a protein-protein interface, a cancer-related transcription factor complex, an antibody-antigen complex and an enzyme. In all cases, PGEL achieves greater structural diversity, better designability, and improved self-consistency, as compared to partial diffusion. Our results establish PGEL as a general strategy for embedding-driven protein generation allowing for systematic, viable diversification of functional motifs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors introduce PGEL, a framework inspired by textual inversion in stable diffusion. It uses RFdiffusion, a pre-trained protein structure diffusion model, to learn motif embeddings that during inference guide the diversification of these motifs. This leads to optimized scaffolds and show improved predicted affinities of the motif, likely due to optimized stabilization and presentation. It resembles “relaxing” a protein structure while keeping the scaffold fixed and thereby allowing the motif to explore and adapt within realistic geometric bounds to yield a diverse ensemble. In practice, PGEL first passes the scaffold coordinates and sequence features through a frozen encoder to produce scaffold embeddings which are then conditionally decoded together with a learnable motif embedding. The decoder, which is a frozen denoiser, outputs a denoised structure, and structure-based losses are computed against the ground truth. The loss is backpropagated to update the motif embedding to effectively calibrate it so that denoised predictions remain close to the true motif structure. Diversity is introduced by masking either specific features across residues or all features for selected residues and controlling how much geometric freedom is allowed. The authors tested PGEL on a monomeric protein (calmodulin), the barstar–barnase complex, and the p53–MDM2 interaction, and found that it produced more designable and self-consistent motifs and designs than partial diffusion when designed using ProteinMPNN. Performance was confirmed through in-silico refolding and designability metrics.

### Strengths
It's a creative and simple idea, plus well adapted from textual inversion to protein diffusion, operating in embedding space rather than directly perturbing atomic coordinates. Motif diversification is a challenging and important problem in protein design, for both de-novo design and optimizing functional proteins. Across three test cases PGEL clearly demonstrates its superiority over using partial diffusion alone. As the authors note, the framework can be easily transferred to other protein diffusion models and extended to tasks beyond motif diversification.

### Weaknesses
I’d like to mention a couple of points: First, a systematic analysis of the masking parameters and their effect on structural diversity would have been a valuable addition to the work. Second, ranking or categorizing motifs by type, for example structured single-motifs, structured multi-motif, and unstructured (loop) motifs, would have been both informative and interesting. And third, the method was only compared to partial diffusion in RFdiffusion, without benchmarking against other existing protein design or motif diversification approaches, which limits the scope of the comparative evaluation.

### Questions
1. The scaffold is kept fixed and the motif diversified. Could you elaborate what motif diversification actually means at the beginning of the manuscript? Ie., are these changes in the residues or chemistry of the motif itself, or are they small relaxations of the residues with respect to the rest of the structure? A stronger biological motivation could also help the reader better understand the rationale behind this task.
2. To me PGEL seems very useful for loops. They are flexible and allow exploration of different motions and conformations. Did the authors test their approach on antibody or antibody-like modalities, or on structures that contain loop regions?
3. Have the authors investigated whether there is a dependence between the amount of scaffold masking and the extent of motif diversification? That is, does masking a larger portion of the scaffold lead to greater deviations of the motif in terms of RMSD or center-of-mass displacement? Are there also differences in the resulting secondary structure outcomes?
4. Can multiple motif embeddings be combined or concatenated to create “chimeric-motifs”?
5. Could you specify whether the model needs to be retrained or refitted for each target motif, or whether it was trained on a broader corpus of motifs and then applied in a general way to all test cases?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work introduces Protein Generation with Embedding Learning (PGEL), a framework for motif diversification in protein design. It addresses the problem with the trade-off in existing methods (like RFdiffusion's partial diffusion) between generating diverse structures and maintaining the functional geometry of a specific protein region. \
PGEL learns a high-dimensional embedding that captures the motif's structural and (implicitly) sequence characteristics. This learned embedding then conditions a frozen, pre-trained protein diffusion model (RFdiffusion) to generate structures with diverse motifs. \
To enhance diversity, PGEL applies random masking (either row or column) to the scaffold's MSA embeddings during generation. This relaxes evolutionary and geometric constraints, allowing the model to explore a wider range of plausible motif conformations while keeping the surrounding scaffold nearly fixed. PGEL outperforms partial diffusion in terms of quality and diversity on 3 test cases.

### Strengths
The authors tackle an important problem of low diversity in protein design. The approach is flexible and can be used with different models.

### Weaknesses
### Key Weaknesses

The method is too computationally expensive for practical use (requiring 2 hours of fine-tuning per protein for motifs up to 50 amino acids). Results and conclusions are drawn from only three selective examples. The authors flip the standard motif scaffolding problem on its head, which significantly complicates understanding of the paper. Furthermore, the comparison is limited to only one model (Partial Diffusion), despite the existence of numerous models that could perform this task with better diversity than RFdiffusion.

### Detailed Claims with Suggestions

1.  **Ablation studies are needed for the novel components.** It's possible that the diversity gains come solely from Embedding Masking, rendering the entire fine-tuning process unnecessary.
2.  **Insufficient experiments.** additional experiments are required for conclusions about the method.
3.  **Limited model comparison.**  It's unclear why standard RFdiffusion and other models capable of scaffolding (e.g., Genie, FrameFlow, DPLM, DiMA) were not included.
4.  **Clarify the problem definition.** The study tackles a scaffolding task but swaps the conditional and generated parts. This task should be renamed to avoid confusion with the established paradigm.
5.  **Validate on established scaffolding benchmarks.** Since it's a scaffolding task, performance should be evaluated on known benchmarks. This could demonstrate, for example, an ability to generate scaffolds with similar quality but greater diversity than RFdiffusion.
6.  **Low diversity.** The number of clusters ≤1% (10 out of 1000 in Table 1) is concerning. Evaluation of the diversity should be revised:
    *   A. Differentiate between novelty and diversity for better model diagnostics. 
    *   B. The TM-score threshold 0.6 for such small motifs might be too strict. Test the maximum achievable diversity by randomly sampling amino acids for the motifs and check the diversity under this threshold.

### Questions
1.  Could you please clarify the type of hierarchical clustering used and the rationale for its selection?

### Soundness
1

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
The authors address a diversity-fidelity tradeoff in protein motif design. Existing methods like partial diffusion in RFdiffusion either produce near-identical structures (low noise) or break functional geometry (high noise). PGEL learns an embedding representation of the target motif within RFdiffusion's frozen network, then generates diverse conformations by stochastically masking the scaffold's MSA embeddings during generation. This shifts perturbations from coordinate space to embedding space. Tested on three protein systems (monomer, binding interface, binder complex), PGEL produces 2-3x more designable structures than partial diffusion.

### Strengths
* Adapting textual inversion from image generation to protein diffusion models is creative and represents a genuine methodological contribution. The idea of learning embeddings in a frozen model's representation space rather than directly perturbing coordinates is interesting.
* The use of hierarchical clustering with TM-score thresholds and the requirement for distinguishability from native structures provides a reasonable assessment of structural diversity.
* The authors transparently discuss the *in silico* nature of validation, computational costs of embedding learning, inherited biases from RFdiffusion, and practical limits on motif sizes.

### Weaknesses
1. **The authors do not demonstrate that embedding learning is necessary.**
The paper's core premise, that learning motif embeddings is essential, is never validated. No comparison to simpler baselines like random embeddings, embeddings from related structures, or zero embeddings during generation (only learning). This is the most critical missing experiment, as it questions whether the convoluted optimization procedure that requires a full structural denoising for a single gradient step, provides any value beyond the masking strategy alone.


2. **Insufficient architectural detail.**
The paper lacks sufficient detail to understand or reproduce the method. How are scaffold and motif embeddings (state, pair, MSA) structured and combined to condition the denoiser? What are the exact dimensions? How does information flow between learned motif embeddings and fixed scaffold embeddings during denoising? These architectural specifics are crucial for reproducibility.


3. **Masking mechanism poorly justified and under-analyzed.** 
No ablation study demonstrates that masking is necessary (comparison to PGEL without masking). The striking 80% preference for column over row masking is merely observed, not explained or investigated systematically. The claimed mechanism of "relaxing geometric constraints" is vague hand-waving without supporting evidence. The choice of uniform sampling for α ∈ [0,1] is unexplored—why not bias toward conservative or aggressive masking rates? Furthermore, there's no analysis of which residues or features benefit most from masking, leaving the mechanism essentially a black box.


4. **Limited baseline comparisons.**
The model is only compared against partial diffusion in RFdiffusion. Adding recent related methods, including mentioned in the background would benefit the soundness of the manuscript. Also, the authors claim the method is "general" and "adaptable to other protein diffusion models," they only demonstrate it on RFdiffusion. The extent to which the approach transfers to other architectures (FrameDiff, Chroma, etc.) remains speculative.


6. **Insufficient ablation studies beyond masking.**
The loss function weights (λ_DM = 0.01, λ_torsion = 0.05) appear arbitrary with no sensitivity analysis demonstrating robustness to these choices. No comparison of different embedding initialization strategies beyond zeros is provided. The paper doesn't analyze individual loss term contributions, leaving unclear which components of the three-term loss are actually necessary for performance. This lack of systematic component analysis makes it difficult to understand what drives the method's success.


7. **Narrow test scope.**
Only three protein systems are tested, all taken from Watson et al. 2023, which severely limits generalizability claims. The evaluation doesn't explore different motif sizes and types beyond 20-40 residues, various protein folds or structural classes, or different functional categories beyond binding interfaces. Importantly, there's no characterization of failure modes or cases where partial diffusion might actually be preferable, presenting an incomplete picture of the method's applicability.


8. **Computational cost-benefit analysis is incomplete.**
While learning time for the 3 examples ranges from 2 minutes to 2 hours, the paper provides no systematic analysis of when this additional optimization cost is justified compared to simpler alternatives. There's no investigation of how learning time scales with motif size and complexity, what memory requirements look like, or whether the method can scale to larger systems. The trade-off between optimization time invested and generation quality achieved remains unquantified, making it difficult for practitioners to decide when PGEL is worth the computational investment.


9. **Statistical rigor is lacking.**
No error bars, confidence intervals, or significance tests. Unclear if results represent single runs or averages over multiple trials. This makes it impossible to assess result robustness or variance.

### Questions
* How are embeddings architecturally integrated? Please provide detailed architectural specifications. What are the exact dimensions of state, pair, and MSA embeddings for both motif and scaffold? How are they concatenated or combined to condition the denoiser? Is there any cross-attention or other interaction mechanism between learned motif embeddings and fixed scaffold embeddings?
* Why does column masking dominate? Can you explain the mechanistic reason for the 80% success rate with column masking versus row masking? What does masking specific residues (columns) versus specific embedding features (rows) actually do to the covariation patterns and geometric constraints?
* Is masking necessary at all? What performance do you get with learned embeddings but no masking during generation? This would isolate the contribution of the masking strategy.
* What determines optimization time? The 2-minute to 2-hour range is large. What specific factors drive this variation? Can you provide a scaling analysis showing how optimization time depends on motif length, structural complexity, or other properties?
* What do the learned embeddings actually encode? Can you visualize or analyze the embedding space? Are there interpretable patterns in how embeddings change during optimization? Can you interpolate between embeddings to predict intermediate structures?

### Soundness
1

### Presentation
2

### Contribution
2
