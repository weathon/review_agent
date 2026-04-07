## Summary
CLIP-Map proposes a mapping-based compression framework for CLIP models, replacing conventional select-based pruning with learnable matrix transformations. The method uses Kronecker-factorized mappings to reduce parameter overhead and a diagonal inheritance initialization to stabilize training, followed by a retraining stage with distillation. Experiments show improved performance over strong baselines like TinyCLIP, particularly under high compression ratios, with gains in training efficiency.

## Strengths
- **Strong and consistent empirical improvements.** CLIP-Map outperforms the select-based TinyCLIP baseline across multiple compression ratios (1%, 10%, 50%) on zero-shot retrieval (MSCOCO, Flickr30K) and classification benchmarks, with particularly significant gains at extreme compression (e.g., +5.3% TR@1 on MSCOCO at 1% compression). The method also requires fewer training epochs, reducing wall-clock time.
- **Innovative adaptation of model-growth techniques to compression.** The core idea of replacing hard parameter selection with a learnable, compressed mapping is novel in this context. The use of Kronecker factorization to reduce the mapping parameter complexity from O(D₁²D₂²) to O(D₁D₂) is a clever and well-motivated design.
- **Thorough ablation studies and analysis.** The paper includes convincing ablations validating the proposed diagonal inheritance initialization (Table 5, Fig. 6) and the choice of mapping-stage duration (Table 4). Visualizations of the evolving mapping matrices (Fig. 5) provide useful insight into the optimization process.

## Weaknesses
- **Incomplete methodological description for depth compression.** While width compression via Kronecker factors is clearly explained, the description of depth compression is insufficient. Equation (2) and the surrounding text state that a depth-compression operator \(L_{depth}\) linearly combines layers, but crucial details—how \(L_{depth}\) is parameterized, initialized, and optimized—are missing from the main text, hindering reproducibility.
- **Lack of quantitative analysis for the core claim of information preservation.** The paper argues that mapping preserves more information from the original model than selection-based pruning, but provides no quantitative evidence (e.g., feature similarity analysis, parameter matrix rank comparisons). This claim remains intuitive but unsubstantiated.
- **Limited and indirect comparison with the broader state-of-the-art.** The primary comparison is with TinyCLIP. Comparisons to other recent compression methods (UPop, MoPE-CLIP, etc.) in Table 7 rely on results reported in their original papers under different training data and settings, limiting the strength of the superiority claim. A direct, controlled comparison is needed.
- **Evaluation scope is constrained to a single dataset and primary architecture.** All main results use YFCC-15M and a ViT-based CLIP. While the appendix shows a proof-of-concept on ResNet and Meta-CLIP, the paper's claims about general applicability would be stronger with more extensive validation across datasets (e.g., larger-scale LAION-2B) and a wider variety of architectures in the main experiments.

## Nice-to-Haves
- A theoretical analysis or bound on the compression error or representational capacity of the Kronecker-factorized mapping.
- An ablation study isolating the contribution of the depth compression component versus width compression.
- Detailed reporting of the computational overhead (FLOPs, memory) of the mapping stage itself, beyond the final model's inference cost.

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **Weakness: "The abstract lacks quantitative results."** – While including specific numbers could be helpful, the abstract's purpose is to summarize contributions; the quantitative results are thoroughly presented in the experimental sections.
- **Weakness: "The method requires predefined compression ratios and targets; how to choose them optimally is not discussed."** – This is a generic issue for all compression methods, not a specific flaw of this work.
- **Weakness: "The figures are cluttered and difficult to interpret."** – This is a formatting/subjective nitpick that does not affect the technical evaluation.
- **Weakness: "Potential overfitting risks due to λ=1.0 are not examined."** – The paper includes an ablation study (Table 10, Appendix A.8) that systematically evaluates λ and selects λ=1.0 based on empirical performance, which is a reasonable justification.
- **Criticism that depth compression is "poorly explained" to the point of being a "significant gap."** – While the description could be more detailed, the core idea (linear combination of layers via a learned matrix) is presented in Eq. (2) and Fig. 3. The greater issue is the lack of implementation details, which is captured in the weaknesses above.

## Novel Insights
The paper's core novelty lies in successfully adapting the paradigm of learnable mapping—previously explored for model growth—to the distinct and challenging problem of model compression for multimodal architectures. It demonstrates that preserving parameters through transformation, rather than selecting a subset, can yield superior performance under aggressive compression, especially when coupled with techniques (Kronecker factorization, diagonal initialization) that address the unique optimization challenges of mapping to a smaller space. No further novel insights beyond the paper's own contributions emerge from the reviews.

## Suggestions
- Provide a complete, reproducible description of the depth compression procedure in the main text, including the parameterization, initialization, and optimization of the \(L_{depth}\) operator.
- Add a quantitative analysis to support the information preservation claim, such as measuring Centered Kernel Alignment (CKA) between features of the original and compressed models.
- Strengthen the empirical validation by including at least one direct, apples-to-apples comparison with a recent state-of-the-art compression method (e.g., MoPE-CLIP) under identical training data and model size settings.