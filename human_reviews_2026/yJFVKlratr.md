# From Pixels to Semantics: Unified Facial Action Representation Learning for Micro-Expression Analysis

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 6, 6

## Abstract
Micro-expression recognition (MER) is highly challenging due to the subtle and rapid facial muscle movements and the scarcity of annotated data. Existing methods typically rely on pixel-level motion descriptors such as optical flow and frame difference, which tend to be sensitive to identity and lack generalization. In this work, we propose D-FACE, a Discrete Facial ACtion Encoding framework that leverages large-scale facial video data to pretrain an identity- and domain-invariant facial action tokenizer, for MER. For the first time, MER is shifted from relying on pixel-level motion descriptors to leveraging semantic-level facial action tokens, providing compact and generalizable representations of facial dynamics. Empirical analyses reveal that these tokens exhibit position-dependent semantics, motivating sequential modeling. Building on this insight, we employ a Transformer with sparse attention pooling to selectively capture discriminative action cues. Furthermore, to explicitly bridge action tokens with human-understandable emotions, we introduce an emotion-description-guided CLIP (EDCLIP) alignment. EDCLIP leverages textual prompts as semantic anchors for representation learning, while enforcing that the "others" category, which lacks corresponding prompts due to its ambiguity, remains distant from all anchor prompts.
Extensive experiments on multiple datasets demonstrate that our method achieves not only state-of-the-art recognition accuracy but also high-quality cross-identity and even cross-domain micro-expression generation, suggesting a paradigm shift from pixel-level to generalizable semantic-level facial motion analysis. Code is available at https://github.com/KinopioIsAllIn/D-FACE.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces D-FACE, a Discrete Facial ACtion Encoding framework that shifts micro-expression recognition (MER) from pixel-level motion descriptors (e.g., optical flow) to semantic-level facial action tokens. Using a conditional vector-quantized VAE, D-FACE discretizes facial muscle movements into domain- and identity-invariant “action tokens.” These tokens are treated as one-dimensional sequences and processed with a Transformer featuring sparse attention pooling to capture localized, discriminative cues. An additional emotion-description-guided CLIP (EDCLIP) alignment connects visual token features with textual emotion descriptions, enhancing semantic interpretability. Experiments on several benchmark datasets show that D-FACE achieves competitive or superior accuracy and exhibits cross-subject and cross-domain generalization.

### Strengths
1. Proposes a novel paradigm shift for MER: from pixel-level motion descriptors to semantic-level, discrete facial action tokens, potentially improving robustness and transferability.

2. Designs a C-VQ-VAE–based facial action tokenizer, the first to discretize motion representations for micro-expressions. Introduces a Transformer with sparse attention pooling to focus on informative local cues, demonstrating improved interpretability and performance.

3. Develops EDCLIP alignment to connect visual action tokens with human-understandable emotion semantics, enhancing explainability.

4. Shows cross-domain and cross-identity generalization, as well as potential for facial expression generation, marking a broader contribution beyond recognition.

### Weaknesses
This paper has a few issues:
(1) Figure 1(b) (and related illustrations) are difficult to follow, limiting accessibility of the proposed conceptual shift, especially the interpretation of token map. A few more descriptive words would help. 

(2) The novelty, while conceptually interesting, relies heavily on adapting existing VQ-VAE and CLIP mechanisms. In the introduction, it is better to further clarify how the method contributes novel insights on top of existing mechanisms. 

(3) Logic clarity in parts (especially empirical analysis of tokens in Sec. 3.2) could be improved; the link between position-dependent semantics and 1D modeling is asserted but not quantitatively validated. Method correctness appears sound but somewhat over-engineered; e.g., the choice of hyperparameters and training stability of the C-VQ-VAE + Transformer pipeline is not discussed in detail.

(4) The reported accuracy improvements are marginal (e.g., small gains over LTR-3O and SRMCL in Table 2), raising questions about practical significance. There is little ablation or visualization proving that the semantic tokens truly capture meaningful, interpretable muscle movements.

### Questions
In addition to the questions relevant to weaknesses, there are a few more:
(1) How generalizable is D-FACE to in-the-wild, low-quality videos, given its dependence on large-scale pretraining on high-quality data?

(2) Could the discretization bottleneck of VQ-VAE lose fine-grained motion nuances crucial for micro-expression detection?

### Soundness
3

### Presentation
2

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
This work is about micro expression recognition. MER is very useful for many applications yet very difficult to detect. This work set the task in a semantic framework, instead of relying on pixel-level MER, as most existing work did. A Transformer was used to process flattened latent feature vector, followed by linear projection and alignment to identify micro expression. Results were good and ablation study was supportive of the necessity of each step.

### Strengths
Strengths include the approach to shift from pixel-wise MER to semantics-based MER, and supportive ablation study.
It sounds like a reasonable idea to discretize the difference between two consecutive frames into identity- and domain-invariant action representations, but the ensuing description of the method did not seem to pursue this idea.

### Weaknesses
The writing is not always easy to follow. It is not clear why authors performed face generation by inserting facial token into various locations at the action token matrix. What was the purpose of this step? Is it just to demonstrate that locations of the same token leads to different facial expression? 
In Eq. 5, what was the linear projection P? Can the authors give the detail? 
A big concern is the impact of the size of codebook on the final performance, Table 5. It is puzzling that the size of the codebook would have such a big effect on the capacity, and it is worrisome that a larger codebook size actually led to reduced capacity, according to the table.
In the method, which part is the identity-invariant action representation and which part is the domain-invariant action representation? It seems these two representation were not described in the algorithm.

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The authors are the first to employ semantic-level facial action tokens for MER and model these tokens as 1D sequences analogous to human language. Furthermore, they design an alignment method to establish explicit connections between action tokens and human-understandable emotions. Experiments demonstrate state-of-the-art accuracy and strong cross-subject and cross-domain generalization.

### Strengths
- The authors introduce a new perspective in the MER field: they argue that previous works operate in the pixel domain, making them highly sensitive to identity variations, and thus shift the focus toward a more robust feature domain.

- The authors experimentally demonstrate why 1D token sequences better represent expressions than 2D token maps with absolute positional information, thereby reinforcing the rationality of their design.

- Their ablation studies on the core components clearly verify the independent contributions of each module, including the Transformer, sparse pooling, and EDCLIP loss.

### Weaknesses
- Lacks ablation comparisons with different combinations of loss function weights.

- Shows sub-optimal performance on grayscale datasets.

### Questions
- The experimental section lacks an ablation study on the weighting of loss components. Would exploring different combinations of loss weights further improve the performance?

- The authors mention that the sub-optimal performance on SAMM is due to training on RGB images. Would training on grayscale images reduce the model’s reliance on color and improve performance?

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
4

### Summary
This paper presents D-FACE, a novel framework for micro-expression recognition (MER). The work's primary contribution is a shift away from traditional pixel-level motion representations (like optical flow), which are sensitive to identity, toward a learned, discrete, semantic-level representation called "facial action tokens." The method uses a C-VQ-VAE pretrained on large-scale video data to "tokenize" facial movements. It then models these tokens as a 1D sequence using a Transformer with sparse attention, motivated by an empirical analysis of token semantics. Finally, it aligns these tokens with human-understandable emotions using a novel CLIP-based loss (EDCLIP), which includes a specific margin-based objective to handle the ambiguous "others" category. The method achieves state-of-the-art results and, more importantly, demonstrates impressive cross-subject and cross-domain generalization in qualitative generation experiments.

### Strengths
1. The central idea of discretizing facial motion into semantic tokens is a well-motivated and significant departure from standard pixel-based MER methods. The authors clearly identify the key weaknesses of prior work (identity sensitivity and lack of semantics) and propose a direct solution.
2. The cross-identity and cross-domain generation experiments in Figure 4 are powerful evidence for the paper's main claim. The ability to transfer an ME from one person to another by only transferring the learned tokens, while preserving the target identity, convincingly demonstrates that the D-FACE representation is successfully disentangled from identity.
3. The framework is well-designed and internally consistent. The empirical analysis in Section 3.2, which finds that tokens behave like 1D sequences, provides a clear justification for using a Transformer (a choice validated in the ablations). The sparse attention mechanism is well-suited for the local nature of MEs, and the EDCLIP module's specific handling of the "others" category is an intelligent solution to a practical problem in emotion recognition datasets.

### Weaknesses
1. The leap from the empirical analysis (Section 3.2) to the 1D Transformer modeling is not fully convincing. The paper states that "token semantics are position-dependent" (e.g., token '2' means something different at position (3,3) than at (1,0) in Figure 3). However, it also claims the "absolute 2D spatial positions... do not correspond to specific locations" and that this motivates flattening to a 1D sequence. This seems contradictory. The observation that meaning is position-dependent could equally (or perhaps more strongly) motivate the use of a 2D Transformer with 2D positional embeddings. The paper would be stronger if it provided a clearer explanation for why a 1D sequence is the correct interpretation of this finding.
2. The C-VQ-VAE tokenizer is pretrained on VoxCeleb, a dataset of unconstrained *macro-expressions* (e.g., people talking in interviews). The model is then fine-tuned and evaluated on *micro-expressions*, which are defined by their subtle, rapid, and involuntary nature. This is a significant domain gap. The authors do not discuss whether pretraining on large, voluntary motions biases the codebook and prevents it from learning the extremely subtle motions unique to MEs.
3. The paper states it adapts a C-VQ-VAE architecture from a robotics paper [1] and "refined the codebook design and latent action length". However, the specific details of these refinements are not provided. This makes the core tokenizer component, which is responsible for the paper's main contribution, difficult to understand and reproduce. And more related works should be discussed to support the adaptation. [2]
4. The action encoder architecture in Figure 2(b) includes a "Causal Transformer" that processes features from the two input frames. The term "causal" implies temporal masking, but it is not explained how this is applied between just two static frames ($I_1$ and $I_2$). This component's function and design are unclear.

[1] Seonghyeon Ye, et al. Latent action pretraining from videos. Proceedings of the International Conference on Learning Representations, 2025.

[2] Chen, Haodong, et al. "Finecliper: Multi-modal fine-grained clip for dynamic facial expression recognition with adapters." Proceedings of the 32nd ACM International Conference on Multimedia. 2024.

### Questions
What are the current inference costs, and are there any optimization options?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes D-FACE, a novel framework for micro-expression recognition (MER) that transitions from pixel-level motion descriptors (e.g., optical flow, frame difference) to semantic-level discrete facial action tokens. The approach employs a conditional VQ-VAE to encode facial motion between onset and apex frames into discrete tokens, modeled as sequences with a Transformer using sparse attention pooling. Moreover, an emotion-description-guided CLIP (EDCLIP) module aligns visual features with textual emotion semantics, improving interpretability and generalization. Extensive experiments on CASME-II, SMIC-HS, SAMM, and CAS(ME)3 demonstrate state-of-the-art performance and superior cross-domain generalization.

### Strengths
1. The paper convincingly reframes MER from a low-level motion estimation problem to a semantic-level action representation problem — an important conceptual contribution.

2. The architecture integrates C-VQ-VAE for discrete motion tokenization, Transformer-based sequential modeling, and EDCLIP for semantic alignment. Each component is well-motivated and supported by ablation studies.

3. Empirical thoroughness: Results on multiple MER benchmarks (CASME-II, SMIC-HS, SAMM, CAS(ME)3) are comprehensive. Ablation studies (Table 6) validate each design choice (Transformer, sparse pooling, CLIP alignment). Cross-identity and cross-domain generation experiments (Fig. 4) provide strong qualitative evidence of generalization.

4. The codebook visualization and token analysis (Appendix A.1–A.2) demonstrate that the learned discrete tokens correspond to interpretable, position-dependent facial actions — a rare property in MER models.

### Weaknesses
1. While empirically effective, the paper could strengthen its argument about why discretization (VQ-VAE) leads to domain invariance — possibly through statistical or theoretical analysis of identity disentanglement.

2. The large-scale pretraining dataset for the facial video tokenizer is only mentioned briefly. It’s unclear: Was identity information explicitly removed or balanced? How was pretraining validated before fine-tuning on MER datasets?

3. The contrastive objective (Eq. 12–13) is well-motivated, but comparisons with existing CLIP-based MER methods (e.g., MER-CLIP, 2025) could include a clearer ablation or embedding visualization to show how EDCLIP differs in practice.

4. The authors claim D-FACE enables generalizable facial action understanding, but experiments are limited to MER datasets. It would strengthen the paper to evaluate transferability to macro-expression or AU-based tasks.

5. On the SAMM dataset, performance drops due to grayscale input. The authors acknowledge this but could have mitigated it with color-agnostic pretraining or domain adaptation.

### Questions
What is the scale and composition of the “large-scale facial video data” used for D-FACE pretraining?

How does the choice of codebook size (K=32) affect semantic granularity? Could larger K provide richer action semantics, or would that harm stability?

Did you explore temporal modeling beyond onset–apex pairs (e.g., full ME sequences)?

How sensitive is EDCLIP performance to the choice of emotion text prompts?

### Soundness
3

### Presentation
3

### Contribution
3
