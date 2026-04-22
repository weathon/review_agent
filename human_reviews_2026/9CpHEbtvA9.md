# reAR: Rethinking Visual Autoregressive Models via Token-wise Consistency Regularization

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Visual autoregressive (AR) generation offers a promising path toward unifying vision and language models, yet its performance remains suboptimal against diffusion models. Prior work often attributes this gap to tokenizer limitations and rasterization ordering. In this work, we identify a core bottleneck from the perspective of generator-tokenizer inconsistency, i.e., the AR-generated tokens may not be well-decoded by the tokenizer. To address this, we propose reAR, a simple training strategy introducing a token-wise regularization objective: when predicting the next token, the causal transformer is also trained to recover the visual embedding of the current token and predict the embedding of the target token under a noisy context. It requires no changes to the tokenizer, generation order, inference pipeline, or external models. Despite its simplicity, reAR substantially improves performance. On ImageNet, it reduces gFID from 3.02 to 1.86 and improves IS to 316.9 using a standard rasterization-based tokenizer.  When applied to advanced tokenizers, it achieves a gFID of 1.42 with only 177M parameters, matching the performance with larger state-of-the-art diffusion models (675M).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes reAR, a new regularization framework designed to improve the generation quality of visual autoregressive (AR) models. The authors argue that existing visual AR methods suffer from a generator–tokenizer inconsistency, which manifests as exposure bias amplification and embedding unawareness during training and inference. reAR introduces two lightweight regularization techniques: Noisy Context Regularization, which injects random noise into the input token sequence to simulate imperfect contexts during training, mitigating exposure bias. Codebook Embedding Regularization, which aligns the generator’s hidden states with the tokenizer’s embedding space through cosine distance loss.

### Strengths
1. The paper identifies a genuine and underexplored problem: the misalignment between generator and tokenizer in visual AR models. This perspective is well-motivated and conceptually interesting.
2. The proposed regularizations are straightforward, computationally light, and compatible with existing AR models and different tokenizers.
3. The approach yields improvements across multiple tokenizers (e.g., TiTok, AliTok) and model sizes, and narrows the gap with diffusion-based methods.

### Weaknesses
1. While the paper frames generator–tokenizer inconsistency as a new perspective, the actual solutions (noise injection and embedding regularization) looks similar to existing techniques in language modeling.
2. For a method positioned as “plug-and-play” and intended to generalize across tokenizers and datasets, the paper applies the embedding alignment regularization to specific layers but does not provide analysis behind this choice. The decision appears empirical, based on trying a few configurations and selecting the one yielding the best FID.
3. While reAR demonstrates improvements across different tokenizers (e.g., TiTok, AliTok), the paper does not provide any analysis of how the proposed embedding regularization interacts with the underlying codebook geometry.

### Questions
1. How sensitive is reAR to the noise schedule or regularization strength λ across different datasets and architectures?
2. Since the proposed method involves embedding alignment, how does it interact with tokenizers of different codebook geometries?

### Soundness
2

### Presentation
3

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
This paper analyzes the performance bottleneck in visual autoregressive (AR) models, attributing it to "generator-tokenizer inconsistency." To address this, the authors propose reAR, a plug-and-play training regularization strategy. Without altering the model architecture or inference pipeline, reAR introduces two auxiliary objectives: 1) Noisy Context Regularization, which mitigates exposure bias by training the model in corrupted contexts, and 2) Codebook Embedding Regularization, which forces the model's hidden states to predict the visual embeddings of both the current and next tokens, making the generator aware of the tokenizer's embedding space.

### Strengths
1. reAR delivers excellent results, significantly improving model performance without requiring architectural changes, and appears to be a general-purpose strategy for vision AR models.
2. As a training-only strategy, reAR is highly versatile. Experiments show it works well not only with VQGAN but also significantly boosts other tokenizers like TiTok and AliTok.
3. The paper clearly defines the "generator-tokenizer inconsistency" problem. It uses well-designed comparison experiments (e.g., "perfect context" vs. "imperfect context" and "error token embedding replacement") to convincingly validate that "exposure bias amplification" and "embedding-unawareness" are indeed critical bottlenecks.

### Weaknesses
1. reAR requires applying regularization at specific shallow (e.g., layer 0) and deep (e.g., layer 15) layers, chosen based on CKA analysis and ablations. This feels somewhat like "alchemy" or fine-tuning, lacking an adaptive or theoretically-driven mechanism to automatically determine the optimal layers for feature alignment.
2. The method introduces additional MLP projection layers and an extra loss term, which inevitably increases training complexity and memory consumption. The paper doesn't explicitly quantify the additional training time overhead introduced by the reAR strategy.

### Questions
1. reAR innovatively regularizes both the "current token" embedding (shallow layers) and the "next token" embedding (deep layers). Is there a potential conflict between these two objectives? (e.g., could the shallow layers lose information needed to predict the *next* token in their effort to align with the *current* token?).
2. Could you provide an ablation study showing that only regularizing the "next" token's embedding (which seems like the more direct objective) performs worse than regularizing both?

### Soundness
3

### Presentation
3

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
This work identifies a key bottleneck in visual autoregressive generation—the mismatch between the generator’s token sequences and how the tokenizer decodes them. The authors propose a plug‑and‑play regularization strategy that forces the autoregressive model to align its hidden representations with the tokenizer’s codebook embeddings and to remain robust under noisy contexts. This approach requires no changes to the tokenizer, inference pipeline or generation order, yet yields substantial gains. The method is validated across different tokenizers, showing that improving generator ‑ tokenizer consistency significantly boosts AR image generation.

### Strengths
1. The paper clearly identifies a fundamental issue in visual autoregressive models — the inconsistency between the generator and tokenizer — and presents a well-motivated solution.

2. The proposed generator–tokenizer consistency regularization is simple, plug-and-play, and does not require changes to the tokenizer, inference pipeline, or generation order, making it broadly applicable.

3. Extensive experiments demonstrate that reAR significantly enhances class-conditional image generation, yielding notable improvements in FID on ImageNet.

### Weaknesses
1. The paper does not report the impact of REAR on downstream multimodal understanding tasks or text-to-image generation benchmarks, leaving the broader utility of the method unclear.

2. While the method improves generation quality, the novelty is somewhat limited, as the approach mainly introduces a regularization term rather than a fundamentally new architecture.

3. The study only investigates reAR’s generalization across different VQ tokenizers, but does not examine its effectiveness when applied to different LLM backbones; additionally, the number of VQ tokenizers included in the experiments is limited.

### Questions
N/A

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
This paper tries to improve (traditional, raster-order) autoregressive image generation.

Problem statement: inconsistency between a generator and a tokenizer, i.e., the autoregressive model might generate a token sequence that is hard for the tokenizer to decode back to an image.

Logical development
* The generated token sequence can be unseen by the tokenizer due to exposure bias between training (teacher forcing) and inference (own predictions). It is more problematic in image models than language models because the possibility of the decoder not seeing the combination of generated visual tokens in the training phase.
* AR models produce token indices and do not care the embeddings produced by the tokenizer.


Solutions
* Expose the model to perturbed context during training.
    * (context = previous tokens)
    * It encourages the model not to rely on clean contexts and improves robustness to imperfect histories at inference.
* Align the generator's hidden states with the tokenizer's embedding space.
    * It helps the visual embeddings being reconstructed from the unseen tokens.
    * Method: SimSiam-like architecture and loss for regressing current embedding and next embedding.

Experiments
* Competitors: many.
* ImageNet
* FID 1.42 with 177M parameters (sota diffusion models = 675M)

### Strengths
Originality:
1. The idea of aligning the embeddings of the generator and the tokenizer is original.
2. Please see weakness 1.

Quality:
1. The experiments thoroughly compare many competitors in Table 1.
2. Please see weakness 2. 
3. Ablation study is thorough.

Clarity:
1. The problem statement, logical development, solution, and empirical supports are clear as described in Summary.
2. Please see weakness 3.

Significance:
1. The inconsistency between the embeddings of generator and tokenizer and its solution are important because they are the fundamental components of AR models.

### Weaknesses
1. Perturbing something is a common practice for robustness. Why is the proposed perturbation non-obvious compared to the literature?
2. Experiments are conducted only on ImageNet.
3. The scope should be specified in more detail such as raster-order autoregressive modeling because the paper does not tackle other autoregressive models such as [Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction].

minor

1. The term "context" should be defined. It is okay-ish to be implied from context (lol) but it can be clearer.

### Questions
1. Figure 1 should be explained in more detail. What is wrong in the image of orange cat? Why are the top and bottom images similar although the token indices are different?
2. Resolving weakness 1 will raise my rating regarding originality.
3. Resolving weakness 2 will raise my rating regarding soundness.
4. Resolving weakness 3 will raise my rating regarding clarity. If the proposed method is applicable to VAR, please help me find the statement in the paper.

### Soundness
3

### Presentation
2

### Contribution
3
