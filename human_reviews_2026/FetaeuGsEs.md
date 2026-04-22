# A Hidden Semantic Bottleneck in Conditional Embeddings of Diffusion Transformers

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 2

## Abstract
Diffusion Transformers have achieved state-of-the-art performance in class-conditional and multimodal generation, yet the structure of their learned conditional embeddings remains poorly understood. In this work, we present the first systematic study of these embeddings and uncover a notable redundancy: class-conditioned embeddings exhibit extreme angular similarity, exceeding 99% on ImageNet-1K, while continuous-condition tasks such as pose-guided image generation and video-to-audio generation reach over 99.9%. We further find that semantic information is concentrated in a small subset of dimensions, with head dimensions carrying the dominant signal and tail dimensions contributing minimally. By pruning low-magnitude dimensions--removing up to two-thirds of the embedding space--we show that generation quality and fidelity remain largely unaffected, and in some cases improve. These results reveal a semantic bottleneck in Transformer-based diffusion models, providing new insights into how semantics are encoded and suggesting opportunities for more efficient conditioning mechanisms.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the properties of conditional embeddings in conditional diffusion models, revealing that they are sparse yet locally uniform across dimensions. The analysis is conducted on several representative class-conditional diffusion models and is extendable to continuous class embeddings and diverse conditioning signals, e.g., video-to-audio and pose-guided image generation. By removing redundant dimensions, slight improvements in FID and CLIP scores are observed.

### Strengths
- The finding that conditional embeddings in DiT are sparse but locally uniform is meaningful, as it reveals redundancy in the conditional representations of current conditional diffusion models. This insight can motivate new directions for future research.
- Both qualitative and quantitative experiments on diffusion models with various types of conditions validate that conditional embeddings in current DiTs exhibit significant redundancy.
- The authors demonstrate that simply removing redundant dimensions in conditional embeddings at inference time can enhance generation quality to some extent, highlighting the potential of optimizing conditional embeddings to improve DiTs.

### Weaknesses
- The proposed hypotheses lack experimental support. For example, regarding the hypothesis about the cause of high cosine similarity in conditional embeddings, the authors argue that DiTs favor embeddings providing stable and robust signals for denoising because they condition on embeddings across all timesteps. Maybe a training log similar to Fig. 12, but for a model trained on fewer timesteps, could validate this hypothesis. Since inference performance is not the concern here, such a model could be trained on a subset of timesteps.
- There is a contradiction between Lines 454–457, which state that high cosine similarity occurs only in transformers and not in U-Nets, and Lines 467–470, which state that similar redundancy can also appear in U-Net-based diffusion models. Moreover, the authors neither present nor discuss experimental results on U-Net diffusion models.
- The authors claim that AdaLN amplifies high-magnitude dimensions of conditional embeddings, thereby preserving semantic differences, but this claim also lacks experimental validation. What happens if AdaLN is replaced with another conditioning mechanism, such as cross-attention?

### Questions
What is the setting of k in Table 2?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This is a very interesting work. This paper makes significant contributions to the understanding of conditional embeddings in diffusion Transformers, a critical component of state-of-the-art generative models.

### Strengths
1. It is the first work to systematically investigate the internal structure of conditional embeddings in diffusion Transformers, uncovering two core emergent properties—extreme angular similarity (exceeding 99% on ImageNet-1K and 99.9% on continuous-condition tasks) and high-dimensional sparsity (only 1–2% of dimensions carrying substantial semantic information). This fills a critical gap in existing literature, which has primarily focused on architectural advances rather than the intrinsic characteristics of conditional encoding.
2. The experiments are well-designed and extensive, covering six diffusion Transformer models (DiT, MDT, SiT, REPA, LightningDiT, MG) on discrete class-conditional image generation (ImageNet-1K) and two continuous-condition tasks (pose-guided image synthesis with X-MDPT and video-to-audio generation with MDSGen). 
3. The demonstration that pruning up to 66% of low-magnitude dimensions preserves or even improves generation quality provides actionable insights for model optimization. This offers a potential path toward more efficient conditioning mechanisms—particularly valuable for resource-constrained applications.

### Weaknesses
1. Although the author reveals the redundancy problem in condition embeddings of diffusion models via empirical analysis, they do not provide any further method to improve the efficiency or performance of existing models based on the observation.
2. More conditions should be considered. For instance, user-generated descriptions or prompts are more general than class-conditional prompts. I suggest the author investigate more types of conditions.
3. Maybe the time embedding and prompt embedding should be considered separately.

### Questions
see weaknesses

### Soundness
4

### Presentation
4

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
This paper presents a systematic empirical analysis of conditional embeddings in transformer-based diffusion models. Through extensive experiments across state-of-the-art models and multiple datasets, the authors show that conditional embeddings exhibit large redundancy and sparsity. The class-conditioned embeddings across different classes can achieve over 99% cosine similarity, which is kind of surprising, and only a small subset of dimensions carries meaningful semantic information. By pruning up to two-thirds of embedding dimensions, they demonstrate that generative quality is unaffected or even improved.

### Strengths
1.  The paper evaluates a broad set of transformer-based diffusion models, including DiT, MDT, SiT, REPA, LightningDiT, etc, on diverse benchmarks. And the paper shows evidence of near-uniform cosine similarity and embedding sparsity. The breadth of experiments convincingly substantiates the paper’s core claims.

2. The figures and results tables in the paper vividly illustrate the redundancy and sparsity of class embeddings, especially the high cosine similarity and the dominance of a few embedding dimensions, and the visual/quantitative impact of pruning. The ablation and t-SNE plots reinforce key claims regarding semantic encoding and redundancy.

### Weaknesses
1. The theoretical analysis, while mathematically sound in defining PR and sparsity, stops short of providing a fundamental theoretical explanation of why such extreme redundancy and cosine similarity emerge in the conditional embeddings of transformer-based diffusion models. The paper’s stated hypotheses remain largely empirical and conceptual, lacking formal proofs or broader generalization to other conditioning modalities or even to transformer architectures outside the diffusion context. For instance, while the AdaLN/linear projection argument is plausible, its theoretical justification is not fully fleshed out.

2. The investigated tasks are representative, but would be better if they could be extended further. For example, would the text-to-audio diffusion models also demonstrate such highly correlated class embeddings?

### Questions
1. Can the authors clarify whether they have empirically tested for similar redundancy/sparsity in conditional embeddings in U-Net based diffusion models? If so, does the semantic bottleneck extend beyond the transformer/AdaLN regime?

2. Could the authors provide detailed ablations on how different conditioning injection techniques (addition vs. concatenation vs. cross-attention), projection, or other schemes affect the redundancy and sparsity? Would, for example, nonlinear projection heads mitigate the observed bottleneck?

3. Although some of the dimensions can be pruned and make little difference in the generation quality, they seem to make little improvement. How would this be useful? 

4. Despite claims of negligible drops or marginal improvements in generation quality after pruning, would pruning induce minor artifacts like edge collapse or structure distortion? Could the authors present more failure case analyses?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper discovered a phenomenon in XL-size DiT and its variants: extremely high cosine similarity and semantic sparsity in conditional embeddings. By unveiling this discovery, the authors aim to gain a deeper insight into how diffusion transformers encode conditioning signals and call for further exploration of more efficient and compact conditioning mechanisms.

### Strengths
Besides discovering the similarity and sparsity from condition embeddings in DiT models, this paper also tries to convert the observation to action by pruning the condition and provides multiple hypotheses for further exploration.

### Weaknesses
By discovering this phenomenon, the authors hope that “future architectures could benefit from compressed or hybrid conditioning strategies that maintain semantic fidelity while reducing computational overhead.” However, the conditional embedding carries not only the condition information but also the timestep information, which is important for image generation diffusion models. It’s not enough to only evaluate the semantic fidelity of the conditional embedding.

Meanwhile, the paper does not present experiments that demonstrate the utility of this discovery, such as proving that pruning indeed "reduces computational overhead".

Furthermore, the observation only builds upon the XL-size model.

### Questions
Apart from what has been mentioned in the weaknesses section:
1. It would be better to also report more evaluation metrics, such as precision and recall values, in class-conditional image generation, following similar evaluation settings in DiT, REPA, etc. 
2. In Section 5, the authors use REPA “as a representative model,” but didn’t comment later on whether the observations on REPA also universally exist in other models. 
3. In Section 5 “Continuous work”, the results are mostly qualitative not quantitative. Could more standard quantitative evaluation metrics, such as FID/SSIM/LPIPS, be added to compare the pruning and no-pruning cases?

### Soundness
2

### Presentation
1

### Contribution
2
