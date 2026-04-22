# Safe Autoregressive Image Generation with Iterative Self-Improving Codebooks

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Unlike diffusion-based models that operate in continuous latent spaces, autoregressive unified multimodal models produce images by sequentially predicting discretized visual tokens. These tokens are derived from a codebook that maps embeddings to quantized visual patterns. The language-like architecture enables unified multimodal models to effectively capture text conditional information for generation, making them promising for text-to-image tasks. This also raises an interesting question: how safe are the images generated in such an autoregressive way? Existing methods that ensure safe generation by operating on diffusion continuous representations fail to generalize well to discrete representations. In this work, we propose iterative self-improving codebooks for safe autoregressive generation. We leverage the understanding and judgment capabilities of the unified multimodal model itself to identify unsafe generated images without human annotation. Subsequently, the inherent representations in the codebook are fixed to eliminate harmful mappings. Our method comprises two steps: first, we use the unified model to identify unsafe generations and construct corresponding harmful and safe image-text pairs. These pairs are used to construct the Harmful Space and guide updates to the codebook, thereby eliminating harmful outputs. Second, we perform adaptive fine-tuning on the codebook within the harmless space using safe image-text pairs to ensure the quality of generated images. These two steps are repeated until no further improvement is observed, producing a safety-enhanced model codebook. Extensive experiments are conducted to verify the effectiveness of our method on five unified multimodal models and on eight harmful-prompt datasets. Without additional external feedback, the safety of models is improved iteratively.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors propose an iterative pipeline that uses a unified autoregressive multimodal model to increase safety in text-to-image generation by preventing harmful concept generation. They use the multimodal model to generate and identify unsafe generations without human labels, and from it construct a “harmful space” from harmful/safe image–text pairs. Next, they update the model’s discrete codebook to remove harmful mappings, and fine-tune the codebook within the null space to retain image quality. These update and finetuning steps are repeated until no further improvement is observed.

### Strengths
1. The paper addresses an important problem and is well-written and well-motivated.
2. The idea of constructing a harmful space by averaging differences of harmful and harmless prompts is novel and interesting.
3. The method is tested across multiple unified multimodal models and several harmful-prompt datasets, and presents improvement in all.

### Weaknesses
1. The visual examples in the paper are limited. More visual examples, as well as visual comparisons with competing methods, could strengthen the paper.
2. The removal of codebook vectors could remove desirable visual modes or bias the model’s outputs (style/content drift), especially for harmless prompts. see q.2 below.
3. The method assumes the multimodal model can successfully distinguish between harmful and harmless content. see q.3 below.

### Questions
1. If an image is generated based on a prompt with a harmful attribute and is not harmful, does it still match the prompt? More visual examples with the required prompts would help (e.g. figure 2 does not contain the prompts, so it is hard to assess whether generations align with the prompt or not)
2. Although the authors reported FID between COCO images and images generated with and without their method, they used I2P prompts, which contain harmful content. Moreover, it does not say anything about prompt alignment.
What happens after applying Safe-CB for "regular" harmless prompts? Do generations still align with the prompts?
3. Did the authors verify that the multimodal model can successfully distinguish between harmful and harmless content?
4. How are the "safer versions" of the harmful prompts generated?
5. Did the authors perform a baseline experiment of simply replacing the harmful prompts with their "similar safer version"? I assume this will reduce the harmful content and will maybe generate similar results to Safe-CB in terms of prompt similarity, given that this is the training data.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Existing methods that aim to ensure safe generation by operating on continuous diffusion representations often fail to generalize effectively to discrete representations. In this work, the paper propose iterative self-improving codebooks for safe autoregressive generation.
The proposed method is able to reduce harmful generations on eight harmful-prompt datasets  while perserve its original capabilities of the models.

### Strengths
- The method leverages the understanding and judgment capabilities of a unified multimodal model to identify unsafe generated images without requiring human annotation.
- The idea of adaptive codebook fine-tuning in a harmless space is interesting
- Extensive experiments demonstrate the effectiveness and applicability of the proposed approach.

### Weaknesses
- L199-200, It is unclear how the paper “constructs a semantically similar but safer version with minimal modification.” The technical details behind this process are insufficiently explained.
- The method section lacks detailed elaboration on the “iterative” and “self-improving” nature of the safe codebook construction. It is ambiguous whether the iterative process occurs only in Step 1 (Sec. 3.2.1), where more harmful concepts are incrementally included, or whether Steps 1 (Sec. 3.2.1) and 2 (Sec. 3.2.2) are repeated iteratively.
- Some visual examples include inappropriate content that should be pixelated or masked.
- The implementation and training details of the proposed method are missing. For example, the choice of k in the top-k singular vectors, learning rate, and other key hyperparameters are not specified.

### Questions
see weakness

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes “iterative self-improving codebooks” to make autoregressive, unified multimodal generators safer. 
The model first uses its own understanding to flag/label unsafe generations, builds a harmful subspace from safe/unsafe prompt–image pairs, projects the codebook away from that subspace, and then fine-tunes in the null space to recover quality. 
Experiments span Janus, VILA-U, Emu3, LlamaGen, OmniMamba and several unsafe-prompt datasets; the authors report reduced unsafe outputs with small FID impact and preserved Geneval/MMMU scores

### Strengths
1. Broad model coverage and multiple safety data sources; qualitative examples help intuition.

2. clear writing and good flow

3. This paper tackles autoregressive visual tokenizers where diffusion-space safety tricks don’t transfer, so technically it is a novel topic and direction.

### Weaknesses
1. Missing semantic-faithfulness metrics on safe generation benchmarks. The paper reports detectors (NudeNet/Q16), FID_g on COCO-30k, Geneval, and MMMU. These do not directly measure text–image semantic consistency post-safety. Add at least one alignment metric—e.g., CLIP-R/CLIP-Score on benign prompts, TIFA, or GenEval’s text–image agreement subcomponents—to ensure the method isn’t “over-sanitizing” or drifting semantics while removing unsafe content

2. The approach assumes you can (a) pre-specify or mine harmful concepts and (b) represent them as low-rank directions in codebook space. This risks coverage gaps (novel, compositional, or pragmatic unsafe intents) and maintenance burden as new categories and jailbreak patterns emerge. Maybe you can evaluate on held-out harmful categories and adversarially paraphrased prompts not used to build the subspace to defend on this.

3. Self-labelling risks & error propagation. The core loop relies on the same unified model to judge safety, create safe/harmful pairs, and then update itself. If the model under-detects a class of harms, the projection space will miss it. You partially compare the model’s judgments with detectors/humans, but the paper still lacks a human-in-the-loop spot-check.

4. missing discussion with SAFREE and other safety methods.

### Questions
1. When concatenating multiple harmful subspaces (sexual + violence + …), how do you prevent capacity conflicts and benign concept collateral damage? Any orthogonalization or sparsity control? 

2. we need to have some masks on unsafe parts in figure  2

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
The paper introduced a novel self-supervised framework design to isolate harmful image generation in autoregressive models by updating and refining the model’s internal codebook. Instead of relying on external filters or human annotations. The method enables the model to analyze its own generations, identify harmful and harmless patterns and creates a harmful latent space to identify the harmful embedding directions that is correlated with undesirable content. The model then removes these harmful components from the codebook and performs fine-tuning within the null space to ensure the quality and without reintroducing unsafe information.

### Strengths
- A novel approach that uses self-supervised methods and not relying on human annotation data or external classifiers. 
- By updating the internal codebook, it is more efficient because it is not updating the whole autoregressive model, reducing the computation cost. 
- The Fine-turning step made sure the quality of generation, and will not reintroduce new unsafe content.

### Weaknesses
- The self-supervised can miss subtle or context-dependent unsafe content. The self-supervised mechanism relies solely on the model’s own judgment of harmfulness without external validation. The unified model can classify generated images as safe or unsafe. Its internal bias or insufficient understanding of unsafe context may lead to false negatives.
- Lack of detail of detail on how Δ (the codebook perturbation) is optimized. The paper briefly mentioned fine-tuning the projected codebook with a perturbation term Δ in the null space to restore visual quality. But it lacks clear details about the optimization process, including how Δ interacts with the null-space constraint in practice. This makes the method’s reproducibility and theoretical stability less convincing.
- Using projection of harmful space might eliminate useful visual features if the features are overlap. With the projection matrix, it projects the harmful space to the codebook and removes the area. It contains the risk that overlapping semantic directions that features are partially correlated with both harmful and unharmful content. This might affect the performance of the generation. Can it use any other method to soft the boundary?

### Questions
Please see the weaknesses section

### Soundness
3

### Presentation
3

### Contribution
3
