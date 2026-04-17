# SoftCFG: Uncertainty-guided Stable Guidance for Visual Autoregressive Model

- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
Autoregressive (AR) models have emerged as powerful tools for image generation by modeling images as sequences of discrete tokens. While Classifier-Free Guidance (CFG) has been adopted to improve conditional generation, its application in AR models faces two key issues: guidance diminishing, where the conditional–unconditional gap quickly vanishes as decoding progresses, and over-guidance, where strong conditions distort visual coherence. To address these challenges, we propose SoftCFG, an uncertainty-guided inference method that distributes adaptive perturbations across all tokens in the sequence. The key idea behind SoftCFG is to let each generated token contribute certainty-weighted guidance, ensuring that the signal persists across steps while resolving conflicts between text guidance and visual context. To further stabilize long-sequence generation, we introduce Step Normalization, which bounds cumulative perturbations of SoftCFG. Our method is training-free, model-agnostic, and seamlessly integrates with existing AR pipelines. Experiments show that SoftCFG significantly improves image quality over standard CFG and achieves state-of-the-art FID on ImageNet 256 × 256 among autoregressive models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces SoftCFG, a plug-and-play inference-time method that improves the stability and fidelity of visual autoregressive models. The method replaces the fixed classifier-free guidance with a soft, uncertainty-weighted guidance that adaptively adjusts the influence of each generated token. Additionally, a Step Normalization mechanism is proposed to bound accumulated perturbations across steps. Experiments on ImageNet and text-to-image benchmarks show consistent FID improvements over standard CFG without retraining.

### Strengths
1. The method is lightweight, easy to integrate into existing AR inference pipelines, and does not require additional training or data.
2. The motivation is clear and relevant to current challenges in autoregressive generation.
3. The paper is well-written, logically structured, concise, and clear, making it easy for readers to understand.

### Weaknesses
1. SoftCFG largely reinterprets existing CFG dynamics rather than proposing a fundamentally new principle. The main innovation lies in weighting and normalization heuristics.
2. The reported FID gains (~0.1–0.2) are within or close to the variance commonly observed across runs and sampling seeds. It is unclear whether these improvements are statistically significant or perceptually meaningful.

### Questions
1. What is the standard deviation of FID across runs? Are the reported improvements statistically significant under multiple random seeds or sampling temperatures?
2. Since SoftCFG strengthens high-confidence tokens, what happens if the model is confidently wrong? Could this amplify hallucination or bias?

### Soundness
2

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
4

### Summary
This paper introduces SoftCFG, a new uncertainty-guided inference method for discrete visual autoregressive (AR) models. The method aims to mitigate the problem of the guidance signal diminishing over time by reweighing the confidence of previous tokens within the unconditional KV cache. And it introduces the Step Normalization to avoid the explosion of the guidance caused by this reweighing step. Experimental results demonstrate that SoftCFG can improve the generation quality.

### Strengths
1. The proposed SoftCFG method is intuitive, easy to understand, and appears simple to implement.
2. The method shows a notable improvement in generation quality on the ImageNet-256x256 dataset.

### Weaknesses
1. **Limited Generalizability:** The paper presents SoftCFG as a general method, yet its effectiveness is only thoroughly validated on a single model. This narrow experimental scope is insufficient to support the claim of generality. Furthermore, the qualitative results shown for the RAR model in Figure 1 are not convincing and do not demonstrate a clear benefit. (Beside, the new versions (10 October 2025) of the baseline model alitok can achieve CFG results comparable to those reported in your paper with SoftCFG. This raises a critical question: is the reported FID improvement a genuine algorithmic contribution of SoftCFG, or does it merely compensate for a sub-optimally tuned CFG baseline? )
2. The 'diminishing guidance' problem may be an artifact of the baseline using only a single class token. Would this problem already be alleviated in standard CFG if the class token were repeated 64 times as the condition, similar to MAR?
3. There appears to be a significant error in Equation 6. The equation as written is inconsistent with the line 8 in Algorithm 1 and does not align with Equation 7. I suspect the correct formulation should be $$z_t^{SoftCFG}=z_t^{cond}+scale*(z_t^{cond}-z_t^{uncond,pertcontext})$$?
4. **Missing Quantitative Results:** The paper text explicitly states in Section 3.1 (lines 349-350, 355-358) that quantitative results (GenEval benchmark, DPG-Bench) for Text-to-Image (T2I) generation are provided. However, no such quantitative results are present anywhere in the paper. This is a major omission that leaves the T2I claims entirely unsubstantiated.
5. **Overall Presentation:** The paper is not well-written. The presentation suffers from a lack of clarity, and the significant issues noted above (e.g., the error in Equation 6, the missing T2I results) make the paper difficult to follow.

### Questions
see Weakness

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
The paper proposes **SoftCFG**, a training-free and model-agnostic inference modification for **visual autoregressive (AR)** image generation models.  
It addresses two common problems when applying Classifier-Free Guidance (CFG) to AR models:  
(1) *guidance diminishing* (conditional signal fading as decoding progresses), and  
(2) *over-guidance* (visual distortions caused by high guidance scales).

SoftCFG introduces **uncertainty-guided token-wise perturbations** to the unconditional branch: each past token contributes to guidance proportionally to its **prediction confidence**, influencing future decoding steps through the cached value vectors.  
A **Step Normalization** mechanism further ensures stability by normalizing cumulative perturbations at each step.  
The method requires no retraining, adds negligible computational cost, and yields improved FID on ImageNet-256 (1.37 → 1.27) compared to vanilla CFG.

### Strengths
1. **Clear motivation and problem framing** – The issues of guidance fading and over-guidance in AR models are well-illustrated with entropy plots and examples.  
2. **Elegant, simple solution** – The token-wise confidence weighting and step normalization are easy to implement and integrate into existing AR inference pipelines.  
3. **Training-free and architecture-agnostic** – Works as a plug-in for existing models like AliTok and LuminaGPT without retraining or modifying the transformer architecture.  
4. **Empirical improvement** – Achieves state-of-the-art FID among AR models on ImageNet-256 with negligible runtime overhead.  
5. **Good ablations and qualitative examples** – The paper studies the impact of StepNorm, guidance scale γ, and scheduling power k, and presents clear visual comparisons showing fewer artifacts.  
6. **Transparency and reproducibility** – The paper includes clear algorithms, theoretical bounds, and a stated plan to release code.

### Weaknesses
1. **Limited novelty** – Conceptually extends prior ideas (adaptive guidance, token-level perturbation) from diffusion models to the AR setting.  
2. **Fragile confidence heuristic** – The reliance on max probability as a proxy for uncertainty can mislead guidance, as shown in the “cat–car” failure case.  
3. **Partial perturbation design** – Only value caches are scaled; effects on keys or multi-head attention routing are not explored.  
4. **Step Normalization rigidity** – The fixed-sum normalization may underutilize guidance in long sequences; no adaptive scheduling is tested.  
5. **Loose theoretical analysis** – The Lipschitz-based bound is general but too weak to predict actual model stability.  
6. **Limited experimental breadth** – Results are restricted to ImageNet-256 and a few text-to-image examples; no tests on higher-resolution or multi-modal AR tasks.  
7. **Hyperparameter fairness** – CFG baselines may not have been re-tuned under identical γ/k sweeps, potentially overstating SoftCFG’s advantage.

### Questions
1. Why only perturb the **value cache (V)** and not keys or queries?  
2. How sensitive is the performance to **confidence miscalibration**? Would temperature scaling or learned uncertainty improve robustness?  
3. Could **Step Normalization** be relaxed or made adaptive for longer contexts?  
4. Are improvements consistent under **different tokenizers** or **generation orders** (e.g., folded or diagonal AR)?  
5. How does SoftCFG behave on **text-to-image benchmarks** quantitatively (e.g., COCO-FID, GenEval metrics)?

### Soundness
3

### Presentation
3

### Contribution
3
