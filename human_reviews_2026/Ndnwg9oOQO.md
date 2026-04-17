# NextStep-1: Toward Autoregressive Image Generation with Continuous Tokens at Scale

- Decision: Accept (Oral)
- Scores: 6, 6, 2, 4

## Abstract
Prevailing autoregressive (AR) models for text-to-image generation either rely on heavy, computationally-intensive diffusion models to process continuous image tokens, or employ vector quantization (VQ) to obtain discrete tokens with quantization loss. In this paper, we push the autoregressive paradigm forward with NextStep-1, a 14B autoregressive model paired with a 157M flow matching head, trained on discrete text tokens and continuous image tokens with next-token prediction objectives. NextStep-1 achieves state-of-the-art performance for autoregressive models in text-to-image generation tasks, exhibiting strong capabilities in high-fidelity image synthesis. Furthermore, our method shows strong performance in image editing, highlighting the power and versatility of our unified approach. To facilitate open research, we have released our code and models to the community at https://github.com/stepfun-ai/NextStep-1.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work introduces NextStep-1, an AR model with flow matching head for image generation. The architecture desire of NextStep-1 is a lot like MAR[1] / FLUID[2] with the diffusion head replaced by flow matching. The work also investigates tokenizer design and figures out the importance of regularized latent space. Experiments on multiple image generation/editing benchmarks show strong performance of NextStep-1 compared to previous diffusion and AR models. 

Reference:
[1] Li, Tianhong, et al. "Autoregressive image generation without vector quantization." Advances in Neural Information Processing Systems 37 (2024): 56424-56445.
[2] Fan, Lijie, et al. "Fluid: Scaling autoregressive text-to-image generative models with continuous tokens." arXiv preprint arXiv:2410.13863 (2024).

### Strengths
1. The paper is well-written and easy to follow. 
2. The paper investigates building strong AR visual generative models which is an important direction. 
3. The paper includes comprehensive experiments on multiple image generation / editing benchmarks and shows strong performance.

### Weaknesses
1. The paper claims state-of-the-art performance of NextStep-1. Though it does show strong performance on multiple benchmarks, it doesn't dominate all the metrics. For example in table 1, there are better baselines in GenEval. 
2. The paper introduces a natural extension to MAR/FLUID, that is replacing the diffusion head with flow matching, and proves its effectiveness. This weakens the methodological contributions in this work.

### Questions
1. In pre-training and tuning, how are the objectives of LM head and flow-matching head balanced?
2. How much compute overhead does the flow matching head add in inference compared to standard AR models?
3. Do the authors compare the convergence rate of NextStep-1 with standard AR models? Since usually flow-matching/diffusion models are less efficient in convergence. 
4. Will channel-wise normalization weaken the expressivity of learned latents since it can break the correlation between different channels? Also, does it lead to instability in training since it's a per-sample normalization?

### Soundness
3

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
4

### Summary
This manuscript introduces NextStep-1, a large (14B) autoregressive (AR) model for text-to-image generation. Unlike prevailing AR models that rely on Vector Quantized (VQ) discrete tokens, NextStep-1's core innovation is its direct autoregressive prediction on **continuous image tokens**.

Its architecture consists of a large AR Transformer (Qwen2.5-14B) and a lightweight (157M) Flow Matching (FM) head. The model processes tokens sequentially: for text tokens, it uses a standard LM head (cross-entropy loss); for each continuous image token (patch), it uses the AR model's hidden state as a condition to drive the FM head, which generates the patch.

The paper's central claims include:

1. This "minimalist architecture" (AR-LLM + lightweight FM head) can achieve SOTA performance on T2I tasks, competitive with top-tier diffusion models.

2. The key to this success is a **novel image tokenizer** (fine-tuned from Flux VAE). This tokenizer, through channel-wise normalization and stochastic perturbation, creates a robust latent space that enables stable training on high-dimensional continuous tokens.

3. The model achieves SOTA or competitive results on T2I benchmarks (e.g., GenEval, WISE) and image editing benchmarks (e.g., GEdit-Bench).

### Strengths
1. **SOTA Autoregressive Performance**: The paper's primary contribution is demonstrating, for the first time, that an AR model based on continuous tokens (NextStep-1) can achieve SOTA performance on T2I tasks, rivaling top-tier diffusion models. This architecture, combining a large AR Transformer for context prediction with a lightweight FM head for continuous token generation, is proven to be a very successful and promising technical direction.

2. **Deep Insights into Tokenizer and Latent Space**: One of the paper's most outstanding merits is its deep analysis of the tokenizer in Section 4.2. The paper correctly identifies a key bottleneck in generative modeling: the quality and properties of the latent space.

3. **Excellent Analysis of CFG Instability**: The paper's insight in Section 4.2 and Figure 3, which attributes instability at high CFG scales to "per-token distributional shift" (rather than 1D RoPE issues), is an excellent observation. The proposed solution (channel-wise normalization in the VAE) is simple and effective.

4. **Strong FM Head Ablation**: The finding in Section 4.1—that the size of the FM head has minimal impact on final quality—is strong evidence supporting the authors' core argument that the AR Transformer is performing the "core generative modeling," while the FM head acts primarily as a lightweight sampler.

5. **Clarity and Completeness**: The paper is exceptionally well-written, complete, and clear. The detailed appendices (e.g., data pipelines in Appendix B, training recipes in Appendix C) are thorough, transparent, and provide significant reference value for the community.

### Weaknesses
1. **Inherent Bottleneck in Inference Latency**: The paper admits in Appendix D and Table A2 that inference latency is a major weakness. The AR sequential decoding is the first bottleneck, and the FM head's multi-step sampling is the second. A 1024-token image requiring 11.31 seconds of accumulated latency (Table A2) is likely far slower in practice than parallel diffusion models.

2. **Significant Challenges in High-Resolution Scaling**: The paper frankly states in Appendix D that the model faces challenges in scaling to high-resolution training (e.g., 1024x1024) and admits that techniques developed for high-res diffusion (like timestep shift) are not applicable to this framework. This severely limits the model's practical ability to "compete" with SOTA diffusion models, which commonly excel at 1024x1024.

3. **Contradictory Information on Stability**: The paper's narrative on stability is confusing. Section 4.2 claims the tokenizer solves CFG instability. However, Appendix D (Limitations) opens by stating that high-dimensional continuous latents introduce "unique stability challenges," showing failure cases like bottom noise, solid color blocks, and grid artifacts (Figure A3). This suggests a trade-off: does solving CFG drift by forcing a normalized VAE latent space introduce new, severe generative artifacts in high dimensions?

4. **"Data-Hungry" SFT Process**: Appendix D notes that the SFT process is unstable and requires "million-sample scale" datasets to show significant improvement, a stark contrast to diffusion models that can be fine-tuned with a few thousand samples. This drastically undermines the model's practical utility for alignment and customization (e.g., LoRA), which is a core strength of modern diffusion models.

### Questions
To validate the paper's core claims and increase its impact, I strongly recommend the authors address the following key questions with experiments:

**Question 1: Does the reliance on million-sample SFT imply a fundamental flaw in the model's "alignability" and "customizability"?**

The paper admits SFT requires massive data. Does this mean the model is far less efficient or practical for aligning with human preferences and performing style customization (e.g., LoRA) compared to the diffusion models it claims to rival?

**Question 2: How should we evaluate the trade-offs of this AR paradigm, given its SOTA quality but practical limitations in speed and scalability?**

Query: The paper demonstrates impressive SOTA quality, yet Appendix D and Table A2 confirm this comes at the cost of inference latency and high-resolution scaling challenges. Could the authors elaborate on how they view this trade-off? Is the goal to prioritize quality and contextual dependency over speed, and what are the most promising future directions to mitigate these latency and scaling bottlenecks?

**Question 3: Regarding the artifacts in Appendix D (Figure A3) for high-dimensional tokens: Are these a side effect of the normalized VAE?**

The paper shows new stability challenges (e.g., bottom noise, solid blocks) in high dimensions. Are these artifacts a direct side effect of the channel-wise normalization or noise perturbation introduced in the VAE? Does forcing a normalized latent space, while solving CFG drift, create new optimization challenges for the AR model in high-dimensional spaces?

**Question 4: Is the 1D Raster-scan order a bottleneck for capturing 2D spatial dependencies?**

When using a 1D row-first serialization, the last token of one row is not spatially adjacent to the first token of the next. Does this hinder the model's ability to learn vertical spatial relationships across rows? Why not employ a more spatially-aware serialization (e.g., Z-order, Hilbert curve) or 2D positional encodings to address this?

**Question 5: Is the 14B scale of the AR Transformer a necessary condition for high-quality generation?**

The paper successfully adapts a 14B LLM (Qwen2.5) for continuous image generation. But is this massive scale essential? Could a smaller AR model (e.g., 2B or 7B), designed specifically for image context, achieve similar results with the FM head? What role does model scale play in the "core generative modeling"?

**Question 6: What is the advantage of this "AR context + FM patch" paradigm over architectures that predict global velocity (e.g., Transfusion, BAGEL)?**

Models like Transfusion or BAGEL use a Transformer to parallelly predict the flow velocity for the entire image. What are the conceptual and practical advantages of NextStep-1's "serial AR prediction + serial FM generation" architecture? Does it trade parallelism (speed) for stronger contextual dependency or finer-grained local control?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes an autoregressive text-to-image model that achieves state-of-the-art performance with minimalist architecture with revealing some key design principles.

### Strengths
1. I really appreciate the research taste of the paper. The paper uses some extreme simple method to achieve state-to-the-art performance.

2. The paper can give many take aways to the sequent researchers.

### Weaknesses
1. I really suggest the authors to hire someone who is adept at academic paper writting to re-write the whole paper. For instance, the current paper is very unclear. For instance, the introduction is too short. The related work paper is too short without fully respecting the former authors. Given this, I have to lower my score to 4. This is a paper needs major revision.

2. Many parts are unclear. For instance, what GPUs are used in training. The pre-training and post-training sections are also unclear.

3. I do not know what the “Next” in the paper presents?

### Questions
NA

### Soundness
4

### Presentation
1

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents NextStep-1, an autoregressive image generation framework that combines patch-level flow matching with a continuous-token VAE. The approach demonstrates strong performance on both generation and editing benchmarks, supported by solid ablation studies and high-quality visual results.

### Strengths
1.The paper is clearly written and technically sound, presenting a coherent and well-motivated method.

2.The ablation studies are particularly valuable—for instance, the analysis clarifying the relative contributions of the backbone versus the flow-matching head provides useful architectural insights.

3.Figures and tables are of high quality and effectively support the claims made in the text.

### Weaknesses
1.Several key technical details remain underspecified. For example, it is unclear whether the autoregressive process decodes one patch at a time and how global consistency is maintained if patches are stitched together.

2.The design of the tokenizer—including its normalization scheme—is described only textually; a schematic illustration would greatly improve clarity.

3.While the paper asserts that “reconstruction quality is the upper bound of generation quality,” it does not explicitly discuss the implications for controllable or conditional generation tasks.

4.The positioning of this work within the broader literature is incomplete: it is not clear whether this is the first AR model to employ patch-level flow matching, nor does the paper compare the trade-offs between alternative design choices (e.g., tokenization strategies, training objectives), making it difficult to assess the method’s ultimate potential or limitations.

### Questions
1.Is the autoregressive generation performed patch-by-patch? If so, how is spatial consistency ensured across adjacent patches in the final reconstructed image?

2.Could the authors provide a diagram illustrating the tokenizer architecture, especially the normalization and latent regularization mechanisms?

3.Given the claim that reconstruction quality upper-bounds generation quality, does this imply that achieving high-fidelity controllable generation fundamentally requires near-perfect reconstruction?

4.Is this the first work to combine autoregressive modeling with patch-level flow matching? A clearer comparison of competing paradigms (e.g., VQ-based AR vs. diffusion vs. continuous AR) would help contextualize the novelty and advantages of the proposed approach.

5.The ablation focuses on the flow-matching head—how sensitive is performance to the design or capacity of the encoder (i.e., the VAE encoder)?

### Soundness
3

### Presentation
3

### Contribution
3
