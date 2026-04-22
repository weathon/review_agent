# From Denoising to Refining: A Corrective Framework for Vision-Language Diffusion Model

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 2, 6

## Abstract
Discrete diffusion models have emerged as a promising direction for vision-language tasks, offering bidirectional context modeling and theoretical parallelization. However, their practical application is severely hindered by a train-inference discrepancy, which leads to catastrophic error cascades: initial token errors during parallel decoding pollute the generation context, triggering a chain reaction of compounding errors and leading to syntactic errors and semantic hallucinations. To address this fundamental challenge, we reframe the generation process from passive denoising to active refining. We introduce **ReDiff**, a **re**fining-enhanced **diff**usion framework that teaches the model to identify and correct its own errors. Our approach features a two-stage training process: first, we instill a foundational revision capability by training the model to revise synthetic errors; second, we implement a novel online self-correction loop where the model is explicitly trained to revise its own flawed drafts by learning from an expert's corrections. This mistake-driven learning endows the model with the crucial ability to revisit and refine its already generated output, effectively breaking the error cascade. Extensive experiments demonstrate that ReDiff significantly improves the coherence and factual accuracy of generated content, enabling stable and efficient parallel generation far superior to traditional denoising methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper addresses the discrepancy between training and inference in discrete, mask prediction-based vision-language diffusion models (VLMs). Specifically, in parallel decoding, early token errors contaminate the bidirectional context and trigger error cascades (repetitions, grammatical errors, and visual hallucinations). The authors propose ReDiff, a refined framework that teaches a diffusion VLM to identify and correct its own errors rather than just “filling masks.” During inference, ReDiff jointly reveals new tokens at each step and refines previously unrevealed tokens, stabilizing parallel generation in a few steps. Experiments with detailed captions show significant improvements over previous diffusion VLMs (e.g., LLaDA-V, LaViDa, MMaDA, FUDOKI) and more elegant quality degradation at increasing speeds, matching or exceeding certain AR baselines on specific metrics.

### Strengths
-  The authors redesign the generation process using a discrete diffusion model from passive noise reduction to active refinement, explicitly targeting the discrepancy between training and inference that leads to error cascades in parallel decoding.The idea of “refining already unmasked tokens while simultaneously unmasking new ones” is an intuitive and clear conceptual change. 
- This article clearly explains why discrete diffusion has problems with parallel decoding (context distortion due to early errors) and develops training/inference methods to counteract precisely this error mode. 
- On various benchmarks for detailed captioning, ReDiff provides improved performance compared to existing diffusion VLMs and deteriorates only slightly with increasing speed (fewer steps), which is typically the range in which error cascades occur. 
- The paper is well-written and easy to understand.

### Weaknesses
- The online self-correction learning highly depends on an external expert (o4-mini). The authors generate approximately 10k draft-refined caption pairs per round, with “a single round” being considered the most effective. Thus, the trained ReDiff cannot be free of the prior knowledge of the external expert, underscoring its marginal performance. In addition, the data/computing costs, input template, and quality control for expert feedback are not quantified. 
- All evaluations refer to detailed image captions. Claims that parallel discrete diffusion improves the dynamics of the “error cascade” would be more convincing with tasks such as short VQA-style answers or informed descriptions. The current main results and tables refer exclusively to image captions and are available within a single diffusion VLM family. 
- ReDiff replaces previously unmasked tokens at each step and simultaneously masks $n$ new tokens, which is the key method. The paper contains tables for different speeds (1/2/4/8 tokens/step). However, there is neither a breakdown by error type (e.g., repetitions vs. factual errors) at increasing speeds, nor a latency profile that isolates the cost of completely re-evaluating the sequence at each step. 
- The authors show that the first online round yields a big boost and subsequent rounds plateau. It is unclear whether this plateau is due to data saturation (too few new errors), expert limits, or catastrophic forgetting of the Stage-I capabilities.

### Questions
To further substantiate the novelty of the algorithm, I would like to suggest that the authors conduct more in-depth ablations on the following points: 
- Synthetic error curriculum and its sensitivity
- Data/computational costs of expert-edited drafts for online self-correction
- Generalization beyond subtitling (other VL tasks/backbones).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduced a type of diffusion model that can detect and correct generation errors. The model training consists of two stages: 1) a model is trained to correct syntactic mistakes; 2) a self-correction loop that teaches the model how to correct its own errors by learning from an expert. The experiments showed their proposed method can ineed address incoherence and factual errors in the generated content of models.

### Strengths
The problem is clearly defined, and the writing is good. 

Secondly, it seems intuitive why the method can address the problem to some extent (e.g., incoherence). For factual errors, the problem might not be fully addressed. It is largely an inherent weakness of data-driven models.

### Weaknesses
The whole method seems to be a combination of knowledge distillation and self-supervised learning, making it less novel to me; The structure of the paper can be improved. I believe it is more appropriate to place the preliminary section outside the method section. However, by doing this, the method will look much less complicated for a top conference. The authors might need to consider how to dig deeper into the problem; Some of the claims about experiments also look quite strong to me, for example, factual error correction. I believe this problem cannot be fully eliminated for data-driven methods...

### Questions
1, Is your method applicable to other architectures beyond diffusion models? It would be good to extend experiments regarding this, making it as general as possible. Maybe in this way, it would be okay that the method was not that novel.

2, There was also always a concern: how much performance gain can we really get for using diffusion models over Transformer-based language models? It would also be great to make a comparison regarding this point, showing the value of similar topics like this paper.

### Soundness
2

### Presentation
3

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
This paper introduces ReDiff, a refining-enhanced discrete diffusion framework that reframes generation in vision-language diffusion models from passive denoising to active refining. This paper first identifies that current discrete diffusion models suffer from error cascades due to the train-inference discrepancy (models are trained on clean data but must generate from their own noisy intermediate outputs). The, it proposes ReDiff as a two-stage training framework that equips the model to iteratively refine its own outputs during inference, significantly improving fluency, factuality, and stability in parallel generation regimes. Extensive experiments on several benchmarks show large improvements over strong diffusion-based baselines and competitive results with top autoregressive models.

### Strengths
1. The paper is well-structured and well-written, with effective visuals to illustrate both conceptual and qualitative outcomes.
2. This paper proposes a novel paradigm that moves from denoising to refining, which reconceptualizes how discrete diffusion models perform generation. The model explicitly learns from its own flawed drafts, rather than synthetic noise alone, which is an elegant and practical innovation.
3. The empirical performance is good over the baselines. The ablation studies also show effectiveness of the design as from the Tables 4-6.
4. Its improvements in few-step inference directly impact the scalability and deployment efficiency of vision-language diffusion systems.

### Weaknesses
1. The self-correction loop relies on an external “expert model” (o4-mini) for generating corrected drafts. While practical, this introduces external bias and resource dependence. The paper could discuss how results vary with different or weaker expert models.
2. The evaluation focuses only on detailed image captioning. Although this is a strong proxy task, extending to other vision-language generation tasks (e.g., dialog, instruction following) would test generalization.
3. The two-stage training (especially online refinement) adds extra rounds of model inference and expert evaluation. Quantitative analysis of training cost versus performance gain would improve transparency.
4. Since the model learns corrections from a specific expert model’s phrasing, stylistic over-alignment could occur. Including human evaluation or stylistic diversity checks would strengthen the claims of general improvement.

### Questions
1. How does ReDiff perform on tasks requiring longer or more structured reasoning (beyond captioning)?
2. Did the authors experiment with varying the ratio of Stage I vs. Stage II data? Could adaptive mixing improve stability?
3. Could ReDiff be extended to autoregressive or flow-matching architectures to improve consistency during token-parallel decoding?

### Soundness
3

### Presentation
4

### Contribution
3
