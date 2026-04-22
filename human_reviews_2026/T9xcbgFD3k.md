# Guidance Matters: Rethinking the Evaluation Pitfall for Text-to-Image Generation

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 2

## Abstract
Classifier-free guidance (CFG) has helped diffusion models achieve great conditional generation in various fields. Recently, more diffusion guidance methods have emerged with improved generation quality and human preference. However, can these emerging diffusion guidance methods really achieve solid and significant improvements?  In this paper, we rethink recent progress on diffusion guidance. Our work mainly consists of four contributions. First, we reveal a critical evaluation pitfall that common human preference models exhibit a strong bias towards large guidance scales. Simply increasing the CFG scale can easily improve quantitative evaluation scores due to strong semantic alignment, even if image quality is severely damaged (e.g., oversaturation and artifacts). Second, we introduce a novel guidance-aware evaluation (GA-Eval) framework that employs effective guidance scale calibration to enable fair comparison between current guidance methods and CFG by identifying the effects orthogonal and parallel to CFG effects. Third, motivated by the evaluation pitfall, we design Transcendent Diffusion Guidance (TDG) method that can significantly improve human preference scores in the conventional evaluation framework but actually does not work in practice. Fourth, in extensive experiments, we empirically evaluate recent eight diffusion guidance methods within the conventional evaluation framework and the proposed GA-Eval framework. Notably, simply increasing the CFG scales can compete with most studied diffusion guidance methods, while all methods suffer severely from winning rate degradation over standard CFG. Our work would strongly motivate the community to rethink the evaluation paradigm and future directions of this field.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper identifies and analyzes a critical yet overlooked evaluation pitfall in diffusion model research-specifically, the tendency of human preference models to favor high CFG scales due to their implicit bias toward color saturation and semantic alignment.
To address this, the authors propose a Guidance-Aware Evaluation (GA-Eval) framework, which introduces an“effective guidance scale”to disentangle the actual improvement from the simple effect of scale amplification. They further design a synthetic baseline called Transcendent Diffusion Guidance (TDG), demonstrating that it can“fool”conventional evaluation metrics without real quality improvement.
Through experiments across multiple diffusion backbones (SD-XL, SD-2.1, SD-3.5, DiT-XL/2) and datasets (Pick-a-Pic, HPD, DrawBench, GenEval, COCO-30K), they argue that many recent guidance methods achieve their gains mainly by exploiting large-scale biases rather than genuine improvements.

### Strengths
①Timely and meaningful contribution. The paper addresses an important meta-evaluation problem in the rapidly expanding field of text-to-image diffusion models. The argument that large guidance scales bias reward-based evaluation metrics is well-motivated and empirically observable.
②Novel evaluation framework (GA-Eval). Introducing the effective guidance scale and decomposing noise updates into orthogonal and parallel components provides an interesting analytical perspective for isolating true guidance effects.
③Strong experimental coverage. The authors compare eight diffusion guidance methods on four major backbones and five evaluation metrics, offering a relatively broad experimental scope.

### Weaknesses
(A) Theoretical ambiguity in ωₑ definition
The derivation of effective guidance scale (Eq.4-7) assumes linearity between the unconditional and conditional noise terms. However, in practical schedulers (e.g., DDIM, ODE-based), the latent evolution is nonlinear.

(B) Limited validation of GA-Eval fairness
The paper claims GA-Eval enables“fair comparison,”but provides no user study or human evaluation to verify that its results better correlate with actual human perception. Without human validation, GA-Eval could itself introduce a new bias.

(C) TDG lacks principled formulation
TDG randomly masks text tokens to produce a“weakened prompt,”then combines multiple noise terms (Eq. 13).
This method appears heuristic and lacks a theoretical foundation for why such hyperplane mixing would improve quality.
The scaling factor∥ϵ_cond−ϵ_uncond∥/∥ϵ_cond−ϵ_weak∥seems ad hoc and could destabilize training or sampling.

(D) Incomplete quantitative analysis of metric bias
The claim that reward models favor high-saturation images is intuitively correct but not statistically demonstrated. No quantitative correlation between color saturation and metric scores is provided. Figure 3 illustrates trends, but lacks regression or significance testing.

### Questions
1.How is the linearity assumption in the derivation of ωₑ justified, given the nonlinear nature of DDIM/ODE schedulers?
2.Have you analyzed ωₑ stability across timesteps or different sampling schedulers?
3.How is GA-Eval’s “fairness” verified without human evaluation? Any correlation with human judgments?
4.Could GA-Eval itself introduce bias (e.g., penalizing certain visual styles)? What is the theoretical motivation for the TDG noise-mixing rule and its scaling factor?
5.How sensitive is TDG to masking ratio or weakened prompt settings? Can you provide statistical evidence (e.g., correlation or regression) showing that reward models favor high-saturation images?

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
2

### Summary
This paper critically examines the evaluation practices in recent text-to-image (T2I) diffusion guidance methods. The authors identify a pervasive evaluation pitfall: human-preference-based metrics exhibit a strong bias toward images generated with large classifier-free guidance (CFG) scales, even when such images suffer from visual artifacts like oversaturation or loss of fidelity. This bias leads to inflated performance claims for many newly proposed guidance techniques.

To address this, the authors make four key contributions: (1) revealing the pitfall; (2) proposing GA-Eval; (3) Designing TDG: A misleading Transcendent Diffusion GuidanceThe work calls for a paradigm shift in T2I evaluation, urging the community to disentangle true algorithmic innovation from artifacts of biased metrics.

### Strengths
The paper is highly original in both problem framing and methodology. While prior works have proposed new guidance strategies, this is the first to systematically diagnose and quantify a systematic bias in human-preference metrics tied to CFG scale. The technical execution is rigorous. The derivation of the effective guidance scale is mathematically sound and generalizable across sampling algorithms (including those that modify latents rather than noise directly). Experiments are extensive: multiple models, datasets, and metrics are tested, and results are consistently interpreted through the lens of GA-Eval. The inclusion of GenEval for fine-grained semantic alignment and COCO/ImageNet for distributional metrics (FID/IS) further strengthens validity.

### Weaknesses
1. Authors could provide more details on how HPS v2, ImageReward, etc., are biased toward high-saturation/high-alignment images
2. While GA-Eval is a diagnostic tool, the paper offers limited direction on how to design guidance methods that truly improve generation beyond CFG scaling.
3. Some recent text-to-image works might need examination in this paper [1,2]. It is intriguing to investigate whether these recent generative reward model are still biased.

[1] Unified Multimodal Chain-of-Thought Reward Model through Reinforcement Fine-Tuning
[2] T2I-Eval-R1: Reinforcement Learning-Driven Reasoning for Interpretable Text-to-Image Evaluation

### Questions
No question

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
This paper critically analyzes recent progress in diffusion guidance for text-to-image generation, focusing on four key contributions. It identifies an evaluation pitfall where human preference models favor larger guidance scales, affecting image quality. The authors propose a guidance-aware evaluation (GA-Eval) framework for fair comparisons between methods and classifier-free guidance (CFG). They introduce the Transcendent Diffusion Guidance (TDG) method, which boosts preference scores but is ineffective in practice. Experiments show that increasing CFG scales can compete with most methods, though all suffer from decreased winning rates compared to standard CFG. The work urges reconsideration of the evaluation paradigm and future directions in this field.

### Strengths
1. This paper emphasizes a significant evaluation flaw in which standard human preference models show a strong bias toward larger guidance scales. This is crucial for the field as it encourages a reevaluation of the contributions of each classifier-free guidance (CFG) method.
2. The proposed guidance-aware evaluation (GA-Eval) framework is both reasonable and robust. It will assist researchers in accurately assessing diffusion guidance methods.
3. The introduced Transcendent Diffusion Guidance (TDG) method replicates the creation of weak conditions found in other methods during the sampling process. It highlights the effectiveness of GA-Eval and enhances the overall persuasiveness of the paper.

### Weaknesses
1. While GA-Eval is mathematically sound, it requires user studies to demonstrate its alignment with human assessments.
2. This paper identifies an evaluation flaw and proposes a guidance-aware evaluation framework. Another approach to addressing this issue is to enhance the reward model and testing benchmarks. Recent reward models, like HPSv3, may rely on vision-language models, and recent benchmarks, such as OneIG, also utilize VLMs to evaluate generated images. These new evaluation tools could potentially diminish the significance of this paper.

### Questions
None

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper critically reexamines recent diffusion guidance methods beyond Classifier-Free Guidance (CFG) and exposes an evaluation bias: human preference models such as HPSv2 and ImageReward tend to favor images with higher CFG scales, which often leads to overly saturated or artifact-prone generations. To address this, the authors propose GA-Eval, a guidance-aware evaluation framework that calibrates guidance scale effects to ensure fair comparisons across methods. They further introduce Transcendent Diffusion Guidance (TDG), a method that appears to outperform CFG under biased evaluations but in reality does not improve generation quality.
The experiments show that, when accounting for this bias, simply increasing CFG scale can match or outperform most recent guidance methods, urging the community to rethink evaluation protocols for diffusion guidance.

### Strengths
The paper makes an insightful observation that commonly used human preference models, such as HPSv2 and ImageReward, exhibit a strong bias toward images generated with larger CFG scales. This bias frequently leads to higher preference scores for oversaturated or artifact-prone images, revealing a critical weakness in current evaluation protocols for diffusion guidance methods.

### Weaknesses
1. Writing clarity: The overall writing could be improved to enhance readability and make the key ideas easier to follow. Some sections, particularly the methodological descriptions, are difficult to interpret without additional context or intuitive explanations.

2. Equations (5)–(7): The derivations and computation steps for Equations (5) through (7) are not clearly explained. It would be helpful if the authors explicitly detailed how these equations are obtained, what intermediate steps are omitted, and what assumptions or approximations are involved in their formulation.

3. Motivation of Transcendent Diffusion Guidance (TDG): The introduction of the proposed TDG method lacks sufficient motivation and a clear conceptual link to the findings presented earlier. It remains unclear why this method should be effective or how it concretely relates to the identified evaluation bias. Providing a stronger theoretical or intuitive justification for TDG would make the contribution more convincing.

### Questions
1. Could the authors clarify how the components that are orthogonal and parallel to $\Delta \epsilon$ are computed in Equation 5-7?     Specifically, what projection formulation or numerical approach is used to separate these components in practice?

2.  When generating images from CFG with the effective guidance scale, do the authors use the original pre-trained diffusion models for all methods and apply $w^e$ computed individually for each guidance variant?

### Soundness
1

### Presentation
1

### Contribution
2
