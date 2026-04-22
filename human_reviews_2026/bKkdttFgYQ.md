# SEEING THROUGH LANGUAGE: HOW TEXT REVEALS OBJECT AND STATE BIAS IN VLMS

- Avg Score: 3.00
- Decision: Reject
- Scores: 0, 4, 4, 4

## Abstract
Vision-Language models (VLMs) have demonstrated strong performance across a variety of multimodal benchmarks though not without internal biases. Little is known about how VLMs balance sensitivity to object identity versus object state. In this work, we systematically investigate object-state bias in VLMs by evaluating a broad set of models spanning diverse architectures and sizes. To enable controlled analysis, we introduce the Benchmark for Biases in Objects and States (BBiOS) dataset containing objects in both their original and transformed states. Across a variety of experiments, we examine model performance on recognizing objects, states, and their interactions. Our results reveal a consistent object bias, where models reliably recognize object categories but struggle to accurately capture states. Furthermore, attempts to steer models toward greater state sensitivity through prompting or injecting oracle information yield only marginal improvements. These findings highlight a fundamental limitation in current VLMs, suggesting that different training strategies or architectural innovations are required to reduce object-state bias in multimodal reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
2

### Summary
The paper presents an empirical study of VLM object and state recognition capabilities. To this extend, the authors introduce a novel benchmark (BBiOS) for the measurement of the introduced "object-state bias".  The benchmark dataset contains images of different vegetables and fruits at various states of processing (like raw, peeled, sliced, fried ...). The benchmark task is then to predict the object categories and their states. The experimental on this dataset evaluation shows that current (open) VLM models are better in predicting object categories than their state.

### Strengths
The paper is well written and easy to follow. The experiments use a large number of open VLMs

### Weaknesses
The reviewer strongly disagrees with the way the authors use the term "bias". While the investigation of all sorts of biases is an important ongoing topic, its inflationary usage as buzzword does not provide real insights. The conducted experiments show that VLMs are better in predicting categories than states (for a very specific dataset). It totally unclear how this represents a systematic model bias. While the formal definition of biases might be a bit weak, it always includes a clear relationship (dependency) between variables. The famous shape-texture bias for example, shows a trade-off between texture and shape information used by CNNs. A racial bias leads to model preferences towards people of a certain skin color (at expense of others). In the presented case, it remains unclear how the ability of models to predict object categories would limit the state predictions.

Taking away the bias claim, the paper simply shows that VLMs are better in predicting fruit types than fruit states - this is not especially surprising since the state prediction task is much more complicated (given the variance of states in images).

### Questions
* What is your definition of BIAS ?
* the paper only shows results for open VLM models. How do state of the art commercial models like GPT-5 behave?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This submission studies the object-state bias in vision-language models (both dual-encoder and autoregressive). The results are based on the collected and annotated BBiOS dataset, containing 16 kitchen objects with 4-6 states each (including one raw state) from a total of 8 states. A study on 23 different models then evaluates the object-state bias in multi-task and forced-choice settings and reveals that 1) models struggle at state recognition as compared to objects; 2) are biased toward objects; and 3) are mostly steerable in the state direction and sometimes not at all.

### Strengths
- Introduces new “Bias in Object State (BBiOS)” dataset based on kitchen ingredients (derived from VidOSC) consisting of 16 objects and 4-6 states each (including one raw state) from a total of 8 states to evaluate object and state bias claiming a balanced coverage across object-state pairs
- Bias investigation is systematically conducted in multi-task and forced choice settings
- Evaluation of 23 diverse VLMs
- Prompt-based steering is evaluated and shown to less effective than shown in previous studies on other cues

### Weaknesses
- Previous bias studies have been based around the methodology that all cues are easily individually recognizable. Yet, models seem to struggle to detect state (as also shown by Newman et al. (2024))–this may introduce a confounder, i.e., models may not be biased–they simply do not recognize the correct state (and then potentially hallucinate the state or fall back to object-only responses). 
- Prompts for “steering” are not tuned; in general no prompt seems to be tuned which may bias the study. For example, the terms "state" and "object" may be simple poorly aligned in VL spaces. It would be good to ablate a few other prompts to show that the findings are not limited by the choice of prompt.
- The prompt template for LLMs is not properly described (Sec. 3.3.2 is lacking details). A few examples would be helpful.
- Also, it is not described how LLM responses are sampled. Using non-greedy sampling may have introduced significant error and mandates a statistical analysis of error between sampled responses in parallel.
- If LLMs are poor at state recognition (Newman, 2024) then they should also introduce an error in the data curation pipeline (L192ff)
- Non-uniform distribution of object/state and  the paper is missing a heatmap two show the joint frequency of object-state pairs (or alternatively numbers/ratios in Table 1)
- The models are poorly documented. Please precisely name the checkpoint (e.g, there are multiple SigLIP variants, Qwen – 10B is probably Qwen-Image-10B etc.), weight source, and inference resolution (transformations) 

Minor:
- Figures are not vector graphics and blurry
- LLaVa -> LLaVA (L99)
- Unnecessary parentheses around citep in intro and elsewhere
- Inconsistent reference capitalization (Fig/fig, e.g, L213; sometime abbreviated or not)

### Questions
- Does in-context learning or chain-of-thought change any of the results?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates object–state bias in Vision–Language Models (VLMs), where models tend to recognize object identity (e.g., "apple") more accurately than object state (e.g., "sliced", "peeled"). The authors introduce the Benchmark for Biases in Objects and States (BBiOS), a curated image dataset of 16 kitchen objects across eight states, collected semi-automatically from VidOSC using LLaMA-3.1 and CLIP for frame selection. The study designs two experimental paradigms—Multi-Task (object/state prediction with or without conditioning) and Forced-Choice (object vs. state classification)—and evaluates 23 diverse VLMs, including CLIP, BLIP, Qwen2-VL, InternVL, and LLaVA families. Results show consistent bias: models achieve high accuracy for object identity but struggle with state recognition, even under oracle conditioning or steering prompts. The paper concludes that object bias is structural and arises from training data and representation design rather than prompting limitations .

### Strengths
1. A well-constructed dataset focusing on controlled object–state variations in realistic visual contexts.
2. Comprehensive evaluation across 23 models spanning diverse architectures and parameter scales.
3. Consistent empirical evidence showing strong object bias and limited steerability through text prompts or oracle information.
4. Thoughtful discussion linking dataset composition, linguistic priors, and multimodal representations.

### Weaknesses
1. The contribution is primarily diagnostic. The paper identifies object bias but does not analyze its causal origin in vision encoders, textual priors, or training objectives.
2. The dataset size is small for large-scale model evaluation, making the generalization of conclusions uncertain.
3. The evaluation lacks human baselines or psychometric reliability checks to contextualize bias severity.
4. The analysis does not disentangle dataset bias (frequency imbalance) from representation bias (model internal weighting).
5. The paper reuses known conceptual framing (object vs. state) without deeper theoretical grounding—similar trends have been observed in ChangeIt-Frames and recent multimodal perception studies [1–3].

[1] Newman et al., “Do Pre-Trained Vision-Language Models Encode Object States?” arXiv 2024. \
[2] Y. Fu et al., “BLINK: Multimodal Large Language Models Can See but Not Perceive,” ECCV 2024. \
[3] Kawaharazuka et al., “Continuous Object State Recognition for Cooking Robots,” IEEE RAL 2024.

### Questions
1. How does the bias magnitude correlate with visual encoder scale or architecture type (ViT-L vs. Swin)?
2. Do object–state biases persist if the textual prompt explicitly disambiguates the state (e.g., “a peeled apple on a plate”)?
3. Could linear probing or concept subspace analysis (e.g., CAV [4]) help localize the state-sensitive directions in the representation?
4. How balanced is the BBiOS dataset in terms of background and lighting? Could visual confounds explain part of the state gap?
5. Is there evidence that models trained on state-rich datasets (e.g., Something-Something V2) show reduced object dominance?

[4] B. Kim et al., “Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV),” ICML 2018.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper aims to analyze the sensitivity of vision-language models (VLMs) to object identity and state. To achieve this, the paper introduces a dataset containing objects in both their original and transformed states. Across a variety of experiments, the paper reveals that VLMs can reliably recognize object categories but struggle to accurately infer states. Moreover, steering models toward greater sensitivity via prompting or injecting oracle information yields marginal improvements. These findings highlight a fundamental limitation in current VLMs. Different training strategies or architectures may be needed to reduce the object-state bias.

### Strengths
- The paper introduces a new dataset for analyzing object biases in VLMs.
- Experiments cover a wide range of VLMs with different architectures and different scales, revealing that the object bias is prevalent and exists across almost all models.
- The paper explores mitigation strategies via prompting and reveals that prompt engineering does not fundamentally solve the object bias in VLMs.

### Weaknesses
- The proposed dataset is relatively small and restricted to a narrow domain, primarily focusing on kitchen-related objects and activities. This setup limits the dataset’s representativeness and generalizability to broader real-world scenarios.

- The paper does not provide substantial new insights for explaining or addressing bias in VLMs. The discussion attributing object bias to the models’ latent state representations and their training data remains rather general and high-level, without offering concrete empirical evidence.

### Questions
Line 193: How do you use a large language model, which does not support images, to do image analysis and select frames?

### Soundness
2

### Presentation
3

### Contribution
2
