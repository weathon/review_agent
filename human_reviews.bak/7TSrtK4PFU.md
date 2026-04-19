# Text-Guided Visual Prompt Tuning for Vision-Language Models

- Decision: Reject
- Scores: 5, 5, 5, 5

## Abstract
Prompt tuning has become a crucial technique for adapting pre-trained vision-language models (VLMs) to various downstream tasks. Recent advancements introduce multi-modal learnable prompts to enhance the creation of task-specific classifiers. Despite their utility, these methods commonly encounter challenges in generalizing to unseen classes, as their symmetrically designed visual prompt struggles to capture task-relevant textual knowledge and lacks the flexibility in adjusting to novel test class distributions. To tackle these obstacles, we propose a novel Text-Guided Visual Prompt Tuning (TGVP) method, which uniquely leverages the robust generalizability of textual knowledge to guide the generation of visual prompt. Our method introduces a simple yet effective Text-Knowledge Guidance Module that dynamically incorporates visual prompt with task-relevant textual knowledge through cross-attention mechanism. The generated text-guided visual prompt endows the visual encoder with semantic awareness and thus enhances both generalization and discriminability of VLMs across various scenarios. Comprehensive experiments demonstrate that TGVP significantly outperforms existing methods in base-to-novel generalization, cross-dataset transfer, and domain generalization tasks, offering a substantial improvement in VLM adaptation.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This article first analyzes the impact of unimodal prompts on the base-to-novel classification. It concludes that using text prompt representations to guide image prompt representations is beneficial. Therefore, the author proposes the TGVP method, which utilizes parameter-free cross-attention to guide the optimization of image prompt representations. The author demonstrates the effectiveness of this method through extensive experiments.

### Strengths
1. The presentation in this paper is quite clear, allowing for a relatively clear understanding of the author's methodological process.

2. The experimental content in this paper is comprehensive, effectively demonstrating the efficacy of the proposed method.

### Weaknesses
1. The author introduces too many variables and formulas in the method introduction section, which may cause some difficulties in understanding the author's method.

2. The author's motivation is derived from the analysis of unimodal prompts, leading to the conclusion of using text to guide visual prompts. However, the visual prompts obtained through the author's method are also unimodal. Therefore, I believe it would be beneficial to add the performance of the new visual prompts in Figure 1 to demonstrate the effectiveness of the author's method.

3. Guiding the representation generation of visual prompts through text has already been applied in MaPLE. Additionally, the author's cross-attention calculation appears to be similar to the parameter-free attention in CALIP[1].

[1] CALIP: Zero-Shot Enhancement of CLIP with Parameter-free Attention

### Questions
1. Referring to weakness 2, I believe it would be beneficial to include the optimized visual features based on the author's method to demonstrate its effectiveness.

2. In the description of the method, it would be helpful to reduce the introduction of new variables and include pseudocode to aid in understanding the author's approach.

3. Personally, I think that given the lack of novelty in the method presented, the paper should reduce the length devoted to describing the method. Instead, it could analyze what causes the differences in generalization capabilities between visual and text unimodal prompts, or explore whether encoder-only VLMs can be extended to decoder-only VLMs. Such analyses would make the work more impactful.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper  proposes a method, named Text-Guided Visual Prompt Tuning (TGVP), to uniquely leverage the robust generalizability of textual knowledge to guide the generation of visual prompt.

### Strengths
-This paper emphasizes that high-level textual semantics are key to facilitating the learning of generalizable visual prompts

-Experiments show that the proposed method performs better than other existing approaches in base-to-novel generalization, cross-dataset transfer, and domain generalization tasks.

### Weaknesses
Cross-modallity attention has been done in CALIP, and the projection from text prompt to visual prompt in Figure 2 has also been done in MaPLe, which looks a bit like incremental;

[1] CALIP: Zero Shot Enhancement of CLIP with Parameter free Attention (AAAI 2023)

### Questions
-Regarding cross-modalality attention on Idea, CALIP: Zero Shot Enhancement of CLIP with Parameter free Attention (AAAI 2023) has already been done (very similar); This paper lacks this reference;

-This paper uses Figure 1 to express his motivation, aiming to demonstrate that the generalization ability of visual prompts is not as good as that of text prompts; However, only two small datasets, Eurosat (remote sensing) and DTD (texture), were displayed. Both datasets are small and very fine-grained. It is interensting that do similar experiments on large dataset, such as ImageNet, or the mean of all 11 datasets;

-In some experimental implementation details, such as line 273, the setting of the number of layers for the visual prompt and the comparative experiment on the number of layers are missing; 260 line EMA method, lacking setting of λ hyperparameter;

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper proposed a novel prompt-tuning method for VLMs. At its core, the proposed method uses visual prompt tokens and the CLS token to attend to the text prompts and get "text guidance" which is subsequently added (through moving average) to the visual prompts to get "text-guided visual prompts".

### Strengths
The paper is well-organized and is easy to read. Experiments includes 4 set of experiments on 11 dataset which is good.

### Weaknesses
The paper seems very similar to MaPLe (Khattak 2023a) and PromptSRC (kHATTAK 2023B) in that they all jointly learn visual and/or text prompts. The related work section briefly mentions them but does not really discuss them adequately. 

In the base-to-novel generalization experiment (table 1) the average improvement under the HM column (harmonic mean of base and novel classes) is 1.27% over 11 dataset. However, a closer look reveals that this improvement is mostly due the EuroSAT dataset which shows 8% improvement . Excluding that dataset, the average improvement over the remaining 10 dataset is only 0.35% which is a very marginal improvement .   

In Table 4 about the domain generalization, the TCP method is missing. Considering that TCP seems to be among top performing methods in other experiments (Table 1-3), including the results of TCP in table 4 will be helpful.

### Questions
See my comments above. 
Also, the proposed method shows a strong performance on the EuroSAT dataset across various experiments. Performance on the other 10 datsets are relatively much lower. A discussion on what is special about the EuroSAT dataset would be insightful.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper introduces Text-Guided Visual Prompt Tuning (TGVP) to enhance the generalization of vision-language models (VLMs) for diverse downstream tasks. Traditional methods struggle to incorporate task-relevant textual knowledge into visual prompts, limiting their adaptability to novel classes. TGVP addresses this by using a Text-Knowledge Guidance Module with a cross-attention mechanism, allowing visual prompts to better capture semantic context. Experiments show TGVP significantly improves VLM performance in generalization, cross-dataset transfer, and domain adaptation tasks.

### Strengths
This paper addresses a compelling challenge in vision-language model adaptation: improving generalization to unseen tasks and classes. Existing prompt tuning methods often overlook the benefits of integrating textual knowledge into visual prompts.  By leveraging textual guidance, TGVP demonstrates superior generalization performance, particularly in base-to-novel class adaptation, cross-dataset transfer, and domain generalization, addressing common limitations in traditional prompt tuning methods.

### Weaknesses
W1:  The performance improvement demonstrated by the proposed method is relatively modest, limiting the practical impact and significance of the contribution. Further analysis or comparison with a broader range of baselines could help clarify the advantages and effectiveness of the approach.

W2: The paper lacks coverage of some important related works, particularly in areas that could provide a deeper contextual foundation for the proposed method. Including a more comprehensive review of relevant studies, especially recent advances in prompt tuning and ensemble learning, would enhance the paper's contribution and situate it more clearly within the broader research landscape.

### Questions
Q1: What is the function of the "project" component in your model? How would altering its structure impact performance?

Q2: Could the authors consider adding the ensemble baselines, such as the WiSE-FT method, to provide a more comprehensive comparison?
WiSE-FT: Robust fine-tuning of zero-shot models

Q3: Could the authors clarify the definition of "P" in Equation (7)? Additionally, could you explain the process for obtaining T^{topk}_{j} and I^{topk}_{j}?

### Soundness
2

### Presentation
3

### Contribution
2
