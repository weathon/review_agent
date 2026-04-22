# Seeing Beyond Redundancy: Task Complexity's Role in Vision Token Specialization in VLLMs

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Vision capabilities in vision large language models (VLLMs) have consistently lagged behind their linguistic capabilities. In particular, numerous benchmark studies have demonstrated that VLLMs struggle when fine-grained visual information or spatial reasoning is required. However, we do not yet understand exactly why VLLMs struggle so much with these tasks relative to others. Some works have focused on visual redundancy as an explanation, where high-level visual information is uniformly spread across numerous tokens and specific, fine-grained visual information is discarded. In this work, we investigate this premise in greater detail, seeking to better understand exactly how various types of visual information are processed by the model and what types of visual information are discarded. To do so, we introduce a simple synthetic benchmark dataset that is specifically constructed to probe various visual features, along with a set of metrics for measuring visual redundancy, allowing us to better understand the nuances of their relationship. Then, we explore fine-tuning VLLMs on a number of complex visual tasks to better understand how redundancy and compression change based upon the complexity of the data that a model is trained on. We find that there is a connection between task complexity and visual compression, implying that having a sufficient ratio of high complexity visual data is crucial for altering the way that VLLMs distribute their visual representation and consequently improving their performance on complex visual tasks. We hope that this work will provide valuable insights for training the next generation of VLLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the visual redundancy problem of large vision language models through controlled experiments. Specifically, it constructs a synthetic benchmark with a set of metrics to measure and understand the problem. Fine-tuning is also employed to investigate the impact of data. Results reveal the connection between task complexity and visual compression.

### Strengths
1. The paper presents a comprehensive suite of metrics accompanied by extensive empirical analyses to characterize visual information compression across layers, providing valuable insights

2. Detailed ablation studies are provided

3. Fine-tuning is included for a deeper understanding

### Weaknesses
1. The paper lacks sufficient justification for the choice of evaluation metrics. A more thorough discussion of the theoretical foundations and practical motivations underlying these metrics would strengthen the methodological framework.

2. The experimental validation is limited to two models (Molmo-7B and Llama 3.2). The generalizability of the findings could be substantially improved by including models from varying parameter scales.

3. The zero-shot analysis relies exclusively on synthetic datasets with simplified characteristics. The extent to which these findings translate to real-world scenarios remains insufficiently addressed. 

4. The practical impact of this work would be considerably enhanced if the derived insights were used to fine-tune vision-language models with improved compression efficiency and reduced redundancy.

### Questions
1. The terminology "Large Vision-Language Model (LVLM)" appears to be more prevalent in the literature than "Vision Large Language Model (VLLM)." 

2. The citation format requires attention to stylistic conventions. Several in-text citations currently employ \cite without parentheses, whereas \citep would be more appropriate. Instances include lines 104-105 where the citations serve as supplementary support rather than grammatical subjects.

3. Typo in line 311 "Figure 2 provides further insights into have visual compression is correlated"

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
This paper investigates visual redundancy and compression phenomena in VLMs, with a focus on the relationship between task complexity and vision token specialization. The authors construct a synthetic dataset to systematically vary visual complexity and propose novel compression/ redundancy measurement metrics, including norm-based and rank-based measures, SVD alignment analyses, linear probe evaluations, and token ablation experiments. They conduct zero-shot and fine-tuning experiments on Molmo and LLaMA-v3.2-Vision, analyzing how task type (referring expression vs spatial reasoning) and dataset complexity influence internal representations and compression behavior. The paper concludes with proposed compression strategies and implications for fine-tuning.

### Strengths
1. Systematic metric design – The work proposes a comprehensive suite of metrics (both norm- and rank-based, plus SVD alignment) to analyze compression and redundancy in VLLMs’ hidden states, offering more granular insight than prior attention-based analyses.

2. Detailed layer-wise analysis – The visualization across layers for different metrics provides an interpretable picture of how visual information is redistributed within models.

3. Task complexity perspective – The link between downstream task complexity and optimal compression levels is well articulated and supported by multiple evaluation angles.

### Weaknesses
1. Synthetic dataset reliance – The main analyses are conducted on a fully synthetic dataset designed by the authors, with limited validation of whether the findings generalize to real-world tasks. The COCO and GQA datasets used in the synthetic data experiments were also only analyzed using the metrics proposed in the paper, rather than through more intuitive computations of prediction accuracy.

2. Evaluation metric coverage vs accuracy gains – The paper heavily focuses on reporting compression/ redundancy metrics but lacks direct evidence that these methods can improve benchmark accuracy when applied in compression policies. A simple empirical demonstration of accuracy improvement would make the contribution more tangible.

3. Architectural limitation in scope – Both Molmo and LLaMA-3.2-Vision adopt CLIP-style fixed-resolution vision encoders with image patching (slice into fixed-size tokens). Newer architectures (e.g., Qwen-VL, GLM-VL) use native resolution and dynamic tokenization according to input resolution, potentially altering redundancy/compression behavior. The generality of the conclusions under these architectures is not assessed.

4. Overlap with prior work’s findings – Some behavioral observations[1] have been highlighted in several earlier VLLM diagnostic studies. The novelty claim would benefit from a clearer positioning relative to these works.

[1] Label Words are Anchors: An Information Flow Perspective for Understanding In-Context Learning

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The aim of this work is to propose an analysis on why MLLM still struggle with visually fine grained tasks even if they excel at tasks involving the global image semantic. The authors explore this via statistical analysis on the visual and text tokens in intermediate layers of the LLM as well as different probing mechanisms like training FC classifiers on top of the tokens and randomly dropping them to see impact on performance. Most experiments are run on Molmo on a synthetic dataset created ad hoc from the authors. Few fine tuning experiments use real data. The findings are that there is a high redundancy within the visual tokens on a LLM and that they tend to be optimized for general vision tasks and not fine grained ones. Moreover, If the model are fine tuning on challenging localization tasks most of the representation changes are in the text part of the model and not in the multimodal one.

### Strengths
+ Authors propose a very detailed statistical analysis to uncover redundancy in the tokens representations within a LLM. A lot of the technique proposed could likelly be re-used for other works interested in uncovering more about the hidden representation of these models.

+ Surprising finding that in the experimental settings of the work fine tuning a model on visual data seems to  overwhelmingly alter text representations while leaving vision representations largely unaltered.

+ Carefully curated creation of synthetic data to support the experimental analysis in the paper.

### Weaknesses
a. **Limited experimental analysis**: Most experiments of the paper are performed using only the Molmo MLLM, would have been interesting to see the analysis expanded to other models trained on different data mixtures and with different architectures. The paper does consider llama, but only for the experiments on probes and visual ablations. The analysis of other decoder based MLLM besides Olmo would have made this submission more strong.

B. **Nice analysis, but limited applicability**: While the work does provide some nice insights the findings are not very actionable and mostly provide experimental evidence of behavior that is quite known to practitioners. Namely: that the amount of tokens that can be dropped from a LLM input is a function of how “hard” a task is and that an effective post-training strategy can involve only the LLM without touching the visual encoder in the model. Also the observation that harder visual task will bring more changes in the model can likelly be linked to the perplexity for the model on those tasks while training. If on average the tasks are harder for the model they could cause higher gradients which in turn results in a bigger shift in the text components of the visual model. 

C. **Small scale controlled experiments**: The paper does a nice job at creating a setting where the claim can be tested and isolated, but it does not verify whether the claim holds on bigger and more realistic settings. For example the token analysis in Sec. 4.1 is all performed only on synthetic images, while the fine tuning experiments in Sec. 4.2 consider fine tuning only on (few) visual tasks, while in practice most MLLM would be fine tuned on a way bigger mixture of visual and textual tasks. Exploring what happen in the more realistic settings would have made the submission stronger.

### Questions
1. What’s the text prompt for the synthetic dataset you generated?

Few typos
L311: “into have”
L481: “are more require”
Across the paper you read few times “muiltimodal” instead of “multimodal”

### Soundness
3

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
2

### Summary
This paper conducts an in-depth study on the problem of visual information redundancy in vision-language large models (VLLMs), and points out that visual information redundancy is one of the important reasons for the poor performance of the model in complex visual tasks, such as fine-grained object recognition and spatial reasoning. The author constructed a synthetic dataset, quantified the complexity of tasks, and found that complex tasks (such as object counting) require more specialized visual tokens, have lower redundancy, and are sensitive to compression. However, simple tasks (such as color recognition) are not sensitive to redundancy and can even tolerate up to 99% token discard. Through fine-tuning experiments on the model, the author found that fine-tuning mainly altered the text representation of the model, while the visual representation changed relatively little. Moreover, different types of tasks (spatial reasoning vs. object localization) affected the internal representation of the model in different ways. Based on these findings, the authors proposed compression strategies and training suggestions for VLLMs, namely, appropriately compressing information in the early layers, carefully compressing in the middle layers, reducing the compression ratio for complex tasks, significantly compressing for simple tasks, and paying more attention to the updates of text and multimodal projection layers during fine-tuning.

### Strengths
- This paper proposes and comprehensively applies multiple quantitative indicators (such as Gini coefficient, stable rank, participation rate, etc.) to systematically analyze visual information redundancy from the two levels of token norm and matrix rank, surpassing previous studies that only focused on attention distribution and providing a more comprehensive tool for understanding the internal visual information processing of VLLMs.

- The experiments precisely controls variables through the construction of synthetic datasets, the negative correlation between task complexity (such as the number of objects and the difficulty of spatial reasoning) and the degree of visual information redundancy was clearly verified for the first time, providing direct evidence for explaining the performance bottleneck of VLLMs in complex visual tasks

### Weaknesses
- Some findings, such as "there is a connection between task complexity and visual compression", are similar with the conclusions given in previous works like PDrop[1]. 

- Fine-tuning experiments are only based on simplified subsets of COCO and GQA (such as objects with only "left-right" relationships), and more complex spatial relationships (such as spatial reasoning in ERQA) have not been tested, which may underestimate the model's redundant performance in real complex tasks.

- The experiment mainly uses syntheti5c data of simple geometric shapes (fixed color/shape/size), lacking complex factors such as texture, occlusion, and lighting changes in real images, which may lead to insufficient generalization of the conclusion in real scenes.

[1] Xing, et al. Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction. CVPR, 202

### Questions
- How to specifically configure the vision token compression on the task with different complexities? And, will the configuration setting be quite different among different types of models?

### Soundness
3

### Presentation
3

### Contribution
3
