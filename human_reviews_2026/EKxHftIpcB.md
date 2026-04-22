# Reasoning to Edit: Hypothetical Instruction-Based Image Editing with Visual Reasoning

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Instruction-based image editing (IIE) has advanced rapidly with the success of diffusion models. However, existing efforts primarily focus on simple and explicit instructions to execute editing operations such as adding, deleting, moving, or swapping objects. They struggle to handle more complex implicit hypothetical instructions that require deeper reasoning to infer plausible visual changes and user intent. Additionally, current datasets provide limited support for training and evaluating reasoning-aware editing capabilities. Architecturally, these methods also lack mechanisms for fine-grained detail extraction that support such reasoning. To address these limitations, we propose Reason50K, a large-scale dataset specifically curated for training and evaluating hypothetical instruction–reasoning image editing, along with ReasonBrain, a novel framework designed to reason over and execute implicit hypothetical instructions across diverse scenarios. Reason50K includes over 50K samples spanning four key reasoning scenarios: Physical, Temporal, Causal, and Story reasoning. ReasonBrain leverages Multimodal Large Language Models (MLLMs) for editing guidance generation and a diffusion model for image synthesis, incorporating a Fine-grained Reasoning Cue Extraction (FRCE) module to capture detailed visual and textual semantics essential for supporting instruction reasoning. To mitigate the semantic loss, we further introduce a Cross-Modal Enhancer (CME) that enables rich interactions between the fine-grained cues and MLLM-derived features. Extensive experiments demonstrate that ReasonBrain consistently outperforms state-of-the-art baselines on reasoning scenarios while exhibiting strong zero-shot generalization to conventional IIE tasks. Our dataset and code will be released publicly.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focuses on hypothetical instruction–reasoning image editing, an interesting task. It proposes a large-scale dataset called Reason5k, with over 5k samples, covering 4 types of reasoning, physical, temporal, causal, and story reasoning. It also proposes a framework called ReasonBrain which includes a multi-modal large language model for instruction reasoning and a diffusion model for image editing. A Fine-grained Reasoning Cue Extraction (FRCE) module and a Cross-Modal Enhancer (CME) are introduced to facilitate reasoning. Extensive experiment results prove the effectiveness of the proposed method.

### Strengths
1. This paper is easy to read, with a clear organization. The figures demonstrate the proposed dataset and framework very vividly. Though it is not the first paper that targets on tackling reasoning instructions, I still think this is an interesting problem. 
2. I think it is good to categorize different types of reasoning, as shown in this paper, they have physical, temporal, causal, and story reasoning (Figure 2). The dataset can make contributions to the community. 
3. The results are extensive and impressive, proving the model's ability to tackle instruction-based image editing with reasoning. I like both the quantitative and qualitative results in the experiment section.

### Weaknesses
1. I like the dataset and I wonder that can we expand it to a more large-scaled one in an easy manner?
2. The novelty of the framework part of this paper is ok, but not very high to me. For example, the similar idea about incorporating a large language model to enhance reasoning ability has been proposed in [1]. I think the authors can make more discussions on the novelty of the model part during rebuttal, especially on how the reasoning part interact with the image editing part. There has been some discussions in the paper now, but I want to see more results, and I think this is what the novelty of the framework lies in.  
[1] LISA: Reasoning Segmentation via Large Language Model

3. Can you can have some discussions or comparison with models such as Qwen-image?
[2] Qwen-image technical report, 2025a. URL https://arxiv.org/abs/2508.02324
Now I think this paper is ok, but if the authors can address the concerns here, it will be a better one.

### Questions
See weaknesses, especially the second point.

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
4

### Summary
This paper introduces Hypothetical Instruction-Reasoning Image Editing, and they claim that this is a  new task for editing images based on complex, implicit instructions. ​Key summary as followings: 

Reason50K Dataset: a large-scale dataset with 51,039 samples across four reasoning categories (Physical, Temporal, Causal, Story) for training and evaluating reasoning-aware image editing models.

ReasonBrain Framework: a model combining MLLM and a diffusion model, with two key modules: (1) FRCE: Extracts fine-grained visual and textual reasoning cues (2) CME: Enhances cross-modal interactions for better semantic representation. ​

Results: their approach outperforms state-of-the-art methods in reasoning-based and conventional image editing tasks, producing more coherent and plausible edits.

### Strengths
they create a large-scale, diverse dataset (51,039 samples) tailored for reasoning-based image editing; and it's nice that they plan to open source the dataset

in the ablation, they show the effectiveness of the component for their model ReasonBrrain

writing is clear, and show the superior results

### Weaknesses
In the main paper, the authors did not provide a detailed description of their dataset creation process, which I consider a significant concern. In my view, the dataset itself is the most crucial component of their work, and thus the paper should clearly explain the rationale and methodology behind each step of its construction. Recently, many papers have proposed new datasets, but only a few of them are genuinely valuable. For instance, it remains unclear why the authors designed their indirect instruction samples in the specific manner they did (e.g., “What would happen if the ice cube were left at room temperature”). Did they conduct any surveys or analyses to confirm that such questions reflect real user behavior, or were these examples generated arbitrarily?

Similarly, the choice of reasoning categories—Physical, Temporal, Causal, and Story Reasoning—raises questions. Why were these particular dimensions selected, and how were the ratios among them determined? Without grounding these design decisions in empirical evidence or user data, the dataset risks diverging from real-world needs and failing to capture the true distribution of reasoning tasks.

Although the ablation study demonstrates that their proposed modules, FRCE and CME, contribute positively to model performance, I am skeptical about their scalability. In practical applications, simplicity and scalability are often more valuable than complex architectural designs, which can be difficult to adopt or reproduce at scale.

### Questions
na

### Soundness
3

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
4

### Summary
This paper proposes a novel task formulation of hypothetical instructions for instruction-based image editing, and attempts to address it by designing new trainable modules within an MLLM-Diffusion model integration framework. The authors synthesized 50K image pairs with corresponding hypothetical instructions for training. Both qualitative and quantitative results presented in the paper demonstrate the superiority of the proposed method over existing works under this setting.

### Strengths
1. The proposed task of hypothetical instructions is novel, and its four-category classification is well justified.
2. The study provides a relatively large-scale training dataset, which represents a valuable contribution to the research community given its substantial volume.
3. The qualitative experiments include comparisons with state-of-the-art closed-source methods, demonstrating superior performance (particularly in handling physical transformations).

### Weaknesses
1. The construction methodology of the Reason50K dataset appears relatively homogeneous. Solely relying on PhyBench to provide target images may limit the diversity of the training data, potentially hindering the method's ability to adequately address all four categories of hypothetical instruction editing tasks.
2. Qualitatively, the method tends to introduce significant alterations in background and object appearance compared to the original image. While this deviation may be less critical in global editing scenarios, it results in visually unsatisfactory outcomes for local edits (e.g., object attribute modifications), as evidenced in Figure 6 (rows 3-4), Figure 8 (rows 1-2), and extensively in the supplementary material.
3. The qualitative ablation results are perplexing. The excessive number of components makes it challenging to determine which element plays a pivotal role, and the explanations provided for the ablation outcomes in the text are not entirely convincing.

### Questions
1. This method only relies on the dataset to achieve better reasoning ability. Is it enough? Since the LLM uses more advanced techniques (COT, GRPO) for better explanation.
2. This method involves many different blocks to learn the detailed tokens, which makes it huge and complex to scale. Any idea?
3.  Does the article report the sizes of different models? How about the backbone (MLLM, diffusion model) influences the final results?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper addresses the limitations of traditional image editing systems in handling complex hypothetical instructions, presenting two significant contributions: first, it constructs a new dataset, Reason50K, which consists of 51,039 source images along with their corresponding complex instructions and target images across four editing scenarios; second, it proposes an innovative framework called ReasonBrain, which integrates multimodal large language models and diffusion models to enhance the reasoning capabilities and quality of image editing. Experimental results demonstrate that ReasonBrain outperforms existing models on reasoning tasks, showcasing its potential in processing complex user instructions.

### Strengths
### originality
- The Reason50K dataset, created within this paper, focuses on complex hypothesis-based instructions, addressing the shortcomings of existing instruction-driven image editing systems.
- The ReasonBrain framework integrates multimodal large language models with diffusion models, developing multiple submodules that aggregate text and image information, thereby enhancing the model's ability to understand and process complex instructions.
### quality
The quantitative experimental results presented in the paper are robust and well-supported by ablation studies, demonstrating the synergistic effects of the different modules.
### clarity
The content of the paper is logically coherent, facilitating an understanding of both the technical approach and the underlying motivations.
### significance
The dataset production process has reduced certain costs and resource consumption, contributing positively to causal inference in image editing.

### Weaknesses
1. Qualitative experiments appear to excessively alter the scenes themselves compared to baseline methods, which may diminish the evaluation of the method's precision in modification capabilities. This issue may arise from the hallucination during the dataset creation process. Compared to existing methods that extract data pairs from video data, it is relatively challenging to maintain consistency between the edits pre- and post-modification.

2. While the current method can effectively adhere to predefined categories of instructions, it seems to lose the ability to handle simple instructions and non-predefined instruction categories.

### Questions
1. Is the last column of the first row in Figure 7 the Full method? The color palette appears noticeably inconsistent before and after, and the results do not seem to align better with cognitive expectations than those in the third column from the end.

2. In Figure 6, the modifications to the background in the second and fourth rows are excessively pronounced, which may impact the presentation of the results and their practical application.

3. The paper mentions the Editworld dataset, but what is the rationale for not including it as a comparative baseline? It seems that Editworld articulates similar motivations and presents comparable experimental results.

4. Additionally, how does this approach compare to more general open-source models, such as Flux Kontext[2] and Bagel[1]?

[1] Emerging Properties in Unified Multimodal Pretraining
[2] FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space

### Soundness
3

### Presentation
2

### Contribution
2
