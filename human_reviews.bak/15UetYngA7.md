# FuseChat: Knowledge Fusion of Chat Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 5, 8

## Abstract
While training large language models (LLMs) from scratch can indeed lead to models with distinct capabilities and strengths, it incurs substantial costs and may lead to redundancy in competencies. Knowledge fusion aims to integrate existing LLMs of diverse architectures and capabilities into a more potent LLM through lightweight continual training, thereby reducing the need for costly LLM development. In this work, we propose a new framework for the knowledge fusion of chat LLMs through two main stages, resulting in FuseChat. Firstly, we conduct pairwise knowledge fusion on source chat LLMs of varying structures and scales to create multiple target LLMs with identical structure and size via lightweight fine-tuning. During this process, a statistics-based token alignment approach is introduced as the cornerstone for fusing LLMs with different structures. Secondly, we merge these target LLMs within the parameter space, where we propose a novel method for determining the merging coefficients based on the magnitude of parameter updates before and after fine-tuning. We implement and validate FuseChat using six prominent chat LLMs with diverse architectures and scales, including OpenChat-3.5-7B, Starling-LM-7B-alpha, NH2-SOLAR-10.7B, InternLM2-Chat-20B, Mixtral-8x7B-Instruct, and Qwen-1.5-Chat-72B. Experimental results on two instruction-following benchmarks, AlpacaEval 2.0 and MT-Bench, demonstrate the superiority of FuseChat-7B over baselines of various sizes. Our model is even comparable to the larger Mixtral-8x7B-Instruct and approaches GPT-3.5-Turbo-1106 on MT-Bench.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduce a fuse-and-merge framework called FUSECHAT, which includes two stages. Pairwise knowledge fusion using a pivot LLM and token alignment to generate target LLMs with identical structure and size, and merging these models via SCE method, which determines merging coefficients based on parameter updates.

### Strengths
In general, the logic of the article is good, and the abstract, main text, and conclusions are consistent. The experiments are sufficiently convincing. The author summarizes the previous work from multiple aspects in the related work section.

### Weaknesses
1. In the Introduction section, there is insufficient explanation of the challenges faced by FUSECHAT. It is not enough to just explain the advantages of knowledge fusion, but the complexity of the work should also be highlighted.
2. The contribution of the work done in this paper is not explained in the Introduction section. 
3. The method section uses too many narrative words and lacks specific formula expressions, which increases the difficulty for readers to understand the article. 
4. In the experiment section, there is a lack of explanation for the adverse results in the experiment.

### Questions
1. First, the challenges of knowledge fusion tasks and the contributions of this paper should be introduced in the Introduction section.
2. The Method section should highlight the work done by the author. Extensive introduction of work that is not their own will make the article appear less innovative, and you can add formulas to further explain Token Alignment. 
3. The introduction of the SCE algorithm is too short, and the reasons for the use of some steps are not introduced, such as the Calculate and Erase steps.
4. Added explanations for poor experimental results in the experimental section, for example, Target LLM performs worse than Pivot LLM and Source LLM in some dimensions.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces FUSECHAT, a framework designed for the knowledge fusion of chat-based large language models (LLMs). The proposed fuse-and-merge framework integrates the strengths of diverse LLMs through lightweight continual training while avoiding the high cost and potential redundancy associated with developing new LLMs from scratch. Experimental results indicate that the proposed model outperforms existing methods across AlpacaEval and MT-Bench.

### Strengths
1.	The motivation is practical and significant, offering a cost-effective solution for integrating capabilities of different heterogeneous LLMs without training new models from scratch.

2. The two-stage framework effectively combines heterogeneous model knowledge through distillation into homogeneous models followed by parameter merging, with a well-designed token alignment strategy.

3. Comprehensive experiments validate the framework's effectiveness, showing competitive performance against different methods.

### Weaknesses
1. The paper's technical contribution appears somewhat limited. The approach can be viewed as a combination of pairwise FuseLLM and model merging (similar to TIES-Merging), both of which have been previously established as effective methods. The improved performance, while notable, follows logically from the combination of these known techniques, making the technical innovation less impressive than desired.
2. Several claims in the paper require further clarification. For instance, the statement on line 92 of the Introduction that "FUSELLM limits its exploration to source LLMs of the same size as the target LLM" appears inconsistent with FUSELLM's design, which can handle different-sized source models. Furthermore, FUSECHAT doesn't present special designs for distilling from differently-sized source models. Additionally, the choice of MinCE for the Fusion function in Equation 2 reduces to single-model distillation of the model with lower CE score in each pair, raising questions about the necessity of the pairwise approach.
3. There are concerns regarding experimental details. The combination weight is 0.9 in Equation 4, which means only 0.1 weight is assigned to distillation loss. Compared to 0.9 for SFT, this setting potentially undermines the significance of the distillation process. Moreover, the modest performance difference between FUSECHAT and Pairwise Fusion shown in Table 1 warrants statistical significance testing to validate the improvements.

### Questions
1.	Have the authors considered individual model distillation instead of pairwise fusion, given the MinCE choice in Equation 2? 
2.	What is the rationale behind the 0.9/0.1 weight distribution in Equation 4? 
3.	Can the authors provide statistical significance tests for the improvements over Pairwise Fusion?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a new framework, FuseChat, to fuse diverse LLMs into a single LLM capable of performing various tasks.  They first apply pairwise knowledge fusion on source chat LLMs to create multiple target LLMs with identical structures. To fuse models with different vocabulary, they introduce a statistics-based token alignment approach to obtain probabilistic distribution matrices. Then, they fuse all the target LLMs within the parameter space by utilizing the proposed new merging method, SCE. In their experiments, they conducted extensive experiments to investigate their framework with diverse source LLMs and evaluation datasets. They also offered a number of model analyses and ablation studies.

### Strengths
1. The paper studies an interesting question of how to fuse multiple chat LLMs into a potent chat LLM. The paper is well-written and well-organized. 
2. The paper has extensive experiments to investigate the effectiveness of their proposed framework and each component in their framework.
3. Their fusion method is also computation-friendly, which doesn't require additional training or dataset.

### Weaknesses
1. They didn't provide a significance test to show if their proposed method significantly outperforms their baselines (e.g. FuseLLM/OpenChat-3.5-7B Multi) or not. Because the improvement in some tasks is small, it would be better to show whether the improvement is significant. 
2. Table 1's caption needs to be improved. It would be helpful if they clarified what bold font and underscore mean in their table and what the percentage means.

### Questions
1. For Figure 3, I wonder if pivot LLM is the original OpenChat-3.5-7B or OpenChat-3.5-7B after fusing training. I also wonder if Target LLM is the OpenChat-3.5-7B after fusing training or the final FuseChat model. Please clarify these.
2. I wonder if you could categorize the samples in your training data into domains by MT-Bench and see how the distribution is in your training set.

### Soundness
3

### Presentation
4

### Contribution
3
