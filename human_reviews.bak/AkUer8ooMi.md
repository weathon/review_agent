# ProReason: Multi-Modal Proactive Reasoning with Decoupled Eyesight and Wisdom

- Decision: Reject
- Scores: 6, 6, 6

## Abstract
Large vision-language models (LVLMs) have witnessed significant progress on visual understanding tasks. 
However, they often prioritize language knowledge over image information on visual reasoning tasks, incurring performance degradation.
To tackle this issue,  we first identify the drawbacks of existing solutions (i.e., insufficient and irrelevant visual descriptions, and limited multi-modal capacities).
We then decompose visual reasoning process into two stages: visual perception (i.e., eyesight) and textual reasoning (i.e., wisdom), and introduce a novel visual reasoning framework named ProReason. 
This framework features multi-run proactive perception and decoupled vision-reasoning capabilities.
Briefly, given a multi-modal question, ProReason iterates 
proactive information collection and reasoning
until the answer can be concluded with necessary and sufficient visual descriptions.
Notably, the disassociation of capabilities allows seamless integration of existing large language models (LLMs) to compensate for the reasoning deficits of LVLMs.
Our extensive experiments demonstrate that ProReason outperforms both existing multi-step reasoning frameworks and passive peer methods on a wide range of benchmarks
for both open-source and closed-source models.
In addition, with the assistance of LLMs,
ProReason achieves a performance improvement of up to 15\%
on MMMU benchmark. 
Our insights into existing solutions and the decoupled perspective for feasible integration of LLMs illuminate future research on visual reasoning techniques, especially LLM-assisted ones.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces PROREASON, a novel multi-modal reasoning framework that decouples visual perception and textual reasoning capabilities in large vision-language models (LVLMs). The framework features a multi-agent system that proactively collects visual information based on questions and performs reasoning through separate specialized components. The authors demonstrate that PROREASON outperforms existing approaches across multiple benchmarks, with improvements up to 13.2% on standard metrics.

### Strengths
I majorly conclude there are two strengths in this paper.

The paper presents a compelling approach to separating visual perception from textual reasoning, addressing a fundamental limitation of current LVLMs. This decomposition allows for more effective handling of each capability and enables the integration of specialized models for different aspects of the task.

The framework's ability to seamlessly integrate existing LLMs for improved reasoning capabilities is particularly valuable, as it allows organizations to leverage their existing investments in language models while enhancing multi-modal capabilities.

### Weaknesses
There are two main weakness about this paper.

1. The paper doesn't thoroughly discuss how the framework handles cases where the Vision Expert and Reasoning Expert disagree or provide conflicting information. Suggestion: Add a section analyzing failure cases and how the framework handles conflicting information between agents.

2. The evaluation focuses on specific benchmark tasks, but doesn't extensively explore how the framework scales with increasing complexity of visual scenes or reasoning requirements. Suggestion: Include experiments with varying levels of visual and reasoning complexity to demonstrate scalability.

### Questions
1. How does the framework handle cases where the required visual information is implicit or requires complex inference from multiple parts of the image? For example, in scenarios where understanding spatial relationships or temporal sequences is crucial?

2. Could the authors elaborate on how the framework might be extended to handle multiple images or video inputs, where temporal reasoning and cross-frame information integration become important?

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
5

### Summary
Briefly, this paper presents a multi-step multi-modal reasoning framework to overcome the limitation of LVLMs for visual reasoning tasks by introducing visual perception (i.e., eyesight) and textual reasoning (i.e., wisdom). The experimental results are strong. Besides,  the authors demonstrate the drawbacks of existing solutions (i.e., insufficient yet irrelevant visual descriptions and limited multi-modal capacities).

### Strengths
+ The proposed method presents an effective way to extract the necessary and sufficient visual details from images for the further multi-model reasoning step. 
+ Extensive and comprehensive experiments demonstrate the superiority of the proposed method.

### Weaknesses
1) Sub-optimal Design and Assumption. In Sect. 2.2, the authors argue that a detailed caption of the given image cannot provide sufficient and relevant information for Visual Reasoning (VR). However, some works (e.g., [s1]) concentrate on optimizing the caption. It might be more reasonable to include an optimized caption for the proposed Action step rather than use Q-I as input. 

2) Insufficient Experimental Evaluation. 
   + Why isn't GPT-4V included for comparison? It is somewhat difficult to fairly evaluate the proposed method on some benchmarks (e.g., HallusionBench) that have already reported the performance of GPT-4V.
   + Since the proposed method includes many steps during the inference, it might suffer from accumulated errors, e.g., wrong answers from the action step. The ablation study should include an extra experiment with a detailed discussion.
   + Missing the experiments on the traditional open-domain VQA benchmark (e.g., GQA).  It seems that the time complexity of the proposed method is high for some simple VQA tasks.

3) Due to the poor presentation of the Summary step, the proposed method might only works well on the VR task with multiple choices provided. 

[s1] Hu et al., PROMPTCAP: Prompt-Guided Image Captioning for VQA with GPT-3. In CVPR 2023.

### Questions
Please refer to the weaknesses section.

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
4

### Summary
This paper discusses the performance degradation problem of Large Vision-language models (LVLMs), which tend to rely more on language knowledge than image information in visual reasoning tasks. The authors first analyze the limitations of current solutions, then introduce the ProReason framework, which incorporates multi-run proactive perception and decoupled vision-language capabilities. It allows integration of the existing LLMs to compensate for the reasoning deficits of LVLMs. In detail, ProReason contains 5 components, including a dispatcher, a vision expert, a reasoning expert, a referee, and a summarizer.  The dispatcher selects a vision expert or a reasoning expert. After several iterations, a referee will decide if the information is enough for a summarizer to answer the question. To demonstrate the effectiveness of the proposed framework, experiments on several benchmarks show that ProReason outperforms existing multi-step reasoning frameworks and passive peer methods.

### Strengths
+ The paper provides a detailed analysis of the limitations in existing models, highlighting how current LVLMs tend to rely more on language information than on visual cues. This analysis points out issues such as insufficient and irrelevant visual descriptions and limited multi-modal capabilities.
+ The proposed ProReason framework can iteratively generate proactive perception and effectively decouple vision and language capabilities.
+ Extensive results on four benchmarks demonstrate ProReason's effectiveness. Additionally, ProReason’s design allows future integration with LLMs to enhance visual reasoning, showcasing the modulization and the potential for LVLM with LLM-assisted.

### Weaknesses
- The authors claim to 'decouple' multi-modal reasoning into visual perception and textual reasoning. However, they do not provide evaluations on key aspects such as the frequency with which the dispatcher selects the Vision Expert versus the Reasoning Expert, the content of the generated memory from each expert, and the relevance between the memory generated by the Vision Expert and the Reasoning Expert and standard answers.
- The authors compute relevance scores and evaluate caption effectiveness across different methods. However, since ProReason answers questions based on information stored in memory, calculating the relevance score between standard answers and the information within the memory could be an effective way to evaluate whether ProReason captures accurate information.
- In Figure 2, the example doesn’t fully clarify each component’s function, and the prompts differ from those in Figures 7 and 8, creating some inconsistency. Also, it would be great to show how memory is stored and given to each component.

### Questions
Questions are in the weaknesses. Here is the short summary:
1. Detailed evaluations on the Vision expert and Reasoning expert.
2. Calculating the relevance score between standard answers and the information stored in memory could enhance the evaluation of ProReason’s information accuracy.
3. Provide more details in the example for greater clarity.

### Soundness
3

### Presentation
3

### Contribution
3
