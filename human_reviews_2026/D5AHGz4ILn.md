# Lego-Edit: A General Image Editing Framework with Model-Level Bricks and MLLM Builder

- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Instruction-based image editing has garnered significant attention due to its direct interaction with users. However, real-world user instructions are immensely diverse, and existing methods often fail to generalize effectively to instructions outside their training domain, limiting their practical application. To address this, we propose Lego-Edit, which leverages the generalization capability of Multi-modal Large Language Model (MLLM) to organize a suite of model-level editing tools to tackle this challenge. Lego-Edit incorporates two key designs: (1) a model-level toolkit comprising diverse models efficiently trained on limited data and several image manipulation functions, enabling fine-grained composition of editing actions by the MLLM; and (2) a three-stage progressive reinforcement learning approach that uses feedback on unannotated, open-domain instructions to train the MLLM, equipping it with generalized reasoning capabilities for handling real-world instructions. Experiments demonstrate that Lego-Edit achieves state-of-the-art performance on GEdit-Bench and ImgBench. It exhibits robust reasoning capabilities for open-domain instructions and can utilize newly introduced editing tools without additional fine-tuning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Lego-Edit, a modular framework for instruction-based image editing. The system consists of a Builder (a multimodal large language model fine-tuned with reinforcement learning) that organizes a library of model-level tools, Bricks, such as segmentation, inpainting, and style-transfer modules. A three-stage progressive RL scheme, supervised fine-tuning, ground-truth-based RL, and GT-free RL guided by a critic mode, gradually improves the Builder’s reasoning and tool-composition ability. Experiments on GEdit-Bench and ImgEdit-Bench show performance gains over strong baselines (e.g., BAGEL, Step1X-Edit, UniWorld-V1). The system also demonstrates zero-shot generalization and the ability to integrate new tools without retraining.

### Strengths
- The paper introduces a well-motivated modular architecture that addresses the limited flexibility of end-to-end image-editing models.
- The three-stage reinforcement-learning pipeline is a strong design choice that contributes to improved reasoning and generalization.
- Experimental results are extensive and include both quantitative benchmarks and visual demonstrations that convincingly illustrate the system’s strengths.
- The implementation details and reproducibility statement are thorough and transparent.
- The framework shows promising potential for extensibility and future multimodal applications.

### Weaknesses
- The contribution is primarily engineering-focused. The paper combines known techniques, MLLM agents, RL fine-tuning, LoRA adapters, into a unified system rather than introducing a new theoretical or algorithmic idea. While the integration is impressive, it offers limited conceptual insight into why or how the components interact optimally.
- The complexity of the architecture (Builder, Executor, Bricks, Critic) makes it difficult to interpret which design choices drive performance gains. More controlled ablations would clarify the marginal value of each component.
- The critic model used in the final RL stage plays an important conceptual role but is evaluated mostly qualitatively; quantitative ablations or alternative critic formulations would strengthen the argument.
- The writing and structure could be improved to highlight the core intuition behind the framework before diving into system specifics.
- The evaluation scope is limited to existing benchmarks. Although results are strong, the paper would benefit from evidence of user studies or real-world use cases demonstrating the practicality of Lego-Edit in diverse conditions.
- Some important limitations and failure modes are only briefly mentioned (e.g., potential tool conflicts, scalability bottlenecks) and deserve more explicit discussion.
- Runtime analysis is incomplete. The paper reports latency only relative to BAGEL but does not compare against other strong baselines such as Step1X-Edit, OmniGen2, or FLUX.1 Kontext. A broader efficiency analysis would help position Lego-Edit’s computational trade-offs relative to state-of-the-art models.

### Questions
1. How dependent is Lego-Edit’s performance on the chosen MLLM backbone (MiMo-VL-7B)? Could smaller or publicly available models achieve comparable coordination ability?
2. Could the authors report quantitative results comparing Stage 2 (GT-based RL) and Stage 3 (critic-based RL) to show how much each stage improves reasoning or compositional success rate?
3. How does the framework scale when the number of available tools increases or when workflows become more complex? Are there computational or latency trade-offs?
4. What are the typical failure modes? For example, does the Builder sometimes misinterpret instructions, or do tool boundaries introduce visual artifacts?
5. Since the paper is positioned in “applications to multiple modalities,” how feasible would it be to extend this framework to non-visual domains such as audio or video editing?
6. Have the authors evaluated how easily new users can add tools or control the Builder’s behavior without retraining? Understanding the system’s real-world maintainability could increase its practical impact.
7. Beyond the single latency result against BAGEL, could the authors provide runtime or computational cost comparisons with other leading systems such as Step1X-Edit, OmniGen2, or FLUX.1 Kontext? Such information would clarify Lego-Edit’s practical efficiency.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper targets on the problem that though have somewhat good editing performance, the current instruction-based image editing model always fail to generalize to unknown instructions. They propose a method called Lego-Edit, it finetunes a Multi-modal Large Language Model (MLLM) by reinforcement learning, enable it to coordinate model-level editing tools. In the MLLM part, it three-stage progressive reinforcement learning training strategy to make the MLLM more adapted to the current task. Extensive experiment results prove the effectiveness of the proposed method.

### Strengths
1. The paper is in a good organization, the idea is easy to understand. 
2. The experiment section is good, with abundant results and clear explanation. So I think the results are convincing.

### Weaknesses
1. Some typos, eg. Line 182, fine-graine. Please check the whole paper. 
2. If I do not mis-understand, since you incorporate more models, can you have some discussions about the train / inference efficiency of the proposed method?
3. I am willing to see more discussions about the novelty of the proposed method, since dividing one editing task to multiple different tasks is not interesting enough I think. 
4. I also want the authors to have more discussions about the limitation and future work of the work.

### Questions
See weaknesses. I think this paper is ok, but we can have more discussions during the rebuttal period. If my concerns are solved, I will be happy to raise the score. I hope the discussions can also bring about some new insights.

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
4

### Summary
This paper proposes a method for fine-tuning MLLMs to establish an automated workflow for instruction-based image editing tasks. By appropriately invoking pre-trained predictive models and editing models, this workflow achieves a variety of editing tasks and even exhibits a certain level of zero-shot capability for flexible editing requirements. Extensive visual results demonstrate the reliability of the method in terms of editing effectiveness, while quantitative results indicate its performance improvement over comparative methods.

### Strengths
1. The paper presents a valuable task formulation: how to orchestrate multiple sufficiently strong single-function perception models and editing models to achieve automated, instruction-compliant, and flexible image editing.
2. The proposed method in the paper is ingenious yet intuitive. Its multi-stage training strategy, tailored for the task, along with the reward design, effectively addresses the challenge of imbalanced training data annotations.
3. The results achieved by the proposed method are satisfactory, positioning it at the forefront of current community standards from both qualitative and quantitative perspectives.
4. The authors' approach to solving the problem is concise and clear, making it easy to understand and facilitating follow-up research.

### Weaknesses
1. Lacking a comparison with manual workflow construction would better highlight the efficiency and automation advantages of the proposed method.
2 . Lacking a dedicated limitations section. A section of limitations would improve the academic rigor and transparency of the work.
3. Authors need to provide a clearer explanation of the dataset construction for each training stage of the Builder to facilitate reader comprehension. If the authors used Qwen2.5-VL to generate ground truth data, why did they opt for the MLLM trained in the paper instead of directly calling the Qwen API in practical applications?

### Questions
1. The tools are not always accurate. How to handle this situation?
2. How to avoid the error accumulation when using the tool chain?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a general instruction-based image editing framework addressing poor generalization of existing methods to out-of-training-domain instructions. It uses a reinforcement learning (RL)-fine-tuned Multimodal Large Language Model (MLLM) called Builder to organize model-level tools (Bricks).

Bricks include predictive models (e.g., RES for object segmentation, ADD-PRED for adding position prediction) and editing models (e.g., INPAINT for inpainting, STYLE for style transfer), each trained independently for flexibility and performance.

Its three-stage progressive RL training: first, Supervised Fine-Tuning (SFT) builds basic capabilities; second, GT-based RL optimizes tool composition; third, GT-free RL uses an MLLM critic for feedback to enhance open-domain instruction handling.

Experiments show LEGO-Edit achieves state-of-the-art on GEdit-Bench and ImgBench. It handles complex multi-step edits, adapts to new tools/feedback zero-shot, maintains non-edited region consistency, and supports Chinese instructions and text editing.

### Strengths
1. This paper proposes a new direction to handle complex human-instructed image-editing tasks that uses existing image-editing models to construct an agent.
2. It shows superior experimental results when compared with other SOTA image generation models.

### Weaknesses
1. The description of training Builder is not sufficient, especially for the reinforcement learning section.  How to obtain an accurate reward when dealing with complex scenarios?
2. How about the results when compared with the Qwen-image and Seedream 3.0/4.0 series, which are end-to-end image generation models? 
3. There seem to be no failure cases presented in this paper. How to deal with the situation where the agent determines the error workflow, which may obtain terrible results? And how about the success rate of the agent?

### Questions
1. What is your opinion on whether the image generation area will pursue a strong end-to-end model or develop an agent that uses specific models?

### Soundness
2

### Presentation
3

### Contribution
3
