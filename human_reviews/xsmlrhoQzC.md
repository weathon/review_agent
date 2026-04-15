# Proactive Agents for Multi-Turn Text-to-Image Generation Under Uncertainty

- Decision: Reject
- Scores: 5, 6, 5, 6

## Abstract
User prompts for generative AI models are often underspecified or open-ended, which may lead to sub-optimal responses. This prompt underspecification problem is particularly evident in text-to-image (T2I) generation, where users commonly struggle to articulate their precise intent. This disconnect between the user's vision and the model's interpretation often forces users to painstakingly and repeatedly refine their prompts. To address this, we propose a design for proactive T2I agents equipped with an interface to actively ask clarification questions when uncertain, and present their understanding of user intent as an interpretable **belief graph** that a user can edit. We build simple prototypes for such agents and verify their effectiveness through both human studies and automated evaluation. We observed that at least 90\% of human subjects found these agents and their belief graphs helpful for their T2I workflow. Moreover, we use a scalable automated evaluation approach using two agents, one with a ground truth image and the other tries to ask as few questions as possible to align with the ground truth. On DesignBench, a benchmark we created for artists and designers, the COCO dataset (Lin et al.,2014) and ImageInWords (Garg et al., 2024), we observed that these T2I agents were able to ask informative questions and elicit crucial information to achieve successful alignment with at least 2 times higher VQAScore (Lin et al., 2024) than the standard single-turn T2I generation.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces a proactive design for text-to-image (T2I) agents that enhances user interaction by prompting clarification questions when user intent is unclear. The approach enables T2I agents to create an interpretable 'belief state'—a representation of their current understanding—that users can review and adjust as necessary. By allowing edits to this belief state, the system promotes transparency and collaboration, enabling the agent to refine its responses to better match the user’s expectations. Overall, this design aims to make T2I agents more interactive, adaptable, and user-centric.

### Strengths
The paper is well-written and easy to understand, presenting concepts and methodologies in a clear and accessible manner. Additionally, the results are strong, demonstrating the effectiveness and reliability of the proposed approach.

### Weaknesses
However, many previous methods have proposed self-correction techniques to improve generated images, as demonstrated in works such as [1], [2], and [3]. This paper, however, does not include comparisons with these related methods. Given this omission, I question the technical contribution of the paper and am uncertain whether it would be a suitable fit for CVPR.

Previous related works have proposed iterative improvements to image quality. Therefore, it would be helpful if the authors could address the following points: 
1) What is the difference between the iterative procedure used in the proposed methods and that of related works? There is no direct comparison with all related works. 

2) It would be beneficial if the authors could compare their methods with these works and explain why the proposed method has a competitive edge. 

3) If users make errors when inputting their data, is the proposed model able to detect and correct these mistakes? 

4) Regarding reasoning ability, if a user requests advanced generation requirements, such as "a robot executes a task," is the proposed method capable of handling spatially-aware generation?

[1] Wu, Tsung-Han, et al. "Self-correcting llm-controlled diffusion models." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

[2] Yang, Ling, et al. "Mastering text-to-image diffusion: Recaptioning, planning, and generating with multimodal llms." Forty-first International Conference on Machine Learning. 2024.

[3] Jiang, Dongzhi, et al. "CoMat: Aligning Text-to-Image Diffusion Model with Image-to-Text Concept Matching." arXiv preprint arXiv:2404.03653 (2024).

### Questions
It would be appreciated if the authors could elaborate on the technical challenges addressed, as well as the potential insights and future impact of this work in the field of computer vision.

========================= 

After reviewing the rebuttal, I have decided to keep my score unchanged.

### Soundness
3

### Presentation
2

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
This paper introduces a proactive approach to multi-turn text-to-image (T2I) generation, addressing the common problem of prompt underspecification, where users struggle to articulate their vision clearly, leading to suboptimal results. The proposed T2I agents actively engage with users by asking clarification questions and updating their understanding through an interpretable belief state that users can edit. The paper details the design of these agents, the use of a graph-based belief state to manage uncertainty, and the development of a new benchmark, DesignBench, for evaluation. Experimental results, including both automated evaluations and human studies, demonstrate that these proactive agents significantly improve image generation quality, achieving higher alignment with user intent compared to traditional single-turn T2I models.

### Strengths
The paper's strength lies in its interactive approach to improving text-to-image (T2I) generation. By using proactive agents that ask clarification questions and utilize a graph-based belief state for transparency, the method addresses prompt underspecification effectively. The combination of user-centric design, comprehensive evaluation, and the introduction of the DesignBench benchmark demonstrates significant improvements over traditional T2I models.

### Weaknesses
1. If the approach involves making prompt engineering more detailed than the initial prompt, it seems intuitive that the resulting image would be more accurately generated as intended. Could you elaborate on what aspects of this method are novel or distinct beyond providing more detailed prompts?
2. How do you determine which characteristics are crucial for accurately generating each image? A clearer and more logical explanation of the criteria or process used for this selection would be helpful.

### Questions
1. Is there a validation process for the ground truth caption? What guarantees that it’s a well-generated caption?
2. I wonder if 15 turns are essential. The number of turns needed would likely vary for each case, so wasn’t this considered? Why 15 turns specifically?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
A new concept, build in a graph-based symbolic belief state for agents to understand its own uncertainty aboutpossible entities that might appear in the image.
It's a nice thing that  this article want to do, but  the solution is so boring. T2I + MLLM directly.

### Strengths
The question does arise as to how to make the generation of T2I models more personalised and more responsive to the specific and potential needs of users.
1. Design and prototypes for T2I agents that adaptively ask clarification questions and present belief states.
2. An automatic evaluation pipeline with simulated users to assess the question-asking skills of T2I agents.
3. DesignBench:a new T2I agent benchmark.

### Weaknesses
The solution is too simple and very uninventive. Put in a new concept and then come back with a self-explanatory statement.
Beliefstate It's a former concept, and there's nothing inherently innovative about it.

### Questions
Focus on D: implementation details and show your novelty. The main text is too shallow.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work addresses the problem of sub-optimal image generation from text-to-image generators due to under-specified or open-ended user prompts. The work proposes building proactive T2I agents that could ask clarifications questions. Furthermore, the understanding is present as a belief state visible and editable by the user. The work also proposes a scalable automated evaluation benchmark. Experimental results suggest that 90% of the human subjects found agents and the belief states useful for the T2I workflow, and the proposed agents significantly improve the VQAScores of the generation.

### Strengths
* In contrast to previous multi-turn T2I systems working on multi-turn user instructions, the proposed system asks question to the users for clarifications, which is a new form of interaction with the users orthogonal to previous works.
* This work proposes an evaluation pipeline that simulates users, which makes the evaluation of proactive agents human-free and much easier. This pipeline could also benefit the development of future agents that ask questions.
* The proposed DesignBench supplements COCO-Captions in the artistic images, which is a dataset that benchmarks the capabilities of text-to-image systems tailored to the needs of designers and artists.
* Writing: the paper has its messages clearly conveyed.

### Weaknesses
* The work frames the problem as the agent updating beliefs according to a fixed world state in the user's mind. However, the user may not have a clear idea in mind (i.e., the user may not have a pre-defined world state). In contast, the user might want to get some inspirations from the system without constraints. This use case has not been considered in the system design. This limitation might affect users such as artists. This was discussed in L534-539, but no suggestions were proposed.
* The work uses LLMs in Ag2 and Ag3 without fine-tuning. However, this work does not explore trained LLMs (VLMs) with either image data or trajectories that include asking questions. This indicates that the LLM is purely exploring in text space. The exploration might be sub-optimal since exploration on text space might be different than exploration with images in mind.
* The output of the system still might not follow user's instructions. For example, one of the generated images in Fig. 1 does not have the rabbit chasing the dog.

### Questions
The reviewer has a question regarding the failure case in Fig. 1:
* Is the system bottle-necked by the alignment and prompt-following capabilities of the text-to-image model or by the capabilities of the agent?

### Soundness
3

### Presentation
3

### Contribution
3
