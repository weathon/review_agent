# MuLan: Multimodal-LLM Agent for Progressive and Interactive Multi-Object Diffusion

- Avg Score: 4.00
- Decision: Reject
- Scores: 3, 5, 3, 5

## Abstract
Existing text-to-image models still struggle to generate images of multiple objects, especially in handling their spatial positions, relative sizes, overlapping, and attribute bindings. To efficiently address these challenges, we develop a training-free Multimodal-LLM agent (MuLan), as a human painter, that can progressively generate multi-object with intricate planning and feedback control.
MuLan harnesses a large language model (LLM) to decompose a prompt to a sequence of sub-tasks, each generating only one object by stable diffusion, conditioned on previously generated objects. Unlike existing LLM-grounded methods, MuLan only produces a high-level plan at the beginning while the exact size and location of each object are determined upon each sub-task by an LLM and attention guidance. Moreover, MuLan adopts a vision-language model (VLM) to provide feedback to the image generated in each sub-task and control the diffusion model to re-generate the image if it violates the original prompt. Hence, each model in every step of MuLan only needs to address an easy sub-task it is specialized for. The multi-step process also allows human users to monitor the generation process and make preferred changes at any intermediate step via text prompts, thereby improving the human-AI collaboration experience. We collect 200 prompts containing multi-objects with spatial relationships and attribute bindings from different benchmarks to evaluate MuLan. The results demonstrate the superiority of MuLan in generating multiple objects over baselines and its creativity when collaborating with human users.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper introduces MuLan, a comprehensive image generation method that utilizes a Large Language Model (LLM) agent for precise control of the generation process. The approach involves decomposing the prompt into a sequence of sub-tasks and generating each object sequentially through a diffusion model. Consequently, the method effectively generates multiple objects in accordance with the prompt.

### Strengths
- The idea of using Large Language Models (LLMs) for planning and Vision-Language Models (VLMs) to provide feedback is quite sensible. 

- This approach allows for the generation of objects that closely adhere to given instructions.

### Weaknesses
1. Using LLMs as planners is not a novel concept. Several methods like RPG have explored this approach before. 

2. In the experimental section, no compared methods leverage LLMs for image planning, although similar methods have been proposed. Only plain text-to-image methods are compared.

3. The entire generation process could be lengthy since each object in the image must be generated in order.

### Questions
1. The method is designed to generate objects progressively rather than all at once, but there is no ablation study demonstrating the benefits of this approach. 

2. Additional baselines need to be compared, particularly those using large language models (LLMs) as planners. 

3. How can we enhance the image quality of the generated outputs? While they successfully follow instructions, the resulting images don't appear to be as appealing as those produced by the base models.

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
5

### Summary
This paper introduces MuLan (Multimodal-LLM Agent), which leverages the reasoning capabilities of Large Language Models (LLMs) to decompose complex prompts into multiple subtasks, progressively generating multi-object outputs with detailed planning and feedback control. Additionally, MuLan incorporates Vision Language Models (VLMs) to provide feedback, thereby enhancing the alignment between prompts and generated images. The authors conducted experiments with 200 prompts involving multi-object scenarios with complex relationships to evaluate MuLan, and the results demonstrate its superiority in generating multiple objects.

### Strengths
1. Utilize LLMs as planners and VLMs as inspectors to enhance generation in complex scenarios.

2. The approach is training-free and model-agnostic.

3. Qualitative results surpass those of SDXL and PixArt-α.

4. Supports human interaction throughout the generation process.

### Weaknesses
1. The results are not competitive enough compare to current open-source models like FLUX and SD3, the method are outdated and lack novelty .

2. As mentioned in L233-243, the rough mask is limited to just four relative positions, which restricts its ability to handle more complex scenarios and reduces its overall flexibility.

3. As mentioned in the limitations, Inference time of MuLan is much higher than base models, however, open-source models like sd3 could already achieve accurate generation in compositional scenarios. It is inefficient to use a mulit-step method which could not show superior advancement as presented in the paper.

### Questions
1. Since MuLan is a training-free framework, why don't you utilize SOTA models like FLUX or SD3, i would appreciate if you could provide more comparisons and results between SOTA models and your methods.

2. The authors deliberately stress the importance of dealing with overlapping problems, however, the paper do not present enough results about overlapping prompts, especially lacks the interactions between human and animals.  Can MuLan achieve accurate and harmony generation for more complex prompts with overlapping entities?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper introduces MuLan, a Multimodal-LLM agent to improve the performances of existing text-to-image generation models, especiallly with multiple objects, spatial relationships and attribute bindings.
The main contributions inlcude, 
* A large language model (LLM) is adopted to decompose complex prompts into a sequence of simpler sub-tasks, each focusing on generating a single object. 
* A vision-language model (VLM) provides feedback to ensure that each object is generated accurately and aligns with the original prompt.

### Strengths
(1) This article adopts LLM to divides text-to-image generation into several steps, it addresses the limitations of existing models in handling multiple objects effectively. (2) The use of an VLM to provide feedback ensures that the generated images maintain consistency to the input prompt.

### Weaknesses
(1) In Section 3.4, the paper mentioned 'MuLan will adjust the backward guidance of the current stage to re-generate the object', but detailed adjustment algorithm or operation is not clearly explained.

(2) The evaluation is not sufficient, more existing works e.g. Ranni[1], Composable[2] should be included.

(3) The baseline models (e.g. SD1.4, SDXL) used in this paper are relatively weak, I highly doubt that if MuLan still works when using more strong base models (e.g. SD3, FLUX)?

(4) The tradeoff between accuracy and efficiency should be evaluated quantitatively, so that we can assess the practical values of this work.

[1] Feng Y, Gong B, Chen D, et al. Ranni: Taming text-to-image diffusion for accurate instruction following

[2] Liu N, Li S, Du Y, et al. Compositional visual generation with composable diffusion models

### Questions
See Weaknesses

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
4

### Summary
The paper presents MuLan, a training-free multimodal language model agent designed to enhance text-to-image (T2I) generation. MuLan addresses challenges in generating images with multiple objects, focusing specifically on controlling spatial relationships, relative sizes, and attribute bindings. By leveraging a large language model (LLM) for planning and a vision-language model (VLM) for feedback, MuLan decomposes complex prompts into sequential subtasks, each handling a single object generation with attention-guided positioning.

### Strengths
+ The paper is well-written and easy to follow.
+ MuLan demonstrates good control over the generation process and produces high-quality images that align with the prompts.
+ MuLan can be applied to human-agent interaction during the generation process.

### Weaknesses
- MuLan increases inference time, especially as the number of objects in a prompt grows, which could limit its scalability in real-time applications.
- As a training-free approach, MuLan is heavily reliant on the capabilities of underlying base models (such as Stable Diffusion).
- In some cases, as shown in Figure 2, the generated images exhibit unrealistic proportions. For example, in the first row, the refrigerator, chair, and table are the same size, and in the second row, the pumpkin and door are also similarly sized, which detracts from the realism of the generated scenes.
- Although qualitative results are emphasized, the absence of metrics such as generation speed or quantitative latency comparisons with baselines makes it difficult to assess MuLan’s practical efficiency.

### Questions
1. How efficient is MuLan, particularly as the number of objects increases in the prompt?
2. If the generated image deviates from the original prompt, how many iterations does MuLan typically require to produce an accurate result?
3. Could you provide more details on MuLan’s performance in handling edge cases, such as generating scenes with objects that have extreme relative sizes or complex occlusions?

### Soundness
2

### Presentation
3

### Contribution
2
