# ChronoEdit: Towards Temporal Reasoning for In-Context Image Editing and World Simulation

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 2, 8, 4

## Abstract
Recent advances in large generative models have greatly enhanced both image editing and in-context image generation, yet a critical gap remains in ensuring physical consistency, where edited objects must remain coherent. This capability is especially vital for world simulation related tasks. In this paper, we present ChronoEdit, a framework that reframes image editing as a video generation problem. First, ChronoEdit treats the input and edited images as the first and last frames of a video, allowing it to leverage large pretrained video generative models that capture not only object appearance but also the implicit physics of motion and interaction through learned temporal consistency. Second, ChronoEdit introduces a temporal reasoning stage that explicitly performs editing at inference time. Under this setting, target frame is jointly denoised with reasoning tokens to imagine a plausible editing trajectory that constrains the solution space to physically viable transformations. The reasoning tokens are then dropped after a few steps to avoid the high computational cost of rendering a full video. To validate ChronoEdit, we introduce PBench-Edit, a new benchmark of image–prompt pairs for contexts that require physical consistency, and demonstrate that ChronoEdit surpasses state-of-the-art baselines in both visual fidelity and physical plausibility. Project page for code and models: https://research.nvidia.com/labs/toronto-ai/chronoedit

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes modeling the physical realism of image editing using video generation models. By leveraging the strong temporal, physical, and motion consistency capabilities of video generation models, the approach achieves impressive editing results. Furthermore, the introduction of a temporal reasoning token to simulate intermediate video frames is a very intuitive idea, and the experimental results are remarkable.

### Strengths
1. Utilizing video generation model to model the image editing task, which introducing great physical prior, achieving great results.
2. The Temporal Reasoning Token simulate the intermidiate step of video and it fills the gap of modeling intermediate changes in image editing, resulting in stronger interpretability.
3. After distilling, the result is still great and the speed is nearly comparable with image editing model.

### Weaknesses
I do not have many concerns regarding the content of the paper itself. However, I am curious about how the video generation based model would perform in multi-turn editing scenarios involving different user instructions, where each round would introduce an additional input frame, which is an interesting attempt for future development.
Additionally, I want to know the true performance of trained video generative model in video generation. This would provide insights into how demanding the requirements are for training video generation models to achieve high-quality image editing. Please show the result in video generation benchmark like VBench series.

### Questions
See the weakness.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose a method termed ChronoEdit using latent diffusion/flow video models for editing images based on text prompts and a reference image. Three variants of the method are proposed in the paper. (1) ChronoEdit - generations of the edited image as a video having two frames (the reference image and the output image); (2) ChronoEdit-Think - where additional "reasoning" frames/tokens are added between these two frames in order to model better physical consistency across time; (3) ChronoEdit-Turbo - distilled version of ChronoEdit for few-step sampling. The paper also presents PBench-Edit, a new benchmark for image editing designed to assess editing in physically grounded contexts. The authors compare the 3 variants to various baselines from the literature on two benchmarks and show an ablation study for various aspects of their method.

### Strengths
* The main novelty of this paper, as I understand it, since I am not familiar with the literature in this area, resides in the idea to frame the problem of image editing based on text prompts as a video generation task using diffusion/flow models. By doing so, one can get better physical consistency. Indeed, I find this idea novel and interesting.
* To gain better control and interpretability, another novel part of the method is to introduce the so-called reasoning tokens, which show that the authors took an extra step in the modeling process of the problem.
* The authors introduce a new benchmark, which I suppose can be valuable for research in this area.
* The method seems to improve over baseline methods on both benchmarks, both quantitatively and qualitatively. 
* For the most part, the paper is written clearly.

### Weaknesses
* While I do appreciate the idea of using video generation models for image editing and the other improvements to the method, in terms of the overall contribution and novelty, to me, this work seems limited. To properly assess its contribution, I believe it will be valuable if the authors could further elaborate on how this work lays a new foundation for future models in this area and if it opens new avenues of research.
* Regarding the experiments:
  - To me, the quantitative results are not clear enough. Specifically, are the numbers in the table bounded between [0,5]? If not, are they bound in another region? How exactly are they measured? In addition, in order to understand the significance of the results, std information should be reported.
  -  The authors present only successful cases for their model, but what are some failure cases of it? Are there common situations in which it systematically fails?
  - One experiment I found missing is showing the performance of the method when using out-of-the-box video generation models (without training), where the first and last frames are set as in this paper.
  - Regarding reproducibility, code wasn't provided. For me, this is a weakness in an empirical paper.
* Minor:
  - In Fig. 3, what is the prompt?
  - In line 311, do you mean $N=8$?
  - It is not clear which version of ChronoEdit, ChronoEdit-Turbo distills.

For now, my tendency is to reject the paper; however, as I am not an expert in this field, I would like to see the author's response, assessment of my fellow reviewers, and the AC before making a final decision.

### Questions
* In line 201, how is it ensured that F', c, and w are integers?
* Was the model from which you generated the training dataset ("video data curation") trained (either in the pre-training stage or fine-tuning) on PBanch?
* All baseline methods seem recent; why is there such a gap in the number of parameters? Specifically, I assume that at least part of the baseline methods also use latent diffusion/flow models. What causes the difference then? And can you evaluate your method using exactly the same network and number of parameters?

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
3

### Summary
The paper introduces ChronoEdit, a novel framework designed to instill physical consistency in generative image editing, a capability deemed crucial for "world simulation" applications (e.g., robotics, autonomous driving). ChronoEdit achieves this by reframing image editing as a two-frame video generation problem, thereby leveraging the learned temporal prior from pre-trained video generative models. For enhanced coherence, the framework features a Temporal Reasoning Stage (ChronoEdit-Think) during inference. Here, "reasoning tokens" (imagined intermediate video frames) are jointly processed with the input to implicitly constrain the final edit trajectory to one that is physically plausible. The authors validate their approach with the new PBench-Edit benchmark, demonstrating SoTA performance in both visual quality and physical realism over existing methods.

### Strengths
1. The idea of repurposing the powerful temporal mechanisms of video generation models to solve a fundamental deficiency (physical inconsistency) in static image editing is a good idea, providing a principled method for incorporating dynamic laws.
2. By focusing on physical consistency, ChronoEdit addresses an important bottleneck for real-world applications like autonomous systems, where geometric or physical inconsistencies are unacceptable.
3. The Temporal Reasoning Stage is cleverly implemented to optimize efficiency by limiting the reasoning steps ($N_r$) and discarding the tokens early in the denoising process. Furthermore, the visualized reasoning trajectory offers a degree of interpretability into the model's "thinking process."
4. The creation of the PBench-Edit benchmark is a valuable contribution, establishing a much-needed standard to evaluate models on physical and temporal coherence, pushing research beyond purely aesthetic metrics.
5. Figure 4 looks great

### Weaknesses
1. Despite efforts to optimize, the ChronoEdit-Think variant still introduces a measurable inference overhead compared to non-reasoning baselines.
2. The framework’s success depends entirely on the video model's ability to perfectly encode and enforce physical laws within its high-dimensional latent space. If the latent representation merely captures statistical correlations of motion rather than strict physical principles, errors in complex or novel dynamics will inevitably persist.

### Questions
1. What editing operations fundamentally cannot benefit from this framework?
2. Is rejection sampling effective for generating physically plausible results?

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
2

### Summary
This manuscript proposes a ChronoEdit method for image editing, which reframes the editing problem as a video generating problem. It introduces a temporal reasoning mechanism at inference to perform editing, wherein reasoning tokens are designed to imagine plausible editing trajectory. A novel benchmark dataset is proposed to evaluate the proposed method.

### Strengths
-	The idea of solving image editing using video generation models is interesting.
-	Experimental results on the benchmark datasets show good performance.

### Weaknesses
-	The proposed method is prone to the video generation models. Is the upper bound of the proposed method is limited to the performance of the video generation models?
-	The computational complexity of the proposed method vs. other image editing methods is expected to be justified.
-	The ambiguity of the “edited” image. As mentioned by the author in L216-232, the author formulates the image editing as a T-frame video generation problem, wherein 0- and T-th frames are defined as the input and edited image. Can the (T-1)-th frame or other frames also be considered as the “edited” image? If not, what are the key differences between the (T-1) and T frames?
-	There are methods [1-3] that apply chain-of-thought techniques to the image editing problem, which also attempt to add an intermediate process into the editing problem. Can you discuss and justify the proposed paradigm vs these methods? 

[1]. Enhancing Image Editing with Chain-of-Thought Reasoning and Multimodal Large Language Models
[2]. ReFocus: Visual Editing as a Chain of Thought for Structured Image Understanding

### Questions
See the weakness

### Soundness
3

### Presentation
2

### Contribution
2
