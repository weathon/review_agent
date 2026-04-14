# EVA: An Embodied World Model for Future Video Anticipation

- Decision: Reject
- Scores: 5, 6, 6, 6

## Abstract
World models integrate raw data from various modalities—such as images and language to simulate comprehensive interactions in the world, thereby displaying crucial roles in fields like mixed reality and robotics. 
Yet, applying the world model for accurate video prediction is quite challenging due to the complex and dynamic intentions of the various scenes in practice. 
In this paper, inspired by the human rethinking process, we decompose the complex video prediction into four meta-tasks that enable the world model to handle this issue in a more fine-grained manner. 
Alongside these tasks, we introduce a new benchmark named Embodied Video Anticipation Benchmark (EVA-Bench) to provide a well-rounded evaluation. 
EVA-Bench focused on evaluating the video prediction ability of human and robot actions, presenting significant challenges for both the language model and the generation model. 
Targeting embodied video prediction, we propose the Embodied Video Anticipator (EVA), a unified framework aiming at video understanding and generation. 
EVA integrates a video generation model with a visual language model, effectively combining reasoning capabilities with high-quality generation.
Moreover, to enhance the generalization of our framework, we tailor-designed a multi-stage pretraining paradigm that adaptatively ensembles LoRA to produce high-fidelity results.  
Extensive experiments on EVA-Bench highlight the potential of EVA to significantly improve performance in embodied scenes, paving the way for large-scale pre-trained models in real-world prediction tasks. The video demo and benchmark information will be available at \hyperlink{https://sites.google.com/view/iclr25-eva}{https://sites.google.com/view/iclr25-eva}.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper introduces a unified framework for video understanding and generation in embodied scenarios called EVA. The model can take as input an image with an instruction, and autoregressively generate future images until the goal is estimated to be completed. This paper introduces a new benchmark EVA-Bench for evaluation, evaluating video generation quality, the finish-thinking estimation accuracy and action description quality. Language evaluation metrics and image generation evaluation metrics are both used for evaluation.

### Strengths
1. The generation performance in both language and video level is significantly better than the compared methods.
2. The introduced EVA evaluates four interesting aspects of captioning, finish-thinking, how-to, and next-step.

### Weaknesses
1. The proposed method largely follows Pandora (https://arxiv.org/pdf/2406.09455), the most critical modification is its application on embodied data, and the reform of tasks. From a benchmark side, I believe this is interesting, but from a model side, this work seems incremental.
2. For the results in Table 1, I hope the authors can give more explanations on the comparison between EVA and "ChatUniVi-loRA", since these methods are very similar on the action description task. There is a big performance gap between ChatUniVi and EVA, however there is no explanation on this.
3. From Table 3, it is clear that the language part has a huge influence on the final score. It is understandable that LLaVa OneVision and qwen2-vl are not trained using the same data. But why the EVAS-V score is not largely influenced by EVAS-L score? Does it imply that one of these scores is not consistent with human understanding?
4. As for the EVA metrics, the video generation metrics largely follows the metrics in Vbench. However, I do not believe all these metrics should be used in the embodied scenario. For example, the aesthetic quality metric. For the Goal Completion Estimation metric, I believe this metric can be better designed. Now it is only the CLIP feature similarity. However I think this similarity is too semantic level and coarse-grained in the embodied AI scenario. This results in the GCE is almost similar for all models. I think spatial-awareness is critical in the GCE metric in the embodied AI scenario, and CLIP feature does not consider this. In the same embodiment with similar object presence, the CLIP features tend to be similar. I believe at least DINO features should be used.

**Post discussion comments.** 
Many thanks to the authors' responses in the discussion phase. After the discussion, my concerns for this work remain. The authors admit the lacks on the model side but fail to downweight it in the revision. This is not good, especially for a  paper with an emphasis on the model in its title. As for the benchmark, there remain concerns on the evaluation metrics. Using misleading metrics for a new benchmark is not the responsible way to contribute to the research community. Since the contributions in this work is also clear, I hope the authors can carefully re-draft and meticulously re-design the benchmark metrics, to make this work in a mature form to appear in a top-tier conference.

### Questions
Please see weaknesses, especially 2,3 and 4.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This work proposed a a unified framework called Embodied Video Anticipator (EVA) for both video understanding and generation. A  video generation model was embedded into a VLM model. Comprehensive testing on EVA-Bench demonstrates EVA’s capability to greatly enhance performance within embodied environments, signaling a strong foundation for deploying large-scale pre-trained models in real-world predictive applications.

### Strengths
1.  Well motivated: Two main challenges, lack of benchmark and failure in the in embodied scenes , are identified.  As a result, 4 meta tasks are unified into the a single framework to deal with multi-level prediction. 

2.  Readability; Well written. Easy to follow. Nice figure.

3.  Originality: A novel world model, the Embodied Video Anticipator (EVA), is designed for video prediction and understanding simultaneously. 

4.  New benchmark: Datasets from various domains are aggregated to build a diverse dataset and introduce a multi-stage pretraining approach.

5.  Good performance: EVA outperform all baselines on various tasks and benchmarks.

### Weaknesses
1.  Ensemble LoRA: Why is the ensemble LoRA injected into the spatial part? If it’s responsible for the task-specific domain, wouldn’t it make more sense to inject it into the temporal transformer?  Since there is already an image encoder, a more precise explanation of the role of the ensemble LoRA would be helpful.


2.  A further analysis of the output of the gating function would be more helpful to demonstrate the effectiveness of the gating mechanism and to prove efficient adaptation.


3.  Embodied Video Prediction: Embodied Video Prediction involves not only first-person perspective videos but, more importantly, egocentric motion, which this work does not seem to explore. If there are examples related to generated videos or tasks in this area, it would be great to have them updated on the website.

4.  Ablation study:  It would be more helpful to see more ablation studies to verify the effectiveness of each newly added module or method.

### Questions
All of my questions are asked in the weakness part

Typo: line 305 Ensamble---Ensemble

Figure 4 Resempler---Resampler?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work aims to address the video prediction problem at the task level, i.e., generating a long video until task completion based on current observations and instruction, which is more challenging than traditional frame-level video prediction. The authors reformulate the task in a coarse-to-fine manner, similar to the approach in "Human Rethinking." Specifically, they decompose the generation of long temporal videos into multiple tasks, including Action-Description, How-To, Finish-Thinking, and Next-Step, thereby generating multiple short videos that are ultimately combined into a long video. Subsequently, based on these four meta-tasks, the authors collected corresponding datasets and obtained the EVA model by adopting multi-stage and efficient fine-tuning methods. Experiments across various dimensions validate the effectiveness of the proposed method.

### Strengths
**Clear Writing and Presentation**

- The paper is well-written and generally easy to follow. The problem it aims to address is indeed a challenge in the field of video generation and the embodied world model. It provides appropriate intuition and effectively builds up motivation where needed, as illustrated in Figures 1 and 2.

**Interesting Decomposition of the Task**

- The author successfully decomposes the task into more specific and actionable components. By employing a self-ask approach, a multi-task model is serially executed at the inference stage to achieve the final goal, which is a simple yet efficient method, as shown in Figure 3.

**Strong Results in the Designed Benchmark**

- EVA demonstrates strong performance compared to existing baselines in the designed EVA benchmark, which encompasses four meta-tasks with their task-specified metrics.

### Weaknesses
**Novelty**

- The model's architecture is very similar to [1][2], as both utilize a Vision-Language Model (VLM) paired with a Video Diffusion Model (VDM). Currently, I have not observed any comparison or discussion between them in the paper (maybe the multi-stage pretraining scheme inspired by human rethinking?). Moreover, multi-stage pretraining and cross-attention alignment have become common training techniques in this field. Furthermore, I am unclear about the differences or advantages of the proposed Ensemble-LoRA compared to other existing LoRA-MOE technologies, like [3]; the authors also lack discussion or comparison with these methods in their experiments.

**Limitations in Evaluation Dimensions**

- Although the authors are tackling a very challenging problem, i.e., task-level video generation over long temporal sequences, all quantitative experiments focus on evaluating the four meta-tasks, without conducting quantitative evaluations on the complete videos. Additionally, since this is an embodied world model, the generated robot manipulation videos can be tested in simulation [4][5] or in real world [5]. For example, RoboDreamer [6], which the authors frequently mention, has performed quantitative evaluations on RLBench. In summary, the experimental results predominantly test short videos and lack comprehensive quantitative evaluations from various perspectives on the overarching problem, i.e., long video generation, they aim to solve.

**Lack of Ablation Studies**

- While the authors have proposed numerous methods (such as decomposing the overall task into four meta-tasks and employing various training strategies for fine-tuning), they have not conducted any ablation studies to verify the necessity of these methods or their specific roles within the framework. For instance, there is no discussion on which dataset/meta-task among the four meta-tasks is more critical for the model's performance, or how the omission of the Action-Description data affects the overall model. Again, if the proposed video generation can be applied to synthesized/real robotic manipulation tasks, it would be easier to verify the performance of this method, as well as how each meta-task helps.

[1] Hu A, et al. Gaia-1: A generative world model for autonomous driving

[2] Xiang, et al. Pandora: Towards General World Model with Natural Language Actions and Video States

[3] Chen Z, et al. Octavius: Mitigating Task Interference in MLLMs via LoRA-MoE

[4] Du Y, et al. Learning universal policies via text-guided video generation

[5] Ko P C, et al. Learning to act from actionless videos through dense correspondences

[6] Zhou S, et al. RoboDreamer: Learning Compositional World Models for Robot Imagination

### Questions
- More convincing evaluations and analyses of the proposed embodied world model, examining how the four meta-tasks help the robotic tasks in addition to their benchmarks in EVA. It would be helpful to clarify the contributions of this paper.
- Please also point out the novelty of the proposed embodied video anticipator, in contrast to the recent state-of-the-art methods.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a framework for video generation and video understanding in embodied scenarios through a world model approach. Inspired by the human thinking, the task of video understanding is broken down into four sub tasks - action description, finish thinking, how-to, and next-step along with video generation. This is introduced as a new benchmark, EVA-Bench. 

The world model function takes visual observations and a question as input to produce a predicted video and a text response. The action description consists of five key elements (subject, verb, object, location, destination), finish thinking does a frame level prediction to check if the task is completed, how-to task turns the instruction into a visual output, and next-step compares the predicted action video with ground truth video. The metrics of these sub tasks is named as EVA-Score which considers a weighted sum of language metric and visual contentment metric. 

The EVA framework is trained as a multi-stage pretraining network with cross-attention alignment and few shot adaptation with LoRA. The method is evaluated the four tasks on multiple metrics and shows an improvement when compared with the baselines. Overall, the contribution of a world model and dividing the video understanding into four subtasks following human thinking is interesting.

### Strengths
1. The paper shows an exhaustive evaluation on multiple metrics.
2. The contribution of designing the video understanding as a world model is interesting.

### Weaknesses
1. The subtasks of how-to and next-step have a possibility of various scenarios. How-to task turns the instruction into a visual output and there can be multiple outputs to perform the same instruction. Similarly, in the next-step task, there can be multiple actions that are possible given the actions that have happened until now. How does the model take into account this possibility of various scenarios during training and in metric evaluation?

2. The task of finish thinking which decides if the task has finished on the basis of frame-level prediction might not be accurate in various scenarios. For example, in cooking scenarios, the end frame is not unique and can take up many variations of the same recipe. How does the current approach take into account the different variations of the generated frame?

3. Since the tasks are sequential, error compounding can also occur over time.

### Questions
Is there any ordering bias in the next-step subtask? Since there are many possibilities for the next step, does the method develop any bias of the dataset when predicting the next step?

### Soundness
3

### Presentation
2

### Contribution
3
