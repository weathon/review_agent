# DMM: Building a Versatile Image Generation Model via Distillation-Based Model Merging

- Decision: Reject
- Scores: 5, 5, 5, 5

## Abstract
The success of text-to-image (T2I) generation models has spurred a proliferation of numerous model checkpoints fine-tuned from the same base model on various specialized datasets.This overwhelming specialized model production introduces new challenges for high parameter redundancy and huge storage cost, thereby necessitating the development of effective methods to consolidate and unify the capabilities of diverse powerful models into a single one.A common practice in model merging adopts static linear interpolation in the parameter space to achieve the goal of style mixing.However, it neglects the features of T2I generation task that numerous distinct models cover sundry styles which may lead to incompatibility and confusion in the merged model.To address this issue, we introduce a style-promptable image generation pipeline which can accurately generate arbitrary-style images under the control of style vectors. Based on this design, we propose the score distillation based model merging paradigm (DMM), compressing multiple models into a single versatile T2I model. Moreover, we rethink and reformulate the model merging task in the context of T2I generation, by presenting new merging goals and evaluation protocols. Our experiments demonstrate that DMM can compactly reorganize the knowledge from multiple teacher models and achieve controllable arbitrary-style generation.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors propose that the ability of expert models to contain many community models should be refined, while avoiding the proliferation of downstream models.This is a good insight. The authors propose methods based on score distillation, feature constraints, and adversarial learning to address the challenges presented. Unfortunately, the authors introduce a more ambitious vision in INTRODUCTION, but actually generalize only style learning.

### Strengths
1 The ideas presented by the author are reasonable and interesting.

2 The ability to distill a strong teacher model to learn more student models is something the community needs and should be tapped into.

### Weaknesses
1 Learnable Embeddings didn't make it clear? How size it is and how to make sure it learns moe-like abilities.

2 The introductory section is suspiciously exaggerated and actually lands in the style section. I think it's more important to explore how it can learn the capabilities of more models, such as animation, and the merging of multiple control conditions is what the community needs.

3 How can we quantify the ability of a distilled model to learn from multiple models without degrading the original model? This is what would like to see.

### Questions
1 Learnable Embeddings didn't make it clear? How size it is and how to make sure it learns moe-like abilities. Is it just the style capabilities available, I think the community needs to migrate more than just style.

2. If it's just the style migration ability, the author should compare it to articles like IP-Adapter,Instantstyle,instanstyle plus,styleshot,CSGO.

3. I believe the author's motivation is reasonable, but there should be programs with broader applicability for more tasks to consider.

4. see weaknesses.

### Soundness
3

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
This paper proposes DMM, a novel approach for compressing multiple models into a single versatile text-to-image (T2I) model. DMM integrates the styles of several models into a unified student model through knowledge distillation and continual learning. A style codebook is designed to conveniently control which teacher model’s style is followed. The authors introduce FIDt, a evaluation metric, on which DMM achieves excellent performance. This method offers a new perspective on model merging tasks.

### Strengths
1. The paper presents an approach to compress multiple models into a single versatile text-to-image (T2I) model. The originality lies in proposing a solution that integrates the strengths of various T2I models. Based on distillation, this method combines multiple T2I models into one. To allow for the specification of which model to utilize, the authors have designed a codebook that enables convenient control. 
2. This work has promising application potential and practical value. Current community models have their unique strengths, and this method offers a valuable approach for optimizing and integrating diverse community models.

### Weaknesses
1. Due to the lack of direct access to the original training data for each teacher model, the authors suggest treating this optimization process as a conventional regression task, using a generalized dataset for training. This approach would be effective if the dataset can adequately capture the core characteristics of the teacher models. However, when attempting to incorporate unique style teacher models, this method may be less effective. The study would be more convincing if the authors explored more distinctive styles beyond the common realistic and cartoon styles.
2. In incremental learning , the paper sets the initial task as four realistic-style models and a new task as four cartoon-style models. This setup may not adequately demonstrate the method's ability to resist catastrophic forgetting in incremental learning. Due to the distinct difference between the initial and new styles ,  it is easy for the model to distinguish between initial and new tasks . This may not effectively demonstrate the method’s ability to handle confusion. To provide more convincing results, the author maybe consider setting the initial task with models of different styles and introducing multiple new tasks, each introducing models of a different style.
3. The paper primarily presents results on a limited set of styles. The authors list eight community models in SDv1.5; however, these largely represent only two broad categories: Realistic and Anime & Illustration & 3D Cartoon. This limited scope makes it challenging to assess the effectiveness of style mixing. To better demonstrate the method's capability in style mixing, it is recommended to experiment with a wider variety of distinct style combinations, such as MoXin, M_Pixel, Miniature World Style and other community models.

### Questions
1, Was the model initialized using the SD1.5 weights, or was it initialized randomly? I did not find a detailed explanation regarding this aspect in the manuscript.
2, The authors demonstrate the selection of a single “style prompt” from N options or a combination of multiple “style prompts”. What would be the outcome if none of these “style prompts” were selected? Would the result resemble the original SD1.5 output?
3, I use fixed seed and prompt to generate images across different models. Sometimes the results differ so much that they appear unrelated, while other times they look like variations of the same image. This phenomenon is also evident in the author's Figure 6. I initially expected the author's model to maintain stylistic consistency with the teacher model, but was pleasantly surprised to see that it also retains similarity in details, as shown in Figure 6. I’m curious about what contributes to this effect. Just the ‘style prompt’ at the timestep embedding level?

### Soundness
3

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
5

### Summary
This paper proposes a model merging paradigm based on score distillation to merge several models into a single versatile model. They present a  distributed training framework to implement the score-distillation, where multiple teacher models are distilled into a single student model (share the same UNet architecture as teacher models) using a MSE loss, a learnable embedding is used to distinguish each teacher model. Meanwhile, they use feature supervision (also via a MSE loss) to facilitate knowledge transfer. Finally, they incorporate an additional GAN objective to distinguish different style distributions.

### Strengths
- This paper is well-written and easy to follow.
- The proposed pipeline is reasonable, using a learnable embedding as style prompt is interesting.

### Weaknesses
- My major concern is about its application. The goal of merging multiple base models (teachers) is unclear. This paper argues that specialized model need redundancy and huge storage cost, but most of style models are in LoRA format (may be merged into UNet and publish as a base model), which is already a light-weighting model and can be used as a plugin.
- For SDXL, it can already cover many styles via prompting, and selected teacher models are usually finetuned to a specific style. In what case we need re-merge these finetuned models into a versatile model?
- As shown in Table 3, the performance gain with weighted merging is minor, while the later is much easier.
- Only eight teacher models are used in experiment. It would be great to analyze the effect of the number of teacher models. For example, would it be a problem to distill 20 models? Especially when some styles are overlapped or similar.

### Questions
- What is the base models in Figure 5?
- In the experiment, the styles from different models are limited and overlapped, like realistic and anime, will it be a problem?
- Instead of merging base model directly, a more practical task is to merging multiple LoRAs, like style or character LoRAs, is this method applicable?

### Soundness
4

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
3

### Summary
This paper proposes a distillation-based model merging technique to learn a single model that can mimic the styles of N teacher models. To that end, learnable style prompts are used that trigger the generation of the style of a specific teacher model. The student is trained to reconstruct the teacher generations using a feature similarity loss (feature imitation), noise reconstruction (score distillation), and a multi-class adversarial loss. Experiments include learning a single model from 8 teacher styles and a continual learning setup.

### Strengths
- Overall, this paper presents a compelling narrative. It begins by addressing a well-motivated problem: the need to merge multiple specialized models into a single model to reduce computational demands in text-to-image generation. 
- The literature review and introduction is thorough, with a well-structured connection between various fields.
- The paper introduces a combination of techniques to improve model merging performance, and combining these approaches yield the best results.
- The proposed evaluation metric is appropriate, as it involves evaluating FIDs across separate tasks and styles.
- The paper includes numerous visual results that support the capabilities of the trained model.
- Additionally, the descriptions of the proposed methods are clear, and the figures are well-presented.

### Weaknesses
My biggest concern is w.r.t. made claims regarding problems of naive merging or other existing methods, lack of strong visual and quant. evidence for improving over the baseline, and the setup of the teacher models.

- The selection of 8 styles, with 4 styles categorized as "realistic" and 2 as "anime" is a weak setup to effectively demonstrate the problem as well as the advantages of the proposed model.
- The demonstration of the issue being addressed and the impact of the proposed solution is missing / low.
  - Claims regarding issues with simple merging lack supporting evidence. Specific conflicts, ambiguities, or patterns of confusion are not demonstrated.  
  - In Table 1, baseline results without any proposed additions (e.g., the initialized student model) are missing and would provide more context. 
  - Table 2 would benefit from including upper-bound teacher performance for comparison.   
  - Visual results for baseline and ablation configurations are missing, which are critical to substantiating claims of improvements and understanding the impact of reducing FID from 80 to 77.  
- The scope is limited to style transfer; extending the experiments to different content domains (e.g., mixing human faces with objects or indoor and outdoor scenes) would strengthen the argument.  
- The discussion on continual learning appears unrelated to the core problem of model merging and feels out of place.  
- Reporting individual FIDs instead of averages in the continual learning setup would provide more insights.  
- In Table 3, the proposed model achieves basically the same performance, but text and results lacks evidence for benefits beyond claimed flexibility. The method for manually determining weights in weighted merging is unclear, as is the sensitivity of this process compared to naive interpolation.  
- The approach still requires access to the fine-tuned models (at least during training), which limits its practicality.    
- The method for determining weights for the total loss and their sensitivity is not adequately explained.
- The term DDPM should be fixed as *Denoising Diffusion Probabilistic Model* rather than "Probability".  
- Generative Adversarial Networks is the correct term (attributed to Goodfellow et al.), not "Generative Adversary Networks" (incorrectly attributed to Song & Ermon).

### Questions
- Clarification is needed on the exact nature of the style prompts: Are these special text tokens appended to the prompts, or is something else trainable in the student model? A similar results might be achievable by tuning prompts and using style-related keywords, as shown by prior work on prompt tuning.

### Soundness
2

### Presentation
2

### Contribution
2
