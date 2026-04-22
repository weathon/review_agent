# Beyond Synthetic Replays: Turning Diffusion Features into Few-Shot Class-Incremental Learning Knowledge

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Few-shot class-incremental learning (FSCIL) is challenging due to extremely limited training data while requiring models to acquire new knowledge without catastrophic forgetting. Recent works have explored generative models, particularly Stable Diffusion (SD), to address these challenges. However, existing approaches use SD mainly as a replay generator, whereas we demonstrate that SD's rich multi-scale representations can serve as a unified backbone. Motivated by this observation, we introduce Diffusion-FSCIL, which extracts four synergistic feature types from SD by capturing real image characteristics through inversion, providing semantic diversity via class-conditioned synthesis, enhancing generalization through controlled noise injection, and enabling replay without image storage through generative features. Unlike conventional approaches requiring synthetic buffers and separate classification backbones, our unified framework operates entirely in the latent space with only lightweight networks ($\approx$6M parameters). Extensive experiments on CUB-200, miniImageNet, and CIFAR-100 demonstrate state-of-the-art performance, with comprehensive ablations confirming the necessity of each feature type. Furthermore, we confirm that our streamlined variant maintains competitive accuracy while substantially improving efficiency, establishing the viability of generative models as practical and effective backbones for FSCIL.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Diffusion-FSCIL, a novel few-shot class-incremental learning (FSCIL) framework that leverages Stable Diffusion (SD) as a unified frozen backbone for feature extraction. Unlike prior works that use SD solely for generating synthetic replay images, this approach extracts four complementary types of features (inversion, synthetic, augmented, and generative) directly from the diffusion process in latent space. The method achieves state-of-the-art performance on CUB-200, miniImageNet, and CIFAR-100, while maintaining efficiency with only ~6M trainable parameters.

### Strengths
1.The idea of using SD as a frozen feature extractor instead of a generative replay buffer is original and well-motivated.
2. The use of multi-scale U-Net features from SD is technically sound and well-justified. The class-specific prompt optimization and controlled noise injection strategies are clever and effective.
3. The paper is well-written, with clear structure and intuitive illustrations.

### Weaknesses
1. While the empirical results are strong, the paper lacks theoretical analysis on why  SD features are more robust to forgetting compared to discriminative models like DINOv2.
2. All experiments are conducted on small-scale image classification datasets.
3. There is no discussion on how the method would adapt to other generative models (e.g., DiT, CLIP-guided diffusion, autoregressive models).

### Questions
1. Why does Stable Diffusion (SD) retain knowledge better than discriminative models like DINOv2 or CLIP in incremental sessions?
2. Why use four types of features instead of learning a single unified representation?

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
The paper proposes Diffusion-FSCIL, a novel framework that leverages Stable Diffusion not merely as a generative replay tool but as a unified backbone for few-shot class-incremental learning. Instead of generating and storing synthetic images, the method extracts four types of multi-scale latent features—inversion, synthetic, augmented, and generative—directly from SD’s U-Net to capture complementary semantic and structural information. These features enable efficient replay, generalization, and knowledge retention using lightweight networks (~6M parameters) while keeping SD frozen. Extensive experiments on CUB-200, miniImageNet, and CIFAR-100 benchmarks show that Diffusion-FSCIL outperforms prior FSCIL methods, demonstrating that diffusion models can act as both powerful generative and discriminative foundations for incremental learning

### Strengths
1. Utilizing a generative model as an encoder backbone for class incremental learning is a novel idea that has been proposed by the author.
    
2. There is no extra buffer space required by the model for the previously learned class for replaying the images to avoid catastrophic forgetting.
    
3. Using the latent features directly for replay during training is an interesting approach since the latent features captures semantics of the class rather than focusing on generating images following exact pixel distribution from data.
    
4. The model trained requires approximately 6M parameters.

### Weaknesses
1. There is a possibility of data leakage since the stable diffusion backbone used by the authors are entirely frozen which has already been trained on very large dataset.
    
2. In line 188, the claim is that DinoV2 is pretrained on CUB dataset, but in DinoV2 paper appendix C table 18 shows that for only for retrieval pretraining that dataset is used and not for pretraining. Which pretrained model did the authors use and if it is not retrieval pretrained then this claim doesn’t hold true and that raises the question on the claim made by the authors.
    
3. In Pilot Study, what is the approach used for training other models in continual setup besides Stable Diffusion Backbone. What parameters of the other models were tuned and how?
    
4. F_{gen} is not present during the Session 0 and introduced from session 1 (line 321) then how is it able to capture features from the classes trained in Session 0 since there is no network to distill from or use weights from. This is also highlighted in algorithm presented in Appendix where in base training F_{gen} is not present.
    
5. For each instance of data, forward pass generates the latent from classes for replay (F_{gen}), how is the class chosen from previous sessions for generating latent. Also, does it cover every single class that has been seen during the previous sessions?
    
6. Is there a particular reason why in CUB-200 there is severe degradation in other SD based models compared to the proposed approach but on miniImagenet or CIFAR 10, the degradation in performance from Session 0 to final session is almost similar.
    
7. For reproducibility, in line 318 what is the number of epochs that the F_{aug} is trained for.
    
8. For reproducibility, how is single label combination is done. What is the prompt or technique used by the authors to combine the labels into single one as claimed in line 745 of Appendix B.
    
9. The claim made in line 750, talks about “By employing single-word embeddings, we ensure that the semantic concept of each label is captured consistently and efficiently eliminating redundancy or semantic dilution that occurs when multi-word labels are split across multiple token.” Is there any experiment or results backing this support since it is dependent on tokenization process? For eg. in gpt-4-32k tokenizer “ring billed gull” uses 4 tokens, “Ring_billed_Gull” (used by the authors) uses 5 token and “RingBilledGull” also uses 5 tokens.
    
10. Some notational clarification needs to be done for eg. In line 146 S is used for denoting total number of session, instance of each incremental session.
    
11. \beta_{l} is used as weighing for each of the aggregation module. Are the weights being shared by each feature type or separate for each feature type.

### Questions
1. What is the performance of model with stable backbone on entire dataset and using similar training used for other models but with backbone of SD.
    
2. Clarification on how F_{gen} is distilled and used is needed. For further details see weakness point.
    
3. For more points refer to weakness

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
5

### Summary
This paper concentrates on Few-shot class-incremental learning (FSCIL), and proposes Diffusion-FSCIL based on Stable Diffusion (SD). Main contributions are summarized as:

a. Proposing  Diffusion-FSCIL, a framework that fully exploits SD as a unified backbone for FSCIL.

b. Extensive experiments to prove the effectiveness of proposed.

### Strengths
a. This paper is well written and easy to follow.

b. It is interesting to focus on Stable Diffusion in CL.

### Weaknesses
There are three main concerns:

a. I respectfully disagree with the motivation of this work. In section of introduction, it is mentioned that classical generative replay (GR) CL methods have the drawbacks that dependeding on separate discriminative backbones. This is not a drawback, because classical GR methods consider the situation that a customer has its own model (i.e., separate discriminative backbone). Classical GR methods aim to mitigate catastrophic forgetting of this model by adopting a generator. Therefore, this is not a drawback.

b. Instead, i think the proposed method is limited to Stable Diffusion (SD). All operations are based on SD. It is ok to propose a SD-bsed methods. But it doesn't mean that the proposed method deals with the issue of classical GR. Instead, classical GR is general to customer's backbone.

c. I think this paper is lack of technical novelty and the motivation of each operation in the proposed method, such as One-step inversion feature, One-step synthetic feature, is not clear. Why does this work use these operations? What is the motivation?

Other concern:

d. Section 5.4 reports the time cost cross incremental sessions. However, the most time cost occurs at base session.

### Questions
N/A

### Soundness
2

### Presentation
3

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
this paper investigates the few-shot class-incremental learning (FSCIL) problem and introduces a diffusion-model-based approach. instead of only using the latent diffusion model (LDM) for generative replays, the proposed method uses LDM for feature extraction, and shows that this LDM-based feature extractor can achieve competitive performance when compared to widely adopted options like DINOv2. by integrating the clean feature, the text-conditioned one-step noised feature, and multi-step noised features, the proposed method achieves good performance on multiple benchmarks.

### Strengths
+ using LDM for feature extraction instead of generative replay is interesting
+ the proposed method and its usage of different clean and noised LDM features makes sense
+ ablation on the different feature components from the ovearll method
+ easy to read with clear figures

### Weaknesses
- the reviewer is not convinced by the claim "DINOv2-G achieves higher initial accuracy due to larger capacity and explicit pre-training on CUB-200" (L188). SD is also trained on web data (most likely including CUB dataset). plus, the proposed method needs at least four forward passes on UNet and one forward pass on VAE encoder, so most like more compute than ViT-G in DINOv2
- following the above point, please consider report the inference FLOPs cost for the overall system and compare with existing methods, since the four forward passes do seem expensive

### Questions
see above

### Soundness
3

### Presentation
3

### Contribution
3
