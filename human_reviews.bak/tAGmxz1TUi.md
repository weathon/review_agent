# A Simple Approach to Unifying Diffusion-based Conditional Generation

- Decision: Accept (Poster)
- Scores: 6, 8, 8, 5, 6

## Abstract
Recent progress in image generation has sparked research into controlling these models through condition signals, with various methods addressing specific challenges in conditional generation. Instead of proposing another specialized technique, we introduce a simple, unified framework to handle diverse conditional generation tasks involving a specific image-condition correlation. By learning a joint distribution over a correlated image pair (e.g. image and depth) with a diffusion model, our approach enables versatile capabilities via different inference-time sampling schemes, including controllable image generation (e.g. depth to image), estimation (e.g. image to depth), signal guidance, joint generation (image \& depth), and coarse control. Previous attempts at unification often introduce complexity through multi-stage training, architectural modification, or increased parameter counts. In contrast, our simplified formulation requires a single, computationally efficient training stage, maintains the standard model input, and adds minimal learned parameters (15% of the base model). Moreover, our model supports additional capabilities like non-spatially aligned and coarse conditioning. Extensive results show that our single model can produce comparable results with specialized methods and better results than prior unified methods. We also demonstrate that multiple models can be effectively combined for multi-signal conditional generation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper allows to generate diverse output modalities (image, depth maps, edge maps, poses) from diverse input modalities using a single generative model. The proposed UniCon model processes image and condition inputs “concurrently in two parallel branches, with injected joint cross-attention modules where features from two branches attend to each other.” LoRA weights are used to adapt the model from the pre-trained stable diffusion model. Experiments show that the approach performs on par with single generative approaches (controlNet, readout guidance, etc) in terms of FID, and depth, pose or edge estimation metrics, and outperform jointnet, a related multi-task generative approach in terms of results and speed.

### Strengths
* The paper is well written and illustrated.
* The approach is well motivated, unifying abilities of multiple generative models into a single one.
* The approach is relatively less computationally intensive than many other diffusion based approaches. 
* The results are convincing.

### Weaknesses
* No open sourcing of the code / models is mentioned.
* There is no human study comparing the effective quality of the approach compared to the others. There are nice qualitative examples in Fig 3, but a human study would be more convincing to demonstrate these examples are representative.  
* One of the most concrete applications of generative image models would be serving downstream tasks, such as classification or depth estimation as demonstrated in this work. However, the estimation performance is far from SOTA performance. 
* There is a confusion in the architecture of Unets, at least a sentence to rephrase line 242: The Unet [2015] [...] does not contain transformer blocks as it is written. Unets employed in diffusion models yes, but the sentence with this reference is confusing for now.

### Questions
1/ Do you plan to release the source code to ease reproducibility of the approach? 

2/ Could you do a human study? 

3/ In figure 3, third column, the fact that both the loose control and Unicon approaches lead o a snowman on the beach would mean their latent space is really similar as both approaches build on Stable diffusion weights. Also the controlNet result (first column) is also very similar to the Unicon one. It makes question the generalization ability of the approach.

4/ Why is there no quantitative comparison with the Loose control results?

5/ How is the weight w_r set up in (7)

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper introduces UniCon, an efficient approach to unify various conditional generation adaptation paradigms by modeling a single joint distribution across multiple modalities. This is done by having two U-Net branches that interact via a joint cross-attention. By also having different noise levels per branch, as previously done in Diffusion Forcing, both joint generation and conditional generation in both directions are possible during inference. To improve parameter efficiency, non-RGB branches are initialized to the RGB branch's weights and adapted using LoRAs instead of training them from scratch.

### Strengths
The proposed method enables more flexible conditional generation than many previous methods (Cond -> RGB, RGB -> Cond, RGB & Cond joint generation) by posing conditional generation as a special case of a more general joint generation setup and combining it with different timesteps per sample as pioneered by Diffusion Forcing. Compared to the approaches prevalent in conditional diffusion adaptation methods, this enables more flexible generation scenarios. The quantitative and qualitative evaluations show competitive performance of the proposed method compared to more specialized approaches. The paper is easily understandable and presents the method clearly.

### Weaknesses
The method, as presented, is limited to only accepting conditioning in the form of 2D images/feature maps. As it has now become common to support those combined with feature vector conditioning, as done in, e.g., Uni-ControlNet [Zhao et al., NeurIPS 2023], Composer [Huang et al., ICML 2023], or CTRLorALTer [Stracke et al., ECCV 2024].

The paper would benefit from additional qualitative examples in the appendix, as is commonplace in similar works.

### Questions
Regarding the weaknesses, I'd recommend the authors incorporate more related work into their quantitative comparisons and related work. Especially Uni-ControlNet seems important to compare against, as it is very closely related (except for the joint generation part).
For additional qualitative comparisons, I recommend following the example set by IP-Adapter, which provides a great overview of the qualitative performance of a wide range of methods.

The ablation study only focuses on RGB generation in a single setting. Have you also ablated effects in other sampling settings?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a framework for unifying different conditional generation tasks involving image-condition correlation in diffusion models. The proposed method learns the join probability distribution of the images and conditions during training and used for conditional generation during inference. The method mainly consist of parallel denoising paths for each condition, interacting with the image denoising path via cross-attention blocks. To maintain paramer efficiency, the base model for image generation is kept frozen, and Low Rank Adaptation (LoRA) is used to adapt the base model to each condition. The proposed method also allows for generation based on multople conditions. The authors evaluate their method on different conditional generation task involving conditions such as depth, edges, images, human poses, and identities.

### Strengths
- The paper is well-written and easy to follow.

- The proposed method provides a parameter-efficient way to model the joint image-condition distribution, which is more versatile for different conditioning tasks compared to specialized conditional methods.

- The authors provide sufficient experiments and comparisons for their method.

- Based on the provided results, the proposed method seems effective in modeling the joint image-condition distributions, and performing conditional generation

### Weaknesses
- Conditional generation using the proposed method requires performing multiple denoising paths, which makes the inference compationally intensive compared to direct conditioning, especially for multiple conditions.

### Questions
- It would be great if authors could provide an analysis on the computational cost of inference using their method.

- In Section 4.2, line 233, authors mention "When the conditional image differs from a natural image, we add a LoRA to the y-branch." How is the branch trained for conditioning on natural images, considering that the base image generation U-Net is fozen and no LoRA is used?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces UniCon, a unified framework for handling diverse conditional generation tasks by learning a joint distribution over correlated image pairs using diffusion models. UniCon employs a simple architecture that adds minimal learned parameters (15% of base model) and uses a single efficient training stage while maintaining the standard model input. Through extensive experiments, the authors demonstrate that their single model can produce comparable or better results than specialized methods and prior unified approaches, while also showing that multiple models can be effectively combined for multi-signal conditional generation.

### Strengths
- Unified framework for handling diverse conditional generation tasks through a joint distribution approach
- Lightweight adaptation of existing diffusion models with minimal parameter overhead
- Clear empirical demonstration of the proposed framework

### Weaknesses
1. **Missing Comparisons/References**

* The paper lacks comparisons with several important recent methods in depth estimation, and this limits our understanding of where the method stands in relation to the current state-of-the-art in depth estimation.
   - ZoeDepth
   - Depth Anything
   - Depth Anything v2 (which is only used as an annotator in this work)

* In addition, it will be helpful to discuss "DAG: Depth-Aware Guidance with Denoising Diffusion Probabilistic Models" [Kim et al.], which appears to be a relevant line of work for conditional generation with depth information.

2. **Technical Ambiguities**
* Section 4.2 should explicitly state the model architecture, e.g., whether the approach uses two pre-trained diffusion models and requires multiple feed-forward passes.
* This paper does not provide sufficient analysis of the inference cost. Based on current understanding, while it uses LoRA, it requires twice as many feed-forward passes, making the inference step complex and resource-intensive.

3. **Joint Distribution Modeling**

* While the paper presents joint distribution modeling as a key contribution, it should better justify why this particular formulation is advantageous over alternatives. The theoretical benefits of learning the joint distribution p(x,y) versus other approaches could be more thoroughly explained and experimented. For instance, their joint modeling might mutually enhance depth-aware image generation through improved representation.

### Questions
1. What is the inference cost of depth prediction compared to other traditional and recent approaches?

2. How does your method compare with ZoeDepth, Depth Anything, and Depth Anything v2?

3. What are the specific advantages of joint distribution modeling over other approaches discussed in the paper, particularly in terms of inference, accuracy, and other metrics?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents a diffusion model-based conditional image generation method as an add-on to current latent diffusion model (e.g., SD1.5). The model consists of joint cross attention layers and LoRA for conditional image branch, jointly denoising a concatenation of generated and conditional images. The training of the model is performed with disentangled noise levels from the two branches, inspired from Diffusion Forcing, allowing flexible conditional generation from different sampling schedules. The final model is compared to various problem-specific conditional model as well as with ControlNet, which solves similar problem with the proposed method with larger add-on network.

### Strengths
The main strengths are in their lightweight configuration, good reported performance, and novelty in using independent timestep scheduling.

- The writing of the paper is clear with comprehensive evaluations supporting the superiority of the proposed method.
- The model is overall lightweight in terms of the size and the training time, compared to previous image conditional add-ons, e.g., ControlNets. This makes the method application-friendly.
- Using disentangled noise level scheduling from Diffusion Forcing in paired generation for controllable image synthesis is quite novel. The demonstration in Figure 5 justifies this extension.

### Weaknesses
Although I believe the current version of the manuscript is above acceptance threshold, there are some limitations that prevents me recommending for higher honors (e.g., Highlight/Oral).

- Although there are five image conditional models trained using the proposed framework, it seems that only three (Depth, SoftEdge, Pose) types are compared quantitatively. This asymmetry in the quantitative/qualitative demonstration makes the manuscript incomplete.
- Moreover, there are also other metrics that compares the conditional generation. For example, for depth-conditional generation, there is $\delta_1$ metric used throughout the methods (please see Table 1 of Marigold and Table 2 of Depth Anything) that compares the method in at least five datasets. For pose-conditional generation, there is multiple metrics such as CLIP Identity Observance and Pose Control Accuracy (as in Table 1 of FineControlNet). We can also use user preference study for the tasks that does not have quantitative measures (such as in identity preservation and appearance preservation). Also there is a CLIP score to report the quality of the generation as well as text-fidelity of the generation. I do not want to recommend showing all those metrics that I have mentioned (these are for examples), but I believe it is required to have a complete table of quantitative metrics at least for models presented.
- It would be better to have a quantitative comparison in generated images from synchronous and asynchronous noise level scheduling in the inference time. The demonstration in the manuscript shows that the method works, but it does not show that this asynchronous generation leads to diversity and quality comparable to synchronous one.

### Questions
These questions are from pure curiosity and not counted in my final scores.

- Does the presented method applicable to applicable/robust to inpainting/editing tasks? Supporting this feature will make the proposed method far more persuasive.
- This is a very minor concern, but do the authors want to extend this model beyond SD1.5, which seems to be deprecated in the generative AI community?

### Soundness
4

### Presentation
2

### Contribution
3
