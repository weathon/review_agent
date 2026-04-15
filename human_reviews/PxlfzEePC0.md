# EC-DIT: Scaling Diffusion Transformers with Adaptive Expert-Choice Routing

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6, 6

## Abstract
Diffusion transformers have been widely adopted for text-to-image synthesis. While scaling these models up to billions of parameters shows promise, the effectiveness of scaling beyond current sizes remains underexplored and challenging. By explicitly exploiting the computational heterogeneity of image generations, we develop a new family of Mixture-of-Experts (MoE) models (EC-DIT) for diffusion transformers with expert-choice routing. EC-DIT learns to adaptively optimize the compute allocated to understand the input texts and generate the respective image patches, enabling heterogeneous computation aligned with varying text-image complexities. This heterogeneity provides an efficient way of scaling EC-DIT up to 97 billion parameters and achieving significant improvements in training convergence, text-to-image alignment, and overall generation quality over dense models and conventional MoE models. Through extensive ablations, we show that EC-DIT demonstrates superior scalability and adaptive compute allocation by recognizing varying textual importance through end-to-end training. Notably, in text-to-image alignment evaluation, our largest models achieve a state-of-the-art GenEval score of 71.68% and still maintain competitive inference speed with intuitive interpretability.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents EC-DIT, a new Mixture-of-Experts (MoE) model specifically designed for diffusion transformers in text-to-image synthesis. EC-DIT leverages computational heterogeneity through expert-choice routing, which enables the model to allocate computational resources adaptively based on the complexity of the input text and the corresponding image patches. This approach supports scaling up to 97 billion parameters, leading to improvements in text-to-image alignment, and generation quality.

### Strengths
1. Investigating methods to scale up diffusion transformers is a crucial research direction, as it directly impacts the effectiveness and potential of large-scale text-to-image synthesis models. 

2. The proposed EC-DIT model's effectiveness is verified by quantitative metrics.

### Weaknesses
1. As shown in Figures 1 and 9, the generated images are relatively simple and lack sufficient complexity. For more intricate images, the aesthetic quality appears subpar, and the benefits of scaling up are not convincingly perceptible. Offering 64-128 more generated images with more complex scenes could better demonstrate the value of the proposed method.

2. Models like SD3 and Flux, which have significantly fewer parameters than EC-DIT-64E, manage to produce images with notably superior visual quality. Please clarify why this discrepancy exists and address what specific advantages scaling up provides if the visual quality is still lacking. This raises questions about the true value that scaling up offers in terms of image quality.

### Questions
Please refer to the weaknesses.

### Soundness
2

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
3

### Summary
This paper explores how to scale diffusion transformers (DiT) to larger model capacities. The authors developed EC-DiT, a MoE-based (Mixture of Experts) DiT model leveraging expert-choice routing, expanding the DiT model parameters to 97 billion. EC-DiT enables context-aware routing and dynamically adjusts computation, inherently ensuring load balancing across experts. Extensive experiments were conducted to verify and analyze the effectiveness of EC-DiT.

### Strengths
- EC-DiT successfully scales the DiT model to 97 billion parameters, demonstrating exceptional performance.
- Expert-choice routing outperforms traditional token-choice routing strategies while maintaining competitive inference speed.
- The paper presents extensive experiments to showcase EC-DiT’s capabilities.

### Weaknesses
- While I recognize the authors’ contribution in scaling up DiT and support the paper’s acceptance, I believe it does not demonstrate sufficient technical innovation beyond prior work, appearing more as a practical integration of existing techniques
- A clearer and more precise definition of ‘Activated Params.’ should be provided. In Algorithm 1, it appears that all parameters are activated. Does this refer to the average number of parameters activated per token?
- I would like the authors to include a comparison of memory usage during training and inference.
- It would be beneficial if the authors could open-source the relevant scripts and model weights upon acceptance.

### Questions
Please refer to the weaknesses part.

### Soundness
3

### Presentation
3

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
This paper proposes a MoE based diffusion transformer framework to scale diffusion models. The basic idea is to combine DIT and MoE. The experimental results demontrate the effectiveness.

### Strengths
i) The technique line is reasonable.
ii) The writing is easy to follow.

### Weaknesses
i) This work should be incremental in essence, as the basic idea is to combine DIT and MoE.
ii) Certain statements exhibit overclaims that need refinement. (see Questions in detail)
iii) Crucial experiment comparisons with diffusion methods utilizing MoE are notably absent. (see Questions in detail)

The rebuttal addresses my concerns, so I raise the score.

### Questions
While efforts have been made to enhance the scalability of diffusion models for improved performance and efficiency, several issues remain:
i) The novelty of this approach requires clarification, especially in comparison to existing diffusion models incorporating MoE, as detailed in the related work section (lines 057-063).
ii) Certain claims should be refined or revised. For instance, the statement: “our approach significantly improves training convergence, text-to-image alignment, and overall generation quality compared to dense and sparse variants with conventional token-choice MoEs (Lepikhin et al., 2020b; Du et al.,2022; Zoph et al., 2022).” First, I do not see the comparison with these methods in experiments; Second, the performance is only comparable with SOAT, thereby questioning the usage of 'significantly.'; Third, the reason of improvement might be mainly attributed to increased model parameters, e.g., in experience, more parameters make the training converge faster. Thus, deeper analysis could be given.
“EC-DIT leverages this global context, combining multi-modal information to optimize token-to-expert allocation.” multi-modal?
“The model resolution is set to 256 × 256, with a patch size of 2.” “This results in the input sequence length of 128 per image.” Patch size is 2*2?
iii) Those existing diffusion models with MoE should be compared in experiments, if they are available.
iv) Fig. 3 only shows the performance on DSG score. Additional evaluation on other metrics should be showed, as the performance across various metrics might exhibit different trends. 
v) Fig. 4 should show more comparisons with other methods (e.g., in Table 2), rather than solely focusing on the proposed methods themselves. 
vi) How to set C in Top-k? There lacks the comparison with the different C values.
vii) The performance is only comparable with SOAT methods, such as SD3. In fact, the parameter quantities are in the same magnitude, SD3 (8B) vs EC-DIT-M-64E(>8B, observed in Fig. 4). Thus, this raises questions about the core contributions of this work. Is the primary focus acceleration? If so, there lacks many comparisons to current methods.

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper focuses on scaling diffusion transformers for text-to-image synthesis. It proposes EC-DIT, a new family of Mixture-of-Experts (MoE) models with adaptive expert-choice routing. By leveraging the computational heterogeneity of image generation, EC-DIT adaptively allocates compute to understand input texts and generate image patches, enabling heterogeneous computation aligned with text-image complexity. It scales up to 97 billion parameters and shows significant improvements in training convergence, text-to-image alignment, and overall generation quality compared to dense and conventional MoE models. Its contributions include introducing EC-DIT for adaptive computation in text-to-image synthesis, achieving promising performance at scale with faster loss convergence and better quality, and conducting comprehensive experiments to validate its superiority and demonstrate effective adaptive compute allocation.

### Strengths
1. EC-DIT can scale up to 97 billion parameters, achieving a state-of-the-art GenEval score of 71.68% in text-to-image synthesis.
2. The adaptive expert-choice routing is a novel design. It enables efficient compute allocation by having experts select relevant tokens, ensuring load balance without an extra load-balancing loss. This approach better utilizes computational resources, adapting to the complexity of different image regions and text, and improving overall performance and efficiency in handling inputs.
3. By embedding timestep and using cross-attention modules, EC-DIT effectively integrates global information, which allows the model to understand the context of the image and its relationship with the text, guiding compute allocation at different stages. It results in hierarchical processing, enhancing image quality and text-to-image alignment, and making the generation process more context-aware.

### Weaknesses
1. The best score for SD3 (Esser et al., 2024) w/ DPO is 74% in the original paper, which is actually higher than that of EC-DIT (71.68%). Even with the value 71% listed in Table 2,  EC-DIT is only gaining marginally. Given its additional overhead, it is hard to see the value of the gain. 
2. The difference between the theoretical and actual inference overhead (e.g., for EC-DIT-M, the theoretical is around 3% but the actual is 23%) indicates that there may be room for optimization in the inference process. The authors may want to explore the factors contributing to this overhead or propose specific strategies for mitigating it, which could further enhance the practical applicability of the model, especially for applications where real-time or fast inference is crucial.
3. The generalizability of the adaptive expert-choice routing is unclear. The performance of EC-DIT is mainly demonstrated in the context of text-to-image synthesis. How would it fit into the task of pre-training mixed-modal, early-fusion language models? It would be interesting to explore its effectiveness and efficiency in other related tasks. 
4. This paper could benefit from more detailed ablation studies on specific components of the proposed model. For example, understanding the individual contributions of the timestep embedding, cross-attention modules, and the expert-choice routing mechanism in more detail. This lack of in-depth analysis makes it harder to understand which parts are truly driving its performance.

### Questions
see weakness

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
This paper proposes a Mixture-of-Experts (MoE) model, called EC-DIT, for diffusion transformers. By applying the prior work of expert-choice routing to Diffusion Transformer (DiT), EC-DIT allocates a fixed amount of tokens to each expert. As the authors suggest, the method has the advantage that tokens on the more complex parts of images can be assigned to more experts, resulting in higher computation allocated to more important tokens and thus higher efficiency.

### Strengths
- The proposed method is well-motivated. An image typically contains regions of different complexity, which require different computation costs during generation. It would be beneficial to use the MoE method to allocate computation to different image regions adaptively. In Figure 6, the visualization of the number of experts for each token shows that more experts are used for the image tokens corresponding to the objects (compared to the background) on the images. The figure provides good support for the motivation of this work.
- The quantitative scores of EC-DIT show its effectiveness (Table 2). The evaluation includes multiple metrics, including GenEval, Davidsonian Scene Graph (DSG) scores, FID, and the CLIP Score.
- The paper is well-written and the method is clearly explained with a pseudocode provided.

### Weaknesses
- For Table 2, can the authors include the FID metric to compare the generated images' visual quality across the models? 
- Can the authors add a column for inference time or throughput (e.g. images/second) in Table 2? It would help the readers better understand the efficiency of different models.
- This paper has a good contribution to the community by showing the benefits of applying expert-choice routing in DiT, although the main approaches are mainly based on existing works, which makes the proposed method slightly less novel. 
- Can the authors make the notations in Section 2.1 regarding the rectified flow clearer? Please explain $p_0$, $x_0$, $x_t$, and $v_\theta$ in more details.

### Questions
- We know that DiT performs the diffusion in the latent space instead of the actual image pixel space. Does it undermine the motivation of this paper to allocate more computations to image regions of higher complexity? Please explain how working in latent space affects their approach to computational allocation.
- From Section 2.1, it seems that the authors use the Rectified Flow method on the DiT architecture with MoE. Can the authors explain why they chose to use Rectified Flow and how it integrates with their overall approach? This would help clarify the role of Rectified Flow in their method.

### Soundness
3

### Presentation
3

### Contribution
3
