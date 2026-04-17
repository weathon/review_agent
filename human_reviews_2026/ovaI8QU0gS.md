# FLAM: Scaling Latent Action World Models with Factorization

- Decision: Reject
- Scores: 2, 2, 2, 4

## Abstract
Learning latent actions from action-free video has emerged as a powerful paradigm for scaling up controllable world models learning.The latent actions offer an extra degree of freedom for users to generate videos iteratively.However, existing approaches often rely on monolithic inverse and forward dynamics models to learn one latent action that controls all, which struggle to scale in complex scenes where different entities act simultaneously. In this work, we propose FLAM, a factored dynamics framework that decomposes the latent state into independent factors, each with its own inverse and forward dynamics model. This structure enables more accurate modeling of complex, multi-entity dynamics and improves the video generation quality in action-free video settings. Evaluated on Multigrid, Procgen, nuPlan, Sports and EGTEA datasets, FLAM consistently outperforms the monolithic dynamics model, demonstrating the superiority of the factorized model.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a Factorized Latent Action Model (FLAM) that decomposes both latent representations and latent actions into multiple factors, aiming to enhance the modeling capability of latent action world models in complex multi-entity scenarios. The approach leverages slot attention to obtain object-centric latent representations and applies shared inverse and forward dynamics networks to each component. Experiments are conducted on Multigrid, Procgen, nuPlan, Sports, and EGTEA datasets.

### Strengths
1. The idea of object-centric decomposition for latent action models is novel and, to the best of my knowledge, has not been explored before.
2. The paper includes extensive experiments across five datasets, covering both synthetic and real-world scenarios, which demonstrates strong empirical effort.

### Weaknesses
1. **Major concern: not good presentation and missing experimental evidence**
   The main text lacks critical experimental results needed to substantiate the paper’s claims. Several key findings are either deferred to the appendix or missing entirely.
   - The results shown in Figure 7, which support a major conclusion of the paper, should appear in the main text. Moreover, quantitative ablation results should accompany this figure to strengthen the argument.
   - The paper mentions that “human users” were asked to specify latent actions, but it does not clarify which latent actions (e.g., action indices or semantic descriptions) were selected for the visualization in Figure 4.
   - Section 4.3 mentions the use of action labels for policy learning, but no corresponding experiments are presented.
   - The paper repeatedly claims (Lines 113, 233, 291) that the learned object-centric representations group entities with similar dynamics rather than visual appearance, yet no qualitative evidence is provided beyond overall quantitative metrics. I strongly suggest including visual showcases illustrating this property.
   - Although not an experimental issue, the discussion on slot initialization in Section 5.5 is insightful and should be moved into the method section, which contributes more directly to understanding the model design.
   - Overall, I strongly recommend a thorough revision of the paper’s structure and formatting to make the contributions clearer and ensure that all major claims are properly supported.
2. The title emphasizes “scaling latent action world models,” yet the experiments do not demonstrate any scaling trend with model size or capacity (see also Question 3). To avoid misleading implications, the paper should either include such experiments or adopt a more conservative title reflecting its actual scope.

### Questions
1. A straightforward baseline could be to factorize latent actions into multiple groups, analogous to how DreamerV2 decomposes latent representations. How would FLAM compare to such a baseline?
2. The number of slots is known to be a critical hyperparameter in slot attention. Have the authors conducted sensitivity analyses on the number of slots or other key hyperparameters?
3. The proposed FLAM yields lower PSNR on real-world datasets. Could the authors explain why? Does the factorization of latent actions inherently limit representational capacity or generalization?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper is focused on the problem of video generation with implicitly discovered actions. The work proposed to improve the state of the art -Latent Action Models - by introducing FLMA that has explicit factors and allows the model to model multiple objects with their actions independently, rather then modelling the entire scene. This give FLAM additional representational power and allows for better video controllability.

### Strengths
* The idea of introducing this additional degree of freedom in modelling, and allow the model con condition different scene parts on different actions is intuitive and potentially effective
* The idea of using slot attention to implicitly learn object-action slots is neat
* I like that the authors get away with not having to explicitly couple slots at timestamp t and t+1, and still get acceptable results

### Weaknesses
* Performance: FLAM achieves inferior results to LAM in PSNR. However, given that the capacity of the representational power of the FLAM is higher, I would expect the opposite. Do the authors know why this could be happening? 
* Generalizability: Given that the visual tokenizer and FLMA itself have to be re-trained on every dataset sugges that the latent action space discovered is limited to just the actions in the dataset. This seems to go against the latest fashion where the goal is to train a single model that works everywhere (eg Genie)
* Visual Quality: For the visual tokenizer, why not use the pre-trained VQ-VAE used in diffusion models (for example SDXL [a])? It is clear from Figures 5, 6 that the tokenizer does not work well because the reconstruction show clear checkerboard and blurriness artifacts. Also, to quantify the reconstruction error, it would be helpful to see some metrics (like PSNR) on top of the visualizations.
* While the authors demonstrate the modelling ability of the model, it would be beneficial to use it for downstream applications, like planning. Such experiments are not present in the paper.

[a] Podell et al., SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis

### Questions
* I am not quite sure what the authors mean by saying that the training objective (5) is qualitatively different from that used in SlotAttention. It is exactly equivalent to the SaVI [a] objective, where the goal is to reconstruct the future frame from the latents. Perhaps this should be explained.

* The authors claim that the slots learn to bind to actions rather than objects. However, is this a desired property? Say, we have a car and a pedestrian turning right, and they may be captured into a single slot, yet, this may lead to “cross-talk” between features of those distinct objects (resulting in undesired artifacts), just because they bind to the same action. It feels like the slots should still be separated by “objectless” and only then bind to actions independently.

* I do not understand how the research question (1) is relevant to the main contribution of this paper - factorization in LAM. It seems to me that it is in fact orthogonal to the paper itself, and this result has been demonstrated before in the video prediction literature. This question would much rather be suitable as model ablation.

[a] Elsayed et al., SAVi++: Towards end-to-end object-centric learning from real-world videos

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a factorized way to learn a latent action world model. The paper leverages slot attention to extract features from pixels, and jointly trains the inverse dynamic model and forward dynamic model for world modeling. Experiments are conducted on game datasets and several real-world datasets.

### Strengths
- This paper proposes a factorized way for better learning of the latent action world model. It is well-motivated.
- The method is evaluated across diverse datasets, validating the efficacy of the proposed method.

### Weaknesses
- The proposed method is not novel, and does not seem to be sound to address the mentioned problem. In the introduction, this paper claims that a single latent action is challenging for tasks that have multiple entities. However, the proposed method proposes to learn multiple actions for each slot $k, k\sim [0,K-1]$, which is not that scalable and clever: 
   - 1) The hyperparameter $K$ should be defined and fixed in advance. It can not handle tasks that include diverse entities. Thus, it can not be scaled like Genie to learn from diverse datasets.
   - 2) The slot attention used in this paper can not guarantee that the learned slot is highly related to a specific **actionable** entity.
   - 3) Overall, this paper just integrates previous slot attention to the context of learning a latent action world model.
- Empirical results do not convince me. This paper does not compare with any baselines, except for the variants of the proposed framework. Related work such as [1] and [2] should be discussed and compared with.
- Only PSNR-based metrics are used for evaluation. It is not sufficient to test the video prediction performance. Metrics like FVD and SSIM should be added.
- For Table 2, the advantage of FLAM does not become more obvious as the number of entities increases.
- No qualitative demos are provided.

[1] Adaworld: Learning adaptable world models with latent actions, ICML 2025
[2] Prelar: World model pre-training with learnable action representation, ECCV 2024

### Questions
Please address my concerns as detailed in the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an approach that factorises the latent action space with a shared codebook to enable better modelling of videos containing multiple entities. The approach involves using slot attention to extract these factorised latent actions and decompose the action space in this manner to achieve linear instead of combinatorial actions. This enables generating diverse videos from the same initial frame.

### Strengths
- The paper is well-written and proposes a neat trick to tackle video generation in cases where there are multiple entities in the video through latent action models. This is an important problem to work on and factorisation of the action space in this manner provides an interpretable solution.
- The promise of the method in generating diverse videos is the biggest strength of the paper.
- The approach is described well and grounded in prior literature.

### Weaknesses
- The lack of a limitations section is a key weakness in the paper: increasing diversity comes at the expense of inconsistency in the video (example in the last point in this section). This is not something that this method addresses and is a limitation, but it would be helpful to the community to have a discussion on this subject so that future work can combine the benefits of factorisation with other work on consistency and fidelity in video generation.
- One critical section that does not appear in the main text is the analysis of the results comparing this approach with the baselines in terms of the benchmark and selection of hyperparameters. While lines 405-408 mention the fact that the number of factors need to be carefully chosen considering the complexity of the scenes, there is no follow-up analysis indicating how this choice is made, and what it implies in terms of the practicality of the method. If the hyperparameter $N$ has to be selected keeping in mind the number of entities appearing in every single video, that makes the proposed method impractical. In addition to this, there is no discussion or evidence on why the number of latent factors was determined to be the reason for the drop in performance using FLAM compared to LAM.
- In Tables 1 and 3, the results for OC+LAM seem to be quite unexpected, given that one key difference between OC+LAM and FLAM is the object-wise reconstruction loss. The additional reconstruction loss should ideally help with PSNR, if not $\Delta$PSNR, but we see consistently worse metrics for OC+LAM compared to both LAM and FLAM. This inconsistency seems quite unsatisfactory and further discussion is warranted, especially since this is an important baseline and closer to FLAM conceptually than LAM.
- In figure 4, it appears that of the two generated next states with the same initial frame, both next frames have certain impossible changes (the white car disappearing, an extra lane being added, etc.). I understand that diverse next frame generation would be much easier with a factored latent space but it is also equally important to discuss any guardrails that this approach has against unrealistic video generation, and if not, the limitations of this approach in that regard.

### Questions
- While extracting factored latents, is this number $N$ fixed? If not, is there an ablation over the number of slots and the impact this has on performance and diversity of generations? Note that this is not the same question as is being addressed in Table 2, if I understand correctly. Table 2 measures the impact of having multiple entities in the video on reconstruction metrics. My question is about the hyperparameter, not the environment parameters: in a video where the number of entities is unknown, how would you set the value $N$? Is there an ablation on this in the following scenarios: 1) $N$ < number of entities in the video, in which case the model might extract different factors in each frame, and 2) $N$ > number of entities in the video, in which case some of the factors might be noisy and potentially adversely impact performance.
- Just as table 2 contains PSNR results as the number of entities in the videos is scaled up, is there a similar table for $\Delta$PSNR?
- If we were to think intuitively, maximising diversity through factorisation would imply that reconstruction metrics might be lower. However, the results in table 1 show that the PSNR is higher than LAM and OC-LAM, which seems counter-intuitive. Could the authors give an explanation for why this is the case?
- A key fact the paper mentions is that this approach is different from an object-centric representation-based approach in that groups with similar dynamics can be clubbed because there is no reliance on superficial visual features. However, in the environments studied, there is no such environment that showcases this key difference. Are there results on such environments in which entities with different visual properties still behave the same way and are thus advantaged from the factored approach versus the object-centric approach?

### Soundness
2

### Presentation
3

### Contribution
3
