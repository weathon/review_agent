# Fast Direct: Query-Efficient  Online Black-box Guidance  for Diffusion-model Target Generation

- Decision: Accept (Poster)
- Scores: 8, 6, 8, 5

## Abstract
Guided diffusion-model generation is a promising direction for customizing the generation process of a pre-trained diffusion model to address specific downstream tasks. Existing guided diffusion models either rely on training the guidance model with pre-collected datasets or require the objective functions to be differentiable. However, for most real-world tasks, offline datasets are often unavailable, and their objective functions are often not differentiable, such as image generation with human preferences, molecular generation for drug discovery, and material design. Thus, we need an **online** algorithm capable of collecting data during runtime and supporting a **black-box** objective function. Moreover, the **query efficiency** of the algorithm is also critical because the objective evaluation of the query is often expensive in real-world scenarios. In this work, we propose a novel and simple algorithm, **Fast Direct**, for query-efficient online black-box target generation. Our Fast Direct builds a pseudo-target on the data manifold to update the noise sequence of the diffusion model with a universal direction, which is promising to perform query-efficient guided generation. Extensive experiments on twelve high-resolution ($\small {1024 \times 1024}$) image target generation tasks and six 3D-molecule target generation tasks show $\textbf{6}\times$ up to $\textbf{10}\times$ query efficiency improvement and $\textbf{11}\times$ up to $\textbf{44}\times$ query efficiency improvement, respectively.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces an inference-time black-box online guidance method for diffusion models. The authors first propose Guided Noise Sequence Optimization (GNSO), an efficient and robust technique for guiding a diffusion model sampling process toward a desired target sample. Building upon GNSO, they present Fast Direct, a novel approach that leverages pseudo-target samples derived from an online, black-box reward model as guidance targets. This approach enables guidance for any reward model, making it highly versatile. Compared to existing black-box online guidance methods, this approach offers the advantage of inference-time guidance, significantly reducing the number of reward model queries required to steer the sampling process toward the desired target.

### Strengths
1. This paper advances the efficiency of reward model queries in steering diffusion models with online black-box reward models. By employing an inference-time guidance method, it avoids the computationally intensive process of model fine-tuning, making it a more practical and efficient solution.
2. From a user perspective, inference-time guidance offers a more immediate and flexible approach to personalization compared to model fine-tuning. This makes the proposed methodology highly promising for real-world applications.
3. The paper proposes a novel approach to online black-box guidance, leveraging reward models to generate pseudo-targets for the GNSO, a technique originally used for same-dimensional target guidance.
4. The proposed methodology has been validated across various domains (image and molecule) and tasks (text-image alignment and drug discovery), demonstrating superior performance and efficiency compared to existing black-box online reward model-based guidance methods.
5. The paper presents its methodology clearly and concisely, making it easy to understand and follow.

### Weaknesses
1. The number of experimental tasks is insufficient, and the number of prompts used in the experiment is limited. A direct comparison with reinforcement learning methodologies for diffusion models, such as those in [1], would benefit from evaluating common tasks like compressibility, incompressibility, and aesthetic scores. Additionally, the manual selection of a small number of prompts by the author raises concerns about potential bias toward the proposed methodology.
2. The comparison with fine-tuning baselines may be unfair due to their intended purpose, which differs from the proposed inference-time guidance approach. However, given the absence of other black-box and online inference-time guidance methods, this comparison may be justifiable.
3. The methodology's optimization of noise during sampling is incompatible with the ODE sampler, a popular choice due to its deterministic nature and high performance [2]. This limitation could hinder the method's practical applicability.
4. In the text-image alignment experiments, the proposed method utilized the EDM sampler, while the baseline employed the DDIM sampler. This experimental setup may favor the proposed method, potentially compromising the fairness of the comparison.

[1] Black, Kevin, et al. "Training diffusion models with reinforcement learning." *ICLR* (2024).

[2] Karras, Tero, et al. "Elucidating the design space of diffusion-based generative models." *NeurIPS* (2022).

### Questions
1. How was the point in time considered complete for guidance iterations determined? Explicitly stating this criterion within the text would provide greater clarity.
2. Are there any comparison results with the inference time guidance method? Although online black box inference time guidance methods are limited, incorporating a comparison with existing methods, even if the performance of the presented methodology is lower, would offer valuable insights. Additionally, considering the paper's focus on tasks that do not allow backpropagation through the reward model, performing neural network-based reward model tasks (compressibility, incompressibility, and aesthetic scores) commonly used in diffusion model reinforcement learning would make it possible to compare with those methods.
3. What is the number of diffusion sampling steps used in each experiment? 
4. What are the specific advantages of guiding $\epsilon_k$ compared to directly guiding $x_k$?
5. The contribution summary in the introduction is not in a clear summary form. It is recommended to reorganize the detailed contributions into a separate paragraph and condense the summary into two or three concise lines.
6. Is the value of $||\epsilon_k||$ used in line 13 fixed as a value pre-calculated in line 3? Or is it a value that continuously changes as $\epsilon_k$ is updated by the loop in line 4?
7. Why was the experiment conducted on $x^{\star} − x_k$ instead of $x^{\star} − \hat{x}_k$ (predicted clean image) in Figure 1 (ablation)? Given that $x^\star$ represents a clean image, comparing it to the predicted clean image $\hat{x}_k$ at the same noise level appears to be a more intuitive approach for assessing the difference.

### Soundness
3

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
3

### Summary
The paper, titled "Fast Direct: Query-Efficient Online Black-Box Guidance for Diffusion-Model Target Generation," introduces an innovative approach to guiding diffusion models in black-box target generation tasks. It addresses scenarios where the objective functions are non-differentiable and where training datasets are unavailable, making it applicable to real-world applications such as drug discovery and human-preference-driven image generation. The authors propose a new method, Fast Direct, leveraging a Guided Noise Sequence Optimization (GNSO) technique. Through extensive experimentation on image and molecular generation tasks, the paper demonstrates that Fast Direct outperforms several baseline methods in query efficiency and generation quality.

### Strengths
Novelty and Relevance: The Fast Direct algorithm introduces a unique approach to guiding diffusion models without requiring differentiable objectives or offline data, making it relevant for applications with black-box or non-differentiable feedback.
Efficiency: The method shows impressive improvements in query efficiency across tasks, which is valuable in settings with expensive evaluations, such as in drug design.
Extensive Experiments: The paper includes rigorous experiments across various image and molecular generation tasks, showing consistent improvements over baseline methods in both query efficiency and task performance.
Generalizability: The proposed approach is designed to be flexible, supporting various black-box objectives and stochastic diffusion schedulers, which suggests it could be adapted to other domains and applications beyond those tested.

### Weaknesses
Limited Theoretical Justification: While empirical results are promising, the paper lacks a thorough theoretical analysis of the robustness and convergence properties of Fast Direct’s update strategy. Providing an in-depth theoretical foundation could strengthen the paper’s contributions and clarify the algorithm's potential limitations.
Baseline Comparisons: Although Fast Direct shows efficiency gains over DDPO, DPOK, and D3PO, a discussion is missing on why these methods were specifically chosen as baselines and how representative they are of the current state of the art in black-box guidance.
Scalability and Computational Costs: The paper could provide more detail on the computational overhead introduced by Fast Direct, especially for high-dimensional tasks like image generation. Scalability analysis in terms of computational cost and time efficiency would be beneficial, particularly as query budgets increase.
Lack of Ablation Studies: The paper does not explore the sensitivity of Fast Direct’s performance to its key hyperparameters, such as step size and batch size. Given the flexibility mentioned for the pseudo-target update, ablation studies could reveal the robustness of these parameters and how they influence the results across tasks.

### Questions
Why were DDPO, DPOK, and D3PO chosen as baselines? Are they the most relevant and competitive in black-box target generation? If there are other methods that could be appropriate, a brief comparison or justification for their exclusion could enhance the robustness of the evaluation.
Have you considered alternative pseudo-target selection strategies? While the paper presents GP and historical-optimal updates, discussing the potential effects of alternative methods (e.g., reinforcement learning-based or clustering-based approaches) might highlight the generalizability of Fast Direct.
Could you provide more insights into the parameter tuning process for Fast Direct? As query efficiency is a primary focus, understanding the sensitivity of Fast Direct to parameters like batch size and step size would clarify the required fine-tuning efforts.
Could you expand on the potential for scalability? Especially in high-dimensional tasks like high-resolution image generation, it would be helpful to know if there are constraints or limitations for Fast Direct in terms of computational efficiency.

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
3

### Summary
This work proposed a new online black-box algorithm, Fast Direct, to provide guidance for diffusion model without the need of pretrained guidance model or differentiable objective function. The important findings of the noise update direction in Guided Noise Sequence Optimization laid the foundation of the Fast Direct algorithm. The Fast Direct showed consistent query efficiency compared to 3 baselines.

### Strengths
1. This work used a data driven method for image generations, different from the three baselines in previous works focusing more on finetuning the diffusion model. This should be a good indicator of originality.

2. The proposed solution was a training free solution, which also led to a more query efficient method.

### Weaknesses
In case of the query being OOD, and the Gemini continues to dislike the generated image, the proposed solution might be less efficient compared to the methods involved finetuning the diffusion model.

### Questions
1. For comparison with three baselines, were there reward models finetuned with Gemini scores? Or were the Gimini scores directly used for finetuning the diffusion model? It was not really clear to me from my reading.

2. In the claim of selecting K instead of k for the update direction in Section 3.1 and the observation from Figure 1, I'd really love to also see the effect of choosing K/2 and K/4 to get a sense how the degree of "far away from the data manifold" correlates quality degradation. 

3. While aiming to deal with the situation of missing a guidance model, could you also add a comparison of the results using this method versus using a trained guidance model? That would make the story more convincing.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
1

### Summary
This paper proposed an LLM-scored/non-differentiable-scored pipeline for generating aligned outputs via diffusion models during the inference procedure. The paper utilizes Gaussian Process (GP) estimation or historical optimal to insert the guidance of the score. With two different generative tasks, the method outperforms several inference-time diffusion guidance methods with fewer iteration steps.

### Strengths
The method utilizes GP/historical optimal as the estimation of the guidance direction, illustrating clear improvements based on the score evaluation. The Bayesian-based optimization in the field of inference-time diffusion guidance is shown to be a useful tool for improving generation quality.

### Weaknesses
Lack of ablation studies of the hyper-parameters used in the proposed pipeline (step-size alpha, lambda). Additional experiments on the chosen hyper-parameters (especially for the step-size alpha since this is the guidance scale) are encouraged.

For both image and molecule generation tasks, the evaluation is based on the same metric used in the optimization, which may enrol incomplete or unfair comparisons to the other baselines. The paper could use the same LLM score during the inferences of the baselines (if it is available for them) or post additional evaluation metrics (such as the score from different LLMs and human evaluation scores) to strengthen the argument.

### Questions
In Figure 1 (also in the figures in the Appendix), the initial image of the proposed method is different to the other baselines, which seems to be a factor that influences the final generation performance. Can the authors clarify here?
(comment: several references could be listed here).

### Soundness
2

### Presentation
3

### Contribution
2
