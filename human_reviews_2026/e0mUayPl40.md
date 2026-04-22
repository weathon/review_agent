# Sparse World Models: Visual World Modeling with Sparse Representations

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 4, 0, 6

## Abstract
World models promise efficient prediction, imagination, and planning by operating in a compact latent space, yet prevailing approaches inherit \emph{dense, entangled} visual features from large pretrained encoders. Such latents conflate unrelated factors and contain redundant dimensions, undermining intervention fidelity, inflating planning cost, and reducing robustness to distribution shifts. We propose \textbf{Sparse World Models (SWMs)}, which learn and plan \emph{entirely in a sparse feature space}. SWMs obtain selectively active codes by training a sparse autoencoder (SAE) to translate dense vision embeddings into an overcomplete but \emph{sparse} vocabulary, and then use these codes for state estimation, dynamics learning, and action optimization. By aligning units to meaningful factors, SWMs enable targeted interventions and attribution, and shrink the optimization search space. We further introduce an evaluation suite that probes feature capacity and links sparsity to planning outcomes. Across studies, sparse representations reduce polysemanticity and maintain planning performance while offering better efficiency and interpretability.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose Sparse World Models (SWM), which make use of a Sparse Autoencoder to disentangle semantically and visually monosemantic features from the dense embeddings of pretrained vision models (like Dino). The authors then investigate whether sparse feature spaces are beneficial for planning. They train an action conditioned latent dynamics model to predict future sparse features from the current sparse features and an action. These sparse features sometimes slightly decrease planning performance, sometimes increase it, but evidence is inconclusive if it actually improves performance in general.

### Strengths
* Thorough evaluation of an interesting idea on several tasks.
* Reasonable hypothesis that sparsity would help planning
* Might have potential for speeding up planning algorithms themselves, but it's a little unclear how this works.

### Weaknesses
* While the sparse representations do not decrease performance substantially, there doesn't seem to be much evidence for them improving the performance for downstream planning tasks.
* The added SAE pretraining that one needs to do between selecting an encoder and performing planning is probably not worth it given that planning in the SAEs latent space doesn't reliably improve performance (perhaps except for the GD method, but this method has the weakest performance overall).
* The benefit of using SAEs is that they improve interpretability, but the authors do not attempt to interpret the features that the SAEs learn here. This would justify using them, if interpretability is indeed the goal. However, if it is, one could just plan in the dense space and train SAEs for interpretability separately.


If the authors can showcase the benefits of planning in a sparse feature space better, and explain how planning is sped up, I would be willing to increase my score.

### Questions
* Action decoding setup is unclear. Is the decoder trained to decode what action the planner executed? This seems circular, since the planner operates in the representation space of the SAE (e.g. minimizes goal distance in the learned sparse representation space).
* How can SWM have lower token_ops count because of its sparsity? In my understanding, the SAE representation is up-projected, and then sparsified. But importantly, the dimensionality of the SAE representation is the up-projected size, not k. I'd appreciate it if the authors could address this and explain how exactly using the SAE representations lead to speed-ups and lower token_ops.
* If TDMPC gets 0 success rate because it doesn't get rewards, why is it included in the comparison? Is this comparison actually informative?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the challenges of inefficient planning, poor interpretability, and a lack of robustness inherent in existing visual World Models, which stem from their reliance on dense, entangled, and redundant visual representations. The paper introduces Sparse World Models (SWMs), a framework that employs a Sparse Autoencoder (SAE) to transform dense features from pretrained encoders (e.g., DINOv2) into an overcomplete yet highly sparse set of activations. Critically, SWMs conduct all subsequent dynamics learning and online planning entirely within this sparse feature space. Experimental results demonstrate that this approach not only maintains (and in some cases surpasses) planning success rates comparable to dense-feature baselines but also significantly reduces feature polysemanticity, enhances attribution accuracy, and substantially improves computational efficiency, accomplishing planning tasks with lower computational overhead and in less time.

### Strengths
The paper presents a clear motivation and well-defined problem statement. The methodology and experimental setup are described with clarity, and the overall presentation is concise and easy to follow.

### Weaknesses
1. The proposed method primarily combines existing sparse autoencoder techniques with established world model architectures, resulting in an incremental contribution rather than a fundamentally new approach. A major concern is that the paper does not convincingly justify the novelty or benefits of integrating sparse representations with existing world models. Table 2 shows no clear performance advantage of SWM over DINO-WM, and Figure 3 indicates that both DINO (without sparse representations) and SWM capture similar dynamics heat map, casting doubt on the necessity and effectiveness of the sparse representations.

2. The paper lacks a thorough discussion and empirical comparison with prior work on robustness under distribution shifts and task-relevant representation learning, particularly recent bisimulation-based approaches [1, 2]. Moreover, the claim that sparse representations improve robustness to distribution shift is not supported by targeted experiments. To substantiate this claim, the authors should include controlled experiments with explicit distribution shifts (e.g., background or visual variations) as in [1,2], and incorporate bisimulation-based world models as additional baselines. These methods have demonstrated strong robustness to distributional changes and effectiveness in extracting task-relevant representations, and their inclusion would provide a more comprehensive and convincing evaluation.

3. The experimental evaluation is limited in both scope and rigor. The tasks (Maze, PushT, Wall) are overly simple and do not sufficiently demonstrate generality, and the paper does not report whether random seeds were used (Tables 1–4), which is important for assessing stability and reproducibility. In addition, there are issues with the experimental setup in Figure 4: using the time to reach a high-performance threshold as a measure of planning efficiency is not reasonable, as the left plot shows the two curves converging roughly simultaneously. Furthermore, the right plot compares SWM-Topk with a latent dimension of 128, but a setting of 384×4 would be more appropriate; the original 128 is unfair for DINOv2.

4. Minor writing and presentation issue: “need reference here” in line 628

[1] Sun R, Zang H, Li X, et al. Learning latent dynamic robust representations for world models. ICML, 2024.

[2] Shimizu Y, Tomizuka M. Bisimulation metric for model predictive control. ICLR, 2025.

### Questions
1. In Eq.(8) is the choice of $z_g$ appropriate? The paper mentions using image observations as the goal representation; however, an image goal is not necessarily unique. For example, in the Wall task, any observation where the ball has already passed the wall could serve as a valid goal observation.

2. In line 478, the authors claim that “Across Maze and Wall, both models recover goal-directed paths.” However, from Figure 5, in the Maze and Wall tasks, neither DINO-WM nor the proposed SWM actually reach the goal. Could the authors clarify this claim?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper introduces Sparse World Models (SWMs), which learn, predict, and plan entirely within a sparse feature space. The method uses a Sparse Autoencoder (SAE) to translate dense visual embeddings from a frozen backbone into sparse codes. Experiments demonstrate that this approach achieves competitive planning success rates while significantly improving computational efficiency and feature interpretability compared to traditional dense-feature world models.

### Strengths
- This appears to be an early example of applying the sparse autoencoder concept to a visual world model.
- Disentanglement in world models is a critical problem.
- Computational efficiency is a critical issue in planning, especially in real-world problems such as robotics, and the model addresses this by maintaining planning success rates that are competitive with the strong, dense DINO-WM baseline, despite employing a highly sparse representation (e.g., 128 active dimensions vs. 384 dense ones).

### Weaknesses
While adapting the sparse autoencoder concept to the world model domain, this paper appears to be a mere application without clearly demonstrating the inherent advantages of sparse representation within the context of world models. Crucially, the evidence that sparse representation actually benefits planning is weak. Overall, the contribution of this work is perceived to be severely lacking, the experiments are often unclear and inappropriate, and they fail to adequately support the core claims of the paper. Details follow:

- The most significant concern is that this study heavily relies on the published paper and open-source code of the pre-existing DINO-WM
(https://arxiv.org/abs/2411.04983). It utilized only three of the datasets (Maze, Wall, Push-T) made public by the original authors, and completely omitted evaluation on the more complex datasets, Granular and Rope, which were also released by them. The baseline performance values reported in Table 3 and Table 4 appear to be directly adopted from the same paper. Also, various planning methods (MPC, CEM, GD) are presented in the DINO-WM paper and codebase. While these points may not be problematic individually, taken together, these choices make the scope of the contribution unclear.

- The current evaluation is significantly limited, relying only on three simple datasets. This narrow scope omits more complex environments like Granular and Rope (released by the DINO-WM authors) or other simulations such as LIBERO (https://arxiv.org/abs/2306.03310). Also in Figure 5, the showcased rollouts appear relatively uncomplicated, reaching the goal looks near-trivial in the depicted instances. As a result, it is difficult to assess from the current evidence whether the sparse representation scales and remains effective in more complex settings.

- The baseline comparison seems inappropriate. The authors’ primary claim is to reduce planning time using SWM while minimizing performance degradation; however, the comparison with the baselines in Table 3 does not feel appropriate for this purpose. Comparing DINO-WM and SWM against DreamerV3 and TD-MPC primarily serves to demonstrate the superiority of the visual token-based world model itself, rather than highlighting the advantage of the sparse representation. I believe the authors should have, at minimum, included comparisons against methods that enhance efficiency in Transformers, such as token reduction techniques (e.g., https://arxiv.org/abs/2404.00680, https://arxiv.org/abs/2409.11923, https://arxiv.org/abs/2506.01392), or other methods based on disentangled features, such as beta-VAE or object-centric representations (e.g., https://arxiv.org/abs/2503.08751, https://arxiv.org/abs/2209.14860,  https://arxiv.org/abs/2502.07600).

- The authors begin the paper with the premise that "DINO's features are polysemantic" but there is no analytical evidence (e.g., analyzing whether a specific DINO unit responds to both object color and background texture) to experimentally support this assumption. Could the authors demonstrate the polysemancity of pretrained vision encoders like DINO?

- The paper implies that clean (monosemantic) features are beneficial for the predictive model or the transition model, but the link is not firmly established. A clean state representation does not necessarily imply that the state transition is easy to predict. The experiments demonstrating this connection are insufficient. Also, Table 3 show that SWM is often outperformed by the dense DINO-WM baseline. If the
authors intend to claim that sparse representation enables faster planning by reducing dimensionality while simultaneously making planning more effective through mono-semanticity, then the experimental results appear to contradict this argument.

- The paper claims that sparse representation increases "Intervention Fidelity". This leads to the expectation that changing only the "object position" unit will truly result in a change only in the object's position. The paper failed to demonstrate a true "intervention experiment", such as "when the object position unit was forcibly changed, only the object position in the next predicted frame actually changed". This core claim remains essentially unverified.

- All analysis is carried out purely in simulation, lacking any reference to real-world images or supporting evidence that the results would generalize to real-world scenarios. While conducting planning experiments in real-world robotics to demonstrate improved planning efficiency with comparable task performance would be highly valuable, it is difficult to expect such validation from the current version of
the paper. Can the authors show some successful cases about test-time optimization without an explicit policy for more complex and long-horizon tasks?

- Since the runtime evaluation is conducted only on the Maze environment, a more comprehensive assessment across diverse environments and varying planning methods and difficulty levels is required. Also, in Appendix A.1, the authors vaguely state that planner settings (horizon, iterations, and episode budgets) "follow the standard protocol used across baselines," but fail to specify the concrete values anywhere. This omission of essential hyperparameters severely compromises reproducibility. Since the planning horizon and sample count significantly impact both performance and runtime, readers cannot verify the results or judge the fairness of the DINO-WM comparison without this information.

- For Maze and Wall tasks, goal and action probing seems infeasible because only single frame is given and the goal is not explicitly visible in the image. The explanation for how a single image can be used to infer the actions and determine the required goal for that image remains unclear.

- The paper does not specify the additional pretraining cost for the SAE, a cost which is not clearly justified by the benefits. Crucially, the SWM exhibits lower planning success than the DINO-WM baseline, suggesting performance degradation due to filtering useful information. Thus, the study presents a typical 'Trade-off' (cost/accuracy for speed) without establishing the necessary clear superiority of SWM to warrant the added complexity and cost of SAE training.

- Since this study was only applied to the DINOv2 ViT-S model, it raises doubts about its scalability to other ViT models. Also, as ViT size increases and feature dimensionality grows, the sparsity hyperparameter k will likely require retuning. Additional experiments are needed to assess this.

- Furthermore, as the sparsity control parameter k would likely increase in complex environments, the paper should include planning success results across varying k; probing alone is insufficient to assess the effect of varying k on planning.

### Questions
- Why there is N/A in Table 4?
- There are some typos: e.g., in page 4, spare autoencoder.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Reinforcement learning algorithms have become increasingly important in the last years. One of the major challenges to deep reinforcement learning algorithms is sample efficiency, especially for long-term tasks. To address this issue an approach is learn a world model to train an agent entirely in imagination, eliminating the need for direct environment interaction during training. To improve efficiency the authors propose to use so-called sparse world models with a sparse feature space. Numerical experiments are performed (on Maze, PushT, Wall) and results are compared DinoV2 and DreamerV3 as a baseline.

### Strengths
* The considered problem is interesting and relevant.
* The approach to improve training efficiency in world models by means of dimensionality reduction via sparse autoencoders has potential.
* The paper is overall well-written and the evaluation is promising.

### Weaknesses
* The paper does not provide analytical results.
* The evaluation contains only few examples (Maze, PushT, Wall) and could be improved by considering more complex problems.
* The current model is requires the DINOv2 backbone.

### Questions
1. Is the approach tailored to be used with DinoV2 (“We pretrain the SAE on fixed DINOv2 ViT-S/14 patch tokens (384-d) extracted from 196×196 images“, ) or is the approach, in principle, of a general kind?
2. Does the proposed model also work for larger problems?
3. What is the speed up in wall clock time in training when using SAE?
4. Besides DreamerV3 (and TDMPC) it would be interesting to compare results against further baselines, such as IRIS (Micheli et al., 2023), TWM (Robine et al., 2023), or Hieros (Mattes et al., 2024).
5. Which hyper parameters are required? How should the number Top-k be automatically chosen in various applications?

### Soundness
3

### Presentation
3

### Contribution
3
