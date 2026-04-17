# Envision the Future in Open-World Dynamic Tasks by a Hierarchical World Model with Residual Enhanced Foresight

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Interacting with proactive agents in open-world environments remains a core challenge for reinforcement learning, as other participants exhibit reciprocal and even adversarial behavior. Effective reasoning representations are crucial in such settings. Although language- or vision-grounded approaches have shown promise, most depend on large-scale pre-training and extensive domain-specific fine-tuning.
Drawing inspiration from the neuroscience phenomenon of active gaze control—where humans proactively direct gaze toward the predicted future location of a dynamic object (e.g., a cricket ball, blade, or oncoming vehicle) well before distinguishing visual features appear—we propose ResDreamer, a brain-inspired hierarchical world model that employs residually connected visual planning representations.
In ResDreamer, each higher-level layer learns on the reconstruction residuals of the layer below, enabling progressive capture of increasingly advanced world dynamics and the construction of a richer internal representation. Every layer incorporates augmented observations that include foresight images that are further modulated by top-down residual prediction signals. This mechanism yields highly informative, predictive, and knowledge-driven visual reasoning representations without external supervision.
Empirical results demonstrate that ResDreamer achieves higher sample efficiency, parameter efficiency, and scalability compared to state-of-the-art baselines, paving the way for more adaptive agents in open-ended, dynamic, and interactive environments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a hierarchical world model, ResDreamer, for dynamic tasks in the open world, such as battle scenarios in Minecraft. The core idea is that, in a hierarchical structure, the high-level model constructs a more comprehensive representation of the world by observing the visual reconstruction residuals from the low-level model and feeding back the modified “vision foresight” to it, thereby enhancing planning and decision-making capabilities. This method does not rely on the language model or external pre-training knowledge; it relies solely on self-supervised interactive learning, which offers good sampling efficiency, parameter efficiency, and scalability.

### Strengths
- Combining the idea of "predictive coding” in neuroscience with hierarchical Reinforcement Learning, the mechanism of using reconstruction residuals as inter-layer communication signals is proposed, which is relatively novel in the existing MBRL literature.
- ResDreamer's information flow design (high-level modeling, low-level errors, low-level receiving high-level corrected visual cues) mimics the prediction error processing mechanisms of the cerebral cortex (e.g., Rao & Ballard, 1999), not only improving model interpretability, but also improving the accuracy of prediction processing. It also provides a new idea for future brain-like RL.
- The results on five high-difficulty combat missions of MineDojo show that ResDreamer (especially the 100M × 2 configuration) outperforms the baseline in both sample efficiency and success rate, and even exhibits the only feasible solution ability in extremely difficult tasks such as confronting Shulkers.

### Weaknesses
- Although Figure 5 contains variants such as “only residual hints,” a key control group is missing: a hierarchical Dreamer who completely removes residual signals and uses only the original image + imagined trajectories. This makes it difficult to determine whether the performance improvement comes from the hierarchy itself or from the specific mechanism of residual enhancement.
- All experiments were conducted in Minedojo's combat subtasks with explicit friend-foe relationships and sparse reward structures. But the challenges of open-world RL also include long-term exploration, tool use, and multi-agent collaboration. ResDreamer's claim to be a “Universal World Model” is undermined by its lack of validation for navigation, construction, or hybrid missions.
- The paper acknowledges that fixed field length is the main limitation (Section 5), but the effect of different H (e.g., H = 4 vs H = 8) on performance is not explored in experiments. If H is too small, it will be invalid. If H is too large, it will be expensive and fuzzy. There is no sensitivity analysis or dynamic adjustment strategy for H, which undermines the method's practicability.

### Questions
- Can you provide a hierarchical Dreamer baseline without residual connections (i. e. only hierarchical but no residual communication)?
This will directly verify whether residual enhancement is a core contribution rather than a benefit of simple stratification.
- Is ResDreamer suitable for non-combat open missions? If the author has a preliminary experiment, even if it does not reach SOTA, please briefly describe it to support its broader applicability.
- The paper mentions "the distribution of lower residual observations shifts during the training process of lower layer models” (Section 4.2), but does not specify whether stabilization strategies such as gradient stopping or curriculum learning are employed. Please clarify the details of the training dynamics.

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
3

### Summary
Reinforcement learning algorithms have become increasingly important in the last years. One of the major challenges to deep reinforcement learning algorithms is sample efficiency, especially for long-term tasks. To address this issue an approach is learn a world model to train an agent entirely in imagination, eliminating the need for direct environment interaction during training. To improve exploration capabilities and sample efficiency the authors propose a hierarchical world model approach (ResDreamer) based on visual planning representations. Numerical experiments are performed (on 5 combat tasks in MineDojo) and results are compared DreamerV3 as a baseline.

### Strengths
S1 The considered problem is interesting and relevant.

S2 The paper is overall well-written.

### Weaknesses
W1 The paper does not provide analytical results.

W2 The evaluation is too small, contains only few handpicked(?) examples and is hence not fully convincing. Further, DreamerV3 is not clearly outperformed.

W3 Comparisons against other strong baselines beyond DreamerV3 are missing.

### Questions
Besides DreamerV3 you should also compare your results against further baselines, such as IRIS (Micheli et al., 2023), TWM (Robine et al., 2023) and Hieros (Mattes et al., 2024), which also uses a hierachical approach.

What are the required computational costs? Does the approach scale to larger problems? In which type of tasks the hierarchical structure particularly pays off?

Which hyper parameters are required? How are they automatically chosen/tuned?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
ResDreamer is a hierarchical world model for RL that passes reconstruction residuals upward so higher layers learn what lower layers failed to explain and provides feed back corrections to improve lower-layer foresight. Each layer consumes an “enhanced observation” consisting of raw pixels, a residual stack, and an imaginary hint built from open-loop predicted frames plus residual corrections from the next layer, with no gradients flowing between layers. The world model optimizes reconstruction of raw and residual signals and uses paired KL terms to balance representation and dynamics prediction. Training rolls out imagined trajectories to supply hints and to learn the actor-critic largely from imagination. Experiments on five MineDojo combat tasks show success-rate curves over 1M environment steps. Compared to DreamerV3, PTGM, and STEVE-1, ResDreamer shows improvements in success with 50Mx2 and 100Mx2 parameters, and performance improves when stacking to three layers. Ablations indicate the residual-corrected foresight is important. The paper concludes that the design scales with roughly linear inter-layer bandwidth and yields better sample and parameter efficiency for open-world dynamic tasks.

### Strengths
1. The idea is sound. Fairly novel algorithmic improvement over Dreamer-v3
2. The paper has managed to convert an idea from neuroscience, “predictive coding” or error based learning into a practical implementation for world models which is impressive.
3. The method is evaluated on a non-trivial benchmark (MineDojo) built on Minecraft style environment.
4. The authors also present a relevant ablation by increasing the hierarchy to 3 levels which confirms the scalability of the method

### Weaknesses
1. My biggest issue with the paper is the poor presentation of the idea and the lack of clarity in communication. I found it fairly difficult to understand the idea despite having extensively dealt with Residual connections, Dreamer and Predictive Coding myself. My concerns are that:

    a. Presentation of figures and mathematical expressions are average at best: In Figure 3, what do the red, green and gray dots represent? Do the background colors (red, green) indicate something? In Figure 2, what does the video symbol represent? Is it observation inputs or open-loop predictions? What do the dotted red circle represent? Perhaps a legend for all figures could help understand them quickly? In Figure 3 why are there blue frames after the yellow frames? I have asked a few other questions below.

    b. The language & communication is not upto the mark for a top-tier conference. I found it generally difficult to parse the paper. For example, o_imag^k is shown in Equation (2) but only defined and discussed before Equation (2). Few grammatical errors: “This has enabled the world model to advance towards the scalable ”ResNet era”. “In the field of visual MBRL, transformer, diffusion model are also known as effective world models”. “They have become the key to our hierarchical world model, enabling it to scale up with linearly increasing communication bandwidth.

2. I have several questions and issues regarding the algorithm, design choices and experiments. I would like the authors to clarify them clearly:

    a. Residual stacking: You state both o_res^k and o_imag^k have shape (h, w, 3*H), but Eq. (3) defines o_res^k only at time t. Do you stack residuals over t+1:t+H as well, or is the stated shape incorrect? Specify exact tensor shapes per layer and per time step.

    b. Normalization operator: Precisely define Norm_k(). Which axes are used for mean/variance. How is the EMA used? Provide the exact formula/expression.

    c. For Eq. (4), please detail the policy and temperature used for open-loop rollouts, the action horizon H per layer, and whether hints are recomputed every step or cached. Confirm that residuals from the upper layer come from an online or target decoder.

    d. You say no gradients pass between layers. Specify the exact detach points for o_raw, o_res^k, and o_imag^k via either the equations or one of the figures.

    e. Compute budget: Please report the number of imaginations per environment step per layer, total wall-clock hours, updates, etc for each method so sample efficiency isn’t confounded by extra compute.

    f. How many random seeds were used per task and algorithm? How are these results aggregated, and how is the confidence interval computed? It seems to me like the confidence intervals are showing a deviation from running average instead of aggregated numbers from multiple seeds. If that is the case, please provide the results for multiple seeds (atleast 3) since RL algorithms are noisy and single seed experiments are unreliable.

    g. Why were the training runs in Figure 4 reported only upto 1M steps? Especially for Combat shulker, Combat wolf and Combat skeleton? It seems like these experiments could have been run for longer, for a more clear idea of the success dyanmics.

3. Why were transformer based world models like IRIS [1] or pre-trained visual encoder based models like Dino-WM [2] not compared? The comparison would help guage the position of ResDreamer over other methods.

4. “However, as far as we know, there is no MBRL method that naturally builds a hierarchical representation learning architecture based on the reconstruction residuals of sensory signals.” I believe there are several works that have attempted this kind of hierarchical representations [3][4][5][6].

### Questions
Asked above.

Conclusion: I was really rooting for the authors when I began reading the paper and found the ideas interesting. But I have so many questions regarding the mechanics, design choices and experiments that I strongly feel it is premature to accept it without carefully assessing the contributions correctly. I would encourage the authors to clarify them during the discussion phase.

*References*

*[1] Micheli, V., Alonso, E., & Fleuret, F. (2023). Transformers are sample-efficient world models (IRIS). In Proceedings of the Eleventh International Conference on Learning Representations (ICLR 2023)*

*[2] Zhou, G., Pan, H., LeCun, Y., & Pinto, L. (2024). DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning. arXiv preprint arXiv:2411.04983*

*[3] Rao, R. P. N., et. al. (2024). Active predictive coding: A unifying neural model for active perception, compositional learning, and hierarchical planning. Neural Computation.*

*[4] Mounir, R., Vijayaraghavan, S., & Sarkar, S. (2023). STREAMER: Streaming representation learning and event segmentation in a hierarchical manner. In Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*

*[5] Hansen, N., et al. (2025). Hierarchical world models as visual whole-body humanoid controllers. arXiv preprint arXiv:2501.01234*

*[6] Mattes, J., et al. (2023). Hieros: Hierarchical imagination on structured state-space sequence world models*

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces ResDreamer, a hierarchical world model that connects multiple Dreamer-style world models via residual signals and imaginary hints. Each layer processes enhanced visual observations that include residual errors from the lower level and imagined future frames. The approach is motivated by predictive coding and aims to build scalable, biologically inspired world models that improve efficiency in complex, open-ended RL tasks such as combat scenarios in MineDojo.

### Strengths
* The proposed hierarchical residual connection mechanism is novel and grounded in predictive coding theory.
* The experimental results suggest improved sample efficiency and task success rates compared to DreamerV3 and other baselines.
* The document has a well-researched and -written literature review in model-based RL and hierarchical representation learning.

### Weaknesses
1. While the concept of “imaginary hints” — visual foresight reconstructed from imagined trajectories — is intriguing, it is not well explained how this component quantitatively drives the observed performance gains.
2. How do the hints differ from the standard Dreamer imagination rollout?
3. Why does adding residual-corrected foresight lead to such large improvements?
4. What is the exact ablation result isolating the effect of imaginary hints versus residual feedback?
5. The text claims that imaginary hints correspond to “dynamic CNN kernels” and “gaze control,” but the analysis in Figure 5 (“Only residual hints”) is the only direct test of this component, yet the explanation of why it performs better is purely qualitative. A more systematic analysis (e.g., measuring predictive accuracy of imagined trajectories or visualizing hint quality over time) would help substantiate the claim.
6. Since the model introduces several interacting ideas (hierarchical residuals, imaginary hints, layer conditioning), it’s difficult to disentangle their contributions from each other.

### Questions
1. Could you clarify how the imaginary hint differs from Dreamer’s latent imagination in implementation?
2. Is the imaginary hint computed online or only during training?
3. Did you experiment with removing the residual correction to see if the improvement persists?

### Soundness
3

### Presentation
3

### Contribution
3
