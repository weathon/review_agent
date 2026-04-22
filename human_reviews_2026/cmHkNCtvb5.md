# Poivre: Self-Refining Visual Pointing with Reinforcement Learning

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 2

## Abstract
Visual pointing, which aims to localize a target by predicting its coordinates on an image, has emerged as an important problem in the realm of vision–language models (VLMs). Despite its broad applicability, recent benchmarks show that current VLMs still fall far behind human performance on this task. A key limitation is that VLMs are typically required to complete the pointing task in a single step, akin to asking humans to point at an object without seeing their own fingers. To address this issue, we propose a simple yet effective self-refining procedure: *Point, Visualize, then Refine* (Poivre). This procedure enables a VLM to first mark its estimated point, then iteratively refine the coordinates if necessary. Inspired by advances of reasoning models in the natural language domain, we employ reinforcement learning (RL) to incentivize this self-refining ability. For the RL training, we design a neat process reward that is not only empirically effective but also grounded in appealing theoretical properties. Our trained model, *Poivre-7B*, sets a new state of the art on Point-Bench, outperforming both proprietary models such as Gemini-2.5-Pro and strong open-source models such as Molmo-72B by over 3%. To support future research, we release our training and inference code, dataset, and the Poivre-7B checkpoint.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Poivre (Point, Visualize, then Refine), a self-refining framework for visual pointing tasks using reinforcement learning (RL). Instead of predicting target coordinates in a single step, Poivre enables a Vision-Language Model (VLM) to iteratively visualize its prediction on the image and refine the coordinates through multiple rounds. The key novelty lies in the PBRS-inspired process reward, which encourages improvement across refinement steps rather than focusing solely on the final outcomes.

### Strengths
1.	The paper identifies an important gap in current VLMs: the lack of self-refinement capability. The proposed “Point → Visualize → Refine” loop is simple yet conceptually appealing, like human correction behavior in pointing tasks.

2.	The pipeline and training process (rollout sampling, reward computation) are clearly described, and the paper includes implementation details that make the work easy to replicate.

3.	The adaptation of PBRS to the visual refinement setting is neat and theoretically grounded. It encourages incremental improvement across iterations, which aligns naturally with the proposed iterative reasoning loop.

### Weaknesses
1.	While the results are promising, the technical contribution remains incremental. The core ideas, visualizing model predictions and refining them iteratively, are intuitive, and the PBRS-based reward formulation is largely a straightforward adaptation of an established concept (Ng et al., 1999). The work demonstrates solid empirical gains but offers limited novelty.

2.	From Table 3, the improvements across multiple refinement iterations are marginal , which are all within 0.5%. This suggests that after training, the model’s first prediction during testing is already very accurate, making further refinement less impactful. This raises the question of whether the iterative training truly contributes to better refinement behavior, or if the model simply learns to predict correctly in the first step. It would be helpful for the authors to analyze the accuracy gap between the initial prediction and the first refinement during training, to clarify whether the model indeed benefits from iterative supervision.

3.	A similar multi-round refinement mechanism could be implemented in a general supervised setting, where the model iteratively predicts and updates based on the distance between the predicted point and the ground truth. Would such a setup achieve comparable effects without using GRPO? In other words, what is the fundamental difference between performing iterative supervised updates and adopting GRPO in this framework? The paper would benefit from clarifying why reinforcement learning is necessary here, and what unique advantage GRPO provides over conventional multi-step supervised training.

4.	The paper lacks detailed visualization or failure case analysis. It would be useful to show when refinement helps or hurts, and whether the model can meaningfully interpret the marker in complex scenes.

### Questions
1.	Have you tried any ablation where the model receives coordinate input directly (without visualization)?

2.	Does the model ever over-correct (move away from the correct target) in refinement steps?

3.	Could a supervised multi-step training baseline achieve similar improvements without RL?

4.	How does the model behave when the initial pointing is already accurate, does refinement still occur?

### Soundness
3

### Presentation
3

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
The authors present poivre, a VLM model capable of pointing to images given natural language queries. The model is trained with reinforcement learning, in order to encourage it to refine its predictions. Rather than assigning rewards only for the final state, the authors propose summing the rewards for how much the model improves its prediction at every step; they show this is equivalent to a formulation that emphasizes the first and last prediction the most. Through experiments, they show state-of-the-art results on pointing benchmarks, show that the model's predictions improve with an additional refinement step, and show the model's ability to extrapolate beyond its training regime to a longer horizon.

### Strengths
* the paper is well-motivated and clearly written. the figures are helpful towards understanding the method and showcasing qualitative results. 
* the composition of ideas seems original, however I am unable to assess this with high confidence as I am not an expert on the literature of visual pointing methods.
* the comparison to single turn RL is interesting! Even with one inference call, the multi-turn objective yields better performance.

### Weaknesses
* T is set to 2 during training, which somewhat undercuts the message that the model is refining its responses. It would be very interesting to see how this dynamic plays out over longer sequences, of T=3 and greater.
* As an extension to the above note, it is difficult to access the utility of the PBRS term with just T=2.
* The "exploration" experiments only try T=3 and cite this as evidence of exploration. Please expand the analysis to include longer horizons, T=4, 5, etc, so we can draw more conclusive evidence of extrapolation.
* It would be interesting to compare to additional self-baselines that rely less on visual reasoning and more on text reasoning; for example, just displaying the input image in the original prompt, and asking the model to make text-based prediction outputs, instead of overlaying the prediction onto the original image, as shown currently in Figure 2. This would help tease apart whether **visual** self-feedback is crucial.
* There are sparse citations to several papers in the field related to sampling-based approaches, such as PIVOT (Nasiriany et al.), MOKA (Liu et al.), and Set-of-Mark Prompting (Yang et al.)

### Questions
Please see the points raised in the "weaknesses" section.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Poivre, a reinforcement learning (RL) framework that enables vision-language models (VLMs) to iteratively refine their pointing predictions through a Point → Visualize → Refine (Poivre) procedure. Rather than predicting a pointing coordinate in one shot, the model visually marks its initial prediction on the image and refines it over multiple rounds. To incentivize this self-refinement, the authors design a potential-based reward shaping (PBRS) mechanism — a “process reward” that encourages progressive improvement across refinement steps, rather than optimizing only for the final outcome.
The resulting model, Poivre-7B, achieves state-of-the-art performance on Point-Bench (67.5% success rate, +2.7% over the best prior open model Molmo-72B) and generalizes to robotics benchmarks (where2place) without domain-specific fine-tuning. The paper includes ablations comparing outcome vs. process rewards, extrapolation to unseen refinement rounds, and a case study illustrating iterative correction.

### Strengths
- Clear conceptual motivation: Identifies the one-shot limitation in visual pointing and proposes a self-refining paradigm inspired by human feedback and test-time scaling.

- Principled reward design: The PBRS-inspired process reward is elegant, grounded in RL theory (potential shaping), and practically effective (+1.3% improvement).

- Strong empirical results: New state-of-the-art on Point-Bench, with additional validation on robotics transfer and extrapolation beyond training rounds.

### Weaknesses
- Limited conceptual novelty. The idea of self-refinement and iterative improvement is well explored in text reasoning (test-time scaling, visual CoT). The contribution lies mainly in applying this paradigm to visual pointing with straightforward reward shaping, rather than introducing fundamentally new algorithms or representations.
- Modest absolute gains. The improvement (≈2–3%) over strong baselines like VisionReasoner-7B or Molmo-72B, while consistent, is relatively small given the added RL complexity and compute cost (~$2000 training run).
- Reward design tuning unclear. The Gaussian-shaped outcome and PBRS coefficients (σ, γ) are selected heuristically; there’s no sensitivity or stability analysis.
- No human evaluation or qualitative failure analysis. The case studies are anecdotal and do not quantify where refinement helps or fails (e.g., cluttered vs. simple scenes).
- Dependence on GRPO. The method’s improvements might rely heavily on the GRPO setup from prior RL works (DeepSeek-R1, VisionReasoner), rather than intrinsic advantages of Poivre itself.
- No runtime or inference cost discussion. Iterative refinement increases inference time linearly with the number of turns; this tradeoff is not analyzed quantitatively.

### Questions
- How sensitive are results to the discount factor (γ) and σ in the process reward? Could these parameters significantly alter performance?

- How many refinement rounds before diminishing returns or overfitting occur (beyond T=3)?

- Can Poivre be extended to spatially dense tasks (e.g., segmentation, affordance maps), not just 2D coordinate regression?

- Did the authors attempt to combine supervised fine-tuning with RLHF/DPO before GRPO?

- What is the inference-time latency cost (in seconds per refinement step) compared to one-shot baselines?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Visual pointing, which aims to localize a target by predicting its coordinates on an image, has emerged as an important problem in the realm of vision–language models (VLMs). Despite its broad applicability, recent benchmarks show that current VLMs still fall far behind human performance on this task. A key limitation is that VLMs are typically required to complete the pointing task in a single step, akin to asking humans to point at an object without seeing their own fingers. To address this issue, we propose a simple yet effective self-refining procedure: Point, Visualize, then Refine (Poivre). This procedure enables a VLM to first mark its estimated point, then iteratively refine the coordinates if necessary. Inspired by advances of reasoning models in the natural language domain, we employ reinforcement learning (RL) to incentivize this self-refining ability. For the RL training, we design a neat process reward that is not only empirically effective but also grounded in appealing theoretical properties. Our trained model, Poivre-7B, sets a new state of the art on Point-Bench, outperforming both proprietary models such as Gemini-2.5-Pro and strong open-source models such as Molmo-72B by over 3%. To support future research, we release our training and inference code, dataset, and the Poivre-7B checkpoint.

### Strengths
1) Extensive experiments show the effectiveness of the proposed method.

### Weaknesses
1) The writing should be improved.
2) The motivation should be further highlighted.
3) The paper organization is poor.
4) Figures in the paper should be enhanced.

### Questions
1) The writing should be improved.
2) The motivation should be further highlighted.
3) The paper organization is poor.
4) Figures in the paper should be enhanced.

### Soundness
1

### Presentation
1

### Contribution
2
