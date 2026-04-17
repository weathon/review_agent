# Steering Language Models with Weight Arithmetic

- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Providing high-quality feedback to Large Language Models (LLMs) on a diverse training distribution can be difficult and expensive, and providing feedback only on a narrow distribution can result in unintended generalizations. To better leverage narrow training data, we propose *contrastive weight steering*, a simple post-training method that edits the model parameters using weight arithmetic. We isolate a behavior direction in weight-space by subtracting the weight deltas from two small fine-tunes---one that induces the desired behavior and another that induces its opposite---and then add or remove this direction to modify the model's weights. We apply this technique to mitigate sycophancy and induce misalignment, and find that weight steering often generalizes further than activation steering, achieving stronger out-of-distribution behavioral control before degrading general capabilities. We also show that, in the context of task-specific fine-tuning, weight steering can partially mitigate undesired behavioral drift: it can reduce sycophancy and under-refusals introduced during fine-tuning while preserving task performance gains. Finally, we provide preliminary evidence that emergent misalignment can be detected by measuring the similarity between fine-tuning updates and an "evil" weight direction, suggesting that it may be possible to monitor the evolution of weights during training and detect rare misaligned behaviors that never manifest during training or evaluations.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces contrastive weight steering for behavior control in LLMs: obtain two small LoRA finetunes that elicit opposite behaviors (positive vs. negative), form a behavior weight vector $w_b=\tau^+-\tau^-$, and steer the base model by adding $kw_b$ to its weights. The method is simple, data-light, and cheap to apply. Experiments show reductions in sycophancy and misalignment with stronger out-of-distribution (OOD) generalization than activation-based steering, and a practical use case for monitoring undesirable drift. Steered models often preserve general capability better than activation steering at comparable control strength.

### Strengths
- The method is simple & practical: A minimal, reproducible recipe (two tiny LoRA runs + one vector addition) with low data/compute cost.
- OOD robustness vs. activation steering: Under matched data and control strength, weight steering more often preserves base accuracy while shifting behavior.
- Useful byproducts: The learned behavior direction doubles as a monitoring signal for emergent misalignment.

### Weaknesses
- Limited novelty: Methodologically close to task arithmetic/task vectors; the contrastive construction is natural but incremental.
- Baselines could be stronger: Most comparisons are to activation steering or prompts. To contextualize tradeoffs, it would help to include training-heavier baselines (e.g., larger SFT/RLHF slices) on the same behaviors and report cost-adjusted outcomes.

### Questions
- Could the authors explain why the double-side difference $\tau^+-\tau^- $ is necessary?: What do we lose by using a single-sided vector, e.g. $ \theta_{\text{positive}}-\theta_0 $? 
- On dataset construction robustness: Could mismatch between positive/negative datasets (style, register, spurious topics) contaminate $w_b$? Is there any robust way to ensure we isolate the intended behavior?
- Data & hyperparameter scaling: How does performance/retention vary with number of examples used to fit $ \tau^\pm $, and hyperparameters such as LoRA rank?

### Soundness
4

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
This paper introduces “contrastive weight steering”, a post-training method for modifying high-level behaviors in large language models (LLMs) by arithmetically combining weight differentials from small-scale, purposefully narrow fine-tunings. By subtracting weight changes induced by opposite behaviors (positive vs. negative fine-tunes), the method isolates weight-space directions corresponding to target traits, such as sycophancy, refusal, or “evilness”. The resulting vector is then added or subtracted from the base (or recently task-fine-tuned) model’s parameters. The authors benchmark weight steering against activation steering, an established, layer-specific behavioral modulation technique, and traditional data augmentation through fine-tuning. The work empirically demonstrates that weight steering often generalizes steering effects better to out-of-distribution queries before harming model competence, and can also serve as a tool for post-hoc behavioral monitoring and drift detection by inspecting task vector similarities.

### Strengths
1. Methodology: the approach is conceptually simple yet effective, contrastive construction of behavioral directions in weight space, building directly upon and extending well-known task-vector strategies (Ilharco et al., 2023).

2. Clarity with Explicit Comparison: The paper systematically compares contrastive weight steering to activation steering and joint fine-tuning, using diverse alignment-relevant behaviors (sycophancy, refusal, evilness) across several standard LLM architectures (Qwen2.5-7B, Llama-2-7B, etc.).

3. Emphasis on Out-of-Distribution (OOD) Evaluation: Unlike much of the prior literature, the paper pays substantial attention to OOD generalization: for example, by constructing steering vectors with one query distribution and evaluating behavioral shifts on distinct domains, multiple-choice setups, or open-ended generations.

4. Visualizations and Interpretability: Several key figures, such as Figure 2 (sycophancy steering tradeoff), Figure 5 (Evil rate vs. accuracy curve), and Figure 9 (cosine similarity heatmap for weight-behavior vectors), provide clear, interpretable evidence of both the strengths and blind spots of the method.

### Weaknesses
1. Oversimplified Method and Ambiguity in Implementation: The proposed method appears conceptually simple, relying on subtracting the negative direction from the positive one and interpolating between them to obtain the final value. However, constructing an appropriate negative direction is often non-trivial and may not be unique (see my Q1), potentially introducing ambiguity in the optimization process. Furthermore, the selection of an appropriate interpolation coefficient k plays a critical role in determining the model’s stability and performance. A more detailed discussion or empirical justification for how the negative direction is defined and how k is chosen would significantly strengthen the technical soundness of the paper.

2. Lack of Insights into Weight vs. Activation Steering: The paper does not provide sufficient explanation or theoretical insight into why weight steering yields better performance than activation steering. Ideally, modifying the weights should be, in principle, equivalent to applying an appropriate transformation to the activations. However, the observed performance difference between the two approaches suggests that there may be underlying mechanisms not yet well understood. It would be valuable to clarify why weight steering behaves differently and appears to offer superior results, with proper insights.

3. Lack of Clarity in Presentation: The paper’s presentation lacks clarity, with several important implementation details either missing or insufficiently explained. For instance, it is unclear how “sycophancy” is formally defined or measured within the experimental setup. Additionally, the paper does not specify whether weight steering is applied to all layers of the model or restricted to certain fixed layers. Providing clearer definitions and methodological descriptions would greatly improve the readability, reproducibility, and overall credibility of the work.

4. Potentially Unfair Evaluation:The evaluation setup may not provide a fair comparison between methods. In several cases, activation steering results in both lower non-sycophancy scores and reduced accuracy (e.g., Figure 3). This outcome is counterintuitive, as both metrics deteriorate under the same adjustment. The authors are encouraged to further investigate this behavior and clarify whether it stems from differences in experimental settings, parameter tuning, or inherent limitations of activation steering. A fairer and more controlled comparison would strengthen the validity of the reported results.

### Questions
1. [Key Issue] The reviewer would like to better understand how the negative direction is determined in the proposed method. In general, while a positive direction can often be uniquely defined, there may exist infinitely many possible negative directions. How is a proper negative direction selected in practice? Moreover, how sensitive is the performance of weight steering to this choice? A more detailed explanation or empirical analysis on this aspect would help clarify the robustness and consistency of the proposed approach.

2. [Key Issue] Following the Q1, the reviewer has concerns on the reported results. Considering the Figure 2 as an example. In the left subfigure, it appears that positive activation steering effectively reduces the non-sycophancy score, whereas in the right subfigure, it shows very limited ability to increase the non-sycophancy score. Could the authors clarify why this asymmetry occurs? Is it potentially due to the selection of an inappropriate negative steering direction, given that multiple negative directions could exist? A more detailed explanation or visualization would help elucidate the underlying cause of this discrepancy.

3. A similar issue can be observed in Figure 3. It appears that applying activation steering leads to a decrease in performance while simultaneously decreasing the non-sycophancy score. This behavior seems counterintuitive, as both metrics deteriorate under this adjustment. Could the authors provide further clarification or analysis on why this occurs? It would be helpful to understand whether this effect arises from the steering method itself, the choice of direction, or potential interactions between the two objectives.

4. Unclear Selection of Interpolation Parameter k:The paper does not provide sufficient details on how the interpolation parameter 
k is selected. Although the authors illustrate the interpolation behavior using color notations, a more concrete description of the actual k values used is necessary for clarity and reproducibility. Furthermore, an ablation study examining the sensitivity of performance to different k values would offer valuable insight into how this parameter influences the effectiveness and stability of the proposed method.

5. In Line 162, the authors mention that they “select the best-performing layer” for activation steering. Could the authors clarify whether the same layer selection strategy is applied for weight steering? Specifically, do you use the same layer identified for activation steering, or is weight steering performed across all layers?

6. Section 3 appears disproportionately brief relative to its importance. The reviewer suggests that the authors expand this section by including key definitions and methodological clarifications. For instance, it would be helpful to formally define how activation steering is formulated and how sycophancy is evaluated. Shifting or adding such explanations to this section would improve the logical flow and make the paper more self-contained and accessible to readers.

7. Some Typographical Error. For example, in Line 309, there is an inappropriate use of quotation marks around the word “evil”. Also, the caption of Figure 2 appears to contain an error. It currently reads “Weight steering is more effective in controlling sycophancy than weight steering.” This seems to be a typographical mistake—perhaps the authors intended to compare weight steering with activation steering.

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces Contrastive Weight Steering (CWS), a post training technique to modify large language model (LLM) behaviors through weight arithmetic rather than activation interventions. The method isolates behavioral “directions” in weight space by contrasting two fine-tunes one inducing a desired behavior (e.g., truthfulness) and another inducing its opposite (e.g., sycophancy). The resulting weight vector is then added to or subtracted from the base model to steer behavior. Experiments across sycophancy, evilness, and refusal show that weight steering achieves better out-of-distribution generalization than activation steering, maintains task performance, and can even mitigate unwanted drift introduced during downstream fine tuning. Additionally, the paper presents preliminary evidence that cosine similarity in weight space may help detect emergent misalignment during training

### Strengths
1. The contrastive weight-space formulation is both simple and effective.
2. Demonstrated across behaviors (sycophancy, evilness, refusal) and architectures.
3. Outperforms activation steering on unseen distributions.
4. Shows that CWS can correct sycophancy induced during task-specific fine-tuning without harming core skills.
5. Provides early evidence that misalignment can be detected by tracking weight-space similarities.
6. Hyperparameters, datasets, and prompts are clearly documented

### Weaknesses
1. The paper does not formally analyze why certain weight directions correspond to behavioral dimensions.
2. Steering coefficient 𝑘, k is tuned manually; adaptive or learning-based selection could improve reliability.
3. Experiments use models up to 7 B parameters larger frontier models (e.g., 70 B+) could test scalability.
4. It remains unclear how interpretable or modular these weight directions are across unrelated behaviors.
5.  The “evil vector” similarity experiment is promising but would benefit from quantitative validation over training trajectories.

### Questions
1. How stable are weight-space directions across model sizes can a vector learned on a 1.5 B model transfer to a 7 B model?
2. How does the choice of fine-tuning layers (e.g., LoRA rank and target modules) influence steering effectiveness?
3. Could monitoring via cosine similarity be used in real-time to stop training before misalignment occurs?

### Soundness
3

### Presentation
3

### Contribution
3
