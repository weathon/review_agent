# Learn to Merge: Meta-Learning for Adaptive Multi-Task Model Merging

- Decision: Reject
- Scores: 6, 6, 2, 4

## Abstract
Model merging in the pretrain-finetune paradigm has proven effective by combining multiple finetuned models into one with multi-task capabilities. Recent merging methods aim to boost merged models’ performance through strategies such as mitigating conflicts, adding trainable modules, and incorporating task-specific components. In most methods, the parameter merging procedure is based on Task Arithmetic, a widely used technique that creates task vectors from each finetuned model and linearly combines them with coefficients into consolidated model parameters. Except for studies specifically focusing on the merging coefficients, many other methods treat them as hand-tuned hyperparameters. However, the merging coefficients, which govern the entire merging process, including the subsequent module training, are empirically crucial for achieving optimal performance and tradeoff across tasks. Thus, this paper proposed an innovative model merging framework called MetaMerging, which constructs the merged model with a unified model and lightweight task-specific adapters. Specifically, the adapters are efficiently trained without labels via feature alignment with fine-tuned models, while the unified model is obtained by merging task vectors with coefficients adaptively optimized through meta-learning, which enhances the generalization and enables more effective adapter training. Extensive experiments on CV and NLP fields show strong performance of MetaMerging on various downstream tasks and demonstrate the effectiveness of meta-learning in our method compared to other parameter merging methods. Our code is available at https://anonymous.4open.science/r/MetaMerging-53A1

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces MetaMerging, a meta-learning-based framework for adaptive model merging in multi-task learning. Traditional model merging methods (e.g., Task Arithmetic, Ties-Merging, Surgery, AdaMerging) often rely on fixed or manually tuned merging coefficients when combining task vectors from fine-tuned models. The proposed method innovatively uses meta-learning to automatically optimize these merging coefficients, improving the generalization of the unified model and facilitating the training of task-specific adapters. Experiments on vision (ViT-B/32, ViT-L/14) and language (GPT-2) models show that MetaMerging achieves higher average accuracy across multiple tasks than prior merging methods.

### Strengths
The key contribution—using meta-learning to optimize merging coefficients—is conceptually elegant and fills a gap in existing merging methods.

This paper is well-written.

Figure 3 is easy to follow.

### Weaknesses
Although comparisons are made to AdaMerging, Surgery, and Pareto Merging, newer or hybrid model merging approaches (e.g., MoE-based fusion or gradient-space merging methods) are not included. The omission limits the claim of state-of-the-art performance.

Meta learning requires backward transfer that needs the calculation of gradient, even second-order gradient, which is infeasible for large-scale language models, such as Qwen3 32B.

The collection of meta-train and meta-test sets is a challenge for powerful models, such as LLMs.

The improvements over Surgery are limited, while the training and memory requirements seem much higher than Surgery.

Table 4 should include the training time of existing model merging methods.

### Questions
See weakness.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper investigates the problem of merging multiple fine-tuned models into a unified multi-task model, aiming to enhance efficiency and performance across diverse tasks. The authors propose a meta-learning-based framework, MetaMerging, which adaptively learns optimal merging coefficients for task vectors, moving beyond fixed or heuristic merging strategies. This method allows the unified model to better preserve and leverage task-specific knowledge, facilitating improved adapter training and multi-task generalization. Experimental results on both vision and language benchmarks demonstrate that MetaMerging consistently outperforms conventional model merging techniques in accuracy and training efficiency. The work offers a principled approach for adaptive model consolidation, contributing to the advancement of scalable and robust multi-task learning.

### Strengths
**Originality:**

The paper proposes a meta-learning approach for merging multiple fine-tuned models, offering a new way of adaptively learning merging coefficients rather than relying on fixed or manual strategies. This creative combination of meta-learning and multi-task model consolidation addresses a practical gap in existing literature.

**Quality:**

Methodologically, the framework is well-founded and experimentally validated across diverse benchmarks in both vision and language domains. The empirical analysis demonstrates consistent improvements over standard baseline methods, with clear, quantitative results supporting the core claims.

**Clarity:**

The presentation is logical, with well-structured explanations, informative figures, and thorough comparative tables. While dense in the technical sections, the overall narrative is coherent and the methodology is transparent for readers familiar with deep learning.

**Significance:**

By providing a principled solution for scalable model merging, the work has practical significance for resource-efficient deployment of multi-task systems. Its adaptive strategy enables better generalization and performance, contributing meaningfully to the progress of the field and addressing real-world challenges in model consolidation and transfer learning.

### Weaknesses
- **Limited Novelty Relative to Existing Adaptive Merging:** Related ideas of adaptive weighting exist in ensemble learning and prior neural model merging. The paper would benefit from clearer differentiation and explicit discussion of how its approach surpasses past adaptive strategies (e.g., with more theoretical justification or unique robustness properties).
- **Generalization Beyond Benchmarks:** The experiments, though thorough on selected datasets and architectures, are limited to standard benchmarks. To strengthen the validity of claims, additional studies on more diverse domains, truly large-scale settings, or real-world multi-task scenarios (with heterogeneous architectures or data) would be valuable.
- **Accessibility and Implementation Details:** The methodology sections (3.1,3.2,3.3) remain technically dense, making it difficult for broader audiences to understand the key concept. I suggest author to provide a brief introduction to basically describe the motivation and purpose and methodology of this section. Being a high-level one for the user to understand the general idea. Then you can describe your detailed procedure or massive computation.

### Questions
1. Can the authors provide evidence or commentary on how their meta-merging approach would handle truly large-scale, real-world scenarios with many diverse tasks, different model architectures, or highly imbalanced datasets? What specific challenges or modifications might arise when scaling beyond the standard benchmarks presented?
2. Could the authors clarify how their meta-learning method differs fundamentally from previous adaptive weighting schemes in model merging and ensembling literature? Are there theoretical advantages, unique robustness properties, or empirical tests that distinguish MetaMerging as more than an incremental improvement?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a model merging technique with meta-learning of merging coefficients for training task-specific adapters. The study followed closely on how model merging problems and experiments have been defined and studied. The writing and organization of the paper are good overall.

### Strengths
1) The work investigates the method & impact of task-specific adapters for a multi-task model, and shows that the proposed method could slightly perform better than the compared works.
2) The method is clearly described for meta-learning and adapters.
3) The experiment is followed model merging literature closely.

### Weaknesses
1) Figures 2 and 3 can be improved; it is difficult to understand without reading the entire paper.
2) Missing SOTA comparison with the data-less model merging method, WUDI merging.
3) The performance improvement against SOTA is marginal for VIT, despite requiring data, gradient, and additional adapters.
4) In the NLP experiment, the proposed method is only compared with weak baselines (all before the end of 2023), and a relatively small GPT2 was used (recent model merging at least uses llama2/3).

### Questions
1) Table 4 should include the running times for all methods compared.
2) It would be good to evaluate out-of-domain generalisation

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes MetaMerging, a meta-learning framework for model merging under the pretrain-finetune paradigm. Instead of treating merging coefficients as fixed or manually tuned hyperparameters, the method meta-learns the coefficients so that the unified merged model yields better downstream task-specific adapter training. Experiments on vision and NLP tasks show improved unified and adapter-augmented performance over prior merging methods.

### Strengths
1, Motivated formulation. \
Clearly identifies the overlooked importance of merging coefficients in task-vector–based model merging and motivates learning them dynamically.

2, Method soundness. \
Uses a meta-learning framework inspired by MAML to optimize coefficients for better downstream adaptation, which is conceptually meaningful and aligned with the merging problem structure.

### Weaknesses
1, Adapter-dependent benefit: \
Improvements are strongest when adapters are added post-merge, raising questions on the intrinsic generalization of the unified model alone vs. the joint effect with adapters.

2, Limited analysis on when meta-merging helps: \
There is insufficient theoretical or empirical characterization of: 

2.1 when adaptive coefficients matter most, 

2.1 how task similarity affects meta-learning benefit, 

2.3 failure cases (e.g., conflicting tasks, negative transfer).

3, Limited novelty vs. MAML:

The contribution mainly adapts MAML to the setting of merging coefficients. The algorithmic innovation is relatively incremental; most complexity lies in applying known meta-learning ideas to merging.

4, Limited evaluation:

There should be some experiments on merging LLMs.

5. Limited baselines:

There should be comprasion with lossless methods like: EMR-Merging, Talls-Mask and Free-Merging.

6, The method still rely on data samples, which is not data-free and may face more challenges for LLMs.

### Questions
1，Adapter structure comparability & design choices:

The performance gains appear closely tied to the adapter modules. Could the authors clarify: What specific adapter architecture is used (e.g., LoRA, bottleneck adapters, MoE heads)? Are adapter capacities kept strictly equal across baselines to ensure a fair comparison?
Have you tested whether the method still holds with different adapter forms (e.g., shared vs. per-task adapters, low-rank variants, different insertion layers)? Since adapters play a key role in the pipeline, a more detailed justification and ablation of adapter design would strengthen the claim that improvements primarily come from better merging rather than from architectural choices.

2，Compared with simple coefficient search:

What is the actual meta-training overhead compared to simple coefficient search (e.g., CMA-ES or Bayesian tuning)? Is the method scalable to very large models or many tasks?

3, How many data samples you used for feature loss computing, if the loss can be replaced by L1 or others ?

### Soundness
3

### Presentation
2

### Contribution
2
