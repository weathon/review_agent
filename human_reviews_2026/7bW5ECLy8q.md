# DAWI: Dual Anchored Weighted Interpolation for LLM Unlearning

- Decision: Reject
- Scores: 2, 4, 2, 2

## Abstract
Large Language Models (LLMs) are trained on vast amounts of text data, and they frequently memorize sensitive or private information that appears in the training corpus. This raises significant privacy, security, and ethical concerns, particularly when such information can be extracted by adversarial prompts or from membership inference attacks. Machine unlearning has therefore emerged as an important research direction, with the goal of selectively removing knowledge of problematic information from a model while preserving its general language understanding and reasoning capabilities. In this work, we focus on unlearning information that is memorized during the fine-tuning phase of training, which commonly happens when inference providers fine-tune models on user interactions. We introduce Dual Anchor Weighted Interpolation (DAWI), a simple yet effective unlearning algorithm that achieves state-of-the-art results on the TOFU benchmark and demonstrates strong unlearning efficacy with minimal hyperparameter tuning, making it practical for real-world deployment.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes DAWI (Dual Anchored Weighted Interpolation), a method for LLM unlearning that interpolates between the base model and fine-tuned model rather than directly updating model weights. By optimizing only the interpolation coefficients, the method restricts parameter updates to a low-dimensional subspace, reducing catastrophic forgetting and training cost. Experiments on TOFU and a newly introduced Math Unlearning Dataset (MUD) show that DAWI achieves effective unlearning while preserving model utility and requiring no retain data or extra models.

### Strengths
1. The method cleverly constrains model weights to lie on the interpolation path between the base model and fine-tuned model. This reduces optimization complexity and avoids full backpropagation while still allowing selective forgetting.
2. On the TOFU benchmark, DAWI achieves competitive or state-of-the-art unlearning performance while preserving model utility, outperforming several gradient-based baselines.

### Weaknesses
1. DAWI constrains parameters to lie between the base and fine-tuned models, but it is unclear whether the method truly removes the unwanted knowledge or merely suppresses it at the output level. The paper does not probe internal representations or test paraphrased / adversarial prompts to verify deeper forgetting.
2. The approach assumes full access to the original pre-trained model. In many real-world settings (e.g., proprietary fine-tuned LLMs), the base model may not be stored or available, limiting practicality.
3. Although DAWI is more efficient than gradient-based baselines, the paper does not provide wall-clock runtime, parameter selection overhead, or memory usage on large-scale models (e.g., 7B or 13B). Scalability claims are not fully validated experimentally.
4. Forgetting is evaluated mainly on exact sequence output or token probability. The method may fail to erase higher-level semantic relations or indirect knowledge (e.g., paraphrases, factual reasoning), especially in Transformer attention layers.
5. The paper does not evaluate whether forgotten knowledge can be easily recovered through few-step fine-tuning on the forget set (relearning attack), which is a standard robustness check in machine unlearning.

### Questions
1. If the forget data is queried using rephrased instructions or semantically equivalent prompts, can the model still reconstruct or imply the forgotten content?
2. Does linear weight interpolation between the base and fine-tuned models always form a meaningful hypothesis space? Could this fail in cases of non-linear parameter trajectories or large domain shifts?
3. Is there any analysis of which layers or heads are most frequently updated? Does the selection correlate with influence of the forget set, or is it uniform?

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
2

### Summary
This paper proposes DAWI, a simple yet effective algorithm that enables LLMs to efficiently forget specific knowledge by interpolating between a base and fine-tuned model, achieving strong unlearning performance with minimal hyperparameter tuning and resource usage.

### Strengths
1. The idea of ​​this paper is novel.
2. It seems that DAWI avoids traditional optimizers, reducing GPU memory and compute costs while maintaining strong unlearning results.

### Weaknesses
1. The paper introduces a new benchmark, MUD, but it does not mention what measures have been taken to ensure the reliability of the benchmark.
2. I think most of the methods and strategies in this paper lack an obvious theoretical basis.
3. DAWI requires access to a pre-finetuned "clean" model. I think it isn't always feasible in real deployment settings.
4. DAWI does not integrate well with standard gradient-based loss functions, reducing its flexibility compared to other unlearning methods.

### Questions
Refer to weakness.

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
This paper introduces Dual Anchored Weighted Interpolation (DAWI), a novel and efficient machine unlearning method for large language models (LLMs) that removes the influence of specific training data without full retraining. Unlike existing approaches, DAWI leverages a base model—unexposed to the undesirable data—and constrains each parameter of the unlearned model to be a convex combination of the base and fine-tuned model's weights. By sparsely updating only the mixing coefficients based on expected loss change, DAWI minimizes drift, avoids over-unlearning, and maintains model utility. Evaluated on the TOFU benchmark and the newly introduced Math Unlearning Dataset (MUD), DAWI achieves state-of-the-art performance with minimal hyperparameter tuning, outperforming methods like NPO, RMU, and SimNPO while being more memory-efficient and faster due to its non-optimizer-based, discrete update strategy.

### Strengths
1.	The method's core design, which anchors the model's parameters between the base and fine-tuned versions, inherently constrains updates and limits catastrophic drift. This is evidenced by DAWI's superior privacy score (Priv) and near-zero "PrivLeak" metric, showing it effectively forgets target data without degrading performance on the retain set or general capabilities.
2.	DAWI achieves state-of-the-art results on the TOFU unlearning benchmark while requiring only a single tunable hyperparameter.

### Weaknesses
1.	DAWI's performance is contingent on having access to a base model that has never been trained on the "forget set" data. This makes it incompatible with common unlearning benchmarks like MUSE and WMDP, where the sensitive information is already part of the model's pretraining knowledge, as a suitable base model does not exist in such scenarios.
2.	Ablation studies revealed that DAWI does not work well when paired with loss functions designed for gradient descent, such as the one used by SimNPO. This indicates that its unique optimization algorithm is specifically tailored to its own proposed loss function, potentially limiting its flexibility and integration with other advanced unlearning objectives.
3.	The paper notes that on small datasets where there is a large capability gap between the base and fine-tuned models, DAWI's discrete optimization steps are sensitive to noise. This suggests the method may be less robust in situations with limited or unreliable data.

### Questions
1.	The paper highlights the advantage of having a base model. However, in many real-world scenarios, the "forget set" might contain information the model already knew from pre-training (e.g., real public figures' data). How can DAWI be adapted or what is its expected performance in cases where a truly "unaware" base model does not exist?
2.	The core of DAWI constrains the unlearned model to a convex hull between the base and fine-tuned models. While this effectively prevents drift, couldn't this also inherently limit the solution space, potentially preventing the model from finding a more optimal state that isn't a simple interpolation of the two anchors?

### Soundness
2

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
5

### Summary
The paper focuses on unlearning for models which have been finetuned from a base model.\
To do so, the authors constrain the unlearned models to lie on the convex hull of the finetuned and the base model.\
The authors introduce a novel loss function for unlearning and propose a discrete update for optimization, to form an unlearning algorithm DAWI.
The paper also provides experiments on TOFU and introduce a new unlearning dataset called MUD (Math Unlearning Dataset).

### Strengths
The idea of having access to a base model may be valuable for developing unlearning algorithms. This can infact be a realistic assumption.\
DAWI's optimization algorithm is a novel algorithm, which can successfully find a model in the convex hull of the base model and the finetuned model, using discrete optimization updates.
which slightly reduces the memory consumption, because of the lack of use of optimizers like Adam.

### Weaknesses
- The paper introduces a new unlearning dataset called MUD. However, the authors do not provide any details of the dataset i.e. how it is created, what kind of trivia questions are present. No examples are provided. 
- The claim for introducing a new dataset is that TOFU finetuning does not represent real world deployment. While this is true, there is no evidence that the MUD setting represents real world deployment. 
- The dataset is called Math Unlearning dataset, however it is written "we chose to avoid any math questions in the retain and forget sets ". So it is confusing if the dataset is introduced to unlearn any math knowledge or not. 
- Experiments on MUSE are missing. The authors mention that this cannot be done because "forget sets overlap with knowledge already acquired during pretraining". However, the goal of the unlearning in this context is to have performance similar to the retrained model (i.e. finetuned only on the retain set). 
- Please include the results for other splits of TOFU i.e. 1% and 5%. 
- In Table 1,  we see that the forget quality of DAWI is 0.581. Since forget quality is a p-value, a p-value of 0.581 is uninformative. This is unconvincing in terms of whether any proper unlearning is occurring. 
- It is written that hyperparameters are not tuned for ablations.  This is not a fair comparison in my opinion. 
- The general writing is unclear, for example, Section 3.4 can be improved for clarity. Please see the Questions section.

### Questions
- What is Gibberish Score ? Is this referring to forget fluency from OpenUnlearning ?
-  Why are metrics like forget quality and privleak missing in MUD ?
- Please provide details about DAWI loss with Adam. How is constraining done in this case ?

### Soundness
3

### Presentation
2

### Contribution
2
