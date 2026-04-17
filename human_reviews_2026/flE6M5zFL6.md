# DiVE-k: DIFFERENTIAL VISUAL REASONING FOR FINE-GRAINED IMAGE RECOGNITION

- Decision: Accept (Poster)
- Scores: 6, 2, 8, 4

## Abstract
Large Vision Language Models (LVLMs) possess extensive text knowledge but struggle to utilize this knowledge for fine-grained image recognition, often failing to differentiate between visually similar categories. Existing fine-tuning methods using Reinforcement Learning (RL) with exact-match reward signals are often brittle, encourage memorization of training categories, and fail to elicit differential reasoning needed for generalization to unseen classes. To address this, we propose $\textbf{DiVE-k}$, $\textbf{Di}$fferential $\textbf{V}$isual r$\textbf{E}$asoning using top-$\textbf{k}$ generations, framework that leverages model's own top-k predictions as a training signal. 
For each training image, DiVE-k creates a multiple-choice question from the model's top-k outputs and uses RL to train the model to select the correct answer. This approach requires the model to perform fine-grained differential reasoning among plausible options and provides a simple, verifiable reward signal that mitigates memorization and improves generalization. 
Experiments on five standard fine-grained datasets show that our method significantly outperforms existing approaches. 
In the standard base-to-novel generalization setting, DiVE-k surpasses the QWEN2.5-VL-7B and ViRFT by 10.04% and 6.16% on the Harmonic Mean metric, respectively. Further experiments show similar gains in mixed-domain and few-shot scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces the DiVE-k framework, which significantly enhances the base model's fine-grained image recognition capabilities by leveraging its top-k prediction results to construct Multiple-Choice Question (MCQ) data, and subsequently fine-tuning the model using the GRPO algorithm for Reinforcement Learning.

### Strengths
1. Using the model's own top-k predictions as training data is interesting and insightful, which serves as a form of hard-negative mining against the model's confusions.
2. DiVE-k achieves significant performance improvements over baseline models across multiple datasets and tasks.
3. The overall writing and presentation of the paper is good.

### Weaknesses
1. Limited comparison with related work: The paper primarily contrasts its results with ViRFT. To comprehensively validate the efficacy of the proposed method, it should be compared against additional approaches mentioned in the "Related Work" section.
2. Lack of diverse backbone models: The current experiments are exclusively conducted on Qwen2.5-VL. It is crucial to demonstrate the generalizability of DiVE-k by performing experiments on different LVLMs, such as confirming performance improvements on models like InternVL.

### Questions
What would the performance be if the obtained data were used directly for Supervised Fine-Tuning (SFT)? This would help in understanding the advantage of using RL.

### Soundness
3

### Presentation
3

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
This paper proposes DiVE-k, a framework to improve fine-grained visual recognition (FGVR) in Large Vision Language Models (LVLMs). The core idea is to address the model's inability to differentiate between visually similar categories. The method first uses an offline step where the base model generates $K$ rollouts to create a top-k set of candidate answers for each image. This set is then used to formulate a Multiple-Choice Question (MCQ). In the second step, the model is trained using Reinforcement Learning (RL) with a simple, verifiable reward (i.e., selecting the correct option letter) to perform differential reasoning among these pre-defined options. The experiments show strong performance gains over baselines like ViRFT and the base QWEN2.5-VL-7B model.

### Strengths
The primary strength of this paper is its strong empirical performance. The proposed DiVE-k method achieves significant improvements in base-to-novel generalization, mixed-domain, and few-shot settings, as shown in Tables 1, 2, and 3. The qualitative examples in Figures 4 and 9 are also compelling, illustrating that the model learns to perform more detailed differential reasoning when forced to choose from a set of plausible, similar options, which is a key challenge in FGVR.

### Weaknesses
Despite the strong results, I have major concerns about the methodological choices and novelty of this work.
* Limited Novelty: The proposed method's novelty is marginal. At its core, it is a two-step process: 1) an offline data curation step that converts an open-ended generation task into a multiple-choice classification task, and 2) a standard RL training step (GRPO) on this new task. The "differential reasoning" appears to be a direct consequence of this prompt reformatting (from open-ended to MCQ), rather than from a new algorithmic insight.
* Questionable Offline Top-k Generation: The decision to use a static, offline set of top-k options generated by the reference model ($\pi_{ref}$) is a significant weakness. This means the policy ($\pi_\theta$) is trained on a fixed set of problems that were defined by an older version of itself. This introduces a distribution mismatch and severely limits the model's learning. The model is not learning to generate better candidates itself; it's only learning to rank a fixed set of candidates provided to it.Lack of a 
* Principled RL Formulation: A more sound and principled approach would be a dynamic, multi-step RL process. For instance, a multi-turn RL agent could first generate its own top-k candidates in a "generation" phase and then, in a "reasoning" phase, select the best one. The entire sequence would then receive a reward. The current method, by decoupling candidate generation (offline) from candidate selection (RL), feels like an ad-hoc pipeline rather than an end-to-end reasoning framework.

### Questions
* The core of the method is the offline top-k generation. Why was this choice made over a dynamic, multi-turn RL formulation where the policy first generates its own options and then selects from them within a single episode?
* Could the authors please provide a comparison against a baseline where the options are generated dynamically by the current policy ($\pi_\theta$) at each training step, rather than fixed offline by $\pi_{ref}$?
* Please provide a comparison against a more complete "multi-turn RL" baseline, as described in Weakness #3. This seems like a more correct and challenging setup for this problem.
*Given that the main change is reformatting the problem as an MCQ, how much of the gain is simply from this prompt engineering versus the RL training itself?

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
DiVE-K formulated finegrained classification task into a self-generated MCQ leveraging the base model reasoning ability. This forces the VLM to perform reasoning using a differentiable rewards, leading to better generalization. The paper is well written and the framework is empirically validated, marking it as a strong contribution. Only concern is, there is no/lack of evidences to show the failure scenarios of the proposed solution.

### Strengths
- **Formualation**: DiVE-K formulated the task using base model's top-k predictions as a source hard-negative examples to construct the MCQ, and leverages model's reasoning ability with a differentiable reward system, making it a highly effective training method.

- **Differentiable Reasoning**- The MCQ format inherently encourages the model to focus on attribute level discriminative reasoning, which beneficial for semantically similar concepts.

- **Reward**: Simple reward based on MCQ index selection overcomes the existing string matching proposal in previous RL based system.

- **Experiments**: The proposed method is empirically evaluated and ablated under different settings. The performance of the DiVE-k is significantly improved with the proposed mechanisms making it as SOTA of the task.

### Weaknesses
- **Senstivity to roll outs and MCQ size**:  The performance of DiVE-k is heavily relies on K (number of rollouts) and m (Size of the final MCQ). While it is stated some processing to keep consistency, but there is no ablation on how variations in K and m affect the quality of the negative set and final performance.

- **Failure cases**: There's systematic analysis/discussions  when this differentiable  reasoning analysis could fail, because this reasoning in next step relies on base model capacity of identifying and including the ground truth in the rollouts. Therefore, it naturally raises question, what is the bottleneck of the proposed solution: initial option mining or differential reasoning chain. For example, in Fig 5, performance on Pets dataset drops when increasing the  top-k generations. Why did that happen?

- **Applicability**: As mentioned earlier, the performance of the base model could influence the final performance. Authors could consider testing the algorithms with different models to demonstrate the proposed solution as model-agnostic. 

- **Evaluation**: Figure 8: Provides a prompt intended to evaluate the fine-grained image classification results. Is this used for only property models or all the experiments?  If it is used for open-sourced model as well, what is the reason?. Given the prediction and groundtruth, performance score can be easily evaluated. Beyond accuracy, there's no other evaluation metric considered in the analysis.

### Questions
1.  Could authors provide computational cost as it involves multiple step pipeline during training and inference?
2. Could authors include zero-shot performance of the CLIP model on same classes (which can be obtained from original paper)? It will inform how models trained with differ in performance on same dataset?
3. Why objective function is moved to supplementary?
4. Does Table 1 has any results with supervised finetuning?If not, could authors include it as well.
5. I’m unsure if this is a typo or an actual output. In Figure 3, all the reasoning steps of DiVE-k compare “X3” and “X6”, but the final prediction is B (which I assume is the second prediction from the top-k). However, the reasoning step itself contains a statement that it cannot be “X5”.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes DiVE-k framwork which uses the top-k generation of base model as a training signal for fine-grained image classification. For each training image, DiVE-k creates a multiple-choice question from the model's top-k outputs and uses RL to train the model to select the correct answer. Experiments on standard base-to-novel generalization and mixed-domain zero-shot base-to-novel generalization demonstrate the effectiveness of the proposed method.

### Strengths
1. It transfers the open-world classification task into closed-world classification task, which is a promising way to settle the problem of brittle exact string-match reward.
2. The evaluation metrics is reasonable for fine-grained classification task. Previous work typically uses string matching to evaluate the accuracy, while this paper uses the LLM to determine whether the prediction and ground truth belong to the same fine-grained category or not.

### Weaknesses
1. A direct way to construct the hypotheses set is to select the most similar top-k categories by CLIP text features, and the advantage of the proposed offline option mining lacks experimental support.
2. Since the framework uses a two-step pipeline with chain-of-thought, it incurs additional computational cost due to the requirement of two forward passes.
3. The final accuracy heavily depends on the recall of the first inference step, which is not presented in the experimental results.

### Questions
1. What is the performance if the ground truth label is already included in the options, i.e., the typical closed-world multiple-choice setting of evaluating LVLM's fine-grained classification performance?
2. What is the performance if the model is trained to obtain options and do differential reasoning in one step?

### Soundness
2

### Presentation
3

### Contribution
2
