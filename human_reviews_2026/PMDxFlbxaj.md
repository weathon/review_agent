# UltraEdit: Training-, Subject-, and Memory-Free Lifelong Editing in Language Models

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 8, 4

## Abstract
Lifelong learning enables large language models (LLMs) to adapt to evolving information by continually updating their internal knowledge. An ideal system should support efficient, wide-ranging updates while preserving existing capabilities and ensuring reliable deployment.
Model editing stands out as a promising solution for this goal, offering a focused and efficient way to revise a model’s internal knowledge. Although recent paradigms have made notable progress,
they often struggle to meet the demands of practical lifelong adaptation at scale.
To bridge this gap, we propose UltraEdit, a *training-*, *subject-*, and *memory-free* approach that is well-suited for ultra-scalable, real-world lifelong model editing.
UltraEdit fundamentally differs from traditional paradigms by computing parameter shifts in one step using only a hidden state and its gradient, making the approach simple yet efficient.
To improve scalability in lifelong settings, UltraEdit employs a *lifelong normalization* strategy that continuously updates feature statistics across turns, allowing it to adapt to distributional shifts and maintain consistency over time.
UltraEdit achieves editing speeds over **7× faster** than the previous state-of-the-art method, which was also the fastest known approach, while using less than **1/4 the VRAM**. This makes it the **only** method currently capable of editing a 7B LLM on a 24GB consumer-grade GPU.
Furthermore, we construct UltraEditBench, the largest dataset in the field to date with over **2M** editing pairs, and demonstrate that our method supports up to **2M** edits while maintaining high accuracy.
Comprehensive experiments on five datasets and six models show that UltraEdit consistently achieves superior performance across diverse model editing scenarios, taking a further step towards safe and scalable lifelong learning.
We will release the code and dataset upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes ULTRAEDIT, a training-free, subject-free, and memory-free framework for lifelong model editing in LLMs. ULTRAEDIT introduces a lifelong normalization strategy to maintain stability across sequential edits. The authors further present ULTRAEDITBENCH, a large-scale benchmark comprising over 2 million editing pairs for evaluating editing scalability. Experimental results demonstrate that the proposed method achieves gains in both performance and efficiency.

### Strengths
* The paper tackles an important and timely problem in scalable and continual model editing. The proposed framework enables LLMs to adapt to new information continuously, making lifelong knowledge updating both efficient and practical.
* ULTRAEDITBENCH, containing over 2M editing samples, represents the largest benchmark in this area and provides a solid foundation for evaluation at unprecedented scale.
* Extensive experiments across diverse LLMs demonstrate that ULTRAEDIT is both effective and lightweight.

### Weaknesses
* **Unreliable evaluation foundation**. Although the authors claim to use LLM-as-judge as a complementary metric, nearly all experiments in the main text rely on **teacher-forcing–based evaluation**, which is known to **overestimate editing performance by feeding the gold answer tokens during decoding**. The key difference between the previous evaluation and the more rigorous WILD framework is not whether an LLM is used as a judge, but whether the evaluation depends on teacher forcing. Prior studies have shown that method performance rankings under WILD may differ from those under teacher-forcing setups. Despite this, the paper reports WILD-based results only once, in Table 13 of the Appendix, which is inappropriate given that evaluation rigor is central to the paper’s claims. Moreover, the results in Table 13 show very low success rates, indicating that the proposed method performs poorly under realistic evaluation and fails to demonstrate convincing scalability or reliability in true lifelong editing settings.
* The evaluation on downstream capabilities is insufficient. The experiments are mainly conducted on a single model (LLaMA-3) and a single dataset (ZsRE), which raises concerns about the generality of the conclusions. Although the paper claims scalability to 2M edits, the reported downstream evaluations only extend to 20K edits, which is orders of magnitude smaller. As a result, how the model’s general abilities behave after large-scale editing remains unknown.
* While ablations show the importance of lifelong normalization, the paper lacks a principled explanation or analysis clarifying why normalization mitigates interference across turns.

### Questions
* Have you evaluated the method under the WILD (non–teacher-forcing) setting on more datesets? How does its performance compare to results reported with teacher forcing?
* After 2 M edits, how does the model perform on downstream tasks? Have you tested beyond 20 K edits to verify robustness?

## suggestions
* In Tables 2 and 3, the paper reports Specificity as the key auxiliary metric, whereas prior studies have shown that Specificity fails to reflect whether other knowledge or task abilities are preserved. Replacing this metric with downstream task performance would be more meaningful.

### Soundness
1

### Presentation
2

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
This paper proposes ULTRAEDIT, a training-, subject-, and memory-free approach to lifelong model editing. Specifically, ULTRAEDIT concatenates the hidden states and the gradients for each edit into a unified “editing feature,” normalizes this feature, and then applies a closed-form least-squares update to selected parameters. The core contribution is a lifelong normalization mechanism that maintains the running statistics (mean/variance) of these editing features so that edits remain stable. The authors also introduce ULTRAEDITBENCH, a new benchmark over 2M editing pairs, derived from Wikidata triples, intended to evaluate efficacy, generalization, and specificity.

### Strengths
1. The paper is well written and easy to follow, and the method is described in a modular way.

2. ULTRAEDITBENCH is quite large and could be a useful community resource for assessing editing methods under large-scale conditions.

3. The evaluations cover multiple datasets and model families, with ablations that probe the contribution of the normalization component.

### Weaknesses
1. Limited explanation for the method. Authors claim that "by calibrating the mean and variance, it mitigates the overwriting of previously acquired knowledge", but they provide neither a theoretical justification nor targeted tests to substantiate this mechanism beyond ablations.

2. Mechanistic explanation is also underdeveloped. The insight behind concatenating hidden states and gradients into a unified feature remains unclear. Moreover, why can normalization stabilize the learning dynamics across many edits? The current version lacks sufficient evidence to support these claims.

3. Some important lifelong editing baselines [1,2,3] are absent, making it difficult to measure the reported gains.


[1]. Aging with grace: Lifelong model editing with discrete key-value adaptors, NeurIPS 2023.

[2]. Towards Lifelong Model Editing via Simulating Ideal Editor, ICML 2025.

[3]. MEMOIR: Lifelong Model Editing with Minimal Overwrite and Informed Retention for LLMs, Arxiv 2025.

### Questions
1. Why does calibration minimize interference? Please provide an explanation—ideally with theory or counterfactual experiments—showing that the normalizing feature is effective at reducing interference with prior edits.

2. What is the intuition (or supporting evidence) that concatenation is superior to using either signal alone?

3. The method appears to scale gradient information by the hidden state to form the target used in the closed-form regression. Why is this a valid surrogate for the targets typically obtained via carefully designed optimization in prior editing methods?

4. AlphaEdit’s results in your tables appear substantially below those reported in its original paper. Could this be due to hyperparameters?

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
3

### Summary
This paper introduces ULTRAEDIT, a training-, subject-, and memory-free approach to lifelong model editing for LLMs. Unlike prior approaches that rely on hypernetworks, subject localization, or external memory, ULTRAEDIT computes parameter shifts in closed form using a concatenation of hidden states and gradients. A key innovation is the lifelong normalization mechanism, which continuously updates feature statistics (mean and variance) across editing turns. This enables stable updates over time. The authors also present a large-scale benchmark for evaluating long-term editing performance. Experiments on several open-source LLMs and datasets show that the proposed method achieves 7 times faster editing speed, 4 times lower memory usage, and supports up to 2M edits with strong performance retention.

### Strengths
The proposed lifelong normalization strategy is a simple yet elegant contribution that addresses the stability bottleneck in lifelong model editing. It adapts running statistics across editing turns, mitigating edit drift and catastrophic forgetting. It also removes the dependency on hypernetwork training, subject localization, or external memory, making it feasible for real-world continual knowledge updates on consumer GPUs.

### Weaknesses
While the method is intuitive and experimental results are convincing, there seems no enough theoretical analysis explaining why lifelong normalization ensures stability.

### Questions
1. It would be perfect if you provide a theoretical justification for why lifelong normalization stabilizes edits across turns.
2. In scenarios involving large-scale, multi-turn editing, what occurs when contradictory edits are applied sequentially? Can ULTRAEDIT address such conflicts? Furthermore, is there a mechanism to revert or forget a specific edit without the need for model retraining?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a training-based yet efficient model editing approach that aims to achieve targeted updates to language models with minimal parameter perturbation. The method is motivated by jointly minimizing the loss on edited samples and the overall magnitude of parameter change. Specifically, it leverages hidden states and their corresponding gradients to construct an optimization problem combining a reconstruction loss and a regularization term on parameter shifts. By leveraging hidden states and their corresponding gradients, the authors derive a closed-form solution that enables an efficient one-step parameter update, achieving rapid and stable model editing without iterative optimization.

### Strengths
1. The paper is well organized and clearly written, making the method easy to follow and reproduce.

2. ULTRAEDIT employs a one-step parameter shift that requires neither iterative updates nor retraining. It appears to be a simple yet effective approach with a clever and well-motivated formulation.

3. The approach seems computationally efficient and memory-friendly, enabling large-scale and lifelong edits while maintaining model stability.

4. Experimental results are comprehensive, covering multiple datasets and models, and show strong performance with minimal degradation of general capabilities. Moreover, the authors introduce a new large-scale benchmark to further evaluate model editing methods.

### Weaknesses
1. While the paper highlights the contribution of the normalization mechanism to feature stability, it lacks deeper theoretical or analysis to substantiate this claim. A more detailed theoretical or analysis could further improve the work.

2. Although efficient overall, the per-module caching of features (H, V) and the required matrix solve can raise peak memory and compute costs as batch size, feature dimensionality, or the number of editable modules grows.

### Questions
How does ULTRAEDIT perform and scale on larger models, compared to the reported results? (If this is resource-prohibitive, feel free to skip.)

### Soundness
2

### Presentation
2

### Contribution
3
