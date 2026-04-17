# Reference-Specific Unlearning Metrics Can Hide the Truth: A Reality Check

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Current unlearning metrics for generative models evaluate success based on reference responses or classifier outputs rather than assessing the core objective: whether the unlearned model behaves indistinguishably from a model that never saw the unwanted data. This reference-specific approach creates systematic blind spots, allowing models to appear successful while retaining unwanted knowledge accessible through alternative prompts or attacks. We address these limitations by proposing Functional Alignment for Distributional Equivalence (FADE), a novel metric that measures distributional similarity between unlearned and reference models by comparing bidirectional likelihood assignments over generated samples. Unlike existing approaches that rely on predetermined references, FADE captures functional alignment across the entire output distribution, providing a principled assessment of genuine unlearning. Our experiments on the TOFU benchmark for LLM unlearning and the UnlearnCanvas benchmark for text-to-image diffusion model unlearning reveal that methods achieving near-optimal scores on traditional metrics fail to achieve distributional equivalence, with many becoming more distant from the gold standard than before unlearning. These findings expose fundamental gaps in current evaluation practices and demonstrate that FADE provides a more robust foundation for developing and assessing truly effective unlearning methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Overall, this is work provides keen insights and moves the machine unlearning field in the right direction. The observation that unlearning models should match gold standard retain models is well-motivated. The work is well-organized, and it fits well in the literature. My main concern is that it is unclear if choice of retain model affects the FADE score. The authors claim that FADE is more robust, but I did not find a sensitivity study which answers this question. Moreover, the authors observe that FADE is typically far better for the original model than for any unlearned model. While FADE appears useful for comparing unlearned models, it calls into question whether FADE is sensitive to behaviors of the retain model that aren’t important in practical scenarios. I believe this is a promising work, and I would be happy to adjust my score if my concerns are addressed.

The authors claim that the core objective of unlearning is that the unlearned model should behave indistinguishably from a retrained model and that current unlearning metrics do not accomplish this. The authors introduce FADE which measures the symmetric KL divergence of the retain model distribution and the unlearned model distribution given some condition. The proposed metric disagrees with current metrics on TOFU and UnlearnCanvas, indicating that current metrics are not good measures of equivalence with (gold standard) retrain models.
The authors find that forget quality (as measured by current metrics on TOFU) is sensitive to choice of reference answer, indicating that current metrics are brittle in evaluating unlearning performance.

### Strengths
Distributional measures of similarity between the unlearned model and a retrained model are a step in the right direction for unlearning metrics, and the authors’ insights regarding this are valuable

The paper is clearly written and easy to follow. It is well grounded in literature.

The observation that unlearning should move beyond static evaluations is valid and moves the field in the right direction.

The consistency of FADE in figure 5 is compelling (compared with differing behaviors of FQ)

### Weaknesses
It is not immediately clear why the distribution of a single unlearned model should fit the exact distribution of a single retrain model. Exact unlearning in the literature (Nguyen et al 2025 “A Survey of Machine Unlearning”) is the case where the distribution of retrain models matches the distribution of unlearned models. Wouldn’t it be more appropriate to measure the expected FADE (or something similar) over distributions of models?

There is no study of the robustness of FADE to choice of reference model. Is this more robust than FQ?

Questions
What is “functional alignment”? It seems to be matching output distributions or something to that effect.

Is FADE robust to choice of reference model (i.e., is it stable when you test it with different randomly initialized and retrained models)?

Line 122 “We expect to achieve more robust unlearning that better withstands such post-unlearning attacks.” If FADE measures how well unlearned models mimic a retained model, why would FADE help make unlearning methods more robust to post-unlearning attacks? My understanding is that the latent information remains inside the model and it is not revealed at the logits/output until after the attack.

What is the variance of FADE across multiple unlearned models compared to the same reference model?

One could argue that it shouldn’t matter if an unlearned model maps the target concept to reasonable concept A vs reasonable concept B, but this could significantly impact FADE. What is the variance of FADE in this case, and should these semantics matter in practical scenarios?

### Questions
What is “functional alignment”? It seems to be matching output distributions or something to that effect.

Is FADE robust to choice of reference model (i.e., is it stable when you test it with different randomly initialized and retrained models)?

Line 122 “We expect to achieve more robust unlearning that better withstands such post-unlearning attacks.” If FADE measures how well unlearned models mimic a retained model, why would FADE help make unlearning methods more robust to post-unlearning attacks? My understanding is that the latent information remains inside the model and it is not revealed at the logits/output until after the attack.

What is the variance of FADE across multiple unlearned models compared to the same reference model?

One could argue that it shouldn’t matter if an unlearned model maps the target concept to reasonable concept A vs reasonable concept B, but this could significantly impact FADE. What is the variance of FADE in this case, and should these semantics matter in practical scenarios?

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a distribution-level unlearning metric FADE, to address the limitations of previous reference-specific approaches. Moreover, its experiments expose the failure of existing unlearning methods under this new metric.

### Strengths
1. This paper effectively reveals the problems that exist in current evaluation metrics.
2. The experiments are comprehensive.

### Weaknesses
1. This paper points out many shortcomings in current evaluation methods. This part is convincing but not surprising. After that, the paper proposes a new method called FADE. The main conclusion from the experiments is that under the FADE metric, existing unlearning methods perform poorly. However, the paper does not go on to propose more effective unlearning methods, making it feel quite incomplete. Based on the presented results, I cannot confirm that FADE is a flawless metric that could be widely accepted. Overall, the paper gives me the impression of lacking significant conclusions and constructive insight.
2. The LLM and T2I parts do not feel like a cohesive whole. Moreover, the FADE mentioned in line 245 and the formula in line 266 appear to be two completely different metrics, yet they share the same name. The discussions of these two parts are also quite disconnected, making the paper difficult to read.
3. From Figure 4, I'm not convinced by the author's claim in Line 228 that unlearned models generate inconsistent images. I think the three images from Ediff/ ESD share a similar style.

### Questions
I notice that retrained models are finetuned for 5 epochs on the retained dataset, but the unlearned models are tuned with LoRA from the base model.   I'm wondering if LoRA fine-tuning itself introduced some stuff you missed. To verify it, you can continue to tune a retained model with LoRA for 5 epochs and denote the resulting models as LoRA-Retain models. Then you can get a new baseline like the dashed line in Figure 5 by averaging over randomness.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper examines current unlearning evaluation metrics for generative models, arguing that prevalent reference-specific metrics fail to assess true unlearning. The paper introduces FADE to quantify distributional similarity between an unlearned model and a retain-only model using a bidirectional likelihood comparison on generated outputs. Experiments on LLM unlearning and T2I demonstrate that unlearning methods may appear successful under traditional metrics but exhibit significant distributional discrepancies not captured by those evaluations, whereas FADE exposes these failures.

### Strengths
1. The paper identifies and convincingly demonstrates the limitations of current unlearning evaluation practices, providing empirical and theoretical support for the claim that reference-specific metrics could result in overestimating unlearning efficacy.
2. The problem and motivation identified in the paper are novel. FADE is implemented in a way that is modality-agnostic, working for both autoregressive language models and diffusion models.
3. The work addresses a significant and growing issue as unlearning gains attention for safety/privacy/ethical AI deployment. The results challenge prevailing practices and set a high standard for future work in the area.

### Weaknesses
1.  The mathematical description of FADE is clear. Still, the paper does not seem to present deeper theoretical guarantees or formal links between FADE and true indistinguishability in the full probabilistic sense.
2. The core of the proposed method lies in using comparative scores to evaluate the differences between the unlearned model and the retain-only model. That said, it seems that one potentially informative control experiment is absent- Namely, assessing the comparative scores between different unlearned models. Incorporating such an experiment could help enhance the generality of the findings and strengthen the rigor of the validation.
3. It appears that the paper does not include a discussion or experimental evaluation of the computational/time/token cost associated with the newly proposed metric, which could be an important aspect. Addressing this point may provide a more comprehensive assessment of the metric’s practicality.
4. Although the writing is generally clear and easy to follow, there appear to be numerous instances that resemble AI-generated text patterns (the frequent use of “–” symbols), though I may be mistaken.
5. In practice, likelihood-based comparison metrics (such as FADE) can be sensitive to the sampling strategy. The paper does not explore whether alternative sampling schemes (diverse decoding, hard negative mining, etc.) could strengthen or weaken FADE as a metric.
6. there’s comparatively little discussion of hyperparameter, optimizer, or architecture robustness

### Questions
1. Could the authors provide comparative experimental results regarding computational cost and time overhead?
2. Would it be possible for the authors to include control experiment results involving mutual scoring between two unlearned models?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper highlights an existing problem in unlearning: unreliable metrics for verifying unlearning for generative models. The paper proposes Functional Alignment for Distributional Equivalence (FADE), which evaluates distributional alignment with retain-only oracles through bidirectional likelihood comparisons over generated samples. The experimental results evaluate the FADE score on language models and text-to-image diffusion models and several well-known unlearning methods. It shows that existing metrics such as unlearning accuracy can be misleading for evaluating the efficacy of unlearning.

### Strengths
1. Grounded definition. The paper frames unlearning as achieving functional alignment with a retain-only model (i.e., behaving as if the forgotten data was never seen). This aligns with the gold standard (exact perfect unlearning) and provides a clear conceptual foundation.



2. Points to a significant gap in existing work. The authors highlight that current evaluation methods rely on reference-specific proxies (e.g., fixed answers, classifiers), which can mask failures and even allow recovery attacks. Thus, the work exposes significant blind spots in prevailing practice.



3. FADE measures full distributional equivalence bidirectionally, not just task-specific correctness. It applies across modalities (LLMs and diffusion models) and detects subtle failures that reference-conditioned metrics overlook.

### Weaknesses
1. Application. While the FADE metric and evaluation are important to emphasize existing problems, FADE requires having a retain-only model. Thus, it is not clear whether it can be practically implemented (or a proxy of it) to improve the unlearning method itself. 

2. Computationally expensive. It is computationally expensive to compute the retain-only model under different seeds since it requires training the model without the forget set samples. FADE requires Monte-Carlo style sampling and likelihood estimates (or denoising approximations for diffusion models), which may be expensive and subject to variance depending on chosen sampling strategies.

### Questions
1. What do we learn from the FADE metric and evaluation for future unlearning methods? It would be good if the authors can comment on how they see their metric applied in evaluations and future unlearning methods.

2. How sensitive is FADE to generation strategy? Since LLM measurement relies on top-k/nucleus sampling, do different decoding strategies change FADE outcomes? How consistent are evaluations across sampling choices?

### Soundness
3

### Presentation
3

### Contribution
3
