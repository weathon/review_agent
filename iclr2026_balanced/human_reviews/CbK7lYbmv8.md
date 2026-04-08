## Human Reviewer 1

### Summary
This paper introduces the Latent Reasoning Tuning framework, a novel approach to enhance the reasoning capabilities of Large Language Models (LLMs) without modifying their parameters. The method involves training a lightweight, external reasoning network (specifically, Qwen3-Embedding-0.6B) to generate a set of latent vectors (256 learned embeddings) that condition the frozen base LLM's output. The framework is trained in two stages: Supervised Fine-Tuning (SFT) on the OpenR1-Math-220k dataset, followed by Reinforcement Learning (RL) fine-tuning on the DeepScaleR-Preview-Dataset.
The authors evaluate their method on several math and general reasoning benchmarks, including MATH and GPQA. The results demonstrate that this approach consistently outperforms the base models (tested on 1.7B and 4B parameter models) as well as other efficient reasoning baselines.

### Strengths
1. The paper addresses the critical and practical problem of improving the inference efficiency of LLM reasoning.
2. A key strength is the framework's efficiency. By only training a small reasoning network and keeping the base LLM's weights frozen, the method is computationally lightweight. This modular design also suggests high flexibility, as the reasoning network could potentially be paired with various pre-trained base models.
3. The authors provide ablation studies to justify key design choices, such as the necessity of the two-stage training pipeline (SFT followed by RL).
4. The method achieves consistent improvements over the selected baselines across multiple benchmarks, validating the effectiveness of the proposed latent reasoning approach.

### Weaknesses
1. One of the major drawback of the proposed method is the loss of interpretability of the reasoning traces. It would be great to provide some analysis on the learned reasoning vectors.
2. Although results show consistent improvements on 1.7B and 4B models, it remains unclear how the reasoning network scale with base model sizes. e.g. would a 0.6B reasoning backbone still sufficient for much bigger LLMs? Table 3 shows that the performance degrade when the number of reasoning token increases from 256 to 512, which seems to indicate that there's a sweet spot for the number of tokens depending on the model capacities, but this needs further experiments with various model sizes to verify. 
3. The paper lacks an appendix and fails to provide crucial details about the hyperparameters used for the experiments. This omission significantly hinders the reproducibility of the work.
4. Although the method avoids the cost of fine-tuning the base LLM, it introduces the training and inference overhead of an additional reasoning model. The paper would benefit from a more direct comparison of the trade-offs (e.g., total training flops, inference latency) against other latent-variable baselines that do involve fine-tuning the base LLM, which would provide a more complete picture of the method's efficiency.

### Questions
1. How interpretable are the reasoning vectors? 
2. Could you provide some insights on the scalability of LRT?

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper proposes a novel reasoning framework, latent reasoning tuning (LRT), which uses an auxiliary network to generate a sequence of latent representations in a single forward pass, and then concatenates the prompt and the latent representations to generate the final answer. Experimental results on various benchmarks show its consistent performance gain over baseline methods, showing its effectiveness.

### Strengths
1. The idea of using latent representations for efficient reasoning makes sense.
2. The method is clean and efficient for training.
3. The performance on different benchmarks compared to baseline methods is good.
4. It reduces the computational cost of generation compared to explicit CoT.

### Weaknesses
1. (minor) In Table 2, it would be beneficial to include the results for the thinking mode, which would make the table more comprehensive and the comparison more straightforward, allowing for a clearer view of the trade-off between computation and accuracy.

2. It would make the paper stronger if the authors could try larger models (such as 7B models) for LRT to show the method is scalable for larger base models. This should be doable (at least for SFT only) since the LRT method only needs to train the reasoning network, which can be smaller than the base model.

### Questions
What is $P\_{ref}$ in line 239 (since $P\_{\theta}$ is fixed during training) ?

### Soundness
4

### Presentation
4

### Contribution
4

### Rating
8

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper proposes Latent Reasoning Tuning (LRT), which replaces the explicit, token-by-token generation of reasoning trajectories in LLMs with compact latent representations. The authors show that reasoning LLMs can maintain high accuracy even when conditioned on fragmented reasoning paths, suggesting significant redundancy in explicit reasoning chains. LRT introduces a lightweight "reasoning network" G_phi that maps input questions to fixed-length latent trajectories, which then condition the base LLM to generate final answers.

### Strengths
The trajectory analysis provides compelling evidence that models are robust to token omission with minimal performance degradation.
It replaces O(k) autoregressive steps with a lightweight model with fixed reasoning size, which should provide efficiency gains.
The approach is modular, allowing switching between latent and explicit reasoning modes.
The results seem to show improvements over "baseline efficient reasoning" methods across several benchmarks.

### Weaknesses
Architecture of G is underspecified: I think there is no clear description of the reasoning network architecture. The paper mentions it uses "Qwen3-Embedding-0.6B" but it's unclear what the actual architecture is. The discussion says the latent reasoner is not trained from scratch, then it means it reuses some LLM parts? This was not clear to me when reading the paper.

Missing efficiency analysis: One of the main motivations for compressing reasoning chains is computational efficiency, but the computational overhead of the latent reasoning is not discussed. No metrics are provided for inference time, FLOPs, or memory usage. Since the approach uses fixed reasoning length and lightweight modules, there should be efficiency gains, but these are not reported or analyzed. It could be a useful addition to the paper.

No statistical significance testing: All tables lack measures of variability (confidence intervals or standard deviations). This makes it unclear whether and when the improvements are statistically significant. This is important for rigorous science.

Concurrent work: The paper mentions other latent reasoning work (Hao et al., 2024; Saunshi et al., 2025; Wu et al., 2025) in only one sentence of the related work section. It is important to better distinguish this work from other reasoning in latent space approaches. Even though most of them are fairly recent, the authors are aware of them and the discussion dismisses these related work too easily.

### Questions
What is the exact architecture of the reasoning network G?
What are the actual efficiency gains at inference time? 
Are the performance improvements statistically significant? Can you provide confidence intervals or significance tests?

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
3

---

## Human Reviewer 4

### Summary
This paper introduces Latent Reasoning Tuning (LRT), a framework that replaces explicit token-by-token reasoning with compact latent representations generated by an auxiliary network. It aims to improve reasoning efficiency by performing implicit reasoning without generating lengthy step-by-step rationales.

### Strengths
+ The problem of overthinking and reasoning inefficiency is indeed practical and highly relevant for today’s LLMs, so it’s valuable to see it studied from this new latent-reasoning perspective.
+ The writing is clear and easy to follow—the method and its intuition are well-explained, making the technical parts digestible even on a first read.

### Weaknesses
+ Even though reasoning efficiency is an important problem, modern inference engines (like vLLM or SGLang) already accelerate long-token generation with KV-caching and routing optimizations. The proposed module doesn’t directly integrate with these frameworks, so it might actually reduce efficiency when generating the same number of tokens—this needs to be considered for a fair comparison with baselines.
+ Using only 512 tokens for distilled-R1 feels too artificial. For genuinely hard reasoning tasks (like AIME-style problems), original responses often exceed 10k tokens, so of course the performance drops drastically. It would make more sense to compare against short-response RL methods (like LC-RL, L1, or even RLVR with the same token cap) for a fairer setup.
+ The 512-token setup feels too synthetic and not representative of real-world reasoning scenarios. Trying longer contexts—even if not as long as full 32k setups—would strengthen the empirical validity.

### Questions
N/A

### Soundness
2

### Presentation
3

### Contribution
1

### Rating
4

### Confidence
4