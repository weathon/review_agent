# TuneShift-KD: Knowledge Distillation and Transfer for Fine-tuned Models

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
To embed domain-specific or specialized knowledge into pre-trained foundation models, fine-tuning using techniques such as parameter efficient fine-tuning (e.g. LoRA) is a common practice. However, as new LLM architectures and pre-trained models emerge, transferring this specialized knowledge to newer models becomes an important task. In many scenarios, the original specialized data may be unavailable due to privacy or commercial restrictions, necessitating direct distillation and transfer of this specialized knowledge from the fine-tuned base model to a different pre-trained model. In this work, we present TuneShift-KD, a novel approach that automatically distills specialized knowledge from a fine-tuned model to a target model using only a few examples representative of the specialized information. Our key insight is that specialized knowledge can be identified through perplexity differences between base and fine-tuned models: prompts where the fine-tuned model responds confidently (low perplexity), but the base model struggles (high perplexity), indicate queries corresponding to the specialized knowledge learned by the fine-tuned model. TuneShift-KD leverages this insight to create a synthetic training dataset intended to transfer the specialized knowledge. Using an iterative process, TuneShift-KD generates more prompts that are similar to the prompts that generated responses with specialized knowledge. TuneShift-KD does not require training discriminators or access to training datasets--it is an automated approach that only requires the initial fine-tuned and base models and a few representative prompts. Our experiments demonstrate that models fine-tuned using TuneShift-KD achieve higher accuracy for the fine-tuned specialized knowledge than prior approaches, enabling both ease of deployment and demonstrably more effective transfer of the specialized knowledge.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces TuneShift-KD, a method for transferring specialized knowledge from a fine-tuned model to a different pre-trained model without access to the original fine-tuning dataset. It addresses scenarios where domain-specific data cannot be shared due to privacy or commercial restrictions. 
This approach uses a small set of seed examples to generate synthetic prompts, then applies a filtering criterion based on perplexity differences between the fine-tuned model and its base model. 
The difference in confidence between the fine-tuned model (high) and its base model (low) is used to select prompts indicative of specialized knowledge.
These filtered examples form a synthetic dataset used for knowledge distillation to the target model. 
This method is automated and architecture-agnostic, requiring no discriminators. 
Experiments on GSM8K, MBPP, and BBH benchmarks show that TuneShift-KD improves target model accuracy compared to prior approaches like Trans-LoRA, even when the original base model is unavailable.

### Strengths
## Originality
This paper introduces a new problem formulation: transferring specialized capabilities from one model variant to another without access to the original fine-tuning data. 
The originality lies in using perplexity differences between fine-tuned and base models as a signal to identify domain-specific knowledge, 
combined with iterative synthetic query generation.
While knowledge distillation and synthetic data generation are established ideas, their combination with perplexity-based filtering for data-free transfer across model families is novel.

## Quality
The proposed methodology is clearly described and includes theoretical reasoning for why perplexity differences indicate specialized knowledge, supported by experiments on multiple benchmarks. 
However, the theoretical foundation is limited to information divergence arguments, and the approach remains heavily dependent on the quality of the fine-tuned model and seed examples.
Experimental results show consistent improvements over prior work, but comparisons are constrained by the lack of open-source baselines.

## Clarity
This paper is well-structured, with clear explanations of the problem, method, and evaluation. 
Figures and tables illustrate the pipeline and results effectively. 
The iterative process and filtering criterion are explained in detail, though some sections assume familiarity with PEFT and LoRA, which may challenge readers without prior knowledge.

## Significance
This work addresses a practical and increasingly common scenario in real-world deployments where fine-tuning data cannot be shared.
Its compatibility with standard model adaptation workflows and ability to operate across different model architectures make it relevant for industry applications.
As it removes a key limitation of prior approaches that required original datasets or structural, its theoretical novelty is limited, but the method offers strong practical utility in real-world deployment scenarios.

### Weaknesses
## Dependence on Fine-Tuned Model Quality
This approach assumes that the fine-tuned model accurately represents domain-specific knowledge. 
If the fine-tuned model contains errors or biases, these will transfer to the target model. 
The paper acknowledges this but does not propose mechanisms to mitigate it. 
A possible improvement is integrating external validation or correctness checks during filtering, such as leveraging trusted knowledge sources or consistency verification methods.

## Limited Theoretical Foundation  
The justification for using perplexity differences relies on entropy and KL divergence arguments, which are presented briefly and lack formal proofs or deeper analysis.
Strengthening the theoretical basis, for example by connecting the filtering criterion to generalization bounds or information-theoretic measures, would improve rigor and credibility.

## Model and Data Dependency  
Performance depends heavily on the fine-tuned model, seed examples, and the target model’s capacity. 
The current paper does not explore strategies to reduce this dependency. Introducing adaptive sampling that considers target model weaknesses or curriculum-based distillation could make the method more robust and efficient.

## Synthetic Prompt Generation Risks  
The method uses few-shot prompting to generate synthetic prompts, but correctness and diversity are not guaranteed. While diversity is discussed, there is no systematic evaluation of prompt quality. Adding quantitative measures of diversity and correctness, or incorporating filtering based on semantic similarity and factuality, would strengthen reliability.

## Evaluation Scope
Experiments focus on three benchmarks and two model families, with additional tests on Qwen. 
However, the currrent paper does not include ablations on initial example set size, generation strategy, or alternative filtering metrics beyond perplexity. 
Broader evaluation, including low-resource domains or multilingual settings, would clarify generalizability.

## No Comparison with Target-Adaptive Filtering (e.g., weakness-aware data selection)
This paper does not consider approaches that select examples based on the target model’s weaknesses, which could improve efficiency. Including such a baseline or discussing its feasibility would provide a more complete picture of the design space.

### Questions
Q1. How does the proposed method ensure that the synthetic prompts generated from a small seed set are representative of the original fine-tuning domain? Could you provide quantitative evidence of prompt diversity and domain relevance beyond t-SNE visualization?  

Q2. The filtering criterion relies on perplexity differences computed using fine-tuned models. How sensitive is the method to the choice of threshold τ, and have you considered adaptive or ratio-based thresholds tied to distributional properties rather than fixed values?  

Q3. Since fine-tuned models may contain errors or biases, what mechanisms could be integrated to prevent propagating incorrect knowledge during distillation? Have you explored external validation or correctness checks in the filtering process?  

Q4. Why was the decision made to use the base model for filtering rather than the target model? Would incorporating the target model’s weaknesses into the selection process improve efficiency and accuracy? Have you considered or tested target-adaptive filtering strategies?  

Q5. The theoretical justification for perplexity difference is based on entropy and KL divergence arguments. Could you provide more formal analysis or empirical evidence that this criterion consistently correlates with domain-specific knowledge across diverse tasks?  

Q6. How does the method handle cases where the fine-tuned model’s specialized knowledge overlaps significantly with general knowledge? Does the filtering process risk discarding useful examples or retaining trivial ones?  

Q7. Synthetic prompt generation uses external instruction-tuned LLMs or the source model itself. How do you ensure factual correctness and avoid introducing noise? Have you evaluated the impact of prompt quality on final distillation performance?  

Q8. Experiments focus on GSM8K, MBPP, and BBH. Could you clarify why these benchmarks were chosen and whether the method generalizes to other domains such as multilingual tasks or low-resource settings?  

Q9. The paper claims compatibility with standard fine-tuning pipelines. Could you elaborate on practical deployment considerations, such as computational cost for large-scale models and scenarios where external LLMs for prompt generation are unavailable?  

Q10. How does TuneShift-KD compare to approaches that directly transfer LoRA weights when models share similar architectures? Would combining weight transfer with your synthetic data approach yield better results?

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
3

### Summary
This paper presents a method called TuneShift-KD, which aims to transfer knowledge from a fine-tuned model to a new target model when the original fine-tuning data are unavailable. The authors compare the perplexity differences between the fine-tuned model and its base model on generated outputs to identify samples that purportedly contain “specialized knowledge,” which are then used for subsequent distillation. The overall reasoning of the paper relies on a heuristic assumption that perplexity differences reflect knowledge differences; however, this assumption lacks rigorous theoretical grounding and comprehensive empirical validation. The core procedure depends on several empirically chosen hyperparameters and the stable behavior of generative models, which limits robustness. Furthermore, the experimental design shows constraints in dataset selection and baseline comparisons, making it difficult to demonstrate the generality and reliability of the proposed approach. Overall, while the work attempts to address a practical problem, the current experiments and analysis do not sufficiently substantiate its effectiveness or novelty.

### Strengths
1. The authors address a genuine and relevant challenge in model transfer and knowledge distillation — how to extract specialized knowledge from an existing fine-tuned model when the original fine-tuning data are unavailable. This problem has practical significance in real-world scenarios, particularly where data access is limited or restricted by compliance constraints.
2. TuneShift-KD avoids the need for additional discriminators or manual labeling by relying on perplexity-based filtering and iterative data generation to construct distillation samples. The overall framework is relatively simple and can be integrated into existing fine-tuning pipelines with minimal modification.

### Weaknesses
1. Lack of theoretical grounding: The core assumption—that perplexity differences can effectively represent knowledge differences between models—has no solid theoretical justification. The authors provide only heuristic reasoning without statistical significance analysis or ablation comparing alternative indicators such as KL divergence or output diversity.

2. Overly heuristic and weakly interpretable method: The key filtering mechanism of TuneShift-KD depends on an empirically chosen threshold (e.g., τ = 1.5), yet the paper does not analyze sensitivity to this parameter or explain why it generalizes across tasks and model scales. This raises concerns about reproducibility and robustness.

3. Limited experimental scope and shallow validation: The evaluation is restricted to a few benchmarks (GSM8K, MBPP, BBH) that mainly test small-scale reasoning or code generation. There is no validation on more complex or safety-critical tasks (e.g., multi-turn dialogue, factual knowledge distillation, or alignment), undermining the claim of general knowledge transfer.

4. Unfair or insufficient baselines: Comparisons with existing approaches such as Trans-LoRA and LoRA-X lack strict control over model size, training steps, and computational budgets, which weakens the credibility of the reported performance improvements.

5. Missing ablation and failure analyses: The paper does not disentangle the contribution of each component in TuneShift-KD or analyze failure cases, leaving unclear whether improvements arise from the core idea or from secondary factors such as data regeneration or repeated distillation.

6. Marginal and statistically unstable improvements: On some datasets (e.g., MMLU), the gains are negligible or inconsistent, yet the paper does not report confidence intervals or variance across runs, making the results statistically unconvincing.

7. Limited novelty: The approach essentially combines known ideas from knowledge distillation and synthetic data generation, with the main modification being a new heuristic for sample selection. It lacks substantial theoretical or algorithmic innovation.

### Questions
1. Please elaborate on the theoretical foundation of the assumption that “perplexity difference reflects knowledge difference.” Is there a more rigorous explanation from an information-theoretic or distributional perspective? If so, could the authors provide mathematical reasoning or additional supporting experiments in the appendix?
2. The key threshold in TuneShift-KD (e.g., τ = 1.5) appears to be chosen empirically. How is this value determined? Does the performance vary significantly under different thresholds? A systematic sensitivity analysis would help demonstrate the robustness of the method.
3. The comparisons with methods such as Trans-LoRA and LoRA-X lack details about computational budget, training epochs, and learning rates. Could the authors clarify these settings to ensure that the reported results are reproducible under equivalent conditions?
4. Please specify under what conditions TuneShift-KD performs best, and in which scenarios (e.g., fact-heavy tasks or open-domain QA) it fails. Providing representative failure cases and analyses would help readers understand the boundaries of the method.
5. Beyond perplexity difference, have the authors explored alternative indicators such as KL divergence, cross-entropy margin, or token-level consistency? If so, please include comparative results to justify the choice of perplexity as the main criterion.
6. How does TuneShift-KD fundamentally differ from existing knowledge distillation or synthetic data generation approaches? The authors are encouraged to more clearly articulate the novel contribution and situate the method within the broader research landscape.

### Soundness
2

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
3

### Summary
The paper proposes TuneShift-KD, a novel approach that automatically distills specialized knowledge from a fine-tuned model to a target model. The method only uses a few examples that represent this specialized information. The key idea is that the specialized knowledge can be identified through perplexity differences between the base and fine-tuned models. The method uses this idea to create a synthetic dataset of training samples and performs SFT on these. The results demonstrate that TuneShift-KD captures a larger share of the fine-tuned knowledge than prior methods, while being easier to deploy in practice.

### Strengths
- The perplexity difference criterion is intuitive. Prompts where the fine-tuned models are confident but base models struggle can capture specialized knowledge. 
- Unlike Trans-LoRA, TuneShift-KD requires no discriminator, which makes the standard fine-tuning process simpler and more practical.
- The method shows accuracy gains over Trans-LoRA in GSM8K, MBPP, and BBH.
- The model is highly automatic, can transfer information across different architectures, and works without the exact base model.

### Weaknesses
- As the authors acknowledge, perplexity/likelihood-based selection of training samples is a well-known technique, known already 15 years ago (see e.g., [1]). Even in an LLM-based distillation context, log-likelihood/entropy-based methods have been used recently (see e.g., [2, 3]). This work is clearly part of the same family of data selection methods, limiting the novelty.
- My interpretation of the results is that the main performance driver compared to Trans-LoRA is the diversity of the prompts. It is unclear if the difference is due to the discriminator vs perplexity-based selection of synthetic data or simply the generator model for prompts.
- The paper frames itself as trying to solve the distillation of domain-specific, specialized knowledge. However, the successful benchmarks are BBH, math, and coding, which can be seen as a big mismatch with the stated motivation. Because these benchmarks are so generic, it is easy to see how Gpt-4o (or Qwen as in Appendix C) can generate relevant prompts. However, if the domain is truly out-of-distribution (such as internal company data), it is unclear if a generic target model can generate relevant prompts even in a few-shot setting to achieve a sufficient factual coverage. The poor performance in MMLU (Table 5) indicates that this issue might be serious. This makes a method like LoRA-X much more attractive, despite its significant drawbacks.
- The assumption that a subset of the data, but not all of it, can be kept to act as seed samples is arguably quite restrictive. How realistic is it, really, especially assuming that you nevertheless retain access to the model weights?
- No error bars, variances, or multiple seeds. Are the results statistically significant?
- Lack of information about the data for fine-tuning the source model. 
- The perplexity difference between the fine-tuned model and the base model could also be partially due to the style of the fine-tuning data (or some other reason causing a generic distribution shift), which could harm the perplexity threshold-based selection in a more realistic setting.

[1] Moore, R. C., & Lewis, W. (2010). Intelligent selection of language model training data. In Proceedings of the ACL 2010 conference short papers (pp. 220-224).

[2] Li, J., Nag, S., Liu, H., Tang, X., Sarwar, S., Cui, L., ... & Tang, J. (2024). Learning with less: Knowledge distillation from large language models via unlabeled data. arXiv preprint arXiv:2411.08028.

[3] Liu, J., Zhang, C., Guo, J., Zhang, Y., Que, H., Deng, K., ... & Zheng, B. (2024). Ddk: Distilling domain knowledge for efficient large language models. Advances in Neural Information Processing Systems, 37, 98297-98319.

### Questions
- If you argue that your perplexity-based prompt selection method is meaningfully different from Li et al.'s teacher confidence/student uncertainty for sample selection (see [2] above), this sample selection method should be a baseline.
- Would it be possible to run experiments with, for instance, LLama2 for prompt generation, to truly isolate whether the perplexity-based selector is meaningfully different from the discriminator in terms of performance (or if it's simply a question of efficiency & ease of implementation)?
- Can you run experiments on synthetic, truly out-of-distribution data and compare your method to Trans-LoRA, LoRA-X, and standard SFT on the original data (as an upper bound) under that setting?
- Trans-LoRA has released at least some code as part of the paper supplementary material (https://openreview.net/forum?id=c3Pakdyi3t). Did you consider this when claiming there's no open-source implementation of Trans-LoRA or is there some issue with this code?
- You write that GANs suffer from mode collapse, which affects Trans-LoRA. That seems unlikely, as the generator in Trans-LoRA is the target model M_t (with only instruction tuning). On the other hand, the objective mismatch makes more sense as an explanation, but could you provide empirical evidence in support of that being the issue (vs the generator model)?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes TuneShift-KD, a method for transferring specialized knowledge from a fine-tuned source model (e.g., LoRA-adapted) to a new target model without access to the original fine-tuning data.
The key insight is to identify “specialized knowledge” regions using perplexity difference between the fine-tuned and base models: prompts where the fine-tuned model is confident (low PPL) but the base model is uncertain (high PPL) are assumed to encode domain-specific expertise.

Using a few seed examples, an instruction-tuned LLM (e.g., GPT-4o) generates synthetic prompts. These are filtered by the perplexity-difference criterion to form a synthetic dataset used for knowledge distillation (via NLL loss) into the target model.
Experiments on GSM8K, MBPP, and BBH benchmarks show consistent gains over Trans-LoRA, a previous data-free transfer method, while being simpler and applicable across model families (LLaMA2, Gemma, Qwen).

### Strengths
The paper addresses a practical and emerging challenge, motivated by real deployment constraints (privacy, cloud-hosted models, hardware vendors): transferring LoRA-fine-tuned expertise when the fine-tuning data are unavailable.

The perplexity-difference criterion is intuitive, theoretically grounded (entropy and KL analysis), and easy to implement.

Broad applicability and strong empirical results

### Weaknesses
While elegant, the main novelty—using PPL difference as a filter—is incremental compared to prior perplexity-based data filtering

Results are limited to small-/mid-scale models (≤13 B). It is unclear whether the method scales or remains stable for larger modern architectures (e.g., 70 B).

Reported improvements (1–7 pp) are modest and could lie within the noise range of evaluation harnesses, yet statistical significance is not reported.

I think the tasks (GSM8K, MBPP, BBH) are general-domain, not truly “specialized knowledge.” Stronger validation would involve real domain-specific fine-tuning (e.g., medical, legal, or code-domain LoRAs) where data unavailability is realistic.

### Questions
How exactly is “specialized knowledge” defined or operationalized? Does it refer to new factual content, new reasoning skills, or simply distributional shift between base and fine-tuned models?

The paper claims improved prompt diversity over Trans-LoRA. Can this be quantified (e.g., via lexical or semantic diversity metrics)?

The paper uses NLL loss on fine-tuned model outputs. Have the authors compared it with logit-based or KL-based distillation (e.g., temperature-scaled soft labels)?

### Soundness
3

### Presentation
3

### Contribution
2
