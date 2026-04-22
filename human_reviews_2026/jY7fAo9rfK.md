# Benchmarking Empirical Privacy Protection for Adaptations of Large Language Models

- Avg Score: 5.50
- Decision: Accept (Oral)
- Scores: 4, 4, 6, 8

## Abstract
Recent work has applied differential privacy (DP) to adapt large language models (LLMs) for sensitive applications, offering theoretical guarantees. However, its practical effectiveness remains unclear, partly due to LLM pretraining, where overlaps and interdependencies with adaptation data can undermine privacy despite DP efforts. To analyze this issue in practice, we investigate privacy risks under DP adaptations in LLMs using state-of-the-art attacks such as robust membership inference and canary data extraction. We benchmark these risks by systematically varying the adaptation data distribution, from exact overlaps with pretraining data, through in-distribution (IID) cases, to entirely out-of-distribution (OOD) examples. Additionally, we evaluate how different adaptation methods and different privacy regimes impact the vulnerability. Our results show that distribution shifts strongly influence privacy vulnerability: the closer the adaptation data is to the pretraining distribution, the higher the practical privacy risk at the same theoretical guarantee, even without direct data overlap. We find that parameter-efficient fine-tuning methods, such as LoRA, achieve the highest empirical privacy protection for OOD data. Our benchmark identifies key factors for achieving practical privacy in DP LLM adaptation, providing actionable insights for deploying customized models in sensitive settings. Looking forward, we propose a structured framework for holistic privacy assessment beyond adaptation privacy, to identify and evaluate risks across the full pretrain-adapt pipeline of LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a comprehensive benchmark to quantify and compare the empirical privacy protection abilities of large language models with differential privacy guarantees. The proposed benchmark leverages robust membership inference and canary data extraction to investigate the potential privacy risks. Unlike prior works, the benchmark considers data distributions of adaptation data and pretraining data. Both in-distribution cases and out-of-distribution examples are carefully studied. Moreover, the benchmark covers various LLMs, adaptation methods, privacy budgets and datasets to perform extensive evaluations. Experimental results show insights on several research questions.

### Strengths
1. The paper is well written with clear motivations. The empirical privacy risk assessment is an important task for DP-protected LLMs.

2. The paper systematically studies the relationship between the pretraining data and the fine-tuning data (adaptation data). Different datasets are considered for both in-domain and out-of-domain setups.

3. Comprehensive experiments are conducted with new robust MIA attacks and exposures.

### Weaknesses
1. My major concern is the evaluation of the utility part. According to the paper, only C.6 and E.2 discuss the privacy-utility tradeoff. Yet, utility is crucial when using various $\epsilon$, under a strict budget $\epsilon = 0.1$, if the model cannot handle the downstream tasks at all, DP tuning will be meaningless. However, the paper only evaluates perplexity and validation loss as the utility indicator. Such utility metrics are not feasible or representative of the utility performance. A more comprehensive utility evaluation should be properly studied.

2. The evaluated LLMs are rather small-scale and outdated. I would encourage OLMo 2 7B to study how the size of LLMs affects privacy performance.

3. Clarity regarding section 6. I am quite confused about the privacy auditing mentioned here. I think more background and explanations on holistic audits would be helpful to understand this section.

### Questions
1. How do the authors justify using perplexity and validation loss as sufficient indicators of model utility under differential privacy? Have the authors explored fine-tuning under DP constraints to assess out-of-domain task utility?

2. How do the authors expect their findings to scale with larger models such as OLMo 2 7B or Qwen2.5/3 architectures?

3. Could the authors define “holistic privacy audits” more precisely? How does it differ from existing auditing frameworks?

### Soundness
3

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
4

### Summary
This work studies privacy risks under DP adaptations in LLMs, using robust membership interference and canary data extraction. They study the effects of overlaps and interdependencies of the pretraining data with the adaptation data, using exact overlaps, in-distribution data and out-of-distribution data. They also study the vulnerability for different adaptation methods and different privacy regimes. The authors find that the data distribution strongly affects the privacy vulnerability, and that parameter-efficient fine-tuning methods, such as LoRA, achieve the highest empirical privacy protection for OOD data.

### Strengths
- This work studies a comprehensive range of datasets, adaptation methods, and different LLMs with a range of sizes.
- The authors run experiments on a range of IID and OOD datasets, and are able to show empirical trends on how the IID settings have higher privacy vulnerabilities than OOD settings.
- This work is highly relevant to practical applications using DP with LLMs.

### Weaknesses
- The experiments could be run on a wider range of privacy parameters (the authors only use $\epsilon=0.1, 8$ and the nonprivate setting).
- The authors discuss the privacy-utility tradeoffs for the different methods in the appendix. However, this is quite an important topic and I think it is worth moving this to the main paper. Similarly, it would also be useful to include some discussion on the costs of the different methods and the tradeoffs with privacy and utility.
- The authors propose a new framework for holistic privacy assessment using adversarial games, but it would help to give examples to demonstrate how this framework would help in practice.
- This work only focuses on open-source models, and it is unclear how to generalize the results to close-source models where the pretraining data distribution is not known.
- The work is empirical and does not give any theoretical explanations, for instance on why some adaptations have lower vulnerability than others.

### Questions
- How could this work apply to closed-source models? Are there any tools or insights from this work that could apply there?
- Are there explanations on why different adaptation methods perform better than others?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper examines how well differential privacy protects sensitive data during large language model adaptation. It studies privacy risks under various data relationships, including overlapping, in-distribution, and out-of-distribution cases, and across several adaptation methods such as full fine-tuning, LoRA, and prefix tuning. The findings show that privacy leakage increases when adaptation data closely matches pretraining data, even without direct overlap, while LoRA offers the strongest empirical protection for out-of-distribution data. The authors also propose a holistic privacy auditing framework that evaluates risks throughout the entire lifecycle of an LLM from pretraining to adaptation to ensure responsible use in sensitive domains

### Strengths
1. The paper provides a thorough empirical evaluation across multiple datasets, adaptation methods, and privacy regimes, offering one of the most systematic analyses of differential privacy in LLM adaptations.

2. The introduction of a structured framework for auditing privacy across pretraining and adaptation stages is a novel and forward-looking contribution, addressing a key gap in existing LLM privacy research.

3. The work directly informs real-world use of LLMs in sensitive domains, helping bridge the gap between theoretical guarantees of differential privacy and practical, empirically validated protection.

### Weaknesses
1. The most contradictory finding concerns Prefix Tuning: in data extraction attacks against adaptation data (RQ3), it was found to be the most vulnerable method. However, regarding the "forgetting" effect on pretraining data (RQ5), it suddenly became the most effective, showing the least leakage. The paper points out these phenomena but does not deeply explain why this trade-off exists. This makes the question of "which method should practitioners choose?" very ambiguous.
2. The paper's core finding is that "distributional closeness is the main driver of risk". However, the paper's definition of "closeness" is purely categorical, namely "Overlap, IID, and OOD. This categorization is very coarse. The paper provides no quantitative metrics (e.g., Wasserstein distance or KL divergence using word embeddings) to measure the actual distributional distance between the pretraining and adaptation data. This makes the core conclusion, while intuitive, methodologically less rigorous. In practical applications, if two datasets are not clearly IID or OOD, the paper's conclusions are difficult to apply for guidance.

### Questions
please see above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work tests how well Differential Privacy (DP) protects sensitive information when adapting Large Language Models (LLMs) to new data. The authors evaluate a variety of DP adaptation methods including Prefix Tuning, LoRA, Full Fine-Tune, and Head Fine-Tune. They evaluate if they can identify (membership inference attack) or extract (data extraction attacks) specific data used during adaptation. They test this for 4 different types of data: overlapping data, in distribution data, out of distribution data and canary data (only used for extraction). They evaluate seven pretrained models such as Pythia, GPT-Neo and OLMo that are all trained on the Pile dataset. The authors provide 5 interesting insights from these experiments such as
- Higher privacy regime is required for data closer to the pretraining distribution for similar empirical privacy.
- Efficient methods like LoRA generally offer better privacy than full fine-tuning
- Common privacy settings (like $\epsilon=8$) aren't strong enough to stop powerful attacks
- The performance of MIAs highly depends on the attacker’s knowledge of the target model and pretraining data

Finally, the authors also provide a new way to holistically audit the privacy of pretraining and private adaptation together.

### Strengths
- The paper evaluates a common paradigm of adapting non-privately pretrained LLMs with sensitive data using privacy preserving mechanisms.
- Very systematic evaluation of empirical privacy risks when varying different parameters like distance of private data from pretraining, type of adaptation and attacker knowledge.
- Covers a large number of models and adaptation mechanisms.

### Weaknesses
- Only 2 epsilons are evaluated, 0.1 which is considered very private and 8 which is considered pretty large. Would be nice to also have something in the middle.

### Questions
- Could the authors clarify measure the distributional closeness of the OOD vs IID datasets used?
- This might be a stretch but I think it might be interesting to compare a DP pretrained model like VaultGemma in this comparison as well. It is recently released and the dataset is not publicly known but it might be a nice addition to this analysis.

### Soundness
4

### Presentation
4

### Contribution
3
