# Watermark Robustness and Radioactivity May Be at Odds in Federated Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 2

## Abstract
Federated learning (FL) enables fine-tuning large language models (LLMs) across distributed data sources.  As these sources increasingly include LLM-generated text, provenance tracking becomes essential for accountability and transparency.   We adapt LLM watermarking for data provenance in FL where a subset of clients compute local updates on watermarked data, and the server averages all updates into the global LLM. In this setup, watermarks are radioactive: the watermark signal remains detectable after fine-tuning with high confidence. The p-value can reach $10^{-24}$ even when as little as 6.6% of data is watermarked.  However, the server can act as an active adversary that wants to preserve model utility while evading provenance tracking. Our observation is that updates induced by watermarked synthetic data appear as outliers relative to non-watermark updates. Our adversary thus applies strong robust aggregation that can filter these outliers, together with the watermark signal.  All evaluated radioactive watermarks are not robust against such an active filtering server. Our work suggests fundamental trade-offs between radioactivity, robustness and utility.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper investigates the behavior of watermarking techniques in federated learning settings. The experiments reveal that watermark signals remain detectable when large language models are fine-tuned on datasets containing a small proportion of watermarked samples. However, an active adversarial server can identify watermarked synthetic samples as outliers and remove them to evade provenance tracking. The evaluation systematically explores these findings and addresses four key research questions.

### Strengths
+ The topic of watermarking in federated settings is interesting and important.

### Weaknesses
- The threat model could be more clearly defined. In particular, it remains unclear whether all users in the federated learning setting employ the same key and watermarking technique when synthesizing data samples.
- The main insight of the paper is that watermarking techniques introduce low distortion. However, more recent distortion-free watermarking methods have been proposed. The paper does not discuss how such techniques could be integrated into provenance tracking or how adversaries might evade them. It would also be valuable to analyze whether synthesized data from distortion-free versus low-distortion watermarking methods affect the model’s training performance differently.
- The evaluation considers only two watermarking techniques, which limits the generalizability of the conclusions. Including a broader range of watermarking methods and datasets would make the analysis more comprehensive and convincing.

### Questions
+ How does the adversarial server determine which clients have watermarked their datasets with synthetic samples and which have not? In particular, what criteria or detection methods are used to identify these outliers?
+ If a larger proportion, or even a majority, of clients employ watermarking to synthesize their data, how can the adversarial server still effectively detect or filter out watermarked samples under such conditions?

### Soundness
2

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
This paper studies the use of LLM watermark radioactivity in federated learning to enable clients to determine whether the model was trained on their data. More specifically, the authors show that a malicious server could distinguish updates from clients using watermarked text and non-watermarked text, thereby evading accountability.

### Strengths
- This is the first work that studies whether the watermark radioactivity properties still hold in the federated learning setting, where multiple gradients from watermarked and unwatermarked data are merged.
- The method is well explained from an intuitive perspective.

### Weaknesses
- The actors' goals are hard to understand. In l.135, the authors write `In FL provenance, an ε fraction of clients, that we denote as watermarking clients, aim to prove that their datasets were used to train the global model.` This raises the questions:
	- If the clients want to know whether their data are used to train the model, why do they send their updates in the first place?
	- Because the clients send their updates, do they not already know that their data are used to train the model?
	- Even if they use the watermark and the server is malicious, they still know whether their data were used to train the model:
		- If the p-value is high, it is because the malicious server successfully removed the watermarked updates, thus the data were NOT used to update the model. This agrees with the p-value result.
		- If the p-value is low, the malicious server was not successful in removing the watermarked updates, thus the data were used to update the model. This also agrees with the p-value result.
	- Lastly, because the clients know the weights of the model, they can easily identify which model in the wild was trained with their data by using a provenance method relying on the weights.
- The evaluation is not complete enough:
	- Utility evaluation:
		- The current evaluation only evaluates the impact of watermarked updates on the model entropy, not the impact of the method per say. If the (watermarked) updates are poor with respect to quality, this is independent of the method.
		- What would be interesting is the impact on utility when using the robust aggregator with only clean data. Moreover, if the data come from non-IID clients, can you lose some impactful capabilities because of the robust aggregation?
		- For measuring utility, using LLM benchmarks (HumanEval, GSM8k, MMLU, ...) would be more meaningful than the entropy of the model.
	- The models used are both small and old. The authors noticed that the larger the model, the higher the radioactivity. What would happen with recent 3B or 8B models? Would the small evasion rate be enough for those models to be radioactive?
	- Using i.i.d. clients is not realistic. I think the method works because there are two distributions among the clients, the watermarked and the non-watermarked. What if each client has its own domain-specific dataset that is either human or synthetic? In this case, would the method easily separate watermarked text?
	- The authors only use two watermarking schemes, and only one is successful. It is unclear whether the attack still works against more advanced watermark schemes.
- The novelty is limited: the malicious server method is standard outlier detection, and watermark robustness is already well-studied (though not in the FL setting). In particular, the authors claim in l.74 that they adapt existing watermarking schemes, but this is not true; the schemes are used as is.
- The definition of robustness is not useful: without putting constraints on the adversary's capabilities, the existence of the adversary is almost certain (an oracle adversary could know the watermark updates with 100% accuracy).

### Questions
- Could the authors explain what is the goal of the different actors, i.e. what are they trying to accomplish and why?
- Could they evaluate the impact of their methods on utility when no watermarked updates are present, especially using LLM benchmarks and non-iid data? (see the weakness section for more details)
- Could the authors show whether the malicious server still succeeds with bigger and newer models?
- What happens if the clients are not iid? Is the method still able to distinguish watermarked updates? Is the utility impacted when no watermarked clients are present because "useful" updates are filtered?
- Could you explain why you dont use distortion-free hashing based schemes like AAR or SynthID-text? Especially given that you acknowledge in l340 that they are radioactive. Could the fact that they are distortion-free bypass the outlier detection method when all clients are synthetic data?
- Can you justify the claims that KTH detection can not aggregate other the prompts (l341 and 353)? Why can't you just concatenate the different prompts given that, as mentioned in l783, `If none of the clients apply watermarking, the data is independent and identically distributed.`?
- OFR is quite high (around 50%). How is this not an issue for utility/training? Especially when no clients are watermarked, OFR is at above 90%.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work investigates watermark radioactivity in federated learning. The authors first demonstrate that watermarked data exhibits radioactive behavior in federated learning systems. They then formulate an active adversary threat model, in which a malicious server attempts to identify model updates originating from watermarking clients and exclude them from aggregation. Several experiments are conducted, showing that existing watermarking methods fail to evade detection by robust aggregation mechanisms.

### Strengths
1. This work addresses the problem of data provenance, which is a critical and timely topic.

2. The authors identify the potential existence of an active adversary aiming to maintain model utility while evading provenance tracking, highlighting an important and realistic threat scenario.

3. Experimental results demonstrate that none of the existing watermarking methods simultaneously achieves radioactivity, robustness, and utility, providing valuable insights into the trade-offs and challenges of watermarking in federated learning.

### Weaknesses
1. Beyond formulating the problem of how an adversary can evade provenance tracking, this work lacks substantial technical contributions. Both the watermarking methods and the robust aggregation mechanisms are directly borrowed from prior works.

2. The motivation for federated learning provenance is not well established. Specifically, as stated in lines 134–135, “In FL provenance, an $\epsilon$ fraction of clients, denoted as watermarking clients, aim to prove that their datasets were used to train the global model.” However, this assumption seems questionable—clients in an FL system already agree to contribute their data for model training, so it is unclear why they would later need to prove their participation. Clarifying this motivation is essential to justify the relevance and practicality of the problem formulation.

3. The experimental evaluation is limited in scope, particularly regarding the diversity and scale of LLMs. More widely adopted and representative models, such as LLaMA and Gemma, are not included in the evaluation. Additionally, the models used in the experiments are relatively small; incorporating larger and more realistic models would provide stronger and more generalizable results.

### Questions
1. In lines 43–45, the authors state that “models fine-tuned on watermarked LLM-generated text exhibit radioactivity in a centralized setting, where watermark signals remain detectable after fine-tuning,” citing Sablayrolles et al. (2020) and Sander et al. (2024). However, in lines 51–53, they claim that “continued fine-tuning on non-watermarked (clean) data can substantially reduce watermark detectability,” citing Sander et al. (2024) again. These two statements are contradictory, as one suggests watermark persistence while the other implies its removal, both referencing the same work. Why does the author make these two contradictory statements in the paper?


2. Many important details about the fine-tuning process on local clients are missing. It is unclear whether clients fine-tune their models using PEFT techniques (e.g., LoRA) or full model fine-tuning. Such details are critical for reproducibility and for understanding the impact of local training on watermark detectability.


3. Figure 1 shows that updates from clean and watermarked clients can be clearly separated in the feature space. This raises an important question: why is a strong Byzantine aggregator still necessary if the two groups are already distinguishable? Furthermore, the visualization suggests a potential for designing a new filtering mechanism. For instance, if the updates naturally form two clusters and clean clients constitute the majority (as assumed in the paper), the smaller cluster could be interpreted as the watermarking clients and filtered out directly.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper explores whether data provenance in federated learning can be achieved through watermarking text produced by large language models. It studies two statistical watermarking methods, KGW+ and KTH+, in a setting where multiple clients collaboratively fine-tune a shared model. When the server simply averages the client updates, the watermark signal persists and the resulting global model still reflects traces of the marked data. When the server instead applies a robust aggregation method that filters unusual updates, the watermark signal disappears while the model's accuracy remains stable. The authors interpret this behavior as evidence of a tension in federated training: model utility, robustness, and detectable provenance cannot all be achieved simultaneously.

### Strengths
The study is thorough in its empirical design. It tests two watermarking schemes across several model sizes in the Pythia family and varies the proportion of watermarked data from small to moderate levels. The main finding is clean: when the server aggregates client updates with a robust method, the watermark signal largely disappears while model performance remains unchanged. This provides a clear picture of how robustness in federated learning can suppress provenance signals that rely on small statistical traces.

### Weaknesses
1. The paper sets out to test whether watermarking can serve as a foundation for data provenance in federated learning, yet the experiments ultimately show that the idea fails once the setting becomes realistic. The limitation lies in the premise more than in the execution. Statistical watermarks are already known to be fragile: their signal weakens with continued training and can be removed with little effort, as demonstrated by Zhang et al. (2023), "Watermarks in the Sand...". In this light, the finding that federated aggregation erases the watermark confirms a known vulnerability rather than uncovering a new phenomenon. A stronger study would begin with a clear argument for why watermarking might succeed in such an environment before investigating its breakdown.

2. The analysis considers only statistical watermarking methods that introduce small sampling biases into the model's output. As the share of watermarked data grows, these biases distort learning and reduce model performance. The authors interpret this behavior as evidence of an inherent trade-off between watermarking and utility, but the effect arises from the particular design of the watermark rather than from any general incompatibility. Cryptographic watermarking approaches based on pseudo-random codes (Christ and Gunn, 2024) define a better design space. Such schemes embed information through computational indistinguishability rather than statistical bias, and therefore would not degrade utility in the same way. Because this line of work is not discussed, the paper's conclusions remain narrower than they appear.

3. The experiments assume that a subset of clients share the same watermark key and the same generative model. This simplifies the analysis but does not reflect realistic federated systems, where clients hold distinct data, use different local processes, and keep their parameters private from the server. The separability observed in the experiments arises from this synchronization rather than from any intrinsic property of federated learning. The assumption also explains why using LLM watermarking for provenance in FL is a bad idea. Provenance requires independence among clients and limited trust in the server, whereas watermarking depends on shared structure and coordinated keys.

4. The central finding, that robust aggregation removes watermark signals while preserving accuracy, is empirically clear but theoretically unsurprising. Robust aggregation is designed to filter outliers, and gradients shaped by watermarking behave like outliers. The paper reports this effect but stops short of analyzing why it occurs or whether a different watermarking design might resist it. The analysis also ends at thirty percent watermarking clients, leaving open what happens at higher or lower proportions and why performance degrades as the watermark share increases. A closer examination of these dynamics, perhaps contrasting statistical and pseudo-random code watermarking, would make the contribution more substantial.

### Questions
1. Why was the study limited to detectable statistical watermarks? How would cryptographically undetectable (PRC-style) schemes behave under the same setup?
2. Why are all watermarking clients assumed to share the same key and generative model? How would heterogeneity in keys or generators affect your results?
3. Have you tested beyond \epsilon = 30% to confirm whether the reported utility–robustness trade-off persists or saturates?
4. What practical insight should readers take from this result, given that statistical watermarks are already known to be non-robust outside federated learning?

### Soundness
2

### Presentation
3

### Contribution
1
