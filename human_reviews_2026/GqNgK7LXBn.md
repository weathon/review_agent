# Private Data Synthesis for Preference Alignment of Large Language Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 2, 6, 4

## Abstract
Preference alignment has become a crucial technique for aligning large language models (LLMs) with human values. However, training on real human preference data raises privacy concerns, as these datasets often contain sensitive user prompts and human judgments. To address this, we propose **DPPrefSyn**, a novel algorithm for generating differentially private (DP) synthetic preference data to enable privacy-preserving preference alignment. DPPrefSyn addresses three key challenges: modeling diverse human preferences via DP clustering and per-cluster DP scoring models; reducing dimensionality with DP-PCA to improve efficiency; and conserving privacy budget by leveraging public prompts. We conduct extensive experiments on three standard benchmarks and compare our method with DP fine-tuning on real data. Our results show that our framework achieves competitive performance under strong privacy guarantees. These results open up new possibilities for preference alignment with privacy protection for a broad range of applications. To the best of our knowledge, this is the first work to generate DP synthetic preference data for LLM alignment.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studied the privacy issue of the dataset in LLM alignment. They consider protecting the information of prompts and preference data. In order to achieve it, the authors proposed an algorithm called DPPrefSyn based on DP-PCA, DP-clustering, and DP-SGD to generate DP synthetic data. Their methods have advantages in data diversity, reducing data dimension, and saving privacy budget by using public data. The authors provided a privacy guarantee for their method and implemented their method on three benchmarks: OpenAssistan,  Anthropic-HH, and the TL;DR summarization task to show that their method also guarantees some competitive utility empirically.

### Strengths
1.  The writing of the paper is clear.
2. The literature review is comprehensive. Most of the related work is discussed.
3. The problem of private synthetic data to protect prompt and preference information in the paper is interesting.
4. The empirical results are comprehensive to implement the methods on different tasks and datasets.

### Weaknesses
1. The authors provided a privacy guarantee for their method. However, the privacy and utility trade-off is important in DP, and there is no utility theoretical guarantee.
2. The key methods of DP-PCA, DP-KMeans, and DP-SGD are proposed in previous work. That means the novelty of the paper is limited.

### Questions
1. Lack some important related work, e.g. [1]. And discuss whether the method in [1] can be used for your task?
2. The paper claims DPPrefSyn “outperforms the utility of fine-tuning without DP constraints” (DP-FT with ε = ∞) while maintaining privacy. Could the authors clarify how this is theoretically possible, given that DP typically introduces noise that reduces utility?
3. The paper relies on the assumption of a linear model for rewards. But in practice, it is mostly non-linear. Discuss how the reward model satisfied this linear assumption in your task.
4. What are the computational costs of DP clustering and DP-PCA for large-scale preference datasets？
5. How to deal with the distribution shift problem for the prompts? That is, if the distribution of prompts in the public set is different from the private set, how can you guarantee the method still works?

[1]. Liu, Xiyang, et al. "DP-PCA: Statistically optimal and differentially private PCA." Advances in neural information processing systems 35 (2022): 29929-29943.

### Soundness
3

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
Authors propose a novel method for differentially-private (DP) alignment. In the first stage, authors essentially learn a DP reward model. This is done by 1) extracting prompt-response pair features using an embedding model, 2) DP-PCA to reduce dimensionality, and then DP-K means to cluster the data, 3) learn linear reward model for each cluster. Then, on public data, authors use this DP reward model to rank responses to generate a new preference data. This new preference data can now be used for different alignment methods such as SFT, DPO, and PPO.

### Strengths
Quality: Authors propose a method based on multiple well-established components in the literature, namely DP-PCA, DP-KMeans, and DP-SGD. This composition allows the authors to develop a principled method with a strong foundation in prior work, giving readers greater confidence in the practical utility of the proposed approach.

Originality, Significance: The proposed method for preference data synthesis departs significantly from previously proposed DP alignment approaches and, as the authors claim, has the merit of being applicable to any preference-based alignment algorithm. This versatility could promote wider adoption of the method and enhance its significance in shaping future research.

Clarity: The writing is easy to follow, and the main ideas are clearly conveyed. The limitations of the proposed method could be discussed more explicitly, however. More on this in the Weaknesses section.

### Weaknesses
My primary concern is that the authors do not control for key differences between the private preference data and the synthetically generated data: (1) the quality of prompts and (2) the quality of candidate responses. Even the dataset volume is not controlled; for instance, the Alpaca dataset used as public data is larger than the portion of OpenAssistant used as private data. Since responses in the synthetic dataset are generated from LLaMA-7B-Chat, they may also be closer in distribution to the Pythia-2.8B policy being fine-tuned. These factors could account for DPPrefSyn’s superiority over DP-FT, making it difficult to conclude that DPPrefSyn is intrinsically a better DP method. In settings where “better” prompts and “better” response generators are unavailable, DPPrefSyn may not outperform DP-FT. In this work, across all three benchmarks, DPPrefSyn with no privacy control ($\epsilon=\infty$) already performs substantially better than DP-FT. Hence, it remains unclear whether DPPrefSyn’s advantage under private settings arises from better algorithmic design or from the use of higher-quality data. I suggest the following to strengthen the empirical claims:

1. Prompt-set flip: swap public and private prompt sets to estimate the impact of prompt quality.
2. Enhanced DP-FT baseline: in the DP-FT setting, use the same LLaMA-7B-Chat generator to produce responses and rank them with public reward models, ensuring the baseline is sufficiently strong for a fair comparison.

Finally, the method assumes that the reward model is a linear function of extracted features. This assumption is uncommon in the preference-alignment literature and deserves clearer discussion. Although the authors partially address this by clustering the data and training cluster-specific linear models, it remains unclear to what extent this bridges the gap between standard practice (fine-tuning a pretrained LLM reward model) and the proposed simplified reward formulation.

### Questions
Could the authors avoid the DP-PCA step by instead:

1. performing non-private PCA on the private data to obtain the projection matrix,
2. applying DP-KMeans on the PCA-projected representations, and
3. using these PCA features (rather than the original high-dimensional embeddings) for DP-SGD reward modeling?

Since the downstream stages (DP-KMeans and DP-SGD) already ensure differential privacy, the overall procedure would remain DP-compliant. While using reduced-dimensional features might modestly affect the quality of the reward model, this alternative would preserve more of the privacy budget for later stages. Hence, it is unclear whether employing DP-PCA provides any tangible benefit compared to simply using non-private PCA for dimensionality reduction before applying the DP algorithms.

### Soundness
2

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
The paper addresses privacy of human-annotated preference data used to fine-tune LLMs for preference alignment. The paper propses a novel method to generate privacy-preserving synthetic preference data as an intermediate on which to apply non-private SFT/DPO.  The method first performs a DP clustering of private preference data, and then trains a scoring model for each cluster to capture preference patterns. The method uses public prompts for the synthetic datasets so that the privacy budget can be allocated for the clustering.

### Strengths
- The paper gives a well-presented intuition of the proposed method.
- The experimental results show impressive successes on on the reference tasks.
- The reusability of the extracted synthetic preference data is a neat trick.

### Weaknesses
- The paper presents results for larger privacy budgets ($\epsilon > 1$) only. It would be helpful to see evaluation vs the totally non-private baseline as well as smaller (<1) $\epsilon$ values.
- There is some concern about using GPT-4o as a proxy for human preference in determining the win rates in the experiments. While I understand that human studies may not be feasible, it would help to provide more explanation of how the win rate is calculated and why the win rate is a good measure.

### Questions
- Is the grid search in the DP-FT a privacy leak?
- Have you experimented with different ways to divide the privacy budget across the method steps rather than equal $epsilon$ values for each? Does this have any impact on the results?

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
4

### Summary
This paper proposes DPPrefSyn, a method to generate differentially private synthetic preference data for LLM preference alignment. The method uses DP-PCA, DP-KMeans clustering of embedding-difference features, and DP-SGD-trained linear scoring models on each cluster. The public prompts plus model-generated candidates are then filtered via the private scoring functions to produce synthetic preference pairs. The resulting synthetic dataset can be used to align models (e.g., via SFT + DPO) without direct access to private preference data. The authors present performance on various datasets and also report empirical membership-inference robustness.

### Strengths
- The pipeline combining DP-PCA, DP-KMeans, and DP-SGD with careful privacy accounting is interesting. 

- Empirical results are interesting and show promise. DP synthetic data can be used across models and preference-optimization algorithms.

-  This work includes MIA evaluation suggesting improved privacy robustness vs DP-SGD baselines and also shown some ablation studies.

### Weaknesses
- The key privacy claim that current preference-alignment pipelines meaningfully leak user privacy  is stated but not demonstrated. The paper does not show privacy leakage of existing RLHF datasets or methods. Without concrete evidence, the privacy motivation feels assumed rather than established in this context. This weakens the necessity argument.

- There is no demonstration of harm or attack feasibility on real preference data. The work evaluates MIA only on their synthetic pipeline. It does not show privacy leakage from non-DP RLHF training, privacy risk in publicly-released human-preference datasets. 

- Synthetic responses are sampled from a strong LLM, private data may reflect lower-quality annotation distributions. This makes utility results hard to interpret as purely privacy-driven improvements.

- Although justified by DP constraints, the choice limits expressiveness. A brief discussion on missed nuance (e.g., contextual preference factors) would help.

- The public prompts may differ from private preference distributions, this work does not provide an analysis of the effect on preference fidelity.

### Questions
Please refer to discussion in weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
