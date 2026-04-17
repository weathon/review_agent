# Semantic Voting: A Self-Evaluation-Free Approach for Efficient LLM Self-Improvement on Unverifiable Open-ended Tasks

- Decision: Accept (Poster)
- Scores: 4, 6, 2, 6

## Abstract
The rising cost of acquiring supervised data has driven significant interest in self-improvement for large language models (LLMs). Straightforward unsupervised signals like majority voting have proven effective in generating pseudo-labels for verifiable tasks, while their applicability to unverifiable tasks (e.g., translation) is limited by the open-ended character of responses. As a result, self-evaluation mechanisms (e.g., self-judging and entropy minimization) are predominantly used to derive pseudo-labels. However, self-evaluation relying on LLMs typically incurs high computational overhead and introduces overconfidence issues due to intrinsic biases. To address these challenges, we propose a novel self-evaluation-free approach for unverifiable tasks, designed for lightweight yet effective self-improvement. Inspired by majority voting commonly employed in verifiable tasks, we propose semantic voting as a novel mechanism that relaxes the principle of hard matching (i.e., exact matching) toward soft matching (i.e., semantic similarity). Soft matching is achieved by leveraging a lightweight sentence embedding model to quantify semantic similarity, thereby mitigating excessive computational burden and intrinsic bias-associated limitations of self-evaluation. Comprehensive experiments demonstrate that our method achieves substantial gains in computational efficiency and overall better performance than self-evaluation methods across diverse model architectures and tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Semantic Voting (SVSI), a novel self-improvement framework for LLMs on open-ended, unverifiable tasks. The core idea is to extend majority voting to the semantic space by generating multiple candidate responses, clustering them to find a semantic consensus, and then using the average semantic similarity as a score to construct preference pairs for DPO training. The method aims to be a computationally efficient and self-evaluation-free alternative to existing approaches.

### Strengths
1. SVSI offers a practical and computationally lightweight alternative to expensive self-evaluation methods (like self-judging) for unverifiable open-ended tasks.

2. The methodology is well-structured and clearly explained.

### Weaknesses
1. For unverifiable, open-ended tasks, diversity is a crucial metric. However, the proposed method may stifle diversity and creativity, as this unsupervised approach could lead to a severe Matthew effect. An answer that is creative, novel, or stylistically unique (yet still correct) is likely to be filtered out or receive a low voting score simply because its semantics differ from the 'mediocre' majority.

2. Its reliance on sentence embeddings makes its generalizability to long-text tasks questionable, where a single vector cannot capture full semantic complexity.

3. The experimental comparison is also limited, omitting more baselines and modern evaluation metrics like LLM-as-a-judge.

4. The cost analysis is potentially misleading; while cheaper than self-evaluation, the method's absolute cost is significantly higher than Supervised Fine-Tuning (SFT), yet no performance comparison is provided to justify this.

5. The pseudo-reward signal appears inherently unstable, as shown by its performance fluctuations with GRPO, suggesting its success may heavily depend on the noise robustness of the DPO framework.

### Questions
1. How does the method impact output diversity? Have you measured whether it causes a Matthew effect, leading to a less diverse distribution?

2. How can the method be adapted for long-text tasks where a single embedding is insufficient?

3. Were other unsupervised methods considered for comparison? For the tasks and models chosen in this paper, Supervised Fine-Tuning (SFT) would likely outperform SVSI with better cost-efficiency. Could you elaborate on the necessity of using SVSI in this context?

4. Can you explain the observed signal instability (e.g., with GRPO) and the method's sensitivity to its components, like the choice of embedding model?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Semantic Voting (SVSI), a lightweight and self-evaluation-free framework for improving large language models (LLMs) on unverifiable open-ended tasks such as translation and summarization. Unlike existing self-improvement methods (e.g., self-judging or entropy minimization), which rely on computationally costly and biased self-evaluation, SVSI replaces hard exact matching with soft semantic matching using sentence embeddings. The method clusters self-generated responses, applies semantic similarity-based voting, and uses the resulting pseudo-preference pairs for Direct Preference Optimization (DPO) training. Experiments across six models (Llama and Qwen series) and multiple datasets show that SVSI achieves comparable or superior performance to baselines while being 100× more computationally efficient.

### Strengths
- Introduces a new paradigm (semantic voting) that generalizes majority voting to open-ended tasks through soft semantic matching.

- Demonstrates a drastic reduction in computational cost (as low as 0.1–5% of self-evaluation methods) without sacrificing performance.

- Extensive evaluation across 6 models and multiple tasks with consistent gains in BLEU, ROUGE-L, and BLEURT.

- Includes flipped-pair tests, ablation studies, and hyperparameter analysis that strengthen methodological credibility.

- Algorithmic clarity and inclusion of implementation details in the appendix enhance replicability.

### Weaknesses
- The semantic similarity metric depends heavily on the choice of the sentence embedding model (SimCSE). Cross-model generalization is not deeply analyzed.

- Evaluations focus mainly on translation and summarization. It remains unclear how SVSI would perform on more subjective tasks (e.g., story writing or reasoning).

- Results rely solely on automatic metrics; human judgments would provide stronger validation of semantic quality improvements.

- As shown in Section 4.5, reward-based training variants exhibit performance fluctuation. The paper could provide further analysis on stabilization strategies.

### Questions
- How sensitive is SVSI to the choice of sentence embedding model? Would using multilingual or domain-specific encoders improve results?

- Could the method scale to even larger sampling sizes (e.g., 256 responses) without diminishing returns or excessive memory cost?

- Did you consider using model-based similarity scoring (e.g., cross-encoder) instead of embedding-based cosine similarity for higher fidelity?

- In the flipped-SV experiments, could the residual improvements over base models be due to pretraining biases or unintentional preference alignment?

- How would SVSI perform in reasoning or dialogue-generation tasks where “semantic similarity” may not correlate with quality?

### Soundness
4

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
This paper focuses on addressing the limitations of existing LLM self-improvement methods for unverifiable open-ended tasks (e.g., machine translation, text summarization), such as high computational overhead and overconfidence bias in self-evaluation approaches (self-judging, entropy minimization) and the inapplicability of majority voting (relying on exact matching) to such tasks. It proposes the
Semantic Voting-based Self-Improvement (SVSI)
framework, a self-evaluation-free solution.

SVSI draws inspiration from majority voting but relaxes hard exact matching to soft semantic similarity matching. The framework first enables an LLM to generate multiple candidate responses for a given input,then uses a lightweight sentence embedding model (e.g., SimCSE) to encode these responses into semantic vectors. After filtering outliers via density-based clustering (HDBSCAN) and retaining only the largest cluster to ensure meaningful semantic consensus, it calculates each response’s "voting score" as the average cosine similarity with other responses in the cluster. The highest and lowest scoring responses form preference pairs, which are used to fine-tune the LLM via Direct Preference Optimization (DPO).

Empirical evaluations across 6 models (Qwen series: 1.5B, 3B, 7B; Llama series: 1B, 3B, 8B) and 6 datasets (4 WMT24++ translation subsets, CNN/DailyMail and PubMed summarization datasets) show that SVSI consistently outperforms self-judging (SJ) and entropy minimization (EM) baselines in most cases, with computational overhead only 0.1%–5% of self-evaluation methods. Additional validations (e.g., flipped-SV experiments to confirm valid semantic signals, out-of-distribution tests to verify generalization) and ablation studies (on clustering and embedding models) further confirm SVSI’s effectiveness, stability, and adaptability.

### Strengths
Targeted Problem-Solving for Unverifiable Tasks:
The paper addresses a critical gap in LLM self-improvement—existing methods either rely on computationally expensive, bias-prone self-evaluation (e.g., self-judging, entropy minimization) or hard-matching majority voting (inapplicable to open-ended tasks like translation and summarization). Its proposed Semantic Voting-based Self-Improvement (SVSI) framework avoids self-evaluation entirely, enabling lightweight and effective optimization for unverifiable open-ended tasks .

Innovative and Efficient Core Mechanism
:
Inspired by majority voting but extended to "soft semantic matching," SVSI uses lightweight sentence embedding models (e.g., SimCSE) to quantify semantic similarity, replacing rigid exact matching. This design drastically reduces computational overhead—its cost is only 0.1%–5% of self-evaluation baselines—while maintaining effectiveness, making it scalable even for larger models .

Practical Engineering Value
:
The framework is operationally transparent (detailed implementation in Appendix B) and requires no complex dependencies—self-generation uses standard sampling, embedding/clustering rely on lightweight tools, and DPO training uses standard hyperparameters. This makes SVSI easy to reproduce and deploy in real-world LLM optimization pipelines .

### Weaknesses
The proposed semantic voting approach relies on a straightforward pipeline: clustering self-generated responses using semantic similarity and calculating voting scores via average cosine similarity between sentence embeddings. This core logic lacks impactful technical novelty, as both semantic clustering and similarity-based ranking are well-established techniques in NLP. Notably, the ablation study confirms that clustering is not universally beneficial—for high-capability models like Qwen-1.5B, removing clustering either preserves performance or slightly improves it, which further underscores the framework’s simplicity and its limited innovative contribution to solving challenges in unverifiable open-ended tasks .

Weak Connection in Related Works
:
The related works section only critiques limitations of existing methods (e.g., majority voting’s reliance on exact matching, self-evaluation’s high computational overhead) but fails to link these flaws to specific components of the proposed semantic voting. It does not clarify how modules like lightweight sentence embedding or soft semantic matching directly address the drawbacks of prior approaches, weakening the rationale for the new method.

Disordered Section Reference in Figures
:
Figure 1, which outlines the SVSI framework, references Section 3.2 (self-generation filtering via clustering) before Section 3.1 (semantic voting)—a misalignment with the logical order of the Methodology chapter, causing confusion about the workflow sequence .

Unjustified Inclusion of Non-Original equation
:
Eq. 5 (pertaining to DPO loss) is directly adopted from prior work (Rafailov et al., 2023) without modifications. The paper provides no reason for including this non-innovative equation in the Methodology section, as it is irrelevant to the core semantic voting mechanism.

Ambiguous Figure 4
:
Figure 4 (comparing semantic voting and flipped-SV) lacks a clear legend. Key elements—such as labels for "SV" and "flipped-SV" or detailed axis annotations for lexical/semantic metrics—are missing, making it impossible to accurately interpret performance differences between the two settings .

Unjustified Clustering Algorithm Selection
:
The paper uses HDBSCAN for clustering self-generated responses but offers no experimental, theoretical, or literature-based justification for this choice. It does not compare HDBSCAN with other common clustering algorithms (e.g., DBSCAN, OPTICS) to demonstrate its suitability for semantic voting .

Sensitive and Opaque HDBSCAN Parameters
:
HDBSCAN’s performance depends on two key parameters (minimum cluster size m and neighborhood sample threshold k), which are shown to be sensitive to task and model variations (e.g., larger m improves performance for most models). However, the paper does not discuss whether these parameters require task-specific tuning or provide guidelines for adapting them to different unverifiable tasks, increasing practical application complexity .

Unjustified Metric Choices
:
For evaluation, the paper uses different metric-model pairs across tasks (e.g., BLEU/n-MQM for translation, ROUGE-L/BLEURT for summarization) but provides no explanation for these selections. It fails to justify why these metrics are optimally suited for measuring output quality in translation vs. summarization, weakening result validity .

Limited Task and Domain Coverage
:
The method is designed for unverifiable open-ended tasks but only tested on translation and summarization. No experiments check for potential negative effects in closed-ended tasks (e.g., arithmetic, multiple-choice), which is critical for assessing generalizability. The narrow task scope also makes it hard to confirm effectiveness across diverse unverifiable scenarios .

No Significant Advantage Over Baselines
:
In Table 1 and Figure 2, SVSI performs on par with or slightly better than baselines (SJ, EM) but shows no substantial, consistent advantage. For example, on Llama-3B, EM outperforms SVSI on multiple translation datasets (e.g., wmt24pp de BLEU: 10.52 vs. 8.71), failing to demonstrate value beyond existing methods .

Stability Issues in GRPO Adaptation
:
As noted in Section 4.5, when adapted to GRPO (SVSI-G), the method exhibits severe performance fluctuations across runs—far less stable than EMPO. This undermines confidence in its applicability beyond DPO training .

Unproven Clustering Necessity
:
The ablation study (Section 4.6) shows clustering is unnecessary for high-capability models like Qwen-1.5B (removing it slightly improves performance) and only beneficial for low-capability models like Llama-1B. This indicates the clustering component— a core part of SVSI—lacks universal necessity .

### Questions
1.Given that the core logic of semantic voting (semantic clustering + similarity-based scoring) is relatively simple and lacks significant technical innovation, and the ablation study shows clustering is not universally useful (e.g., it does not harm Qwen-1.5B’s performance when removed), how does the paper justify the novelty of the proposed framework compared to existing semantic similarity-based or clustering-based methods in LLM self-improvement?

2.In the related works section, the paper only critiques the limitations of existing methods (e.g., majority voting’s reliance on exact matching, self-evaluation’s high overhead) but fails to link these flaws to specific modules of semantic voting (e.g., lightweight sentence embedding, soft matching). Why does the paper not explicitly explain which module addresses which drawback of prior work, and how to verify the causal relationship between these modules and the solved problems?

3.Figure 1 (the SVSI framework overview) references Section 3.2 (clustering-based filtering) before Section 3.1 (semantic voting), which contradicts the logical order of the Methodology chapter. What is the reason for this disordered section reference, and does it reflect a misunderstanding of the framework’s own workflow?

4.Eq. 5 (the DPO loss formula) is directly adopted from Rafailov et al. (2023) without any modifications,
yet it is included in the Methodology section. Since this equation is not an original contribution and is
irrelevant to the core semantic voting mechanism, why does the paper choose to present it, and what
value does it add to explaining the proposed method?

5.Figure 4 (comparing semantic voting and flipped-SV) lacks a clear legend, making it impossible to
accurately identify which curve represents "SV" and which represents "flipped-SV," and unclear how
the axes correspond to lexical/semantic metrics. Why is such a critical result figure missing key
annotation information, and how does this affect the credibility of the conclusion that "flipped-SV
underperforms"?

6.The paper selects HDBSCAN as the clustering algorithm for filtering self-generated responses but
provides no experimental comparison (e.g., with k-means, DBSCAN) or theoretical justification (e.g.,
alignment with semantic voting’s needs) for this choice. What evidence supports that HDBSCAN is
more suitable for semantic voting than other common clustering algorithms?

7.HDBSCAN’s performance depends on sensitive parameters (minimum cluster size m and
neighborhood threshold k), and the paper shows larger m generally improves performance. However,
it does not explain whether these parameters need to be tuned for different tasks (e.g., translation vs.
summarization) or models. How should users determine the optimal m and k values when applying
SVSI to new unverifiable tasks?

8.For evaluation, the paper uses different metric pairs (BLEU/n-MQM for translation, ROUGE-L/BLEURT
for summarization) but provides no explanation for these choices. Why are these specific metrics
selected instead of others (e.g., using BLEURT for translation or n-MQM for summarization), and how
to ensure the selected metrics can fairly reflect the method’s performance across tasks?

9.The method is designed for unverifiable open-ended tasks but only tested on translation and
summarization. No experiments are conducted to check if SVSI causes negative effects (e.g., reduced
accuracy) in closed-ended tasks (e.g., arithmetic, multiple-choice). Why is the scope of experimental
verification so narrow, and how to confirm the method’s safety when applied to mixed task scenarios?

10.In Table 1, SVSI shows no significant advantage over baselines (e.g., on Llama-3B’s wmt24pp de task,
EM’s BLEU score is 10.52 while SVSI’s is 8.71). The paper claims SVSI is "effective," but why does it
not address the cases where it underperforms baselines, and how to prove its superiority in practical
application scenarios?

11.Section 4.5 notes that when SVSI is adapted to the GRPO framework (SVSI-G), it exhibits obvious
performance fluctuations, which are more severe than EMPO’s stability. Since the paper emphasizes
SVSI’s "efficiency and robustness," why does it not solve the stability issue in alternative training
paradigms, and what limitations does this reveal about the method’s adaptability?

12.The ablation study (Section 4.6) shows clustering is unnecessary for high-capability models like
Qwen-1.5B (removing it slightly improves performance) and only beneficial for low-capability models
like Llama-1B. As a core component of SVSI, why does clustering lack universal necessity, and how to
adjust the framework to avoid redundant steps for high-capability models?

13.SVSI relies on sentence embedding models (e.g., SimCSE) to calculate semantic similarity. If the
embedding model fails to capture task-specific semantics (e.g., medical terminology in PubMed
summarization), the pseudo-label quality declines. How does the paper plan to address the
dependence on embedding model performance, and is there a way to reduce the method’s sensitivity
to embedding model selection?

14.Semantic similarity metrics cannot distinguish between "semantically consistent but low-quality" and
"semantically distinct but high-quality" responses, leading to "conformity bias" (prioritizing majority
consensus over individual accuracy). Does the paper acknowledge this limitation, and what
improvements can be made to avoid discarding accurate but niche responses?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces Semantic Voting–based Self‑Improvement (SVSI) for tasks without verifiable answers. The method samples multiple answers, embeds them, and filters them down to the largest semantic cluster. Within this cluster, it ranks answers by their average cosine similarity to all other answers. The top-ranked ("winner") and bottom-ranked ("loser") answers are then used as a preference pair for DPO training. This approach uses a simple sentence encoder and clustering, avoiding expensive LLM-as-a-judge or entropy calculations to create preference pairs much more cheaply. Tests on translation and summarization tasks using six different models show SVSI consistently improves over the base model and generally performs as well as or better than self-judging or entropy-based methods. A control experiment confirms the signal is valid, as it degrades performance. The method is also tested as a dense reward for GRPO, where it's competitive but shows higher variance. Ablations show it's robust to different embedders and clustering tools.

### Strengths
* The mechanism is simple, general, and easy to explain, acting as a "soft majority vote" without needing a separate judge model.

* Building preference pairs is computationally cheap, and the cost doesn't scale with the size of the base model, unlike LLM-as-a-judge.

* The paper provides broad empirical evidence across multiple models, tasks, and metrics, including a useful "flipped-signal" control test.

* Improvements generalize to out-of-distribution datasets, suggesting the model is learning a real capability, not just overfitting to the test set.

### Weaknesses
* The core assumption (consensus = correctness) can be a flaw. The method rewards conformity and could reinforce common errors while discarding correct but niche answers found in minority clusters.

* Some baselines, like self-judging, perform poorly or collapse, making it unclear how much of SVSI's lead is due to its own merit versus the fragility of the alternatives.

* The evaluation is limited to narrow tasks (translation to English, summarization). It hasn't been tested on long-form generation, conversation, safety, or creative tasks. The embeddings used are also English-centric.

* The compute savings focus only on building the pairs, not on the sampling of answers, which is the most expensive part. True end-to-end time savings aren't clear.

* When used as a dense reward for RL, the method seems less stable than discrete rewards.

### Questions
* How well does the "top vote" answer actually align with human judgment? A correlation study can be helpful to separate consensus from true quality.

* Could the "majority-error" problem be fixed by retaining multiple clusters? Could "hard negatives" from outside the main cluster be used to improve DPO training?

* How does this method scale to much larger models and longer, more complex outputs? How does it perform on non-English tasks?

### Soundness
3

### Presentation
3

### Contribution
3
