# From Shortcuts to Reasoning: Robust Post-Training of Theory of Mind with Reinforcement Learning

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Theory of Mind (ToM) is a must-acquire skill for modern foundation model systems to operate effectively and safely in the real world. Recent works have explored honing ToM via post-training; however, we show that such progress is confounded by a pervasive “shortcut” issue: tasks can reach up to 99% accuracy by simply exploiting spurious causal correlations, leading to a false sense of ToM. Motivated by this, we first develop a framework to systematically examine ToM datasets for shortcuts and provide guidance for future development. We find that questions reducible to pure state tracking (e.g., “belief”) are especially shortcut-prone compared to mind questions (e.g., “intention”) where reasoning beyond tracking is required. Using four shortcut-free datasets across three ToM contexts, we then comprehensively study whether reinforcement-learning fine-tuning with verifiable rewards and explicit reasoning (Thinking-RFT) elevates ToM beyond supervised fine-tuning (SFT). Our key findings are: 1) Thinking-RFT effectively improves ToM in all scenarios (+6% vs. SFT), particularly in complex higher-order reasoning (+10% vs. SFT) and multimodal cases (+7% vs. SFT), and generalizes notably better to unseen domains and higher-order queries while being more robust to counterfactuals. 2) ToM benefits specifically from the joint effect of reasoning and RL: Thinking-RFT outperforms No-Thinking-RFT by 7% on average. 3) RFT works by learning to ground its reasoning on anchor cues (keywords/state changes) that correspond to causal factors. We believe our study is useful for developing effective and robust ToM post-training datasets and advancing critical ToM capabilities in foundation models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work investigates potential shortcuts in existing ToM datasets, and train LMs via RL on the shortcut-free datasets.

The main findings are as follows:

(1) several datasets are prone to shortcut exploitation and are therefore unsuitable for training;

(2) when trained on shortcut-free datasets, the performance follows the order thinking-RFT > no-thinking-RFT > SFT;

(3) thinking-RFT exhibits strong generalization to unseen domains and higher-order ToM questions; and

(4) RFT improves performance by learning to ground its reasoning in cues that correspond to causal factors.

Overall, the paper attempts to address an interesting question, but it lacks important methodological details and sufficient evidence to support its claims, especially (1) and (4). The writing is generally clear, though there are some typos. I will outline my specific concerns below.

### Strengths
- The writing is mostly clear and easy to follow.

- A major contribution of this work lies in its attempt to identify and analyze shortcuts in existing ToM datasets. Training on such datasets may encourage models to exploit superficial correlations rather than develop genuine ToM capabilities.

- The finding that RFT training outperforms SFT is not entirely surprising, since similar trends have been observed in other domains. However, this work reaches different conclusions from Lu et al. (2025) that explored similar questions.

- I appreciate the use of mechanistic analyses to investigate where RFT demonstrates its advantages, although the presented evidence is not fully convincing.

Reference:
Lu, Y. L., Zhang, C., Song, J., Fan, L., & Wang, W. (2025). Do Theory of Mind Benchmarks Need Explicit Human-like Reasoning in Language Models?. arXiv preprint arXiv:2504.01698.

### Weaknesses
- The paper reaches a different conclusion from Lu et al. (2025) but does not discuss the differences in detail. It seems the authors suggest that Lu et al. (2025) trained on shortcut-prone datasets, which may explain their opposite findings. Please provide a more detailed comparison and discussion in the related work section to clarify the source of these discrepancies.

- Some terms and methods are not clearly illustrated.
  - In Section 2.1, what exactly is the “stratified seed set”? For example, does it include (x, y) pairs where x is a question and y is the answer? How is the set “stratified”? 
  - How are the heuristics implemented and combined? These parts should be described more clearly for reproducibility and clarity, since it's suppose to support a crucial claim of this work.

- Not enough evidence to support the claims:

  - The paper claims that training on shortcut datasets can harm ToM abilities, but there isn’t enough evidence to support this. It would be more convincing if the authors trained the same model on these shortcut datasets using the same method and compared the results, instead of only showing a qualitative figure with reasoning-trace errors (Figure 2).

  - Using procedurally generated data is not very convincing for testing generalization (e.g., generating data from 'apartment' to unseen 'outer space' for MMToM), since such data still follow similar logic as the training data. It would be stronger to test on truly out-of-domain datasets, e.g., the shortcut-prone datasets mentioned earlier, to see if RL really improves generalizable ToM abilities.

- Typos for revising the manuscript:
  - Line 118: casual -> causal
  - Line 152: there should be '.' before 'On'
  - Line 155: mentioned section 2 -> mentioned in Section 2
  - Line 159: Table 2 -> Figure 2

### Questions
- Regarding generalization, could you test on existing evaluation datasets instead of constructing new ones derived from the training data? This would make the generalization claim more convincing.

- In Section 5.2, you mention that 'we manually mark the minimal cues in each narrative that establish the causal hinge between agents’ intentions and outcomes.' This appears to suggest a quantitative evaluation across multiple data, yet the paper only provides a single qualitative example. Please include quantitative results or clarify how the experiments were truly performed to better support the crucial claim of 'RFT improves performance by learning to ground its reasoning in cues that correspond to causal factors'.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper aims to tackle the generalizability and counterfactual robustness of foundation models on theory-of-mind (ToM) reasoning tasks. They first design a principled filtering approach that identifies the shortcut issue in the existing ToM benchmarks, especially for the datasets like Hi-ToM higher-order queries. After identifying the shortcut-free dataset, the authors further make a comprehensive comparison among the performance of zero-shot and different post-training approaches, including SFT and RFT (with or without thinking tokens). The results demonstrate that RFT with thinking outperforms other post-training approaches in most tasks. RFT enjoys not only the best accuracy on different ToM benchmarks, but also better generalizability to higher-order reasoning cases and better causal consistency to counterfactual probing. The qualitative studies on the attention map also reveal that the model after RFT is more capable of capturing the semantics in the contexts.

### Strengths
* The paper is generally well-written and easy to follow. The clarity on the experiment settings is good. 
* The shortcut perspective on the current ToM benchmark is interesting and worth studying deeper for the ToM research community. 
* The empirical results on different post-training approaches, as well as the qualitative studies are clearly presented and can justify the major claim of contribution in this paper.

### Weaknesses
* **Incoherent contribution in section 2 and 3:** It's a little bit unclear what is the necessary logical connection between section 2 and 3. Section 2 only identifies those dataset that potentially has confounded shortcuts that should be precluded from post-training. However, these datasets with shortcut are also held away from evaluation set in the final experiments with generalization. 
* **Missing baselines:** There exists quite a few baselines in solving ToM-QA tasks, such as SimToM [1], AutoToM [2]. However, the paper misses these baselines that are currently leading approaches in the MMToM benchmark. 
* **Limited analysis on the higher-order generalizability:** Since the authors only use shortcut-free dataset like OpenToM to conduct post-training and evaluation, the claim on 'generalizability to higher-order queries' does not seem to be sufficiently supported as they only care about the generalization from first-order to second-order ToM. A more comprehensive studies on third-order and fourth-order ToM (like the ones in Hi-ToM) will be interesting. 
* **Missing evaluation on cross-dataset generalizability:** It is hard to justify whether the results the authors present are coming from better overfitting to a single dataset, or indeed a better capability in general reasoning across different ToM datasets. Therefore, it is necessary to add the evaluation on different ToM datasets after the SFT/RFT, rather than a small test split which is less different in the QA domain. 
* **Limited scale of evaluation:** The OpenToM dataset, the authors only select 100 samples from each category as evaluation. They also missed the evaluation on the test split provided by the MMToM leaderboard. These limits the contribution of the proposed method. 
* **Limited scale of qualitative analysis**: The author only demonstrates one pair of qualitative comparisons on the attention map. More qualitative examples can be provided in the appendix to make the claim of causality-coherent reasoning more solid. 
* **Missing analysis on the 'spurious correlation'**: The authors mentioned their motivation comes from the observation that the model 'simply exploiting spurious correlations'. However, the analysis terminates after section 2 right after they preclude the Hi-COM and other datasets in the finetuning and evaluation dataset. It will be more reasonable if they can conduct **quantitative** analysis on how (a) the data filtering (judged by simple rules and lexical association), as well as (b) the SFT/RFT-style post-training, can help mitigate such spurious correlation exploitation, even on those benchmarks with shortcuts. 

> [1] Wilf, Alex, et al. "Think twice: Perspective-taking improves large language models' theory-of-mind capabilities." ACL 2024.
>
> [2] Zhang, Zhining, et al. "Autotom: Automated bayesian inverse planning and model discovery for open-ended theory of mind." *ICLR 2025 Workshop on Foundation Models in the Wild*. 2025.

### Questions
See the weakness section for the main questions I have and the comments. I would consider re-adjust my assessment if the majority of them get resolved during the rebuttal phase.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies how RL–based post-training can enhance ToM abilities in large language models. The authors identify that existing ToM datasets often contain shortcut patterns that allow models to achieve high accuracy without genuine reasoning. They propose a framework to audit such shortcuts and select four datasets for further study. Using these datasets, the paper compares Reinforcement FT, Supervised Fine-Tuning, and zero-shot baselines across several ToM tasks. The results show that RFT consistently outperforms SFT, particularly in higher-order reasoning, unseen domains, and counterfactual settings. Further analysis indicates that explicit reasoning steps and reinforcement learning jointly improve ToM robustness and generalization.

### Strengths
- Original framing of the ToM shortcut issue.

- The paper evaluates models on narrative, conversational, and multimodal ToM tasks.

- By connecting ToM, with RL techniques, the paper bridges human mental reasoning and computational learning frameworks in a creative and interdisciplinary manner.

### Weaknesses
- The introduction needs revision. It's hard to quickly grasp what challenge the paper aims to address from the current content.
Since Table 2 is mentioned early, it might help to first provide clear examples in the introduction so readers can easily understand the problem.

- The authors should double-check the references to figures and tables to ensure that all mentions correctly correspond to the intended visual content.

- It seems the authors did not re-implement or compare with previous ToM training methods; their comparison is limited to RFT, SFT, and zero-shot.
For a more complete experimental design, prior ToM methods should also be retrained on the authors' cleaned datasets and compared directly with RFT.

- Line 422–423: The explanation is unclear.
The authors should describe in more detail how to interpret Figure 5, and explicitly explain what the example is doing.

- The claim that explicit reasoning improves performance is not very novel, many NLP works have shown this.
What exactly is its effect in this specific task?
The authors should further analyze the reasoning content itself, both good and bad cases.

### Questions
- Line 62–63: "More importantly, the trained model produces incoherent and illogical reasoning traces as shown in Table 2."
How can this be observed from Table 2?

- Line 159–164: "As shown in Table 2, RFT mixes 'Jack's' own observation with the 4th order query being asked, ignoring intermediate ToM."
How can this be seen from Table 2?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
“From Shortcuts to Reasoning” examines how reinforcement fine-tuning (RFT) can enhance Theory of Mind (ToM) reasoning in language models. The authors first audit existing ToM datasets and reveal that many contain shortcuts—spurious cues that let models perform well without genuine reasoning. To address this, they curate shortcut-free datasets and compare RFT with supervised fine-tuning (SFT). Experimental results show that RFT yields stronger performance, particularly on second-order, multimodal, and counterfactual reasoning tasks. The paper concludes that reinforcement learning promotes more robust, causally grounded reasoning—but only when the data itself demands true inference rather than pattern matching.

### Strengths
- The paper provides a thoughtful and systematic critique of current Theory of Mind  benchmarks. It convincingly shows that many datasets contain *shortcuts* — spurious correlations or superficial cues that allow models to achieve high scores without genuine reasoning. This diagnostic insight is valuable, as it highlights that performance on standard ToM tests often overestimates a model’s true inferential ability.
- Through detailed experiments and analyses, the paper explores where RL-trained models perform better. It examines multiple dimensions — reasoning depth (first- vs. second-order ToM), modality (text vs. multimodal), and robustness (counterfactual consistency, generalization to unseen contexts).

### Weaknesses
- While the paper thoroughly shows that RL outperforms SFT, much of it confirms a result that is already broadly recognized in the field — that RL tends to yield higher task-specific performance. What’s missing is a deeper analysis of **how** RL-trained models differ qualitatively from SFT models in their reasoning traces. The attention map analysis is only an **indirect indicator**, and without a direct comparison between RL and SFT attention patterns, it’s difficult to understand whether RL’s gains reflect genuinely improved reasoning or simply better optimization.
- A key claim of the paper is that RFT improves reasoning only on shortcut-free datasets, while shortcut-prone ones hinder learning. However, this is **not empirically demonstrated**—the authors audit shortcut datasets but never show RFT results on them. As a result, the claim remains **conceptually convincing but experimentally unverified**; direct comparisons would have provided stronger evidence for RFT’s limitations.

### Questions
- Where is the result for comparing RFT including the datasets with shortcuts?
- What does the attention map look like for SFT?

### Soundness
2

### Presentation
2

### Contribution
2
