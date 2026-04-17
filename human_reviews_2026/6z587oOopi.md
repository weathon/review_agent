# Dynamic-anchored Preference Optimization for Human-Like Moral Alignment

- Decision: Reject
- Scores: 8, 4, 2, 4

## Abstract
Preference optimization has become a widely used approach to align large language models (LLMs) with human values. Direct Preference Optimization (DPO) provides a simple and reward-model-free solution, but it relies on static binary preference pairs and a fixed reference policy, which limits its ability to capture multi-dimensional moral signals and makes it sensitive to conflicting prompts. To address these limitations, we propose \textit{Dynamic-anchored Preference Optimization (DAPO)}, an extension of DPO that incorporates moral preference reconstruction and adaptive-weighted optimization. It introduces: (1) a dynamic-anchored triplet construction mechanism grounded in Moral Foundations Theory (MFT), which enables exploration of both benevolence reinforcement and malevolence suppression; (2) a value-guided pairwise loss with heuristic adaptive weighting to balance training signals while reducing reliance on a fixed reference policy. Experiments on benchmarks covering emotional understanding, moral reasoning and factual consistency show that \textit{DAPO} consistently improves accuracy and robustness compared to DPO-based methods. Further sensitivity analyses demonstrate that \textit{DAPO} provides a practical extension to DPO, making preference optimization more reliable and effective for moral alignment.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents an enhancement to the Dynamic Preference Optimization (DPO) process, aiming to align binary preference signals with multi-dimensional moral signals. Building on existing literature that indicates Large Language Models (LLMs) often produce biased responses to opposing prompts, the authors seek to anchor model preferences more explicitly along five moral dimensions derived from Moral Foundations Theory. These dimensions include care/harm, fairness/cheating, loyalty/betrayal, authority/subversion, and sanctity/degradation. To achieve this, the authors introduce a novel framework termed Dynamic Anchored Preference Optimization (DAPO).




Under the DAPO framework, for each prompt, a standard response is generated. The prompt is then modified across each moral dimension to produce a "good" prompt and an "evil" prompt, resulting in corresponding good and evil responses. These elements—prompt, modified prompts, and their respective responses—form the foundational training tuple. The methodology targets three distinct objectives: the "general moral preference objective," capturing the preference margin between good and evil responses; the "benevolence reinforcement preference objective," capturing the preference margin between a good response and the standard response to a good prompt; and the "malevolence suppression preference objective," focusing on the preference margin between the standard and evil responses to an evil prompt. A sophisticated weighting schedule, based on training steps, integrates these objectives into the training goal.




The training data is sourced from the MentalChat dataset, with evaluations conducted on EmoBench, MoCA, TruthfulQA, Ethics, ToxiGen, and MMLU-Pro. Results demonstrate superior performance of the DAPO method compared to alternatives like DPO, TDPO, and SimDPO. The paper also includes sensitivity analyses exploring DAPO's efficacy across varying levels of "evilness," alongside a detailed mathematical analysis and theoretical bounds of the new objectives. Overall it provides a pretty compelling argument towards grounding the preference signal along the pre-defined moral dimensions.

### Strengths
- The paper presents an innovative approach by utilizing multi-dimensional grounding of the preference signal, as opposed to a binary preference signal. It also includes an effective method for generating data using model-driven, pre-defined prompting.
- A comprehensive suite of evaluation benchmarks and analysis demonstrates the method's effectiveness.
- Thorough ablation studies provide compelling evidence for the significance of each sub-objective within the method.
- The paper offers rigorous mathematical analysis and establishes theoretical bounds for the new objective.
- An extensive appendix includes valuable supplementary information, such as prompts and example analyses.
- Evaluation on MMLU-Pro suggests potential generalizability of the method to other domains.

### Weaknesses
While the paper demonstrates a strong methodology, it lacks a clear analysis of its applicability to real-world scenarios involving human-annotated preference datasets. There is a limited comparison in this context, which could benefit from further exploration to enhance the study's practical relevance.

### Questions
- The interpretation of sensitivity analysis across different levels of "evil-levels" appears to be complex and somewhat ambiguous. Could you clarify the statistical significance of these measurements? It appears that other methods may be equally effective at resisting higher levels of challenging prompts. Could you provide further clarification on this point?
- Based on the generated dataset, DAPO clearly outperforms other methods. However, could you provide insights into the magnitude of DAPO's advantage when applied to a human-annotated dataset? Are there any indicators or metrics available for this aspect?

### Soundness
4

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
5

### Summary
The paper proposes DAPO, a DPO-style method for aligning LLMs with human moral values. It builds triplets for each question using prompts grounded in MFT. They define three pairwise preference objectives and combine them with a heuristic adaptive-weighting schedule that varies over training. Empirically, on several benchmarks, DAPO outperforms DPO/SimPO baselines. Sensitivity analyses indicate better robustness under adversarial “evilness” prompts.

### Strengths
1 $\textbf{Solid experimentation and ablations.}$ The paper evaluates across diverse benchmarks and includes clear ablations for each sub-component. Results generally favor DAPO.

### Weaknesses
1 $\textbf{Insufficient background on MFT and prompt design.}$ Section 3.1 states that prompts are designed from Moral Foundations Theory, but the paper does not explain MFT nor detail how it informs the prompt design. Readers unfamiliar with MFT will struggle to follow the construction. I recommend adding a short MFT background in Related Work or Preliminaries to keep the paper self-contained.

2 $\textbf{Prior methods placed in Method section.}$ Section 3.2 is largely a recap of DPO/SimPO. This belongs in Related Work/Preliminaries, not in the proposed method. Moving it would streamline Section 3 and highlight the new contributions.

3 $\textbf{Offset idea and novelty.}$ Section 3.3 introduces an offset-like margin. Similar ideas exist (e.g., DPO with an Offset [1]). The paper should precisely contrast its margins with previous work either mathematically or intuitively, and clarify what is new. Also, fix the typo in Eq. 5 (missing “[”).

4 $\textbf{Technical contribution and novelty.}$ The method largely reduces to triplet preference learning with different loss weights. The rationale for using triplets is unclear: given an original query $x$, why and how should one generate an additional ‘virtuous’ and ‘evil’ prompt? The claim of increased mutual information appears to follow from properties of the component preference losses, which are not novel to this work, thereby weakening the paper’s technical contribution.

5 $\textbf{Weights in Section 3.4 lack justification.}$ The adaptive weighting schedule $\lambda_1, \lambda_2, \lambda_3$ appears heuristic. Please explain the design choices, and discuss whether equal weights or alternative schedules harm/help performance.



[1] Amini, Afra, Tim Vieira, and Ryan Cotterell. "Direct preference optimization with an offset." arXiv preprint arXiv:2402.10571 (2024).

### Questions
Please see my weakness. I am mainly concerned about the technical contribution and the novelty of the proposed method. In what ways do DAPO's triplets go beyond prior DPO preference or its variants? I would consider raising my point if those concerns are fixed.

### Soundness
3

### Presentation
3

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
Direct Preference Optimization (DPO) provides a simple, reward-model-free solution, but it relies on static binary preference pairs and a fixed reference policy, which limits the model’s ability to capture multi-dimensional moral signals and makes it sensitive to conflicting prompts. To address these limitations, the paper proposes Dynamic-Anchored Preference Optimization (DAPO), an extension of DPO that integrates moral preference reconstruction and adaptive-weighted optimization.

### Strengths
1. Novel triplet preference modeling: A new triplet-based preference modeling approach is proposed, extending the application of DPO to moral alignment tasks, and employing an adaptive weighted fusion strategy to balance different moral signals.
2. The authors' writing in the methodology section is relatively clear.

### Weaknesses
1.Lack of analysis of existing moral preference alignment methods:
The related work section lacks a discussion and analysis of current approaches to moral preference alignment.

2.Insufficient analysis of the adaptive weighting strategy:
Although the paper includes an ablation study of the “without AW” setting, it does not explore the sensitivity of the temperature parameter τ or the dynamics of weight changes. The design of the weighting function is based on a heuristic approach, lacking sufficient theoretical justification or empirical validation.

3.Computational cost issue:
The proposed triplet structure requires three forward passes for each input (benevolent, neutral, and malevolent), along with the adaptive weighting step, resulting in significantly higher computational cost compared to DPO. The paper does not provide any runtime or computational complexity comparison data.

4.Major deficiencies in the experimental section:
① Lack of statistical significance testing: Although six benchmarks are covered, the paper does not report significance tests or variance across multiple runs; the claimed 3–5% performance improvements may fall within normal fine-tuning variance for LLMs.
② Insufficient baseline comparison: The baselines are limited to the DPO family (DPO, SimPO, TDPO) and lack broader comparisons with alignment methods such as RLHF[1] or PRO[2], making it difficult to assess whether DAPO truly advances the field of preference optimization.
③ Lack of human evaluation: The claim of “human-like moral alignment” is not supported by human evaluations; all experiments rely solely on automatic metrics, leaving it unclear whether the model is genuinely aligned with human values.

**Reference**

[1]  Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.

[2]  Feifan Song, Bowen Yu, Minghao Li, Haiyang Yu, Fei Huang, Yongbin Li, and Houfeng Wang.Preference ranking optimization for human alignment. In AAAI, 2024.

### Questions
Please refer to the “Weakness” section for related questions.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a new method called "Dynamic-anchored Preference Optimization" (DAPO) aimed at addressing the moral alignment of Large Language Models (LLMs). The authors note that standard preference optimization methods (like DPO) rely on static binary preference pairs (chosen vs. rejected) and a fixed reference policy, which limits their ability to capture multi-dimensional moral signals and makes them sensitive to conflicting prompts.

### Strengths
The paper's core contribution, the "dynamic-anchored triplet", is highly original. Using a 'standard' response $y$ as a dynamic anchor—acting as a lower bound for 'benevolence reinforcement' 12and an upper bound for 'malevolence suppression' —is a clever and principled extension of the standard DPO binary comparison paradigm. This design is conceptually well-suited for the multi-dimensional, nuanced problem of moral alignment.

### Weaknesses
1. The core of the DAPO framework is the 'standard response $y$, which is used as a dynamic anchor. The authors describe it as a "standard human response" 262626, but the data collection section (4.1) indicate that other responses ($y_w, y_l$) are generated by LLMs. So, how is $y$ actually generated? Is it the LLM's response to the original question $x$ (without any virtuous/evil prompt)?

2.Lack of ablation study comparing the MFT framework to simpler alternatives: The paper's data generation strategy is deeply tied to Moral Foundations Theory (MFT) and its five dimensions. While the authors demonstrate that MFT-based DAPO outperforms baselines like DPO, they do not provide a crucial ablation study. This missing experiment should compare the full, MFT-based DAPO against a "simplified DAPO" variant. This variant would use the same DAPO triplet loss structure but rely on generic 'positive' (virtuous) and 'negative' (evil) prompts, rather than the 5 specific MFT dimensions. Without this comparison, it is difficult to disentangle whether the performance improvements stem from the novel DAPO loss framework itself or just MFT prompt.

### Questions
Please see above.

### Soundness
3

### Presentation
3

### Contribution
3
