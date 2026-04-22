# SPLIT-VLM: Salience-Guided Partitioning towards Local Coverage for Importance-Aware Token Dropping in Vision-Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 6, 2, 2

## Abstract
Large-scale vision–language models (VLMs) excel at multimodal reasoning, yet efficiency collapses when vision tokens—often orders of magnitude more than text—dominate compute and memory. Prior token-reduction strategies typically trade off salience (which is prone to position bias and incurs extra computation) against diversity (which can under-cover salient regions and is sensitive to hyperparameters). We present SPLIT, a theoretically grounded framework that jointly preserves salience and diversity while aggressively eliminating redundancy. SPLIT (i) estimates token importance via temporal shifts of hidden states across layers—eschewing attention scores and their biases; (ii) assigns adaptive region-level budgets to guarantee localized coverage; and (iii) selects tokens using a diversity score that prioritizes distinctive, non-redundant representations. Our analysis shows that adaptive budgeting yields tighter coverage guarantees than uniform allocation, and our selection rule maintains diversity without costly tuning. Empirically, SPLIT consistently outperforms state-of-the-art on image and video understanding benchmarks. On image understanding with LLaVA-1.5-7B, SPLIT preserves over 99\% accuracy with 192 vision tokens and about 92.8\% with only 64 tokens, demonstrating robust performance under severe token budgets. These results indicate that SPLIT delivers scalable, attention-score-free token reduction that makes multimodal reasoning substantially more efficient without sacrificing accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
SPLIT is a theoretically grounded framework for efficient token reduction in large-scale vision–language models, jointly preserving salience and diversity while eliminating redundancy. By estimating token importance via temporal shifts, assigning adaptive region-level budgets, and selecting distinctive tokens, SPLIT achieves robust performance under severe token constraints. Experiments show that it outperforms prior methods on image and video benchmarks, maintaining high accuracy even with drastically reduced vision tokens.

### Strengths
1. The paper is clear and practical, proposing SPLIT to efficiently balance token importance in vision–language models. The manuscript is well-structured and presents the method and results in a coherent and accessible manner.

2. I personally find it very interesting to apply shift bias from temporal domain to token pruning.

3. Experimental validation is sufficient. The authors conduct comprehensive experiments on various tasks and show **improvements**, to validate the effectiveness of the method. Moreover, the ablation study is detailed, particularly in the efficiency analysis section.

### Weaknesses
1. In table 3, compared with GreedyPrune, SPLIT performs notably better under the 64-token setting. However, its advantage diminishes under the 128- and 192-token settings. 

2. Equations 4 and 7 are missing periods.

3. Does the diversity-based selection in SPLIT overlap with previous methods, such as DivPrune and CDPruner[1]?

[1] Beyond Attention or Similarity: Maximizing Conditional Diversity for Token Pruning in MLLMs, NeurIPS 2025.

### Questions
See above Weaknesses.

### Soundness
3

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
This paper proposed SPLIT, a novel training-free vision tokens selection framework, which first measures token importance via cross-layer hidden-state shifts instead of attention, then allocates adaptive region-level budgets to ensure local coverage, and selects tokens with a simple diversity score to avoid redundancy. On image/video benchmarks, it consistently matches or surpasses SOTA; with LLaVA-1.5-7B it retains >99% accuracy at 192 tokens and ~92.5% at 64, delivering scalable, attention-free token reduction without sacrificing accuracy.

### Strengths
1. The proposed SPLIT is training-free and can be seamlessly integrated into several different architectures of vision–language models. The method is novel and clearly reproducible.

2. The experiments are comprehensive and convincing on various benchmarks, models, and tasks, and maintain a great trade-off between accuracy and efficiency.

3. The paper is well-organized, with intuitive figures and sufficient experimental tables, and is easy to follow.

### Weaknesses
1. Baseline comparison with other SOTA methods. More methods should be compared, for example, the VisPruner[1], CDPruner[2], and Visionzip[3] in video.

2. Small typo: In the appendix, lines 842, 869, 896, 923, the token numbers are duplicated; please check it again.

[1] Zhang, Q., Cheng, A., Lu, M., Zhuo, Z., Wang, M., Cao, J., ... & Zhang, S. (2024). [CLS] Attention is All You Need for Training-Free Visual Token Pruning: Make VLM Inference Faster. ICCV, 2025.

[2] Zhang, Q., Liu, M., Li, L., Lu, M., Zhang, Y., Pan, J., ... & Zhang, S. (2025). Beyond Attention or Similarity: Maximizing Conditional Diversity for Token Pruning in MLLMs. NeurIPS, 2025.

[3] Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. Visionzip: Longer is better but not necessary in vision language models. CVPR, 2025.

### Questions
1. In the video experiments, you appear to compute token salience and diversity within each frame independently using the same approach as in the image setting. Does SPLIT explicitly account for temporal relationships or cross-frame consistency when selecting tokens?

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
3

### Summary
The paper introduces SPLIT-VLM, a framework that makes VLMs more efficient by intelligently dropping redundant visual tokens. It solves the limitations of prior methods by uniquely balancing token importance with spatial coverage. The method partitions an image, assigning budgets to regions using a novel temporal shift importance metric to guarantee full coverage . Within each budget, a robust diversity score selects the most informative tokens.

### Strengths
1.	This paper proposes a novel, attention-free metric to efficiently measure token importance.
2.	This paper introduces a strategy that guarantees full image coverage by assigning token budgets to different regions.
3.	This paper proposes a robust diversity score-based method for selecting the most representative tokens without sensitive hyperparameters.

### Weaknesses
1.	The performance improvement of the proposed method is rather marginal. Compared with DivPrune and DART, it shows no significant advantage in either performance (as is particularly evident in Table 6 and 7) or efficiency.
2.	The ablation study is not sufficiently convincing. For example, in Figure 5, it is difficult to discern any clear advantage of the Adaptive strategy over HiRED.
3.	There is a lack of clear results or examples demonstrating that the proposed local budget allocation combined with the diversity score effectively compensates for the information that previous methods tend to miss, thereby improving understanding performance.

### Questions
1.	In lines 415-417, the authors claim that random selection achieves performance comparable to HiRED. This statement is confusing, as it is not clear how such a conclusion can be drawn from Figure 5.
2.	The importance of global coverage for understanding tasks remains unclear-what if the informative content of an image is concentrated in only a specific region? The paper lacks concrete examples or results to substantiate the claimed superiority of the proposed method over previous approaches in this respect.
3.	Why is the parameter lambda in Equation 10 set to 0.5? How would changing its value affect the results?

### Soundness
3

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
5

### Summary
This paper targets VLM inference efficiency. It first analyzes the limitations of prior saliency-based and diversity-based approaches, then proposes SPLIT, a framework that combines both ideas. The paper provides a theoretical analysis of the method and conducts extensive experiments across different model architectures and multimodal tasks. Results show that SPLIT outperforms selected baselines, with additional efficiency and ablation studies.

### Strengths
1.	The paper presents both theoretical and empirical perspectives. The proposed method has a theoretical basis and shows solid performance.
2.	The experiments are extensive and cover multiple VLM architectures and multiple multimodal tasks.
3.	The paper provides detailed efficiency analysis, showing that the method brings practical improvements to VLM inference efficiency.

### Weaknesses
1.	The paper lacks discussion of key related work. To avoid LLM attention bias, methods such as VisionZip[1] and VisPruner[2] prune after the vision encoder. Other methods such as CDPruner[3] and MoB[4] also combine saliency and diversity. These works are not discussed or compared.
2.	The paper does not demonstrate how uneven token-budget distribution across regions affects pruning performance, which weakens the motivation.
3.	The metrics used for adaptive region-level budget allocation are highly heuristic. The paper does not explain why tokens with high change rate should be considered more important, and no visualization analysis is provided.

[1] Yang S, Chen Y, Tian Z, et al. Visionzip: Longer is better but not necessary in vision language models. CVPR 2025.
[2] Zhang Q, Cheng A, Lu M, et al. Beyond text-visual attention: Exploiting visual cues for effective token pruning in vlms. ICCV 2025.
[3] Zhang Q, Liu M, Li L, et al. Beyond Attention or Similarity: Maximizing Conditional Diversity for Token Pruning in MLLMs. NeurIPS 2025.
[4] Li Y, Zhan H, Chen T, et al. Why 1+ 1< 1 in Visual Token Pruning: Beyond Naive Integration via Multi-Objective Balanced Covering. arXiv 2025.
[5] Li A, Duan Y, Zhang J, et al. TransPrune: Token Transition Pruning for Efficient Large Vision-Language Model. arXiv 2025.

### Questions
1.	The idea of allocating budget based on change rate is very similar to TransPrune[5]. Can the authors clarify the difference?
2.	In budget allocation, how is the sum of token-budget allocations guaranteed to equal B? In Eq. (7), the sum appears to be 2B. Is a factor of 1/2 missing?

### Soundness
3

### Presentation
2

### Contribution
2
