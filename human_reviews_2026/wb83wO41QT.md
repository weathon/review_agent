# Noise-Aware Generalization: Robustness to In-Domain Noise and Out-of-Domain Generalization

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 4, 6, 4

## Abstract
Methods addressing Learning with Noisy Labels (LNL) and multi-source Domain Generalization (DG) use training techniques to improve downstream task performance in the presence of label noise or domain shifts, respectively. Prior work often explores these tasks in isolation, and the limited work that does investigate their intersection, which we refer to as Noise-Aware Generalization (NAG), only benchmarks existing methods without also proposing an approach to reduce its effect. We find that this is likely due, in part, to the new challenges that arise when exploring NAG, which does not appear in LNL or DG alone. For example, we show that the effectiveness of DG methods is compromised in the presence of label noise, making them largely ineffective. Similarly, LNL methods often overfit to easy-to-learn domains as they confuse domain shifts for label noise. Instead, we propose Domain Labels for Noise Detection (DL4ND), the first direct method developed for NAG which uses our observation that noisy samples that may appear indistinguishable within a single domain often show greater variation when compared across domains. We find DL4ND outperforms DG and LNL methods, including their combinations, even when simplifying the NAG challenge by using domain labels to isolate domain shifts from noise. Performance gains up to 12.5% over seven diverse datasets with three noise types demonstrates DL4ND’s ability to generalize to a wide variety of settings.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces Noise-Aware Generalization (NAG) to tackle noisy, diverse real-world data by improving in-domain noise handling and out-of-domain generalization. The authors show that distinguishing noise from domain shifts is challenging, and naive combinations of LNL and DG fail because domain shifts interfere with noise detection. They propose using cross-domain comparisons as a signal for identifying noise, leveraging its lack of intrinsic class features. Experiments validate that this approach significantly improves performance and offer insights for further advancing NAG.

### Strengths
1. The paper is well-motivated and seems to be reproducible.
2. This paper is well-organized and easy to follow. Specifically, in Sce. 3, the authors provide an in-depth analysis of the causes and implications of the proposed task, and in Sec. 4, they present concrete solutions, which are highly insightful.
3. This paper addresses real-world challenges and significantly improves task performance.

### Weaknesses
1. Existing works [A][B] have discussed the presence of real-world noisy labels in domain adaptation, which this paper seems to overlook. The related work section should be improved to include comparative discussion with these methods.
2. The formula proposed in Section 3.1 is explored with a toy experiment in Section 3.2. In my view, the authors could provide deeper theoretical analysis to make the argument more convincing.
3. The experiments lack some qualitative analysis to illustrate specific cases, which would help demonstrate the method’s advantages and enhance understanding of the task.
4. Currently, NAG is evaluated on synthetic noisy data. Introducing more realistic “asymmetric noise,” as studied in noisy label learning, could improve the practical applicability of NAG.
5. In my opinion, the abstract focuses almost entirely on the NAG task itself, while neglecting to highlight the insightful ideas behind the proposed method.

[A] Feng, Yanglin, et al. "ROAD: Robust unsupervised domain adaptation with noisy labels." Proceedings of the 31st ACM international conference on multimedia. 2023.

[B] Yin, Ziniu, et al. "RoDA: Robust Domain Alignment for Cross-Domain Retrieval Against Label Noise." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. No. 9. 2025.

### Questions
Please refer to the strengths and weaknesses of the paper.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies the joint setting where data contains both domain shifts and label noise, which the authors call Noise-Aware Generalization. The idea is that existing domain generalization and noisy-label learning methods usually handle one problem at a time. The proposed method, DL4ND, uses cross-domain feature comparison to detect mislabeled samples. After a warm-up stage, it selects low-loss samples as clean, computes class-domain prototypes, and relabels high-loss samples based on the closest prototype from another domain. The method can be combined with standard DG approaches such as ERM++, SWAD, or SAGM. Experiments on seven datasets show consistent but relatively modest improvements.

### Strengths
1. The topic is relevant and realistic since many real-world datasets contain both domain shifts and noisy annotations.  
2. The method is simple and easy to integrate into existing frameworks.  
3. Experimental coverage is broad, with multiple datasets and several DG and LNL baselines.  
4. The paper is well written and figures are clear.

### Weaknesses
1. The novelty is limited. The proposed approach mainly reuses existing ideas from noisy-label learning such as loss-based sample filtering, GMM separation, and prototype relabeling. The only new component is comparing features across domains, which is a small modification conceptually.  
2. The framing of Noise-Aware Generalization as a new task feels overstated, since similar scenarios have appeared in prior DG or LNL discussions.  
3. The paper lacks deeper analysis or theory explaining why cross-domain comparison works better.

### Questions
1. How does the proposed cross-domain comparison differ fundamentally from prototype-based noisy-label methods such as DivideMix or UNICON?  
2. What motivates defining Noise-Aware Generalization as a new task rather than treating it as DG with label noise?  
3. Can you provide visualization or analysis to support the claim that cross-domain comparison improves noise detection?  
4. How sensitive are results to the relabeling frequency and the accuracy of domain labels?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Learning with Noisy Labels (LNL) addresses label noise and multi-source Domain Generalization (DG) handles domain shifts to boost downstream performance, but prior work mostly explores them in isolation, with limited efforts to mitigate label noise’s impact on DG.
Many applications require robustness to both label noise and distribution shifts (defined as Noise-Aware Generalization, NAG), posing challenges: LNL’s assumption that distribution shifts equal label noise fails, and DG’s neglect of label noise harms training. A naive NAG solution uses domain labels to separate shifts but wastes cross-domain information, while the proposed DL4ND improves noise detection by leveraging greater variation of noisy samples across domains. Experiments on seven diverse datasets show DL4ND significantly enhances performance.

### Strengths
1. This paper is easy to follow.

2. The performance of the proposed method is good. The experimentail results are relatively extensive.

### Weaknesses
1. It seems that there is no practical application of NAG in real world, so is it meaningful to address this new setting? The authors should discuss the potential value in real world application.

2. More related methods [1-4] in the filed of Robust Domain Adaptation under Label Noise should be reviewed and discussed. 

[1] Y. Shu, Z. Cao, M. Long, and J. Wang, “Transferable curriculum for weakly-supervised domain adaptation,” in Proc. AAAI Conf. Artif. Intell., 2019, vol. 33, pp. 4951–4958.

[2] Z. Han, X. Gui, C. Cui, and Y. Yin, “Towards accurate and robust domain adaptation under noisy environments,” in Proc. 29th Int. Joint Conf. Artif. Intell., C. Bessiere, Ed., 2020, pp. 2269–2276. 

[3] Y. Zuo, H. Yao, L. Zhuang, and C. Xu, “Seek common ground while reserving differences: A model-agnostic module for noisy domain adaptation,” IEEE Trans. Multimedia, vol. 24, pp. 1020–1030, 2022. 

[4] Junbao Zhuo, Shuhui Wang, Qingming Huang. Uncertainty modeling for robust domain adaptation under noisy environments. IEEE Transactions on Multimedia. pp. 6157-6170. 2023.

### Questions
Please refer to the Weakness.

### Soundness
3

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
3

### Summary
This paper studies the joint problem of robustness to label noise and domain shifts, termed Noise-Aware Generalization (NAG). The authors propose DL4ND, a method that detects noisy samples by performing cross-domain comparisons among low-loss examples. The idea is that comparing samples across domains helps identify intrinsic class features and avoid confusion caused by domain-specific artifacts. Experiments on several benchmark datasets demonstrate that DL4ND improves both in-domain and out-of-domain performance over prior LNL and DG methods.

### Strengths
- The paper is easy to follow and logically developed.
    
- The proposed approach is reasonable and intuitively well-motivated, combining insights from both LNL and DG.
    
- Extensive experiments across multiple datasets validate the effectiveness of the method and provide detailed ablation analysis.

### Weaknesses
- Although the setup is novel, it appears somewhat artificial, and its practical real-world scenarios are unclear.
    
- The method heavily relies on the multi-domain assumption. It is uncertain how DL4ND would perform if only one source domain with noisy labels were available.
    
- The analysis in this manuscript (such as lines 251–258) is mostly heuristic and lacks rigorous theoretical justification or stronger empirical evidence/observation.
    
- The experimental evaluation could be further strengthened, for example by involving more diverse noise types.

### Questions
- Have the authors evaluated whether low-loss samples truly correspond to clean labels in practice?
    
- How robust and reliable is the clean/noisy distinction under different training dynamics or noise levels?

### Soundness
2

### Presentation
2

### Contribution
2
