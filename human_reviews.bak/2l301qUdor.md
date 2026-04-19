# BOSE-NAS: Differentiable Neural Architecture Search with Bi-Level Optimization Stable Equilibrium

- Decision: Reject
- Scores: 3, 5, 5, 6

## Abstract
Recent research has significantly mitigated the performance collapse issue in Differentiable Architecture Search (DARTS) by either refining architecture parameters to better reflect the true strengths of operations or developing alternative metrics for evaluating operation significance. However, the actual role and impact of architecture parameters remain insufficiently explored, creating critical ambiguities in the search process. To address this gap, we conduct a rigorous theoretical analysis demonstrating that the change rate of architecture parameters reflects the sensitivity of the supernet’s validation loss in architecture space, thereby influencing the derived architecture's performance by shaping supernet training dynamics. Building on these insights, we introduce the concept of a Stable Equilibrium State to capture the stability of the bi-level optimization process and propose the Equilibrium Influential ($E_\mathcal{I}$) metric to assess operation importance. By integrating these elements, we propose BOSE-NAS, a differentiable NAS approach that leverages the Stable Equilibrium State to identify the optimal state during the search process and derives the final architecture using the $E_\mathcal{I}$ metric. Extensive experiments across diverse datasets and search spaces demonstrate that BOSE-NAS achieves competitive test accuracy compared to state-of-the-art methods while significantly reducing search costs.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes a new operation importance evaluation metric in network architecture search. The authors first introduce the concept of stable equilibrium state, which shows the stability of the bi-level optimization process in differentiable NAS. By analyzing the supernet training dynamics, the metric named equilibrium influential is proposed for fair differentiable NAS. The experimental results show that the proposed metric and search method can achieve competitive accuracy with significantly reduced search cost.

### Strengths
+ The experimental results clearly show the effectiveness and the efficiency of the proposed method.

### Weaknesses
- The writing can be improved. The abstract and the introduction are redundant. For the abstract, there are too many contents to introduce the background. For the introduction, many details especially the experimental results don’t have to be elaborated. I think demonstrating the main results is enough to show the effectiveness of this method.

- The technical soundness can be further verified. There are some strong assumptions without verification or explanation. For example, the assumptions to transit (6) to (7) should be verified. Why they have little effect on $\alpha$?

- Some exact calculations can be put in the Appendix part.

- The reason why the proposed method has less search cost should be analyzed in the result analysis, which is an important benefit from the new metric.

- The performance of the proposed method underperforms the SOTA NAS methods such as IS-DARTS. More clarification is required for the performance analysis.

### Questions
See Weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
Differentiable Architecture Search (DAS) often faces the issue where the magnitude of architecture parameters fails to reflect the true importance of operations. This paper addresses this problem by proposing BOSE-NAS, a DAS method guided by the Stable Equilibrium of architecture parameters (i.e., the point where the rate of change of the architecture parameters is minimal). The authors provide relevant experiments to support their method. However, the experimental section has several issues, such as limited improvement in performance and a lack of ablation studies.

### Strengths
1.	The paper is easy to read.
2.	The problem of DAS is clear.

### Weaknesses
This paper was submitted to NeurIPS 2024, compared with NeurIPS 2024, there are still some important issues that need to be addressed.
1. The ablation studies are not convincing. To be specific, in Figure 3, we can clearly see that the proposed method is sensitive to hyperparameters.
2. There still exist some typos/grammatical errors in the paper.
3. The format of references is still wrong.
4. Exploring the reasons behind the success of these techniques and providing intuitive explanations would contribute to the overall scientific contribution of the work.
5. I don't understand the theoretical analysis. Why use " Influence Function"? What relationship between " Influence Function" and your method? why validate the "reliability" of your proposed metric? Please provide detailed motivation and clear proven process in step by step. What is the difference between stability and reliability? Please provide a step-by-step proof process for validating their metric. And, clarification on the relationship between the Influence Function and their method.
6. In page 7, "I(z, L)" denotes?
7. The main limitation of this paper is that proposed method lacks comparison with larger datasets (i.e., COCO2017, VOC), and more competitors (i.e., β-DARTS++, Λ-DARTS).
8. Pls to prove your statement of generalizability.

[1] β-DARTS++: Bi-level Regularization for Proxy-robust Differentiable Architecture Search
[2] Λ-DARTS: MITIGATING PERFORMANCE COLLAPSE BY HARMONIZING OPERATION SELECTION AMONG CELLS

### Questions
pls see weaknesses

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
In this paper, the authors propose BOSE-NAS, a novel differentiable neural architecture search method that addresses critical challenges in existing differentiable architecture search (DARTS). The core idea of BOSE-NAS  is around the the concept of a ‘Stable Equilibrium State’, which offering insights into the validation loss trajectory across architectural spaces to stabilise the supernet’s bi-level optimisation process. The proposed method introduces a novel metric called Equilibrium Influential (EI) to evaluate the importance of operations during the architecture search phase. By choosing operations based on the EI metric at the Stable Equilibrium State, BOSE-NAS uses bi-level optimisation to find the optimal architecture operations.

### Strengths
1. The introduction of Stable Equilibrium State is somewhat novel and interesting, the theoretical analysis of architecture parameter dynamics provides a solid foundation for understanding the bi-level optimisation in differentiable NAS.

2. The Equilibrium Influential (EI) metric for operation evaluation is an innovative approach and offers a more reliable measure of operation importance to the bi-level optimisation process in the differentiable NAS. 

2. The proposed BOSE-NAS achieves competitive performance as well as less computational overhead in benchmark datasets like CIFAR-10 and CIFAR-100, compare with other differentiable NAS methods.

### Weaknesses
1. The propose method heavily depends on the accurate identification of the Stable Equilibrium State, specifically, the EI metric evaluates each operation independently, which could overlook potential dependencies among network operations within the architecture. This could make the proposed method not always generalise well.

2. The biggest concern of the proposed method, e.g., EI metric and the concept of Stable Equilibrium State, are the limited use scenario. It may not be easily applicable to non differentiable NAS methods, e.g., the evolutionary or pruning-based search algorithms.

### Questions
1. Although the problems within the bi-level optimisation process of differentiable NAS have been widely studied for years, e.g., BONAS [1], the proposed EI metric and Stable Equilibrium State still bringing some new insights to the NAS research. But differentiable NAS are often sensitive to the hyper-parameters, I wonder how sensitive is the Stable Equilibrium State identification process to the choice of hyper-parameters such as the learning rate and batch size? Can authors provide some ablation studies? It would be helpful to understand how the proposed method handles changes in the hyper-parameters, as well as its robustness.

2. The proposed methods are only applied to the differentiable NAS, however, the interest of NAS research has been largely shifted to training-free NAS methods, as they are offering more flexibilities to different search algorithms and search spaces, as well as better performance and much less computational overhead compare with differentiable NAS, e.g., Zen-NAS [2] and SWAP-NAS [3]. Can author discuss the potential adaptation that extend the concept the Stable Equilibrium State and EI metric to non-differentiable NAS methods?  



[1] Han Shi, Renjie Pi, Hang Xu, Zhenguo Li, James T. Kwok, and Tong Zhang. Bridging the gap between sample-based and one-shot neural architecture search with bonas. NeurIPS 2020.

[2] Ming Lin, Pichao Wang, Zhenhong Sun, Hesen Chen, Xiuyu Sun, Qi Qian, Hao Li, and Rong Jin. Zen-nas: A zero-shot NAS for high-performance image recognition. ICCV 2021.

[3] Yameng Peng, Andy Song, Haytham. M. Fayek, Vic Ciesielski and Xiaojun Chang . SWAP-NAS: Sample-Wise Activation Patterns for Ultra-fast NAS. ICLR 2024.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focuses on Differentiable Architecture Search (DARTS). They conduct theoretical analysis over DARTS and propose a concept called Stable Equilibrium State. Upon it, they propose an effective framework called BOSE-NAS to identify the optimal state during the searching procedure. Experiment results show that the proposed method shows competitive results over state-of-the-art methods.

### Strengths
1. I think this paper focuses on a very important problem. DARTS is a very crucial framework in NAS, but it has some well-known problems. It is very important to have some theoretical analysis on this framework. 
2. This author provides large-scale theoretical analysis, focusing on very important aspects, such as the stability of bi-level optimization, the loss trajectory, etc. I think the analysis is insightful. 
3. The proposed method can reduce the search costs.

### Weaknesses
1. I think the figures in this paper can be polished to be more clear (maybe in the camera ready version). 
2. The accuracy of the proposed method is just comparable with sota, but not superior to sota. I think it is not a serious problem, but I just list it as one weakness.

### Questions
I think overall this paper is good. Currently I give 6 since I have not checked the proof very carefully. I am willing to raise the score to 8 if the proof is proved to be right by other reviewers.

### Soundness
4

### Presentation
3

### Contribution
3
