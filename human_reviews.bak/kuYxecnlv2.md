# Addressing Label Shift in Distributed Learning via Entropy Regularization​

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 6

## Abstract
We address the challenge of minimizing "true risk" in multi-node distributed learning.\footnote{We use the term node to refer to a client, FPGA, APU, CPU, GPU, or worker.} These systems are frequently exposed to both inter-node and intra-node "label shifts", which present a critical obstacle to effectively optimizing model performance while ensuring that data remains confined to each node.
To tackle this, we propose the Versatile Robust Label Shift (VRLS) method, which enhances the maximum likelihood estimation of the test-to-train label importance ratio. VRLS incorporates Shannon entropy-based regularization and adjusts the importance ratio during training  to better handle label shifts at the test time.
In multi-node learning environments, VRLS further extends its capabilities by learning and adapting importance ratios across nodes, effectively mitigating label shifts and improving overall model performance. Experiments conducted on MNIST, Fashion MNIST, and CIFAR-10 demonstrate the effectiveness of VRLS, outperforming baselines by up to 20\% in imbalanced settings. These results highlight the significant improvements VRLS offers in addressing label shifts. Our theoretical analysis further supports this by establishing high-probability bounds on estimation errors.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes the Versatile Robust Label Shift (VRLS) method, which enhances the maximum likelihood estimation of the test-to-train label density ratio, for multi-node distributed learning. It incorporates Shannon entropy-based regularization and adjusts the density ratio during training to better handle label shifts at the test time. Experiments conducted on MNIST, Fashion MNIST, and CIFAR-10 demonstrate the effectiveness of the proposed VRLS. The main contributions of this paper are:

- It proposes VRLS to enhance the training of the predictor of the conditional distribution by incorporating a novel regularization term based on Shannon entropy.

- It establishes high-probability estimation error bounds for VRLS, as well as high-probability convergence bounds for IW-ERM with VRLS in nonconvex optimization settings.

### Strengths
- The problem studied in this paper is interesting.

- The paper is well-organized, which is easy to follow.

- This paper provides some novel perspectives for multi-node distributed learning.

### Weaknesses
- In my opinion, the concept of "underlying uncertainty" needs to be further elaborated. Specifically, why does the failure of a learned predictor to inherently capture the underlying uncertainty have an impact on model performance? What is the Role of "underlying uncertainty" in multi-node distributed learning?
- The motivation for the introduction of Shannon entropy-based regularization also should be further explained by the authors. What is the advantage of the leveraged technology compared with others?
- For the experimental evaluation, this paper compares the proposed method with some baseline methods. The experimental results seem promising. However, the comparison methods in this paper are all proposed before 2021, and lack the latest algorithms (e.g. [1]-[2]). I think the comparisons between the proposed method and the latest algorithms are necessary.



[1] [GradMA: A Gradient-Memory-based Accelerated Federated Learning with Alleviated Catastrophic Forgetting. CVPR 2023: 3708-3717.](https://ieeexplore.ieee.org/document/10203330/)

[2] [Multi-Level Branched Regularization for Federated Learning. ICML 2022: 11058-11073.](https://proceedings.mlr.press/v162/kim22a.html)

### Questions
1. The concept of "underlying uncertainty" needs to be further elaborated. Specifically, why does the failure of a learned predictor to inherently capture the underlying uncertainty have an impact on model performance? What is the Role of "underlying uncertainty" in multi-node distributed learning?
2. The motivation for the introduction of Shannon entropy-based regularization also should be further explained by the authors. What is the advantage of the leveraged technology compared with others?
3. The comparison methods in this paper are all proposed before 2021, and lack the latest algorithms. The comparisons between the proposed method and the latest algorithms are necessary.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the challenge of inter-node and intra-node label shifts in distributed learning. Test-to-train density ratio estimation plays an important role in addressing the label-shift problem. The authors proposed the Versatile Robust Label Shift (VRLS) method, which incorporates a regularization term based on Shannon entropy. This regularization term significantly improves the estimation of the conditional distribution $p^{tr}(\mathbf{y}|\mathbf{x})$, thereby enhancing the estimation of the label density ratio. The authors conduct experiments on MNIST, Fashion-MNIST, and CIFAR-10 datasets to verify the effectiveness of the proposed method. The authors also provide theoretical analysis for the estimation error and convergence.

### Strengths
1. The authors proposed a simple but effective method to address the problem of label shift in distributed learning.
2. The authors conduct experiments to verify their proposed method. The MSE for the density ratio of the proposed method is lower than that of baselines. The test accuracy of the proposed method is also better than that of baselines.
3. Bounds on ratio estimation and convergence rates are analyzed theoretically.

### Weaknesses
1. The definition of inter-node and intra-node label shifts is ambiguous. I suggest that the authors define them more clearly.
2. In Lines 196-198, why penalizing the Shannon entropy of the softmax output can improve the approximation of $p^{tr}(\mathbf{y}|\mathbf{x})$? In other words, why the smoother output is a better approximation of $p^{tr}(\mathbf{y}|\mathbf{x})$ (Line 207-208)?
3. What is the meaning of $\lambda_k$ in the risk of IW-ERM?
4. Lack of sensitivity test for the hyperparameter $\zeta$ in Equation (2).
5. The font sizes in Fig 1 and Fig 2 are inconsistent.

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes the Versatile Robust Label Shift (VRLS) method to address label shift in distributed learning systems. VRLS employs entropy regularization to improve density ratio estimation between training and test label distributions, making it resilient to both intra-node and inter-node label shifts. The paper further extends VRLS to a multi-node setup through an Importance Weighted ERM (IW-ERM) framework, which balances label distribution estimation across nodes while preserving data privacy. Empirical results show VRLS outperforming baselines by up to 20% in imbalanced settings, and theoretical bounds confirm VRLS's estimation accuracy and convergence rates.

### Strengths
1-VRLS introduces a unique entropy regularization term that enhances density ratio estimation by penalizing degenerate solutions, thus yielding well-calibrated predictions. Unlike existing methods such as MLLS and BBSE, which do not adequately capture uncertainty, VRLS adapts the estimation process to real-world variability. This improvement aligns with recent work in calibration under distribution shifts (e.g., Guo et al., 2017), offering both theoretical and practical robustness.

2-The paper’s thorough empirical evaluation spans single-node and multi-node scenarios, with tests on MNIST, Fashion MNIST, and CIFAR-10. In distributed settings, VRLS with IW-ERM adapts effectively, providing significant accuracy improvements across nodes while handling inter-node and intra-node label shifts. Compared to methods such as FedAvg and FedProx, which struggle with label shift, VRLS maintains robust performance in both settings, a key distinction from prior approaches.

3- Theoretical error bounds and convergence rates for VRLS are rigorously established. The authors show that the VRLS method’s estimation error decreases proportionally to the square root of test sample size, a valuable addition to the literature where empirical results often lack formal guarantees. This theoretically grounded approach helps ensure VRLS's performance consistency in both high and low label-shift scenarios, a substantial contribution over prior studies that assume stability without formal validation.

4- By using locally computed density ratios without requiring data exchange, IW-ERM maintains data privacy across nodes, a marked improvement over traditional federated approaches. This reduction in communication rounds, coupled with enhanced convergence rates, renders IW-ERM with VRLS practical for real-world applications where privacy and resource efficiency are paramount, setting it apart from resource-intensive federated models like FedBN.

### Weaknesses
1- The paper effectively shows VRLS's reliance on Shannon entropy regularization but does not explore how it compares to other regularization techniques, such as focal loss or maxent loss. A comparative analysis across regularization methods on the same datasets, particularly with focal loss regularization, would help clarify VRLS's robustness and calibration capabilities. Additionally, a theoretical discussion on the effects of different regularizations on model calibration and uncertainty estimation could deepen the understanding of VRLS's effectiveness.

2-VRLS primarily addresses scenarios under symmetric or Dirichlet-based label shifts, with limited discussion on its performance with highly non-uniform or asymmetric label shifts. Extending empirical analysis to include more diverse label distributions would provide a clearer picture of VRLS’s generalizability to complex, real-world shifts.

3- The theoretical error bounds rely on assumptions that may not fully align with real-world distributed data characteristics, such as bounded data spaces and sub-Gaussian noise. Providing examples of scenarios where these assumptions might fail, along with discussing VRLS’s potential performance in such cases, would add rigor to the theoretical claims. Additionally, testing VRLS with artificially unbounded or heavy-tailed data distributions would offer practical insights into the method’s robustness beyond the idealized assumptions, adding depth to its theoretical grounding.

4- The paper’s focus on label shift in classification limits its applicability to other supervised learning tasks, such as regression or multi-label classification, where label distributions differ. Addressing how VRLS might generalize to other tasks, or clarifying any modifications needed, could extend the method’s relevance across broader distributed learning applications.

5- The VRLS method’s reliance on entropy-based regularization (inspired by Shannon entropy) is theoretically motivated by preventing degenerate solutions. However, there is limited theoretical analysis on whether this regularization could inadvertently introduce biases, especially in settings where label shifts deviate from the assumptions about class-conditional stability. This could be explored further in the theoretical results or discussed as a potential limitation.

### Questions
1-  Could the authors discuss how VRLS would perform with other regularization methods, such as focal loss, or compare its performance with maxent loss for calibration? Would such alternatives affect calibration or robustness?

2- Can the authors provide empirical or theoretical evidence on VRLS’s performance under non-uniform or more complex label shift distributions, such as class-conditional label shift?

3- How does VRLS handle scalability in networks with significantly more nodes or nodes with varying computational capacities? Are there adjustments to VRLS or IW-ERM to address such scenarios?

4-  Given that assumptions like bounded data and sub-Gaussian noise are not always valid in practice, could the authors discuss how VRLS might be adapted or how results may vary when these assumptions do not hold?

5-  VRLS is designed for classification, but can it be adapted for regression tasks where label distributions may also shift? If so, what adjustments would be needed?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper solves the problem where distributed data nodes have different label distributions in their training and test data. The authors propose VRLS - a method that estimates how these distributions differ between nodes and use this information to improve model training. Their approach maintains data privacy while achieving 20% better accuracy than existing methods.

### Strengths
1. This paper clearly articulates its research focus by addressing both inter-node and intra-node label shifts in distributed learning systems, which represents a more comprehensive approach than prior works.
2. The paper provides extensive theoretical proofs to support the proposed algorithm, though I have not thoroughly examined all of them.

### Weaknesses
1. The paper lacks clear logical flow and detailed algorithm explanations. While the title emphasizes distributed learning, Section 4 (covering VLRS in distributed environments) is surprisingly brief - only one-third of a page. It simply states that minimizing equation (IW_ERM) enables VLRS in distributed settings, without proper elaboration. In contrast, Section 5 on VLRS bounds spans two full pages. It's puzzling why the authors prioritized theoretical bounds over making their work more accessible to readers.
2. Lines 193-194 contain an inconsistency where the authors claim "Our proposed VRLS objective aims to better approximate the conditional distribution ptr(y|x)", but equation 1 (VRLS objective) actually estimates r. The conditional distribution approximation corresponds to equation 2.
3. The authors fail to adequately explain why minimizing equation 2 leads to better conditional distribution estimation. While Ω(fθ) represents entropy and promotes more dispersed neural network outputs as claimed, there's no clear justification for how this ensures f(x) better estimates ptr(y|x).
4. The paper suffers from poor notation management. Symbols like Ω(fθ) should be defined at first appearance rather than later, as this significantly increases the time needed to understand the paper.
5. The experimental validation is unconvincing, relying only on synthetic datasets (MNIST and CIFAR-10) with small images and just 10 classes. The evaluation should include larger datasets like ImageNet and real-world scenarios. The lack of real-world validation raises concerns about whether the paper's assumptions are practical.

### Questions
same as the third point of weakness: why minimizing equation 2 leads to better conditional distribution estimation?

### Soundness
3

### Presentation
2

### Contribution
2
