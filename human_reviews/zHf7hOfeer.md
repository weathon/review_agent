# Zero-cost Proxy for Adversarial Robustness Evaluation

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Deep neural networks (DNNs) easily cause security issues due to the lack of adversarial robustness. An emerging research topic for this problem is to design adversarially robust architectures via neural architecture search (NAS), i.e., robust NAS. However, robust NAS needs to train numerous DNNs for robustness estimation, making the search process prohibitively expensive. In this paper, we propose a zero-cost proxy to evaluate the adversarial robustness without training. Specifically, the proposed zero-cost proxy formulates the upper bound of adversarial loss, which can directly reflect the adversarial robustness. The formulation involves only the initialized weights of DNNs, thus the training process is no longer needed. Moreover, we theoretically justify the validity of the proposed proxy based on the theory of neural tangent kernel and input loss landscape. Experimental results show that the proposed zero-cost proxy can bring more than $20\times$ speedup compared with the state-of-the-art robust NAS methods, while the searched architecture has superior robustness and transferability under white-box and black-box attacks. Furthermore, compared with the state-of-the-art zero-cost proxies, the calculation of the proposed method has the strongest correlation with adversarial robustness. Our source code is available at https://github.com/fyqsama/Robust_ZCP.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a zero-shot proxy applied directly to the initial weights of a deep neural network (DNN), serving as an upper bound for adversarial loss. This approach enhances the adversarial robustness of the network without requiring adversarial examples. Theoretical justification for the zero-shot proxy is provided based on the Neural Tangent Kernel (NTK) and the input loss landscape. Experimental results demonstrate a 20x speedup over state-of-the-art robust neural architecture search (NAS) methods, with no loss in performance against black-box and white-box attacks.

### Strengths
+ The preliminaries on the Neural Tangent Kernel (NTK) and input loss landscape are clearly presented, making the methodology easy to understand. All symbols and equations are well-defined. 

+ A robust theoretical justification for the proposed zero-shot proxy is provided by establishing the corresponding loss as an upper bound for adversarial loss.

+ The efficiency of the search process using their technique is clearly demonstrated in the experimental section.

+ The authors introduce a novel dataset called Tiny-RobustBench, which may prove valuable to the research community.

### Weaknesses
- In the theoretical analysis section,  $\lambda_{\min}(\hat{\Theta_{\theta_o}})$ is approximated based on empirical observations to $-\lambda_{\min}(\Theta_{\theta_o})$, which weakens the theoretical foundation of the work. I wonder if the authors could establish the relationship between the two NTKs in strict mathematical terms, such as showing that the approximation holds true within a specific margin of error with a certain probability. Currently, the analysis relies on an empirically driven approximation, which undermines its theoretical robustness.
- The performance of the black-box and white-box attacks is demonstrated using a single dataset (CIFAR-10) in Tables 1 and 2. The authors should consider providing performance comparisons across multiple datasets for the chosen baselines in both tables. Additionally, Table 2 should clearly present a comparison of search efficiency.
- I am curious about the rationale for selecting only specific models (ResNet-18 and PDARTS) for comparison in Table 3. Would it be possible to include models from Table 1 to evaluate their transferability in Table 3?
- There is inadequate justification for why their approach exhibits better transferability in Table 3. It would be beneficial to provide justification grounded in their theoretical contributions, particularly highlighting the unique advantages that their proposed techniques offer to enhance transferability.
- The authors should discuss the assumptions made in their analysis and their validity within the context of the study. For instance, when transforming from Equation 1 to Equation 3, it would be helpful to clarify the assumptions involved and how they apply in their setting. Specifically, I am uncertain whether the assumption of infinite-width DNN parameters is valid; if it is not, what implications does that have for this paper? I encourage the authors to include these details, even if they are standard, as doing so would enhance the paper's readability and facilitate the assessment process.

### Questions
Please refer to Weaknesses Section

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a zero-shot proxy that requires no training and depends only on the initial neural network weights to find the robust architecture. Using this zero-shot proxy reduces the neural architecture search speed drastically. Experiments are performed on multiple datasets with varying resolution and number of classes.

### Strengths
- The reduction in the search cost is very significant. 
- The paper is well-written with typos.  
- From Table 3, the transferability results are good. 
- A novel dataset will be released by the authors, which will help advance research in the robust NAS direction.   
- Comparison with the latest state-of-the-art methods.

### Weaknesses
- While the search cost is reduced significantly, the increase in the adversarial robustness is minimal (less than 1% in most cases for Table 1). A similar observation can be derived from the results in Table 4.    
- Minor typos: "8/2550" instead of "8/255" on line 301, "We" instead of "we" on line 63.

### Questions
While the contribution in terms of speed reduction is significant for the proposed method, my concern is the increase in adversarial robustness as compared to the baselines is very minimal in the majority of the cases.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper targets a lightweight proxy to assess the adversarial robustness of Neural Architecture Search (NAS) networks. The proposed proxy is represented as the product of two terms (Eq. 4). The first term, based on the intuition that adversarial and natural accuracy are reversely correlated, is introduced to consider adversarial accuracy and is experimentally validated. The second term is proposed through a theoretical approximation using the Neural Tangent Kernel. The proposed proxy is experimentally validated.

### Strengths
- The paper is easy to read through, well presented.
- The paper proposes relatively strong method for the proxy of searching adversarially robust architectures.
- The paper includes numerous validations including performance under white-box, black-box attacks, datasets, numerous datasets, and ablation studies.

### Weaknesses
1.  To read through, I felt that the paper has to focus on two points : 
- Is the proposed method superior in terms of cost-efficiency? (zero-cost)
- Is the proposed method superior in terms of robust accuracy?
- For now, RQ1 and RQ2 looks a bit similar(line 255-256), before looking at the experimental results.

2. I feel the cost-efficiency related explanation is slightly short. The authors could include detailed explanations for (line 183-185.) such as , the reason the authors chose to iterate samples over generating adversarial samples itself and why is it efficient.

### Questions
- In the preliminatries, 3.1., the formulation is bit awkward; if the function $f_{\theta_t}$ is a function $\mathbb{R}^d \rightarrow \mathbb{R}$, then the output will be a single scalar, but in the equations the l2-norm calculation is applied … I also checked the reference paper (Xu et al., (2021)), but it is slightly different. 

- The overall experiments on Kendall’s tau seems to be not high (Fig.2,Fig.4). (Relatively high, but not significant in absolute terms). Can the authors explain this in further details?

- The cost analysis is done with the GPU search day metric. How is the cost calculated? Under which GPU?

[Post-Rebuttal] I have reviewed the author's rebuttal and the feedback from other reviewers. Most of my concerns have been addressed, and given the paper's contributions, I would like to maintain my current rating.

### Soundness
3

### Presentation
2

### Contribution
3
