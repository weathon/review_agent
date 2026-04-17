# Dual-Priv Pruning : Efficient Differential Private Fine-Tuning in Multimodal Large Language Models

- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
Differential Privacy (DP) is a widely adopted technique, valued for its effectiveness in protecting the privacy of task-specific datasets, making it a critical tool for large language models. However, its effectiveness in Multimodal Large Language Models (MLLMs) remains uncertain. Applying Differential Privacy (DP) inherently introduces substantial computation overhead, a concern particularly relevant for MLLMs which process extensive textual and visual data. Furthermore, a critical challenge of DP is that the injected noise, necessary for privacy, scales with parameter dimensionality, leading to pronounced model degradation; This trade-off between privacy and utility complicates the application of Differential Privacy (DP) to complex architectures like MLLMs. To address these, we propose Dual-Priv Pruning, a framework that employs two complementary pruning mechanisms for DP fine-tuning in MLLMs: (i) visual token pruning to reduce input dimensionality by removing redundant visual information, and (ii) gradient-update pruning during the DP optimization process. This second mechanism selectively prunes parameter updates based on the magnitude of noisy gradients, aiming to mitigate noise impact and improve utility. Experiments demonstrate that our approach achieves competitive results with minimal performance degradation. In terms of computational efficiency, our approach consistently utilizes less memory than standard DP-SGD. While requiring only 1.74% more memory than zeroth-order methods which suffer from severe performance issues on A100 GPUs, our method demonstrates leading memory efficiency on H20 GPUs. To the best of our knowledge, we are the first to explore DP fine-tuning in MLLMs. Our code is Our code is avaliable in : https://anonymous.4open.science/r/Dual-priv-pruning-AE7E.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper introduces Dual-Priv Pruning, a new framework for differentially private (DP) fine-tuning of Multimodal Large Language Models. Dual-Priv Pruning addresses the challenges of computational overhead and model degradation due to noise injection in DP-based finetuning, which scales with parameter dimensionality. The proposed Dual-Priv Pruning framework employs two techniques: (1) visual token pruning and fusion to reduce input dimensionality by removing redundant visual information tokens, and (2) gradient-update pruning to apply noisy gradients during DP-SGD optimization selectively. Experimental results show that Dual-Priv Pruning efficiently reduces the computational cost during DP-based finetuning.

### Strengths
1.	Pioneering Approach. Dual-Priv Pruning is the first framework to address DP fine-tuning specifically for MLLMs, filling a critical research gap.

2.	Efficiency Gains. Dual-Priv Pruning achieves significant memory reduction (14.34% less peak GPU usage) and computational efficiency compared to standard DP-SGD.

3.	Better privacy-utility trade-off.  Dual-Priv Pruning maintains competitive performance under strict privacy budgets (ε ≤ 3) despite noise injection during DP optimization.

### Weaknesses
1.	Lack of Novelty. The work doesn’t invent a new approach for DP. It is more like an engineering approach that combines existing pruning methods and DP training methods.

2.	Limited Generalizability: Relies on specific assumptions (e.g., selection mechanism for visual tokens), which may not apply universally across all MLLM tasks.

3. Doubtful experimental results. Under ϵ = inf in Table 1, non-private performance is reported for DZPO, DP-SGD and Dual-Priv. As it is a non-private performance, why does the overall performance differ so much? For example, under ScienceQA and ϵ = inf setup, DZPO has an accuracy of 22.16, DP-SGD has an accuracy of 81.10, while  Dual-Priv has an accuracy of 84.60. Do they use the same base VLLM? I am so confused about the reported results.

### Questions
I am not familiar with the VLLM optimization and the evaluated tasks. So my review comments may not be professional, and I set my confidence score as 2. If Question 1 can be addressed properly, I am willing to raise my rating.

1. Under ϵ = inf in Table 1, non-private performance is reported for DZPO, DP-SGD and Dual-Priv. Why does the overall performance differ so much? Do they use the same base VLLM?

2. Will empirical attacks be conducted to verify the robustness of Dual-Priv Pruning?

3. What is the baseline VLLM performance without fine-tuning? What is the baseline VLLM performance after normal non-private fine-tuning?

### Soundness
2

### Presentation
2

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
Dual-Priv Pruning is a novel framework for differentially private fine-tuning of multimodal large language models (MLLMs) that addresses the computational challenges and privacy-utility trade off. The proposed method combines two machines: 1.  visual token pruning using attention mechanisms to select and compress the most informative visual tokens, and 2. gradient-update pruning that selectively updates only the most significant parameter blocks after adding DP noise. The proposed method achieves substantial reductions in memory and computational overhead by reducing the context length of each input. The experimental results show improved accuracy and significant GPU memory savings across a range of benchmarks and privacy budgets.

### Strengths
1. This paper address an important challenge of privacy utility trade off in a multimodal LLM fine-tuning setup.
2. The proposed method employs two levels of pruning, one for each a) reducing memory overhead, and b) reducing the impact of DP noise on utility.
3. The experimental results show the improvements in accuracy resulted by utilizing dual-priv pruning as compared to DP-SGD (first order DP fine-tuning) and DPZO (zeroth order DP fine-tuning) for various benchmarks and privacy settings.
4. The paper also presents ablation studies on memory usage and empirical privacy results via MIA.

### Weaknesses
The proposed mechanism introduces additional hyper-parameters that need to tuned such as 1) selected layers of the vision encoder for computing the importance scores, and 2) K and |C| values for pruning in step-1 (token selection) and step-2 (gradient pruning). Tuning these parameters to get reasonable trade-offs can introduce heavy computational overhead.

### Questions
1. "we first compute the multi-head self-attention maps within a selected layer of the vision encoder." How do we decide which layers or how many layers to use for scores computation? Is this a hyper-parameter that needs tuning?
2. Section 5.5 indicates computational efficiency analysis but only presents the memory usage numbers. Dual-priv pruning reduces computational cost by token pruning but at the same time there are additional computations introduced to select the important tokens and contextual token fusion. What are the compute savings of the end-to-end pipeline of the proposed method in-terms of FLOPs as compared to DP-SGD?
3. We could potentially reduce the number of parameters by reducing the LoRA rank. How does dual-priv pruning compare to DP-SGD at iso-parameters? For more context, let's say that we train an MLLM with 1) setup 1: Dual-Priv with LoRA rank (r) = 128 with 50% pruning and compare it with, 2) setup 2: DP-SGD with LoRA rank (r) = 64. Assuming both these setups do not have mechanism 1 (token pruning). The goal here is to understand if top-k gradient pruning is significantly better than simply reducing the LoRA rank. 
4. The proposed mechanism seems generic and can be applied to any large model with transformer architecture. Is there any part of the algorithm that is addressing multimodal specific challenges? 
5. How does "Dominant Token Selection via CLS Attention" compare with random selection?
6. In table Table 5, can you add an additional datapoint where you split mechanism 1 into a) w/ token pruning, and b) token pruning + contextual fusion? The proposed algorithm has the following pieces: token pruning + contextual fusion + fusion noise + gradient pruning. To have a comprehensive ablation study, I would suggest adding results on all combinations: (0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1) and (1, 1, 1, 0). 
7. Token pruning depends on the existence of CLS token. Is the proposed method extendable to generation tasks?
8. The final set of visual tokens is a concatenation of $v_{cls}$, $V_d$ and $C$. Do the proposed method ensure that the ordering of the tokens is maintained? The wrong ordering of tokens might not impact the performance of models with visual inputs but in my opinion, this will impact performance of models with text inputs. Will the proposed method work equally well for text to text LLMs?

### Soundness
3

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
This paper studies differentially private training of Multi-modal LLMS (vision language models). DP-SGD is the most widely used algorithm for training models with differential privacy. DP-SGD adds noise to the gradients at each step, where the noise scales with The authors improve upon this baseline with two techniques that increase model accuracy:

(1) reduce dimensionality of the input image by selecting the most relevant tokens and fusing the remaining tokens using a clustering technique. 

(2) masking the (noisy) gradient update so that the gradient update only occurs for the parameters with strongest signal (about 80% of parameters). 

The authors evaluate accorss several visonal language question answering benchmarks, including benchmarsk in the medical domain where privacy is a more relevant concern. There are consistent accuracy improvements over the DP-SGD baseline (~4 pct points and ~8 pct points for 2 of the tasks, and at about 1-2 pct point in 3 of the tasks). The reduced input dimensionaly also leads to memory improvements of 14%.

### Strengths
- First paper to consider private training of VLLMs, opening up a new avenue for research 
- Paper proposes some interesting techniques for improving the utility of differentially private training when working with high-dimensional data. These techniques might be useful in other settings for differentially private traning beyond VLLMs
- Comprehehensive evaluation with open source code
- Consistent improvement over the baseline method. 
- Great, easy to follow presentation

### Weaknesses
The first technique, which reduces dimensionality of the input image by selecting the most relevant tokens and fuses the remaining tokens using an averaging+clustering method,  does not have a differential privacy guarantee. Instead noise is added to the fused tokens heuristically. Unless I am missing something, the E2E algorithm is not technically differentially private and I think this should be emphasized further in the limitations/intro. 

I agree that for practical privacy guarantees and as shown by your MIA results this might not matter as much.

### Questions
Couuld you provide more intuition for why adding nosie to the fused non-dominant tokens helps with accuracy? You also say that this noise should be of the same magnitude as the noise added to the gradients. Maybe I am missing something but isn't the gradient computed wrt to the fused tokens as well, so why do we need double the noise?

The sentece in line 199-200 is also confusing. It seems to imply that because the pruning is text-agnostic we do not need to worry about using part of the privacy budget, but your privacy statement is wrt both the image and text.

Willing to increase my score upon clarification of these questions and addressing of the weakness I mention.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a dual pruning algorithm to optimize the differential privacy process of MLLMs. The first stage uses visual CLS to discard less important visual tokens, and the second stage uses gradient pruning for parameter updates.

### Strengths
1.	The motivation behind the problem is clear: MLLM computation is computationally expensive. Differential privacy also suffers from reduced utility as dimensionality increases.
2.	The method design is simple and relatively easy to implement.
3.	The experiments and ablation studies are relatively comprehensive.

### Weaknesses
1. It is unclear whether importance scores should be used as the evidence for discarding tokens. This seems to be a common technique used in engineering, and the author also seems to have demonstrated the significance of discarding them. However, simply discarding tokens based on the importance of attention seems to lack rigorous justification.

2. It's unclear why the author conducted accuracy experiments on the Q&A dataset: the author proposed a new method for differential privacy, but testing its accuracy on different Q&A datasets seems strange, because differential privacy itself is not designed to achieve higher accuracy. I think the author wanted to convey that the discarded tokens are redundant tokens, and that even after discarding them, the method still maintains high accuracy on Q&A datasets, is that correct?

3. Experiments conducted using only LLAVA-7B and its medical fine-tuned model have limited effectiveness. The authors should consider incorporating other models into the experiment.

4. Anonymous code cannot be accessed.

5. The text contains numerous typos and content that could easily mislead readers. For example, in line 269, "Mechanism 2" should be "Mechanism 1". Requiring readers to infer these errors from the context increases the reading effort.

### Questions
See the weaknesses

### Soundness
3

### Presentation
2

### Contribution
2
