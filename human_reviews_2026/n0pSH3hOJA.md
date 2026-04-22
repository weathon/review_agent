# Modality-Aware Quantization: Balancing Visual and Textual Fidelity in Multimodal Compression

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 4, 4

## Abstract
Vision-language models (VLMs) have achieved remarkable capabilities across multimodal tasks, yet their deployment remains constrained by substantial computational requirements. While post-training quantization (PTQ) offers a practical solution for model compression, existing methods fail to address a fundamental challenge in VLM quantization: the inherent heterogeneity between visual and textual representations. In this work, we identify and formalize a critical failure mode where visual tokens, despite their lower semantic density, dominate the quantization optimization process due to their extreme value distributions and numerical prevalence. This dominance systematically degrades the preservation of semantically-critical language tokens, severely impacting model performance. We present a theoretically grounded framework that proves this trade-off through formal analysis and introduce an adaptive optimization pipeline that dynamically balances cross-modal heterogeneity. Our method leverages activation-scale statistics and gradient-sensitivity priors to construct layer-wise modality weights that counteract visual dominance while preserving linguistic fidelity—all without altering the inference computation graph. Extensive experiments demonstrate state-of-the-art performance across diverse quantization regimes: on Qwen-VL-Chat with W4A8 quantization, we achieve 59.27% on TextVQA, surpassing the previous best method MQuant by +2.70%. Most notably, under extreme W4A4 quantization where existing approaches fail catastrophically, our method maintains robust performance (55.72% on TextVQA), proving that aggressive multimodal compression is both achievable and practical for real-world deployment. The code is available at this anonymous link:
https://anonymous.4open.science/status/MAQ-23971iclrAnonymous

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates a key challenge in post-training quantization (PTQ) of vision-language models (VLMs): the heterogeneity between visual and textual representations, which causes the visual token dominance problem—visual tokens, though less semantically dense, dominate quantization due to their extreme value distribution and larger number, reducing the fidelity of crucial text tokens and degrading performance. To address this, the paper theoretically proves the trade-off between visual and textual quantization losses (Target Weakening Principle) and proposes a computation-invariant adaptive optimization framework that uses activation-scale statistics and gradient-sensitivity priors to assign hierarchical modality weights, effectively mitigating visual dominance without changing the inference computation graph.

### Strengths
This paper combines MQuant and FlatQuant and achieves competitive performance under the W4A4 quantization setting.

### Weaknesses
1. The innovation of this work is very limited. LayerNorm->RMSNorm and FlatQuant are both existing and commonly used technologies, yet this paper presents them as core contributions. Additionally, the heuristic method of weighting the language and visual branches in step 3 also has limited innovation.
2. FlatQuant was originally designed for LLMs. The paper directly applies it to cover the three major components of VLMs but fails to analyze the effect of its rotation-reconstruction mechanism on handling the heavy-tailed distribution of visuals. Moreover, it lacks reporting on the additional time and machine overhead caused by introducing FlatQuant.
3. The paper only compares with MQuant, a representative method, and lacks comparisons with methods such as MBQ and VLMQ, which limits its credibility. I have reproduced MQuant, and it can still maintain good performance under the W4A4 quantization setting for the language part, but the accuracy reported in this work is 0, which I find questionable.
4. The generalization performance of the model is limited. The experiments are only based on two 8B-scale VLMs (Qwen-VL-Chat, InternVL2-8B) and do not test larger parameter-scale VLMs such as Qwen-VL-14B and LLaVA-13B.
5. The typesetting and writing of the article need improvement. The text in Figure 1 is unclear, while the text in Figure 2 is too large. The overall writing focuses on token differences, yet FlatQuant is probably the component that contributes the most to accuracy.
[1] Mquant: Unleashing the inference potential of multimodal large language models via full static quantization. 
[2] Flatquant: Flatness matters for llm quantization.

### Questions
1. If only method 3 is used, what is the performance on W4A8 and W4A4?
2. Is the W4A4 reported in the article static quantization or dynamic quantization? If it is dynamic quantization, comparing it with MQuant may be unfair.
3. Can you report the parameter optimization time and analysis with some other methods?
4. Is GPTQ performed after FlatQuant?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the problem of post-training quantization (PTQ) for Vision-Language Models (VLMs). The authors identify and formalize a key challenge: during the quantization optimization process, visual tokens, which have a larger numerical range and greater quantity, tend to "dominate" the loss function, thereby degrading the precision of textual tokens crucial for model performance. To tackle this, the paper proposes a modality-aware quantization framework. This framework employs an adaptive weighting scheme to balance the reconstruction errors of visual and textual tokens, with the weights determined by the ratio of activation scales and a gradient sensitivity prior. Experimental results show that the proposed method achieves state-of-the-art performance across various quantization settings, and notably maintains model usability in extreme W4A4 compression scenarios where other methods fail.

### Strengths
1.  The paper tackles a critical bottleneck in the practical deployment of VLMs—their computational and memory overhead. The "visual dominance" problem it introduces is a keen and intuitive observation, offering a new and valuable perspective on VLM quantization.
2.  Centered around the core problem of "visual dominance," the proposed modality-aware weighting scheme is well-motivated and logically sound. Combining activation scales and a gradient prior to dynamically adjust the loss is a concise yet effective design.
3.  The method achieves strong performance across multiple benchmarks. Its most notable strength is its robustness under extreme low-bit (W4A4) quantization, which validates the effectiveness of the approach and significantly advances the practicality of using multimodal models in resource-constrained environments.

### Weaknesses
1.  Insufficient justification for a key hyperparameter: In the methodology, the authors introduce a "gradient disparity prior" parameter, `β`, and set its default value directly to 0.1. The paper does not explain the selection process for this value, nor does it provide a sensitivity analysis. This makes a part of the method appear to rely on a "magic number" that has not been thoroughly validated, which undermines its "adaptive" nature and generalizability.
2.  The ablation study is not comprehensive enough: The current ablation study (Table 2) only compares "with weighting" versus "without weighting." While this demonstrates the overall effectiveness of the strategy, it is not deep enough. The authors' final weight is a product of two components: the activation scale ratio `α` and the gradient prior `β`. A more convincing ablation study should disentangle the individual contributions of these two components to understand why the method is effective, and whether they produce a synergistic effect.

### Questions
1.  Regarding the gradient disparity prior parameter `β`, how did the authors choose the default value of 0.1? Was any hyperparameter search or sensitivity analysis performed to justify this choice? Does this parameter need to be re-tuned for different models or tasks?
2.  Could the authors provide a more detailed ablation study to separately show the contributions of the activation scale ratio (`α`) and the gradient prior (`β`) to the final performance improvement? For instance, what is the performance when using only `α` (i.e., `β=1`) or only `β` (with a constant `α`)? This would help clarify the working mechanism of the method.

### Soundness
3

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
4

### Summary
This paper addresses post-training quantization (PTQ) for VLMs by identifying a failure mode where visual tokens dominate optimization due to extreme distributions and numerical prevalence, systematically degrading language token preservation. The authors propose an adaptive optimization pipeline combining activation-scale statistics with gradient-sensitivity priors to construct layer-wise modality weights. Experiments show improvements, particularly under extreme quantization regimes where existing methods fail catastrophically.

### Strengths
- The paper has well-motivated problem identification of the visual token dominance issue.
- The paper is well written and easy to follow.

### Weaknesses
Please see my questions and concerns below.

### Questions
- Theorem 1 only provides an inequality relationship but gives no guidance on how to _choose_ $\alpha_2$ to achieve desired $L_A(\Theta_2)$. How do you select the optimal weighting in practice?
- The inequalities in Theorem 1 seem not tight. Can you characterize when they become equalities? What does this reveal about the optimization landscape?
- The proof implicitly treats the optimization as if comparing global optima, but VLM quantization is highly non-convex. How does non-convexity affect the validity of your conclusions?
- Is there any experiment directly validating Theorem 1? Show $L_A(\Theta_2) \leq L_A(\Theta_1)$ and $L_B(\Theta_1) \leq L_B(\Theta_2)$ with measured values.
- Eq.5 assumes errors from visual and textual tokens add independently. What if there are interaction effects? Cross-modal dependencies?
- During iterative optimization, activation statistics change. Do you recompute $\alpha$ at each iteration or fix it initially?
- FlatQuant assumes rotational invariance, but your modality weighting seems break symmetry. Are these approaches compatible?
- Are rotation matrices learned jointly with quantization parameters? If so, how does modality weighting affect rotation learning?

### Soundness
3

### Presentation
2

### Contribution
2
