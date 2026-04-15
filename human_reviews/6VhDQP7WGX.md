# Inference Optimal VLMs Need Fewer Visual Tokens and More Parameters

- Decision: Accept (Poster)
- Scores: 8, 5, 5, 6, 5

## Abstract
Vision Language Models (VLMs) have demonstrated strong capabilities across various visual understanding and reasoning tasks, driven by incorporating image representations into the token inputs of Large Language Models (LLMs). However, their real-world deployment is often constrained by high latency during inference due to the substantial compute required by the LLM to process the large number of input tokens, predominantly arising from the image. To reduce inference costs, one can either downsize the LLM or reduce the number of input tokens needed to represent the image, the latter of which has been the focus of many recent efforts around token compression. However, it is unclear what the optimal trade-off is given a fixed inference budget. 
We first characterize this optimal trade-off between the number of visual tokens and LLM parameters by establishing scaling laws that capture variations in performance with these two factors. Our results reveal a surprising trend: for visual reasoning tasks, the inference-optimal behavior in VLMs is achieved by using the largest LLM that fits within the inference budget while minimizing visual token count - often to a single token. While the token reduction literature has mainly focused on maintaining base model performance by modestly reducing the token count (e.g., $5-10\times$), our results indicate that the compute-optimal inference regime requires operating under even higher token compression ratios. Based on these insights, we take the first steps toward designing token compression algorithms tailored for high-compression settings, utilizing prompt-based compression of tokens. Our work underscores the performance and efficiency benefits of operating in low visual token regimes and the importance of developing tailored token reduction algorithms for such conditions.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper addresses a crucial trade-off between visual token count and language model size in Vision-Language Models (VLMs). The authors argue that for visual reasoning tasks, optimal inference performance is achieved by maximizing the LLM size within a given inference budget, even if it means drastically reducing the number of visual tokens. They propose a novel method, "QueCC" for compressing visual tokens at inference time, demonstrating significant improvements in accuracy and efficiency.

Quoting from the abstract: "for visual reasoning tasks, the inference-optimal behavior in VLMs is achieved by using the largest LLM that fits within the inference budget while minimizing visual token count — often to a single token."

And is shown with a logical experimental setup along with strong baselines. The authors also outline the potential shortcoming and also show that visual recognition and textual recognition from images have different goals and the more tokens are often needed in the second case to hold on to accuracy.

### Strengths
The paper tackles a highly relevant problem in the rapidly evolving field of VLMs. Balancing computational resources between visual processing and language modeling is critical for achieving optimal performance.

1) Well-designed Experiments: The authors conduct thorough experiments on various visual reasoning benchmarks, including GQA, CLEVR, and SNLI-VE. They systematically vary the number of visual tokens and LLM sizes, providing valuable insights into the relationship between these factors.

2) Strong Empirical Results: The proposed token compression method consistently outperforms baseline models, achieving state-of-the-art results on several benchmarks. The gains are particularly impressive at extremely low visual token counts, demonstrating the effectiveness of the approach in resource-constrained scenarios.

3) Clarity and Presentation: The paper is well-written and easy to follow. The authors clearly explain their motivation, methodology, and results, making the contributions accessible to a broad audience.

### Weaknesses
Have no major weaknesses from a technical standpoint. However, the related work can have a bit more coverage. 

While the paper provides a comprehensive overview of token compression techniques, it could benefit from a more in-depth discussion of related work on adaptive compute -- be it in Dynamic Sparsity, Elastic models (MatFormer, Flextron etc.,) and early exits.

### Questions
The paper is clear and achieves a strong threshold for the problem defined. While one can further improve the paper, I think the paper as is is strong enough to be accepted to ICLR. 

I am happy to champion the paper unless other reviewers find something glaring I am missing.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper discusses the challenge of high inference latency in VLMs. The authors explore the optimal balance between the number of visual tokens and LLM parameters for a fixed inference budget. They find that the optimal method in visual reasoning tasks is achieved using the largest possible LLM and minimizing visual tokens, even just one token. So they also propose a new approach using prompt-based compression for high-compression settings.

### Strengths
1. The paper discusses balancing the number of visual tokens and LLM parameters for a fixed inference budget, which is a novel perspective for studying effective methods to accelerate speed.
2. They also propose a method that makes the paper more comprehensive. 
3. The writing is clear and concise.

### Weaknesses
1. The method is quite similar to LLaMA-Vid, but the paper didn't compare it in its experiment and didn't show the differences between them.
2. Experiments are insufficient. The paper could also compared with VoCo-LLaMA[1] and LLaMA-Vid[2], which are also efficient in high-compression settings. In addition, they lack an ablation study.

[1] Ye, Xubing, et al. "VoCo-LLaMA: Towards Vision Compression with Large Language Models." arXiv preprint arXiv:2406.12275 (2024).

[2] Li, Yanwei, Chengyao Wang, and Jiaya Jia. "Llama-vid: An image is worth 2 tokens in large language models." European Conference on Computer Vision. Springer, Cham, 2025.

### Questions
1. Could you explain more about the discrepancy of the result between previous work and your work in **Log-Linear Relation between Error and Number of Visual Input Tokens**? Why limited downstream benchmarks lead to the discrepancy?
2. The method is quite similar to LLaMA-Vid, but the paper didn't compare it in its experiment and didn't show the differences between them. Have you tried to compare QueCC with LLaMA-Vid and VoCo-LLaMA?
3. Is the Convolutional Downsampling necessary? By adjusting the number of MLP output tokens, I can also adjust how many tokens to compress. There is a lack of ablation study proof for both User Query Information Injection and Convolutional Downsampling.
4. In User Query Information Injection, if this is the first time for visual token and text token entering this LLM, there is no way to get the last hidden state of the text, how can I perform such an operation?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors unveil the inference time scaling law, which characterizes the optimal tradeoff between the number of visual tokens and LLM parameters. The law reveals that the larger LLM matched fewer vision tokens under the fixed inference budget is the most optimal. Besides, the paper proposes a prompt-based VLM compression manner dubbed QueCC and conducts comprehensive experiments on it.

### Strengths
1. The authors do not limit the compression of VLM to vision, but dynamically explore the relationship between LLM and visual tokens. The motivation of the paper is sufficient.
2. The practical guideline provides a novel insight into VLM efficiency. The author carefully points out the limitations of the scaling law, which is not applicable to OCR tasks.
3. The authors conduct comprehensive experiments on various metrics and show an improvement. Furthermore, the discussion is in detail.

### Weaknesses
1. The employment of cross-attention in QueCC to compress information is common [1].
2. LLaMA-VID [2] and VoCo-LLaMA [3] have already done token compression in extreme regimes, which is impressive. The author should compare their performance with QueCC. It seems that QueCC is inferior to VoCo-LLaMA on benchmarks such as GQA and MME.
3. There is a lack of ablation experiments, especially the analysis of depth-wise 2D convolution and the injection of text-embedding.
4. The authors’ scaling law seems to have no direct relationship with the method they proposed.

[1] Jaegle, A., Gimeno, F., Brock, A., Vinyals, O., Zisserman, A. and Carreira, J., 2021, July. Perceiver: General perception with iterative attention. In *International conference on machine learning* (pp. 4651-4664). PMLR.

[2] Li, Yanwei, Chengyao Wang, and Jiaya Jia. "Llama-vid: An image is worth 2 tokens in large language models." In *European Conference on Computer Vision*, pp. 323-340. 

[3] Ye X, Gan Y, Huang X, Ge Y, Shan Y, Tang Y. VoCo-LLaMA: Towards Vision Compression with Large Language Models. arXiv preprint arXiv:2406.12275. 2024 Jun 18.

### Questions
1. The author uses the scaling law of [1] as an analogy. They replace the length of the text token in [1] with the length of the vision token and get the reversed conclusion. However, since section 3.3.2 points out the influence of text token length, it is more reasonable to include it in the discussion of Formula 2.
2. Although this paper exceeds the page limit, it will not affect my judging.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper explores optimizing VLMs to reduce inference latency by balancing model size and visual token count. The authors demonstrate that, for visual reasoning tasks, using the largest feasible LLM within a fixed inference budget, while drastically minimizing visual tokens (often to a single token) yields optimal performance.

### Strengths
This paper provides valuable insights showing that larger LLMs enhance visual reasoning performance more than reducing visual tokens. Also, the introduced compression algorithm QueCC demonstrates better performance on benchmarks with high compression, proving its effectiveness.

### Weaknesses
While the paper demonstrates that larger model sizes can be more effective than increasing token counts for visual reasoning tasks, this approach appears less effective for OCR-related tasks, as acknowledged by the authors. Given that many real-world applications require fine-grained visual understanding, the proposed compression method may not fully address these demands, as evidenced by its performance on the TextVQA benchmark in Table 1. Although the authors provide valuable insights, their claim that "only one visual token is needed" may be overly naive, given the trend in VLM research toward advancing fine-grained visual comprehension capabilities.

---

The authors somewhat addressed my concerns, although some efficiency perspective seems insufficient.

### Questions
Refer to Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper investigates the balance between the number of visual tokens and the size of LLM  in Vision Language Models to optimize inference costs. The authors discover that for visual reasoning tasks, using the largest feasible LLM with just one visual token is the most compute-efficient. They introduce a prompt-based token compression method for high compression ratios, which focuses on selecting relevant tokens based on user queries. Experiments show that their approach outperforms other compression techniques at extreme token reductions, highlighting the importance of developing algorithms for extreme token compression. The findings suggest that for visual reasoning, prioritizing a larger LLM over more visual tokens is crucial for maintaining performance within limited inference budgets.

### Strengths
1. The idea of establishing the scaling law among visual token numbers and model sizes is interesting. They explore optimizing inference costs in VLMs by using a single visual token with the largest possible LLM within a given budget.
2. The paper offers a thorough analysis of the trade-offs between LLM size and the number of visual tokens, covering various scenarios and use cases. This comprehensive approach provides a deeper understanding of VLM optimization.
3. The paper is well-organized and written in a clear and concise manner.

### Weaknesses
1. The main concern is the generalization of the scaling law. The paper focuses on visual reasoning tasks and may not fully explore other types of tasks where a single visual token might not be sufficient. For instance, tasks that require detailed image analysis might not benefit as much from such extreme token compression. 
2. While the scaling laws developed in the paper are insightful, they are based on a specific set of experiments and models. The findings are heavily dependent on the specific LLMs and VLMs used in the experiments. Different architectures might yield different optimal points in the trade-off between the number of visual tokens and LLM size, which could limit the applicability of the results. It's unclear how these laws would apply to other VLM architectures or if they would hold as new more complex models are developed in the future.
3. The proposed prompt-based token compression method adds complexity to the VLM pipeline. This could make the approach more difficult to implement and integrate into existing systems compared to simpler token reduction techniques.
4. The paper does not discuss the training costs associated with the proposed compression method. It's possible that training such a method to be effective, especially at high compression ratios, could require significant computational resources.

### Questions
1. Will one visual token be enough for other tasks (e.g, VLMs for detection)? The focus on minimizing the number of visual tokens to a single token might risk overfitting to the specific datasets used in the experiments. It's uncertain how well this extreme compression would generalize to unseen data or datasets with different characteristics. 
2. Will the proposed scaling law generalize to other VLM architectures? The authors only conducted experiments on one type of VLM.
3. What is the inference time and performance trade-off? Does the scaling law still hold? The computed FLOPs can be inaccurate.

### Soundness
3

### Presentation
3

### Contribution
2
