# Selective Prompt Anchoring for Code Generation

- Decision: Reject
- Scores: 6, 5, 6, 5

## Abstract
Recent advances in large language models (LLMs) have transformed software development by automatically generating code based on users' requests in natural language. Despite these advancements, challenges remain in generating buggy code and fully aligning with user intent. Our empirical study reveals LLMs tend to dilute their self-attentions on the initial prompt as more code tokens are generated. We hypothesize this self-attention dilution issue is one of the root causes of inaccuracies in LLM-generated code. To mitigate this issue, we propose **S**elective **P**rompt **A**nchoring (SPA) to amplify the influence of the selected parts in the initial prompt, which we refer to as "anchored text", during code generation. Specifically, SPA calculates the logit distribution difference with and without the anchored text. We prove this logit difference approximates the anchored text's contextual contribution to the output logits. SPA creates an augmented logit distribution by linearly combining the original logit distribution and the logit difference. We evaluate SPA with five LLMs on four benchmarks. Our results show that after tuning on a few dozen instances, SPA consistently improves Pass@1 on new tasks by up to 7.6% across all settings. Notably, with selective text anchoring, a small version of DeepSeek-Coder (6.7B) can achieve better performance than an original much larger version (33B). Our code is available at https://anonymous.4open.science/r/Selective-Prompt-Anchoring-74E7.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors identify that LLMs tend to dilute their self-attention on the initial prompt as more code tokens are generated, leading to inaccuracies in the generated code. To address this, they propose a novel approach called Selective Prompt Anchoring (SPA), which amplifies the influence of selected parts of the initial prompt, referred to as "anchored text," during the code generation process.

Key contributions :

**1 Identification of Attention Dilution:** The authors conduct an empirical study revealing that LLMs' attention to the initial prompt diminishes as more code is generated, which they term as "attention dilution."

**2 Proposal of Selective Prompt Anchoring (SPA):** SPA is introduced as a model-agnostic method to optimize LLMs' attention by amplifying the contextual contribution of selective prompt text towards each generated token.

**3** SPA calculates the logit distribution difference with and without the anchored text, approximating the contextual contribution of the anchored text to the output logits. It then creates an augmented logit distribution by linearly combining the original logit distribution and the logit difference.

### Strengths
**1 Rigorous Theoretical Foundation:** The authors provide a detailed theoretical proof for the concept of Selective Prompt Anchoring (SPA), demonstrating how it can approximate the contextual contribution of anchored text to the output logits. This theoretical underpinning adds depth to the methodology and supports the validity of the approach.

**2 Comprehensive Experimental Validation:** The paper backs the theoretical contributions with extensive experimental validation across multiple LLMs of varying sizes. The consistent performance improvements observed on different benchmarks and models showcase the robustness and generalizability of the SPA approach.

### Weaknesses
**1** The authors assume that self-generated tokens by the model are potentially incorrect, which leads to the idea that anchored text should only come from the initial prompt. However, it's possible that some self-generated tokens could also be considered as anchored text, which SPA does not account for currently.

**2** It's not clear how SPA compares to other state-of-the-art methods which focused on anchor-attention[1] improvements . More comprehensive benchmarking against a broader range of methods could strengthen the paper's claims.

**3** The selection of anchored text is crucial for the effectiveness of SPA, yet the paper does not provide a method for identifying the most informative tokens within the prompt. The authors conduct experiments in a general manner, but a more nuanced approach to selecting anchored text could potentially improve results.

**4** The experiments are conducted on HumanEval and MBPP datasets, which may not fully represent the complexity and diversity of real-world programming tasks. Testing SPA on more challenging datasets like OpenEval[2] or BigCodeBench[3] could provide a better understanding of its performance under more demanding conditions.

**5** The paper does not address the generalization of SPA across different programming languages. It's unclear if the same experimental conclusions would hold for languages other than those tested in the paper.

[1] [Anchor-based Large Language Models](https://aclanthology.org/2024.findings-acl.295) (Pang et al., ACL Findings 2024)

[2] [Chain-of-thought in neural code generation: From and for lightweight language models](https://ieeexplore.ieee.org/abstract/document/10634302/) (Yang et al., TSE 2024)

[3] https://bigcode-bench.github.io/

### Questions
**1** What are the advantages of SPA over Anchor-LLM with improved attention mechanism? (I looked at the open source code and it seems that SPA only supports greedy search at the moment)

**2** How well does SPA generalize to other programming languages besides python?

**3** Why there is no consideration on how to choose the optimal anchored-text?

### Soundness
3

### Presentation
3

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
This paper proposes that as the generated code tokens increase, LLMs often dilute their self-attention on the initial prompt, which is one of the fundamental reasons for inaccuracies in the code generation. To solve this issue, the paper introduces Selective Prompt Anchoring (SPA), a model-agnostic approach that amplifies the influence of selective prompts. The results demonstrate that, after tuning on a few dozen instances, SPA improves Pass@1 on new tasks by up to 7.6\%.

### Strengths
1. SPA, as a method that requires no training, can be applied to various models, demonstrating its broad applicability. 
2. The method has been validated across multiple code generation models, with experimental results showing consistent performance improvements among various models.

### Weaknesses
1. This paper proposes to optimize the results by enhancing attention to anchored text. Similar methods have already been explored in natural language contexts, and it is recommended to include them in the "Related Work" section. 
2. The authors should specify which specific information should be selected as anchored text in section 3.5, or provide a method for segment identification.

### Questions
1. It is recommended to include experimental comparisons with other LLM-based optimization approaches, rather than solely comparing with baselines. 
2. The authors mention identifying and anchoring the most informative tokens in longer prompts, thereby excluding trivial information. However, I didn't see any methods related to identifying fine-grained informative tokens  in the paper. 
3. The SPA method amplifies the influence of specific parts of the prompt. Will this approach change the model's behavior and compromise the initially correct output?

### Soundness
2

### Presentation
2

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
The paper proposes SPA, an approach to improve code generation in LLMs by addressing attention dilution, where models lose focus on the initial prompt during extended generation. The authors demonstrate the limitations of current LLMs in maintaining prompt relevance over generated sequences, potentially leading to inaccuracies. SPA amplifies the attention on selected prompt tokens, significantly improving performance across different LLMs and benchmarks.

### Strengths
+ a new approach to improve LLMs for code generation 
+ bring significant performance improvement

### Weaknesses
- generalizability is limited to code generation with Python
- missing prompt optimization-based baselines
- the randomness in the process of tuning anchoring strength should be explored
- the impact of prompt length is unknown

### Questions
1. In Table 1, some SPA-optimal models perform slightly behind the SPA-turned models, for example, pass@10 of DeepSeek-Coder on HumanEval+ and MBPP. Are there any reasons for this? 

2. The authors used 15% of sampled data from a benchmark to tune the anchoring strength. If picking the 15% data was a random process, how stable is SPA with different sample sets?

3. As shown in Figure 4, the examined LLMs share similar trends regarding the performance of different anchoring strength values on the same dataset, is it possible to share/reuse anchoring strength among different LLMs?    

4. Many existing studies optimize the prompts to make LLMs more focused on the key information of the provided prompts. Are there specific reasons that authors do not compare SPA to these prompt-optimization-based approaches?

5. Does SPA perform significantly differently on short prompts compared to long prompts?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors propose a new method to improve the code generation quality of LLMs by enhancing the attention mechanism. To show the effectiveness of their method on HumanEval(+) and MBPP(+) datasets, they use 1/5 tasks in the datasets to set the hyperparameters and the other 4/5 tasks to evaluate the performance.

### Strengths
- The overall structure is clear and easy to follow.
- The authors conduct experiments on several open Code LLMs like CodeGen, DeepSeek-Coder and Code Llama with different model sizes.
- The authors further provide ablation studies to validate the effectiveness of their methods from different perspectives.

### Weaknesses
The contributions of the paper could be very limited, and the efficacy of the research is still questionable without further experiments.
Several weaknesses include:
- The motivation of the paper is not strong enough. For example, the "attention dillusion" phenomenon is not a new phenomenon and has been discussed in the literature [1-3]. It appears that the lack of attention is quite common in the existing LLMs, not just in code genaration. The authors should provide more motivation as to why they only focus on the code generation task.
- The efficacy of SPA is not very convincing. Essentially, SPA requires tuning/searching the hyperparameters (e.g., anchoring strength) on each benchmark, which makes it impractical.
- Although there is an ablation study on cross-dataset evaluation, it is still not enough to validate the effectiveness of SPA, as HumanEval and MBPP are in the same algorithmic paradigm. Widely-used open-domain code benchmarks like BigCodeBench [4] should be used to further validate the effectiveness of SPA.
- Section 5.3 shows that the anchoring weight affects the performance of SPA significantly, and results in various performance across models and datasets. It is unclear whether the proposed method can generalize without hyperparameter tuning.
- The evaluated Code LLMs are quite outdated. The authors should include more recent models like StarCoder2 [5] and DeepSeek-Coder-V2 [6].
- As pointed in A.5, SPA additionally takes 2 to 3.5 times longer than regular inference, with extra memory usage. This limitation further limits the usability of SPA.
- The current evaluation only focuses on Python-only and function-level code generation, and it is unclear how the proposed method can generalize to other programming languages and code generation tasks.


[1] Shi, F., Chen, X., Misra, K., Scales, N., Dohan, D., Chi, E. H., ... & Zhou, D. (2023, July). Large language models can be easily distracted by irrelevant context. In International Conference on Machine Learning (pp. 31210-31227). PMLR.

[2] Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2024). Lost in the middle: How language models use long contexts. Transactions of the Association for Computational Linguistics, 12, 157-173.

[3] Zhang, Q., Singh, C., Liu, L., Liu, X., Yu, B., Gao, J., & Zhao, T. Tell Your Model Where to Attend: Post-hoc Attention Steering for LLMs. In The Twelfth International Conference on Learning Representations.

[4] Zhuo, T. Y., Vu, M. C., Chim, J., Hu, H., Yu, W., Widyasari, R., ... & Von Werra, L. (2024). Bigcodebench: Benchmarking code generation with diverse function calls and complex instructions. arXiv preprint arXiv:2406.15877.

[5] Lozhkov, A., Li, R., Allal, L. B., Cassano, F., Lamy-Poirier, J., Tazi, N., ... & de Vries, H. (2024). Starcoder 2 and the stack v2: The next generation. arXiv preprint arXiv:2402.19173.

[6] Zhu, Q., Guo, D., Shao, Z., Yang, D., Wang, P., Xu, R., ... & Liang, W. (2024). DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence. arXiv preprint arXiv:2406.11931.

### Questions
See the weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
2
