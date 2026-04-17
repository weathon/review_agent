# DTR: Towards optimal token compression with data-driven token ranking for efficient visual-language model inference

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Token compression is crucial for vision-language models (VLMs) inference due to its tremendous computational complexity. Although substantial works with various model-driven methods have been done to mine importance rankings among tokens for compression~(e.g., rank according to attention scores or matrix ranks), they are all constrained by one-sided handcrafted information, thus being trapped in local optimum. To utilize comprehensive information for global optimum, we present a Data-driven Token Ranking (DTR) framework, which trains a plug-and-play token-ranking model with self-gathered token-ranking data for VLM token compression at runtime. Specifically, first, we propose a dataset construction method to efficiently gather importance rankings of tokens based on original VLM datasets. Then we present a training method to build a token-ranking model for predicting a ranked-list of token importance based on input vision and text tokens. Finally, the ranking model can be plugged in the model, then filter tokens with an user-defined token number at runtime for acceleration. Extensive experimental results across 8 mainstream benchmarks show that DTR achieves the state-of-the-art token compression performance compared with 8 challenging comparatives. Moreover, a comprehensive analysis shows that DTR as well as data-driven methods possess tremendous potential, which can comprehensively outperform the vanilla VLM with much fewer tokens.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper addresses the computational inefficiency of vision-language models caused by excessive visual tokens during inference. It proposes DTR, a data-driven token ranking framework that replaces handcrafted model-driven compression heuristics with a learned token importance predictor. The core contributions include: (1) a method to construct token-ranking datasets using greedy search over token subsets, (2) a token-ranking model (TRM) trained to predict token importance rankings, and (3) a plug-and-play integration scheme for runtime token filtering.

### Strengths
1. The method is widely evaluated on 8 diverse benchmarks against 8 strong baselines.

2. The proposed TRM is plug-and-play, requiring no architectural changes to VLMs.

3. The ablation study is abundant and comprehensive.

### Weaknesses
1. The paper is not well-written and hard to read, with several typos that I cannot understand. For example, the citation of the paper all comes with author(year) but does not mention the name of the paper, which is weird.  Also, line371 does not contain any information. What do you mean by adding that? I suggest that the author comprehensively revise the paper including but not limited to the typos noted above.

2. The results are not exciting. First, the results when retaining 64 tokens are substantially inferior to other methods, which challenges the utility of the method. Also, although improvements are achieved when the number of tokens becomes fewer, most comparison methods are uniform or random, which is not exciting. More methods are supposed to be incorporated to validate the effectiveness.

3. As you mention in line465, the overhead of DTR is a crucial question of the method, which may offset gains for small batches or simple images. Based on the results, we are unable to know whether the performance gain comes from additional computation or the effectiveness of the method. I recommend the author explain more about this, including but not limited to the actual latency, comparing under the same flops other than the same token and so on.

4. The paper claims the "global optimum token compression", but it requires training without analytical support beyond empirical results. I would like to see more theoretical proof of how your token compression is optimal but no discussion is shown about this in the paper, which significantly reduces the persuasiveness of the article. Also, the choice of greedy search over optimal combinatorial search lacks theoretical guarantees on ranking quality.

5. ​​Sparse analysis of multimodal interactions​​: The role of text tokens in guiding visual token ranking is underexplored.

### Questions
1. The greedy algorithm for forward passes are computational expensive, especially for large N. How do you deal with it and are there any quantitative results?

2. Is the TRM latency overhead amortizable across batches, and what are optimal batch sizes for real-world deployment?

3. Given the high computational consumption, what is the scalability of the proposed method on larger models, like 32B, different architectures, like Qwen?

4. The diversity of evaluated datasets is somewhat narrow and more results are expected to be conducted to validate this. For example, how does DTR perform on tasks requiring fine-grained spatial reasoning (e.g., object counting), long context understanding and could task-specific ranking models help?

5. DTR relies heavily on the quality of ranking. Are there failure modes where DTR's rankings degrade VLM performance (e.g., adversarial images), and how can robustness be improved?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposed Data-driven Token Ranking (DTR): a plug-and-play ranking model trained on automatically collected token-importance orders from standard VLM datasets. At runtime, DTR predicts a ranked list from the input vision–text tokens and filters to a user-specified budget for acceleration. Across 8 mainstream benchmarks, DTR delivers state-of-the-art compression, and analysis indicates substantial headroom—often matching or surpassing vanilla VLMs with far fewer tokens.

### Strengths
1. Using a novel two-stage algorithm,  offline and online; offline: the end-to-end loss for ranking the selected token lists to create a token ranking dataset and train a TRM for automatically selecting the top related tokens; online: inference with the plug-and-play TRM, combined with a user-defined number of tokens.

2. The upper bound of the method is surprisingly achieved SOTA in a really high pruning ratio of the vision tokens.

3. The paper is very well written and easy to read, with a clear logical flow.

### Weaknesses
1. Generalization yet to be verified: The paper lacks experiments on different models and numbers of parameters; they only conduct experiments on the LLaVA-7B; effectiveness on other architectures (e.g., LLaVA-OV[1], InstructBLIP[2], Qwen-VL[3]) and other numbers of parameters(e.g., LLaVA-13B) remains to be validated.

2. Baseline selection is not accurate: The comparison with existing methods is not entirely fair, as some baselines are not aligned in settings or optimization conditions. For example, the baselines are all training-free methods, which are substantially different from the training setting in this paper. Therefore, more methods should be compared, such as PDrop[4], M3[5], FastVLM[6], and so on.

3. Compare to other SOTA baselines: I found another SOTA baseline, QueCC[7], that also claims they select a very minimum vision token and still gain great accuracy.

[1] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. ArXiv, 2024a.

[2] Wenliang Dai and Junnan Li and Dongxu Li and Anthony Meng Huat Tiong and Junqi Zhao and Weisheng Wang and Boyang Li and Pascale Fung and Steven Hoi. InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning. ArXiv, 2023a.

[3] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023.

[4] Long Xing, Qidong Huang, Xiaoyi Dong, Jiajie Lu, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, Jiaqi Wang, Feng Wu, et al. Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction. CVPR, 2025.

[5] Cai, Mu and Yang, Jianwei and Gao, Jianfeng and Lee, Yong Jae. M3: Matryoshka Multimodal Models. ICLR, 2025.

[6] Pavan Kumar Anasosalu Vasu, Fartash Faghri, Chun-Liang Li, Cem Koc, Nate True, Albert Antony, Gokul Santhanam, James Gabriel, Peter Grasch, Oncel Tuzel, Hadi Pouransari. FastVLM: Efficient Vision Encoding for Vision Language Models. CVPR, 2025.

[7] Li, K. Y., Goyal, S., Semedo, J. D., & Kolter, J. Z. Inference Optimal VLMs Need Fewer Visual Tokens and More Parameters. ICLR, 2025.

### Questions
1. Upper bound–TRM gap at 32 tokens. The paper reports that the "upper bound” yields about a +29% relative improvement, whereas the learned TRM preserves about 94% of the baseline at 32 tokens. Could you diagnose the sources of this gap? Is it caused by an insufficient modeling or training of the TRM?

I will consider raising my score if all my concerns are solved.

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
5

### Summary
This paper studies token compression of vision-language models (VLMs) inference. Most existing works focus on model-driven idea to  mine importance rankings among tokens for compression, with one-sided handcrafted prior. Differently, this paper presents a Data-driven Token Ranking (DTR) framework, covering offline token-ranking construction, offline token-ranking model training, online model insertion and token filting. Experimental results are carried out across 8 mainstream benchmarks, to show the effectiveness of DTR.

### Strengths
[+] The manuscript is well written, with clear logics.

[+] The symbol definitions are clear, and the image visualization is complete. 

[+] Many experiments are conducted to analyze the effectiveness of each component.

### Weaknesses
[-] For the offline/online phase of DTR, there is a core assumption that offline data and online data are approximately distributed. However, in practical scenarios, such as on rare MLLM benchmarks, this assumption may not necessarily hold true. The above issues will result in limited generalization of this work, thereby reducing its impact on the community.

[-] Although in the deployment phase, this work and existing methods (training-based, training-free) are similar in terms of speed. However, the offline phase of this paper clearly requires more cost. The reviewer suggests conducting comprehensive evaluations for the overall process in terms of time and cost, and comparing it with existing methods, in order for the community to better understand the practicality.

[-] In Table 1, existing methods need to be divided into training-based and training-free. As this work requires training, such division makes it easier for readers to make fair comparisons. In addition, please provide a detailed analysis of differences between training-based token compression, so that readers can better understand the innovation.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
