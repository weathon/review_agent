# Less is Not Worse: Effective Reasoning Without Complete Reasoning Traces

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Large language models (LLMs) often produce lengthy reasoning traces with substantial token redundancy.
While reasoning processes are widely adopted to tune LLMs as a post-training regime, it has been underexplored whether LLMs truly learn from the complete trajectory, particularly in supervised fine-tuning (SFT). We argue that, for mid-size LLMs commonly trained with SFT for reasoning, using full reasoning trajectories may harm performance because their limited capacity increases susceptibility to redundant intermediate steps.
To investigate, we first analyze the redundancy in thinking trajectories through attention maps and controlled token-removal studies, both of which show that intermediate tokens contribute minimally to reasoning quality.
Our analyses suggest that the most redundant segments typically appear in the middle of reasoning traces, whereas the earlier and later segments are crucial for generating high-quality final outcomes.
We further posit that avoiding redundant intermediate information leads to exploiting the capability of LLMs to infer concise and coherent intermediate steps by utilizing the known start and end points.
Based on the insights, we propose MidCut, a method that removes redundant middle steps during both training and inference. We demonstrate the effectiveness of MidCut in two scenarios for LLM reasoning: (1) SFT trained on s1K and OpenThoughts datasets for reasoning; and (2) decoding strategy for a test-time application.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes MidCut, a method that removes redundant middle steps during both training and inference. The author demonstrate the effectiveness of MidCut in two scenarios for LLM reasoning, a new SFT training for reasoning; and a new decoding strategy for a test-time application.

### Strengths
The author finds a new index, or called rule, to reduce redundant reasoning patterns, and it can also improve the model’s ability to carry out intermediate reasoning when needed.

### Weaknesses
There are so many papers which aims at finding the overthinking or underthinking in LRM. So many index and so many rules. Actually, the algorithms behind this are so similar. There are far too many such papers, making it even impossible to compare this type of work with similar ones. While this is indeed a very serious issue with LRM, I believe the current paper is more like a homework assignment than a paper accepted by a conference. It falls below the typical bar of ICLR.

### Questions
There are far too many such papers, making it even impossible to compare this type of work with similar ones. While this is indeed a very serious issue with LRM, I believe the current paper is more like a homework assignment than a paper accepted by a conference. It falls below the typical bar of ICLR.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces MidCut, a method that removes intermediate reasoning steps during both training and inference, enabling large language models to use more compact reasoning traces without sacrificing accuracy.

The central insight is that redundant content predominantly appears in the middle portions of reasoning sequences, whereas the early and final segments are essential for maintaining reasoning quality and correctness.

Experimental results demonstrate that MidCut-SFT enhances accuracy while substantially reducing the number of training tokens, and that MidCut-Decoding—by trimming 25% to 75% of intermediate reasoning steps—achieves nearly identical performance compared to full reasoning traces.

### Strengths
1. The paper provides an insightful analysis of the attention weight patterns within reasoning traces, offering empirical evidence that supports the motivation and effectiveness of the proposed approach.
2. The proposed MidCut-SFT and MidCut-Decoding methods present complementary training- and inference-time solutions, respectively, enabling improvements in both model training and inference efficiency.

### Weaknesses
1. The experimental evaluation lacks diversity in model architectures. All experiments are conducted on the Qwen series models (Qwen2.5-32B, Qwen3-8B, Qwen3-4B), leaving the generalizability of the MidCut approach to other model families (e.g., LLaMA series) unverified.
2. The selected datasets (AIME, MATH, GPQA-D) are domain-specific and heavily focused on mathematical and scientific reasoning. The effectiveness of MidCut on more general-purpose or language-centric tasks remains unexplored.
3. The paper focuses primarily on accuracy metrics, without providing experimental evidence on efficiency-related aspects such as training time, inference throughput, or latency (e.g., first-token generation time). These measurements are essential to fully validate the claimed improvements in efficiency.

### Questions
1. Has the effectiveness of the MidCut approach been verified on other model architectures, such as the LLaMA series? It would be valuable to understand whether the proposed method generalizes beyond the Qwen family.
2. How does MidCut perform on broader and simpler tasks (e.g., TruthfulQA)? Would aggressively trimming intermediate reasoning traces lead to noticeable performance degradation when tasks require shorter or less complex reasoning?
3. What are the concrete advantages of MidCut-SFT in terms of actual training efficiency, such as total training time or resource consumption?
4. How does MidCut-Decoding affect inference efficiency, particularly in terms of throughput and first-token generation latency?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the redundancy and its impact in the Chain-of-Thought (CoT) reasoning used to train Large Language Models (LLMs). Through attention-based and token-removal analyses, it is found that the middle part of the reasoning trajectory typically contains the most redundant segments, while the beginning and end segments are crucial for generating high-quality final results. Based on this insight, the paper proposes a simple method called MidCut to synchronously prune redundant intermediate steps during both training and inference. The authors demonstrate the effectiveness of MidCut in two scenarios: MidCut-SFT (a data preprocessing technique for training) and MidCut-Decoding (an inference-time strategy). The results indicate that this method can improve both inference performance and training efficiency.

### Strengths
1. Simple and Effective Method: The work systematically reveals the existence of redundant information in reasoning trajectories and, consequently, proposes a very simple, low-cost, and easily reproducible method. It performs only "region-level" trimming without requiring additional scorers or RL controllers, bringing consistent benefits across multiple models/datasets, proving its practical effectiveness.
2. Sufficient Empirical Validation: The paper provides strong experimental support for the redundancy of intermediate steps through attention weight analysis and knockout experiments, demonstrating that LLMs pay less attention to intermediate steps and that removing them has a minimal impact on the final answer quality.
3. Comprehensive Experiments: The authors conduct a comprehensive evaluation of MidCut on various models (Qwen series), datasets (s1K-1.1 and OpenThoughts3), and challenging reasoning benchmarks (AIME24, GPQA-D, MATH). MidCut-SFT consistently outperforms the baseline (training with full trajectories) and other trimming strategies (e.g., random removal or LLM-based compression), strongly supporting the method's value. The inference stage also benefits, as MidCut-Decoding achieves almost the same accuracy as the full trajectory when trimming 25%–75% of the middle segment, implying direct computational savings on the deployment side.

### Weaknesses
1. The paper emphasizes that MidCut benefits medium-sized LLMs due to their limited capacity, but lacks direct comparative experiments with larger or smaller models to clearly delineate the scope of MidCut's applicability across different model scales. If experiments showed the advantage of MidCut disappearing on larger models, it would more strongly support "capacity limitation" as the key factor.
2. The best-performing variant, "step-level filtering," relies on preserving the first and last $$n$$ steps. Appendix B.3 sets $$n=100$$ and $$n=200$$ for the two datasets, respectively, but lacks explanation for this choice or related experimental analysis. How sensitive is model performance to variations in $$n$$? How should the optimal $$n$$ be selected for a new dataset? Including an analysis of this hyperparameter sensitivity would make the paper more convincing.
3.  A variant of MidCut-SFT is similarity filtering, aimed at removing repetitive reasoning patterns. However, Table 3 shows that similarity filtering performs worse than simple step-length or token trimming. Analysis should be provided for the failure of similarity filtering to explain why content-based semantic filtering is less effective than simple position-based trimming.

### Questions
1. Section 4.2 and the conclusion mention that MidCut-Decoding can reduce computational overhead and latency. However, the paper only shows its impact on accuracy but does not provide actual efficiency improvement data (e.g., percentage reduction in generated tokens during inference, specific latency reduction times, or throughput improvements). Supplementing experiments with this data is suggested to more fully demonstrate its "effectiveness."
2. The discovered "U"-shaped attention curve (i.e., the beginning and end are important, the middle is not) is very similar to the "Lost in the Middle" phenomenon in long-context processing. Is there a possible connection between the two phenomena? Including a discussion on this could increase the depth of the paper.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a simple yet effective method to improve training accuracy and inference efficiency by simply remove intermediate reasoning steps of offline collected SFT trajectories and online generated thinking steps. MidCut-SFT improves accuracy than the LLM-based compression method and random compression with fewer training tokens. The authors only report the accuracy preserving of MidCut-Decoding for inference. What are the the inference efficiency and other advantages of MidCut-Decoding?

Given the widely studied underthinking and overthinking mechanism of LRMs, removing intermediate reasoning during SFT data processing is straightforward. More clarification of the novelty is needed.

Four simple reasoning trajectory filtering methods are mentioned in the main part but not compared in the main results. In addition, it is not clear how "ours" is defined in Table 1.

### Strengths
Motivated by the relatively lower importance of intermediate thinking segments, the authors propose to remove the redundant middle steps during both training and inference to improve the training effectiveness and inference efficiency. The method is simple but effective. 

The proposed MidCut-SFT is more effective than the LLM-based compression and random compression baselines on MATH and science datasets.

### Weaknesses
1. LLM based compression only compress the center content of the whole reasoning trajectories by 50%. It would be more fair to instruct LLMs to directly process and compress the whole trajectories with the same compression ratio settings.

2. The settings of the "Base" baseline are not clear. Is it the open-sourced LLMs or the fine-tuned versions of them using the whole and long trajectories?

3. More evaluations on general datasets are recommended to analyze the effects of the proposed SFT data processing method.

4. Data pre-processing and cleaning is extremely critical for LLM training. More comparison of other existing data filtering methods is recommended to validate the effectiveness of the proposed filtering method.

### Questions
See the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
