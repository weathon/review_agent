# Mitigating Memorization in Language Models

- Avg Score: 7.33
- Decision: Accept (Spotlight)
- Scores: 8, 6, 8

## Abstract
Language models (LMs) can “memorize” information, i.e., encode training data in their weights in such a way that inference-time queries can lead to verbatim regurgitation of that data. This ability to extract training data can be problematic, for example, when data are private or sensitive. In this work, we investigate methods to mitigate memorization: three regularizer-based, three fine-tuning-based, and eleven machine unlearning-based methods, with five of the latter being new methods that we introduce. We also introduce TinyMem, a suite of small, computationally-efficient LMs for the rapid development and evaluation of memorization-mitigation methods. We demonstrate that the mitigation methods that we develop using TinyMem can successfully be applied to production-grade LMs, and we determine via experiment that: regularizer-based mitigation methods are slow and ineffective at curbing memorization; fine-tuning-based methods
are effective at curbing memorization, but overly expensive, especially for retaining higher accuracies; and unlearning-based methods are faster and more effective, allowing for the precise localization and removal of memorized information from LM weights prior to inference. We show, in particular, that our proposed unlearning method BalancedSubnet outperforms other mitigation methods at removing
memorized information while preserving performance on target tasks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper focuses on addressing the challenge of language models (LMs) memorizing sensitive or private data during training, leading to potential issues such as data leakage. The primary contributions of the paper are:

1. TinyMem: The introduction of TinyMem, a suite of small, computationally efficient GPT2-style models that enable rapid prototyping and evaluation of memorization mitigation strategies.

2. Comprehensive Evaluation: An empirical evaluation of three main strategies for mitigating memorization: regularizer-based methods, fine-tuning-based methods, and machine unlearning-based methods, with five new unlearning-based approaches being proposed.

3. BalancedSubnet: Among the newly introduced methods, the BalancedSubnet unlearning method is highlighted for its superior performance in reducing memorization while preserving model performance across different datasets.

4. Applicability to Larger Models: The paper demonstrates that mitigation techniques developed using TinyMem can be successfully applied to production-grade models, such as the Pythia series, validating their effectiveness in real-world settings.

The paper provides a detailed analysis of these strategies, focusing on the trade-offs between mitigating memorization and maintaining model accuracy

### Strengths
1. This paper first introduces TinyMem, a new suite of lightweight models specifically designed for rapid testing of memorization mitigation strategies. Additionally, the proposal of five new unlearning-based methods, including the innovative BalancedSubnet approach, adds contribution to the field. This work also highlights a novel way of applying unlearning-based techniques to real-world, production-grade models, bridging the gap between research and application.

2. Extensive experiments involve 17 memorization mitigation strategies, including both existing and newly proposed methods. The experiments are carefully designed to compare regularizers, fine-tuning methods, and unlearning-based strategies under diverse training conditions. The comprehensive analysis across different model sizes, data types, and training durations demonstrates the robustness of the findings. The use of Pythia models to validate the transferability of the methods adds credibility to the results.

3. The paper is well-structured and clear, with a logical flow from the problem of memorization in LMs to the proposed solutions and experimental results. The introduction of key concepts such as TinyMem and BalancedSubnet is explained effectively, and the figures, while not in vector graphic format, are useful in illustrating results.

### Weaknesses
This paper seems to be missing some important unlearning baseline methods, e.g., Gradient Ascent + Descent and Gradient Ascent + KL divergence [1].

[1] Jin Yao, Eli Chien, Minxin Du, Xinyao Niu, Tianhao Wang, Zezhou Cheng, and Xiang Yue. 2024. Machine Unlearning of Pre-trained Large Language Models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8403–8419, Bangkok, Thailand. Association for Computational Linguistics.

### Questions
Can you describe the difference between the setting of mitigating memorization and knowledge editing? It seems like the gradient-based methods of these settings are similar.

### Soundness
4

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
This paper points out that developing and evaluating memorization mitigation strategies for open-sourced LLMs is a critical challenge due to the lack of memorized data or a big-time cost. To address these issues, the authors first propose a suite of GPT2-style models, namely TinyMem, to rapidly develop and evaluate memorization mitigation strategies. Based on TinyMem, this paper investigates three classes of existing methods, including training-time regularizer-based strategies, post-training fine-tuning-based strategies, and post-training unlearning-based strategies. Additionally, this paper also proposes five new unlearning-based methods. The experimental results show the effectiveness of these proposed methods.

### Strengths
1. This paper focuses on the critical problems, namely, the challenges of developing and evaluating memory elimination methods directly on large models.
2. This paper innovatively proposes a series of small GPT2-style models (TinyMem), which is used to conduct memorization injection and memorization mitigation experiments in a quick and computationally efficient way. 
3. This paper proposes five new unlearning-based methods, and among them, BalancedSubnet outperforms other methods at removing memorized information while preserving performance on target tasks.

### Weaknesses
1. I am afraid that the evaluation results of different methods on TinyMem cannot truly reflect the performance of these methods in the memorization elimination of LLMs. On the one hand, the results in Table 1 and Table 2 show that some unlearning-based methods do not perform consistently in different types of tasks and memorization elimination. So how to effectively compare the comprehensive performance of different methods? On the other hand, the types of memorization within LLMs that need to be mitigated in practical applications are more complex and diverse. The two types of memorization (Noise and Backdoor) may be too simple compared to the real harmful memorization. The evaluation based on these two memorizations may hardly represent the ability of different methods to eliminate different types of memory within LLMs.
2. The innovation of the five proposed methods and their differences from previous methods should be introduced in detail. Additionally, although the algorithms are listed in the appendix, they still lack detailed introductions, which are difficult to understand.
3. Lack of detailed analysis of the reasons for the good performance of BalancedSubnet.

### Questions
1. In Line 066, "the few existing models with known memorized data are large" lacks detailed examples and references.
2. In lines 143-144, why test whether the next n−k tokens are matched with the clean sequence? Shouldn't it still match with the noise sequence? Or only disturb the first k words of one sequence?
3. Why is there a big difference in the mitigation performance and time of the same method in Nosie and Backdoor? Is there a difference in the difficulty of eliminating different types of memorization? The time consumption of the BalancedSubnet method in Table 1 is 0.00. How is it calculated here? Is there an error?
4. It is recommended to bold the best results in all tables.
5. The tenses in Lines 327-336 are not consistent.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This manuscripts presents a comparison of 17 methods for reducing memorization in large language models (LLMs) as well as a testbench 
 for measuring model memorization comprising of a set of small models called TinyMem. The authors find that the BalanceSubnet method is the most effective at reducing model memorization while preserving performance.

### Strengths
The authors present a comprehensive comparison of 17 methods for reducing memorization.

Applying the TinyMem test bench to larger models (Pythia) demonstrates the generalizability of the result

### Weaknesses
No major methodological weaknesses, however I find this study to be a straightforward comparison of several (existing and new) techniques without any particular novel insights. While this type of work is useful, the authors should extend the work by investigating why the best method works well or extracting practical insights from the results (beside stating that the method is most effective).

### Questions
What does the test accuracy in Table 1 represent?
The perplexity scores in Table 2&3 are quite high - can you describe how these are calculated and on what dataset?

### Soundness
3

### Presentation
2

### Contribution
2
