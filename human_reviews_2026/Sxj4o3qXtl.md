# ATGen: Adversarial Reinforcement Learning for Test Case Generation

- Decision: Accept (Poster)
- Scores: 4, 2, 6, 4

## Abstract
Large Language Models (LLMs) excel at code generation, yet their outputs often contain subtle bugs, for which effective test cases are a critical bottleneck. Existing test generation methods, whether based on prompting or supervised fine-tuning, rely on static datasets. This imposes a “fixed-difficulty ceiling”, fundamentally limiting their ability to uncover novel or more complex bugs beyond their training scope. To overcome this, we introduce ATGEN, a framework that trains a test case generator via adversarial reinforcement learning. ATGEN pits a test generator against an adversarial code generator that continuously crafts harder bugs to evade the current policy. This dynamic loop creates a curriculum of increasing difficulty that continuously challenges the current policy. The test generator is optimized via Reinforcement Learning (RL) to jointly maximize “Output Accuracy” and “Attack Success”, enabling it to learn a progressively stronger policy that breaks the fixed-difficulty ceiling of static training. Extensive experiments demonstrate that ATGEN significantly outperforms state-of-the-art baselines. We further validate its practical utility, showing it serves as both a more effective filter for Best-of-N inference and a higher-quality reward source for training code generation models. Our work establishes a new, dynamic paradigm for improving the reliability of LLM-generated code.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents ATGen, which formulates unit test generation in an adversarial RL setting. The test generator model is optimized for IO acc and attack success, two important quality measurements for unit tests, and another code generator model is used to generate adversarial code that passes the current test case but fails on the gold test. This forms a loop that enables self-improving test case generation, and results show that ATGen improves both IO acc and attack success by a large margin and which can transfer to better code generation.

### Strengths
1. The proposed method is sound, and the formulation of both test generation and adversarial setup is sensible.
2. Although it can be costly, the adversarial setup captures the incremental progress needed for the test generator to improve, and the authors explore different ways of sampling to reduce costs.
3. ATGen generates test cases that largely improve IO acc and attack success, which are two important factors to measure the test quality.

### Weaknesses
1. The author mentioned the cost during adversarial code sampling and proposed two modes, but there is no analysis showing the cost and performance comparison.
2. The main evaluations focus on IO acc and attack success, while the results and discussion on improving downstream code generation are less focused. Best-of-N using the generated test case is a good choice, but more code generator models and datasets are expected to see the effectiveness and generalizability.
3. Lacking analysis on how to combine the three types of reward (i.e., how to decide their weight) and what their ranges are.

Please see the questions for other minor weaknesses.

### Questions
1. In line 196, why does raising an execution error also get a positive attack reward? Doesn't it mean that the generated code is flawed and has nothing to do with the quality of the generated tests?
2. What are the ranges of all three rewards, and how do the authors decide the weights for each of them?
3. ATGen shows large improvements on both IO acc and attack success, but I wonder what the intersect performance would be, i.e., the union of IO acc and attack success, compared to the baselines. As UTGen and the authors mentioned, there is inherently a trade-off between these two.
4. What is the code generator model used in the "ATGEN as a Best-of-N Filter" analysis? More code generator models can be helpful to prove the effectiveness and generalizability.
5. The adversarial setup here only asks the model to generate adversarial code, but does not update for better code generation quality. Is it possible to optimize the test generator and code generator jointly?
6. Missing a period in line 454.
7. Can the generated test case improve frontier models like GPT-5 / Claude 4.5 Sonnet?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents ATGEN, a RL-based framework for test case generation, which trains a generator to optimize both output accuracy and attack success. By introducing an adversarial training paradigm, ATGEN enables the discovery of more complex and subtle code flaws compared to traditional static datasets. The authors demonstrate the utility of ATGEN in both code generation inference and trainin, showcasing its potential to improve test case generation in dynamic environments.

### Strengths
Clear and Readable Writing: The manuscript is well-structured, with clear and concise writing that facilitates easy comprehension.

Straightforward Motivation: The motivation behind the work is solid, with a direct and convincing argument for applying Reinforcement Learning (RL) to test case generation.

### Weaknesses
Lack of Novelty in Techniques: The application of RL to test case generation is conceptually straightforward and has been explored in previous works. The reward function design, while effective, does not introduce significant challenges or innovations.

Lack of Precision in Metrics: The metric "Attack Rate" is used in the evaluation but primarily reflects recall (but not real recall) rather than precision. It would be beneficial to include more comprehensive statistics, such as precision, to provide a clearer and more balanced evaluation of the model's performance. The datasets, metrics, and baselines used for more comprehensive evaluations can be referred to in TrickCatcher[1].

Limited Evaluation Against Related Work: Some relevant studies that have already explored RL in test case generation, e.g. ACECODER[2], are not included in the main results of the evaluation section. 

Minor Spelling and Grammar Errors: There are a few minor spelling and grammatical mistakes that need to be addressed for improved clarity and professionalism.

[1] Kaibo Liu, Zhenpeng Chen, Yiyang Liu, Jie M. Zhang, Mark Harman, Yudong Han, Yun Ma, Yihong Dong, Ge Li, and Gang Huang. (2025). LLM-Powered Test Case Generation for Detecting Bugs in Plausible Programs. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 430–440, Vienna, Austria. Association for Computational Linguistics.
[2] Zeng, H., Jiang, D., Wang, H., Nie, P., Chen, X., & Chen, W. (2025). Acecoder: Acing coder rl via automated test-case synthesis. arXiv preprint arXiv:2502.01718.

### Questions
Potential Circular Reasoning in "Adversarial Loop": The concept of the "adversarial loop," which is central to the paper’s contribution, raises concerns about potential circular reasoning. Specifically, using a more powerful large language model (LLM)—which may inherently have better code generation capabilities—to train the test generator may limit the generator’s potential. This creates a fixed difficulty ceiling, as the performance of the test generator will be ultimately bounded by the capabilities of the LLM used during training. A more detailed discussion on how this limitation is addressed or mitigated would be beneficial.

Clarification Needed on Dataset Usage for SFT Baselines: The manuscript does not clearly explain how the dataset is utilized in the context of supervised fine-tuning (SFT) baselines. Further clarification on this point is necessary to ensure the reproducibility and fairness of the results.

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
This paper presents ATGEN, a novel framework for automatic test case generation that uses adversarial reinforcement learning to improve the reliability of code generated by LLMs. Unlike existing test generation approaches that rely on static datasets, ATGEN introduces a dynamic adversarial loop between two agents: a test generator and an adversarial code generator. The code generator continuously produces harder buggy programs, while the test generator learns to craft tests that can detect these increasingly sophisticated bugs.

### Strengths
The paper is very well written.

The approach that uses adversarial reinforcement learning to solve the test generation problem is novel and very interesting.

The evaluation is well designed.

### Weaknesses
# Limitations of evaluation benchmarks

The benchmarks used for evaluation appear to be simplified and not fully representative of real-world programming challenges. There are two issues with these benchmarks: 1) they are designed for code generation, not test generation; 2) they have the data leakage issue where their test cases might have appeared in the training data of Qwen models. Consequently, the claimed improvements might not translate to realistic software engineering scenarios. It is recommended that the authors evaluate their approach on benchmarks that are designed for test generation purposefully, especially contamination-free benchmarks, such as UnLeakedTestBench (https://arxiv.org/pdf/2508.00408).

# Lack of discussion and comparison with Mutation Testing

The approach adopted in this paper uses buggy versions to trigger more effective test generation. The idea is similar to mutation testing, a widely studied test assessment method. In mutation testing, faults are injected purposely to check whether the tests are strong enough to detect the bugs. I recommend discussing such a connection in the paper. It is also interesting to explore the effectiveness of using mutants (with more subtle changes) instead of faulty versions generated by LLMs to conduct adversarial RL. 

# Coverage is not reported

The approach does not report code coverage in the evaluation, which is an important metric for measuring the effectiveness of test inputs, and is widely adopted in industry.

### Questions
What is the coverage of the generated tests?

What are the potential risks of data leakage in the evaluation process?

How does the model perform on test generation benchmarks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces ATGen, a reinforcement learning (RL) framework for automated test case generation that integrates an adversarial training loop. The test generator is trained to produce input-output pairs that are both correct (Output Accuracy) and bug-revealing (Attack Success). Meanwhile, a code generator adversarially creates progressively harder buggy programs, forming a dynamic curriculum. This design aims to overcome the “fixed-difficulty ceiling” of prior static datasets (e.g., UTGen). Empirically, ATGen outperforms strong baselines (UTGen, GPT-4 series, Qwen models) on APPS and Codeforces subsets. The authors also show downstream applications: using ATGen as (1) a Best-of-N filter to select higher-quality generated code, and (2) a reward signal for RL-based code generation training.

### Strengths
1. Novel and interesting method: The combination of adversarial reinforcement learning and test generation is creative and technically relevant, representing a step beyond static SFT-based approaches like UTGen.


2. Good results on benchmark datasets: The model achieves improvements in both IO Accuracy and Attack Rate, especially on APPS and Codeforces tasks, showing clear empirical gains.


3. Clear structure and thorough experiments: The paper is well-organized with detailed experiments, ablation studies, and multiple baselines


4. Practical downstream validation: Using ATGen-generated tests as a reward signal for RL-based code generation is an excellent demonstration of the method’s broader utility and potential to generalize beyond testing.

### Weaknesses
1. Limited domain and dataset generalization: The evaluation focuses only on a single type of domain and dataset, algorithmic coding problems from APPS and Codeforces. It would be helpful to show ATGen’s performance on more real-world software domains, such as repository-level test generation and API testing. Moreover, it would be interesting to see if training ATGen on one dataset (e.g., APPS) generalizes to another dataset (e.g., HumanEval).
2. Unclear technical details, as listed below:
    - The choice and training of the base model for UTGen comparison are not fully described. It is unclear whether UTGen and ATGen share the same base model and training dataset, making fairness difficult to assess.
    - The evaluation assumes that the “ground-truth test suite” (T_gold) is complete. Without quantitative measurement such as coverage metrics, it’s uncertain if this actually holds.
    - I could not find which buggy programs are considered when computing Attack Rate. Did the paper specify it somewhere?
3. The non-adversarial model (ATGen w/o Adver) is already significantly better than the baselines. The adversarial training only brings very marginal improvements.
4. Lack of qualitative examples: No examples of generated tests or adversarial programs are shown. Such examples would help illustrate what kinds of bugs ATGen detects or fails to detect, and how its generated tests differ qualitatively from UTGen’s.


5. Clarity and novelty overlap: While the adversarial RL setup is novel for test generation, parts of the approach resemble existing self-play or adversarial curriculum ideas. The paper could better situate itself relative to frameworks like self-play RL in AlphaZero-style.

### Questions
Please address my comments in the "Weaknesses" section

### Soundness
3

### Presentation
3

### Contribution
2
