# Small Drafts, Big Verdict: Information-Intensive Visual Reasoning via Speculation

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Large Vision-Language Models (VLMs) have achieved remarkable progress in multimodal understanding, yet they struggle when reasoning over information-intensive images that densely interleave textual annotations with fine-grained graphical elements. 
The main challenges lie in precisely localizing critical cues in dense layouts and multi-hop reasoning to integrate dispersed evidence.
We propose Speculative Verdict (SV), a training-free framework inspired by speculative decoding that combines multiple lightweight draft experts with a large verdict model. 
In the draft stage, small VLMs act as draft experts to generate reasoning paths that provide diverse localization candidates; in the verdict stage, a strong VLM synthesizes these paths to produce the final answer, minimizing computational cost while recovering correct answers.
To further improve both efficiency and accuracy, SV introduces a consensus expert selection mechanism that forwards only high-agreement reasoning paths to the verdict.
Empirically, SV achieves consistent gains on challenging information-intensive and high-resolution visual question answering benchmarks, including InfographicVQA, ChartMuseum, ChartQAPro, and HR-Bench 4K. By synthesizing correct insights from partially accurate reasoning paths, SV achieves both error correction and cost-efficiency compared to large proprietary models or training pipelines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces SV, a training-free framework that combines multiple lightweight draft experts with a large verdict model. They use small VLMs to generate reasoning paths as candidates, and use strong VLMs to produce the final answer. They claim SV achieves consistent gains over many benchmarks.

### Strengths
1. Enhancing visual reasoning ability is a highly important and actively studied research problem.

2. The paper is easy to follow, the figures are clear, and the code is open-sourced.

3. The experiments are detailed, and the ablation study in Section 4.4 is very comprehensive.

### Weaknesses
1. My main concern with this work lies in the evaluation. If this issue can be properly addressed, I would be willing to raise my score. Currently, Table 1 compares the performance of several VLMs, and SV claims to outperform all of them. However, I would like to see a cost analysis. Although Appendix B provides some numerical results, it lacks a comparison of the computational cost between SV and the baseline methods. How much additional cost does SV introduce compared to the baselines? Under the same computational cost, would using only the verdict yield a similar effect? Likewise, under the same cost, would conducting debates among multiple draft models achieve comparable results?

2. The literature review is missing several important works. In the area of vision-language model reasoning, this paper is not limited to tool-related methods; therefore, it should also include more general VLM reasoning studies and cite some representative works in that domain.

### Questions
1. Address the issues mentioned in the weaknesses.

2. The formatting on page 17 could be improved.

### Soundness
3

### Presentation
3

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
This paper addresses the difficulty of VLMs in reasoning over information-intensive images that combine dense text and fine-grained graphics (e.g., infographics, charts), and introduces Speculative Verdict (SV), a training-free framework for information-intensive visual reasoning tasks. Inspired by speculative decoding, SV operates in two stages. The first is a draft stage where multiple lightweight VLMs generate diverse reasoning paths; and the second is a verdict stage where a large VLM synthesizes these paths to produce the final answer. The authors introduce a consensus expert selection mechanism that forwards only high-agreement reasoning paths to the verdict model. SV is evaluated on several benchmarks (InfographicVQA, ChartMuseum, ChartQAPro, and HR-Bench 4K) and demonstrates consistent improvements over strong baselines while maintaining cost efficiency.

### Strengths
- Originality: The paper presents a novel adaptation of speculative decoding for visual reasoning quality improvement rather than its original purpose of inference acceleration. 

- Quality: 
  1. The proposed approach is effective. 
  2. The experimental evaluation is comprehensive, covering multiple benchmarks and comparing against strong baselines. 
  3. The ablation studies provide insights into the importance of different components.

- Clarity: 
  1. The paper is well-structured and clearly written. 
  2. The figures effectively illustrate key concepts and results. (BTW, I like the animal icons in Figure 2.)
  3. The methodology is described in sufficient detail.

- Significance: The paper addresses a significant challenge in multimodal AI with a cost-effective solution that outperforms more expensive alternatives, which has practical implications for deploying such systems.

### Weaknesses
1. The tables lack clarity: The tables lack direct comparison between baseline and SV, such as directly providing the increment from GPT-4o (line 332) to SV+4o (line 341), and also the increment on Qwen2.5VL-72B, so that readers can easily compare the performance gain regarding different base models, instead of refering to other places in the paper (such as refering to line 377). Althought I like the figures and plots, the tables really need improving. 

2. Limited analysis of computational efficiency: This paper claimis that SV is more cost-efficient, but doesn't provide detailed metrics comparing computational costs (e.g., FLOPs, inference time, memory usage), and the budget cost.

3. Restricted Draft Model Pool: The evaluation is restricted to a fixed draft model pool, limiting understanding of how SV would perform with a more diverse set of draft models.

4. Insufficient Analysis of Failure Cases: The paper would benefit from a more detailed analysis of cases where SV fails to understand the approach's limitations.

5. Limited Comparison to Advanced Ensemble Methods: The paper compares SV to majority voting but doesn't compare to more advanced ensemble methods that could be applied to this problem.

6. Potential Overfitting to Specific Benchmarks: While evaluating on multiple benchmarks, they all focus on similar types of information-intensive visual reasoning, making generalizability unclear.

### Questions
1. Why SV improves more dramatically with GPT-4o (11.9/6.6/11.4/4) compared to Qwen 72B (2.5/7.5/2.3/2.5), especially for InfographicVQA and ChartQAPro? 

2. Could you provide more detailed metrics on the computational efficiency of SV compared to baselines? For example, inference time, FLOPs, or memory usage would help quantify the efficiency claims.

3. How sensitive is SV to the choice of draft models? Have you experimented with draft models of different sizes and architectures beyond the ones reported?

4. In what types of cases does SV fail? A more detailed analysis of failure modes would help understand the limitations.

5. How does SV compare to more advanced ensemble methods beyond majority voting? For example, methods that learn to weight or combine the outputs of multiple models.

6.How well does SV generalize to other types of visual reasoning tasks beyond the information-intensive benchmarks evaluated?

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
3

### Summary
This paper addresses Large Vision-Language Models’ struggles with information-intensive images—difficulty localizing critical cues in dense layouts and multi-hop reasoning. It proposes Speculative Verdict (SV), a training-free framework combining lightweight draft experts and a large verdict model. Small VLMs generate diverse reasoning paths in the draft stage; a strong VLM synthesizes these for final answers in the verdict stage. SV adds a consensus expert selection to forward only high-agreement paths. Experiments show SV gains on benchmarks like InfographicVQA, achieving error correction and cost-efficiency vs. large models.

### Strengths
1) This paper accurately pinpoints VLMs’ core flaws in information-intensive images—poor dense cue localization and error-prone multi-hop reasoning—and clarifies limitations of existing solutions, ensuring relevance.
2) The “Draft-Verdict” two-stage structure (lightweight experts for coverage + large VLM for synthesis) and consensus selection balance accuracy and efficiency, with clear alignment to solving target challenges.
3) Experiments on diverse benchmarks (InfographicVQA, HR-Bench 4K) and comparisons with various baselines, plus error correction data (47-53%), fully validate performance and cost-efficiency.

### Weaknesses
1) How about the comparison of the proposed method with specialized models? 
2) The inference speed is not presented in the experiments section. Does it add much computation cost to the baseline method thus slow down the inference1 speed, and if yes, could you give the speed?
3) On HR-Bench 4K, SV w/ GPT-4o Verdict performs worse than SV w/ Qwen2.5-VL-72B-Instruct Verdict, and even worse than several Open-source VLMs, please explain why?

### Questions
See Weaknesses.

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
5

### Summary
This paper presents a prompting framework for high-resolution image reasoning tasks, where small LVLMs generate draft reasoning trajectories and answers, and a large verdict model incorporates all reasoning and produces the final answers. The reasoning trajectories are selected based on the consensus score (i.e., the absolute difference between the model's own answer and the answer generated by the other model).  Experimental results show the framework achieves better results on high-resolution/dense-layout benchmarks such as InfoVQA and HRBench.

### Strengths
- The investigated problem of solving dense-layout image reasoning tasks using ensemble learning is of great practical value.
- The performance is promising, surpassing tool-based methods such as DeepEyes.

### Weaknesses
- Unclear Connection to Speculative Decoding
The paper's framing as "speculative decoding" is confusing. Traditional speculative decoding aims at inference acceleration, whereas this work operates more as an LLM-as-a-Judge paradigm where candidate answers are evaluated by a verdict model. The paper lacks discussion and comparison with existing judging frameworks (e.g., [1, 2]), which weakens its positioning within the literature.

- Limited Technical Contribution:
Viewing this work through the LLM-as-a-Judge lens, the technical novelty appears limited beyond modifying the aggregation process with consensus scores. Despite the ablation in Figure 7, several critical aspects remain underexplored:
  - What is the distribution of agreement/disagreement among models?
  - How does normalization affect the results? Could overconfident models skew the consensus?
  - Since answers are generated with reasoning trajectories, does estimating NLL on answers alone introduce inaccuracy due to off-policy estimation?

- Insufficient Motivation and Analysis:
The paper would benefit from deeper investigation to strengthen its claims:
  - A detailed analysis of reasoning trajectory patterns across different models
  - Exploration of whether using a smaller model as the verdict would yield similar performance gains
  - Token efficiency comparison against standard LLM-as-a-Judge frameworks, since both approaches can leverage prefilling



[1] LLaVA-Critic: Learning to Evaluate Multimodal Models, CVPR 2025
[2] VL-RewardBench: A Challenging Benchmark for Vision-Language Generative Reward Models, CVPR 2025

### Questions
## Questions

- The reasoning trajectory analysis is interesting in Figure 1 and Section 3.2. I am wondering if there is any quantitative analysis of reasoning type distribution between draft models and their influence, e.g., which models are more complementary?

- Why is the ablation study about the verdict scale only conducted on a subset of InfoVQA?

- Will some of the small reasoning models produce a long reasoning trajectory, and then the final prefill stage exceeds the verdict model's length limit? 
## Format
- Table 7 -> Figure 7.

### Soundness
2

### Presentation
3

### Contribution
2
