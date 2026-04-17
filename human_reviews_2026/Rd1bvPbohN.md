# QuadCal: Calibration for In-Context Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 2

## Abstract
Large language models (LLMs) are increasingly being applied to high-stakes domains with high consequences for errors such as healthcare, drug discovery, law, and finance. However, they are often unstable and highly sensitive to prompt design, which can introduce contextual bias into their predictions. To mitigate this bias, various calibration methods have been developed to prevent overconfident and incorrect predictions. Existing techniques are either confidence-based, relying on heuristics to quantify bias, or likelihood-based, which is theoretically grounded but introduces unnecessary computational overhead. In this work, we introduce QuadCal, a novel supervised likelihood-based calibration method that is up to 40% faster and outperforms the existing likelihood-based approach. Specifically, QuadCal leverages Quadratic Discriminant Analysis (QDA), a supervised algorithm that directly models class-conditioned distributions, making it more efficient. We evaluated calibration methods on GPT-2 models and the more recent Llama and Gemma’s instruction-tuned (IT) models, which are harder to calibrate. Empirically, we show that on average over seven different natural language classification datasets, QuadCal outperforms existing methods on GPT-2 models and is competitive with earlier methods on IT models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In-context learning is widely applied to high-stakes domains with high consequences for errors. Existing calibration methods may  introduce contextual bias or introduce unnecessary computational overhead. The paper proposed a supervised likelihood-based calibration method that is up to 40% faster and outperforms the existing likelihood-based approach. Experiments across classification datasets valiate the efficiency of the proposed method.

### Strengths
1. The paper introduces a novel supervised likelihood-based calibration method which achieve low computational cost and higher macro-average accuracy.

2. The methodology is easy to understand and follow.

3. Comprehensive empirical analysis and ablation studies in the experiment section.

### Weaknesses
1. Tables 1 show that the results of QuadCal and the baselines are quite similar, especially for the advanced LLMs like Llama-3.2-1B or 3B. It would be helpful if the authors could further clarify the advantages of the proposed methods.

2. The paper should compare QuadCal with the advanced and recent baseline methods from other studies, such as LPC[3] and SupICL[4].  Such an analysis could offer valuable insights into the strengths and weaknesses of the proposed approaches.

3. The author should complete the full 9 pages. The figure and its illustration should be shown on the same page, such as Figure 1. Section 3.1 should be titled Motivation or Methodology rather than Background.

4. The paper validate the proposed method only on simple classification datasets. The authors should add more complex downstream tasks such as question-answering tasks or code generation tasks.

5. The authors should report the results of the Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Adaptive ECE, and Brier Score, which are widely used to evaluate the performance of calibration methods.

6. The absence of code makes it to reproduce the results claimed in the paper or verify the method's effectiveness on the tasks.

7. It would be useful to report the performance of in-context learning using a calibration method based on Linear Discriminant Analysis.

8. It would be useful to compare the computational cost with more baselines such as CC and BC.

9. Which dataset is used in Figure 1?

10. Missing the details of used dataset, such as dataset sizes, prompts.

11. Missing analysis on the selection strategy used in ICL: It would be useful if the authors also studied how different selection strategies (e.g., EPR [1] and DPP [2]).

12. Missing the analysis of different prompt formation.

13. Missing the analysis of larger model sizes (such as 7B, 14B, 30 B and 70B) and API (such as GPT-4o, Claude and Deepseek).

[1] Learning to retrieve prompts for in-context learning.

[2] Compositional exemplars for in-context learning

[3] Enhancing In-context Learning via Linear Probe Calibration

[4] Large Language Models are Miscalibrated In-Context Learners

### Questions
See Weaknesses

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a supervised likelihood-based calibration method for large language models in-context learning.  The whole frame work is built on ProCa but replacing Gaussian Mixture Model with Quadratic Discriminant Analysis.  Experiments across multiple models, e.g. GPT-2, Llama and Gemma and different classification tasks demonstrate its performance.  The results indicate that smaller or non-instruction-tuned models particularly benefit more from this post-hoc  likelihood-based calibration process.

### Strengths
1. The paper is well-written and easy to follow. 
2. The experiments are fairly designed, including multiple model families and various task domains, as well as different in-context learning shot settings. 
3. The main claim, such as faster runtime and improved average performance, is supported by experimental results and comparisons. 
4. The discussions section is good regarding when different calibration methods, either confidence-based or likelihood-based, are preferable.

### Weaknesses
1. The observation that larger size or instruction-tuned language models tend to be better calibrated is not new and has been reported in prior work, such as “_Language Models (Mostly) Know What They Know._” So, findings in this paper mainly confirm existing understanding rather than providing new insights.
2. The inclusion of statistical testing is good, but it seems unnecessary to me.  Moreover, the paper does not present detailed test statistics and full testing reports, even though the authors discussed and compared the results and significance levels.
3.  Expect for the reduction in computational cost, the improved accuracy of the proposed approach is not very significant for the majority of tasks, which limits its practical advantage. 
4. Although positioned as a theoretically grounded Bayesian approach, this work does not clearly articulate how or why its likelihood-based formulation provides deeper theoretical justification compared to confidence-based methods like BC, which achieves comparable performance with greater implementation efficiency.  
5. The construction of the estimate set is not well explained. And all of the methodological explanation relies only on a toy example (Figure 1), which makes the implementation details unclear.

### Questions
Please see the weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces QuadCal, a supervised likelihood-based calibration method for in-context learning (ICL) that addresses contextual bias in large language models. The key innovation is replacing Prototypical Calibration (ProCa)'s unsupervised GMM clustering + Hungarian algorithm approach with Quadratic Discriminant Analysis (QDA), which directly models class-conditioned distributions using ground-truth labels. The authors evaluate QuadCal against existing calibration methods (CC, BC, ProCa) on seven NLP classification datasets using GPT-2 models and instruction-tuned Llama/Gemma models across 0/1/4/8-shot settings.

### Strengths
1. While the core idea of substituting GMM with QDA is straightforward, the application to ICL calibration is sensible and previously unexplored.
2. The key insight, that supervised learning eliminates the need for expensive cluster-to-label mapping, is valid and practically useful.
3. The paper clearly motivates the problem (contextual bias in ICL, computational overhead in ProCa) and positions QuadCal as a solution.
4. For practitioners deploying LLMs in high-stakes domains, this work offers a useful, faster alternative for likelihood-based calibration.

### Weaknesses
1. In Table 1, the performance gain seems to be trivial for the majority of the models. In fact, BC outperforms QuadCal a few times. QuadCal significantly outperforms ProCa in only 26% of 168 settings, where 66% show no significant difference, and ProCa is significantly better in 8% of cases. For a paper whose main claim is improved performance, these statistics are weak and are insufficient to claim a meaningful contribution.
2. The datasets used are pretty saturated and could already have been covered by some models' pre-training and post-training.  It is generally a good practice to test on ICL tasks that are new, hard, and require the model to learn novel logic or strategies, since the value of ICL is to learn new tasks at inference time. For example, tasks like BBH, BB-extra-Hard, ZebraLogic, GPQA, MMLU-Pro, and other logical puzzles or algorithmic reasoning challenges should be the main evaluation focus. The current set of datasets does not support a generalized claim and undermines the impact of the contribution.
3. There are some potential missed baselines. There is no comparison with Linear Discriminant Analysis, which would be faster than QDA, and test whether class-specific covariances are necessary. Also, there are no comparisons with simpler supervised methods like logistic regression on log-probabilities or ensemble approaches.
4. The discussion notes that different methods work better for different tasks but provides no principled explanation:
- Why does QuadCal excel on AGNews and TREC but not RTE?
- What properties of these datasets drive the differences?
- No analysis of class balance, class separability, or output probability distributions
5. The core contribution, replacing GMM+Munkres with QDA, is primarily an engineering substitution rather than a fundamental methodological advance. While the paper claims QDA avoids ProCa's computational overhead, the theoretical justification is superficial. Why should QDA outperform GMM for this specific problem? Under what data distributions or class structures would QDA be preferred? The statement that ProCa's restriction to n clusters makes GMM "functionally similar" to QDA (lines 183-186) needs rigorous proof. GMM with n components can still capture multimodal within-class distributions, while QDA assumes unimodality.

### Questions
1. The observation that larger IT models benefit less from calibration is interesting but underdeveloped:
- Are larger models inherently better calibrated? Are there ECE comparisons on the model sizes?
- Is this specific to instruction-tuning or general to scale?

### Soundness
2

### Presentation
2

### Contribution
2
