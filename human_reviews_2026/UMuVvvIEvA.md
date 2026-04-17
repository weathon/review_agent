# Fast Intent Classification for LLM Routing via Statistical Analysis of Representations

- Decision: Withdrawn (Treated as Reject)
- Scores: 8, 4, 2, 2

## Abstract
Intent classification in Large Language Models (LLMs) involves categorizing user prompts into predefined classes. For instance, given a user prompt, the system must determine whether it primarily concerns mathematics, coding, or general text processing. Such classification enables routing prompts to specialized models optimized for specific domains, improving both accuracy and computational efficiency.
In this work, we conduct a systematic study comparing training-free vs training-based approaches for intent classification. For this purpose, we introduce two lightweight, training-free methods based on statistical analysis of internal model representations and compare them against MLP classifiers and linear probes. The training-free methods use key statistical metrics from hidden features, enabling intent inference during the initial forward pass with minimal overhead. Our comprehensive empirical evaluation reveals that 1) both training-free and training-based methods saturate easy benchmarks (mathematics vs. coding vs. natural language), 2) Training-based classifiers have an advantage on harder classification tasks (e.g. Java vs Python), and 3) Training-free methods are more robust to out-of-distribution and ambiguous prompts.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces two novel training-free methods, VecStat and NormStat, for intent classification in LLMs. The main idea is to leverage statistical analysis of internal model representations during the LLM's prefill phase to categorize user prompts into predefined classes (e.g., mathematics, coding, general text). This classification enables efficient routing of prompts to specialized LLMs, aiming to improve both accuracy and computational efficiency in production systems. 

The authors systematically compare their training-free approaches against traditional training-based methods, such as MLP classifiers, across various LLMs (1B to 32B parameters) and diverse datasets, encompassing both coarse-grained (Level-1) and fine-grained (Level-2) intent classification tasks. Their empirical evaluation highlights scenarios where training-free methods are most effective, particularly NormStat for coarse-grained classification and VecStat for fine-grained tasks. 

A key benefit emphasized is the inherent uncertainty quantification provided by their statistical methods, which often outperforms the overconfident predictions of training-based models on ambiguous prompts. The paper also provides theoretical analyses to explain when each method is preferable, considering factors like feature directionality and magnitude, and discusses computational, memory, and calibration costs.

### Strengths
This approach of training-free methods addresses key challenges in LLM routing, such as the need for rapid updates when classes change and the avoidance of extensive retraining, which can be computationally expensive and time-consuming. The methods operate during the prefill phase with negligible extra cost. This is crucial for deployment in production LLM systems where inference latency is a major concern. Another notable strength is the ability of NormStat and VecStat to provide well-calibrated class probabilities and uncertainty estimates, even on mixed-intent prompts. This is a considerable advantage over many training-based methods that often require post-hoc calibration to mitigate overconfidence. The visual evidence in Figure 2 clearly demonstrates this benefit. The paper presents an extensive empirical analysis across seven LLMs (from 1B to 32B parameters) and multiple benchmark datasets for both coarse-grained and fine-grained classification. This thorough evaluation provides valuable insights into the performance characteristics of each method under different conditions.

### Weaknesses
While the methods are training-free for classification, their effectiveness inherently relies on the quality and discriminative power of the LLM's internal representations. If these representations are not well-separated for different intents, the statistical methods might struggle, as suggested by the lower performance of NormStat on certain fine-grained tasks (e.g., mathematical subfield classification). For mathematical subfield classification (a Level-2 task), both NormStat and VecStat fall short compared to training-based approaches (Table 2, page 7 and Table 8, page 24). This suggests limitations for tasks requiring more complex, non-linear discrimination that supervised training can capture. While the methods are "training-free" for the classifier, they still require "calibration data" to compute per-class baselines. It would be beneficial to explicitly clarify the differences in data requirements and costs (e.g., annotation effort, quantity) between "calibration data" for their methods and "training data" for traditional classifiers to fully convey the practical advantages.

### Questions
1. How do VecStat and NormStat adapt to entirely new, unseen intent classes? Do the authors think of a hierarchical approach where coarse-grained (Level-1) intents are identified by statistical methods, and then fine-grained (Level-2) or novel intents are handled by a fallback mechanism or a different classification strategy?
2.  The theoretical analysis for VecStat assumes a diagonal covariance matrix. How might the performance or theoretical guarantees of VecStat change if the covariance structure is more complex and non-diagonal, reflecting richer correlations between features?
3. How robust are these statistical methods to prompts designed to intentionally mislead the intent classifier? Given the training-free nature, are there specific vulnerabilities to adversarial attacks, and have the authors considered any countermeasures?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper come up with two training-free methods based on statistical analysis of internal model representations to classify the prompt and route them to specialized models. The two methods NormStat and VecStat works well in practice while requiring negligible additional cost with theoretical guarantee.

### Strengths
- This paper is well-written.
- The experiment design is very fine-grained, covering many cases like ambiguous prompts, etc.

### Weaknesses
The motivation of this work is not very clear to me. Why is MLP not sufficient? In table 1, it mentioned that MLP needs extra calibration — I am wondering comparing to LLM pretraining, is this a bottleneck? Why do we need training-free methods in this scenario? Under what kind of situation, this method could be very useful? I think much more clear background needs to be provided in both introduction and related work. Also, MLP’s performance seems pretty nice in the experiments, which made me more confused. I assume the training samples for training a MLP is pretty easy to get, and training won’t take a long time, too.

### Questions
see weakness

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces VecStat and NormStat, two "training-free" methods for classifying user intent during the prefill phase of LLM inference. These methods rely on collecting simple statistics from internal hidden layer(s) and comparing them to pre-calculated class baselines.

The authors provide a theoretical analysis to justify these statistics. Extensive empirical evaluations were conducted across multiple LLMs and two different levels of task granularity. Results were compared with lightweight supervised probes (MLPs).

### Strengths
Originality: this work offers a fresh perspective on feature representation analysis with respective to the intent classification problem.

Quality: this work provided rigorous theoretical grounding. With some simplification, the authors provided sound proof to supports the intuition that NormStat is good for detecting large shifts in activation magnitude (e.g. between very different domains), while VecStat retains the necessary directional information for finer-grained distinctions.

Clarity: the work is well written and easy to follow. Extensive empirical evaluations were provided.

Significance: this work offers a practical, actionable guidance for designing intent classification system.

### Weaknesses
Novelty is not strong enough: the application to routing is useful, but the technical innovation is relatively limited for a top-tier conference, especially considering the discriminative performance compared with peer methods. Both VecStat and NormStat work okay on easy intent classifications, but failed significantly on hard ones like Level-2 Math (I was not convinced that Level-2 Code/Natural Language is "hard".). A simple linear probe might actually have a lower memory footprint while potentially offering better discriminative performance.

Over-claim of "training free": A core selling point is that these methods are "training-free." However, they require a "calibration" phase with 2k labeled data per class. Distinguishing this "calibration" from "training" a simple probe is largely semantic.

Lack of evidence for "FAST": The computational argument against standard lightweight probes is weak. The paper lacks latency comparisons to substantiate the "FAST" claim.

Lack of strong baselines: independent routers should also be included in comparison, especially considering the labeled samples used in "calibration".

### Questions
1. Could you provide latency measurements comparing VecStat/NormStat against the Avg-MLP baseline? I suspect the difference is negligible relative to the attention mechanisms.

2. How does a simple linear Probe (e.g., logistic regression on z_avg) compare to the baselines? You used 2-layer MLPs, but a linear probe may be sufficient for these distinct representations and would have a lower memory footprint.

3. Could you elaborate more about the fundamental difference between calculating statistics over labeled samples vs. "training" a simple classifier?

4. Could you add one more baseline: a router-based solution? Is a simple independent router enough to solve the problem?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper studies LLM routing. Instead of using a trained probe, the paper proposes VecStat and NormStat based on feature statistics for intent classification, reducing the latency.
The paper provides a theoretical analysis that VecStat performs better when prompt types differ in feature directions, while NormStat is better when they differ in overall magnitude.
The paper further evaluates the proposed methods against training-based methods on LLMs, showing superior uncertainty quantification for mixed-intent prompts compared to training-based approaches, while achieving competitive accuracy.

### Strengths
(1) The paper proposes a more efficient method than the probe, which uses statistics to perform intent classification.

(2) The paper develops a theory to distinguish the two proposed methods, i.e., under which conditions, which method will have better performance.

(3) It is easier for the proposed method in this paper to obtain a reliable uncertainty quantification.

### Weaknesses
(1) In Section 2, the method section, I assume the paper should introduce the proposed method. After reading, though there are a bunch of analyses, I did not find any formulation on how the classification is performed. It's recommended to have Figure 1 to deliver the method.

(2) The paper uses baselines Avg-MLP and Tail-MLP. I'm not sure whether they are good baselines or not. Is there any existing work on this? Are the two methods good surrogates for all baselines?

(3) The accuracy in Tables 2 and 3 is most saturated, which makes the results less convincing.

### Questions
(1) Following Strength (2), do we know when training-based methods are better than statistical method?

### Soundness
2

### Presentation
3

### Contribution
2
