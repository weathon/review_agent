# When Reasoning Meets Its Laws

- Decision: Reject
- Scores: 6, 6, 2, 6

## Abstract
Despite the superior performance of Large Reasoning Models (LRMs), their reasoning behaviors are often counterintuitive, leading to suboptimal reasoning capabilities. To theoretically formalize the desired reasoning behaviors, this paper presents the Laws of Reasoning (LoRe), a unified framework that characterizes intrinsic reasoning patterns in LRMs. We first propose *compute law* with the hypothesis that the reasoning compute should scale linearly with question complexity. Beyond compute, we extend LoRe with a supplementary *accuracy law*. Since the question complexity is difficult to quantify in practice, we examine these hypotheses by two properties of the laws, *monotonicity* and *compositionality*. We therefore introduce LoRe-Bench, a benchmark that systematically measures these two tractable properties for large reasoning models. Evaluation shows that most reasoning models exhibit reasonable monotonicity but lack compositionality. In response, we develop an effective finetuning approach that enforces compute-law compositionality. Extensive empirical studies demonstrate that better compliance with compute laws yields consistently improved reasoning performance on multiple benchmarks, and uncovers synergistic effects across properties and laws.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces LORE, a unified theoretical and empirical framework for analyzing and improving reasoning behaviors in Large Reasoning Models (LRMs). The authors formalize two key laws, Compute Law and Accuracy Law. In addition, they propose two tractable empirical proxies, monotonicity and compositionality, and design LORE-BENCH, a benchmark to evaluate whether LRMs follow these properties. The authors then propose SFT-Compo, a simple fine-tuning method to enforce compositionality.

### Strengths
1. The idea of defining formal “laws” governing reasoning is original and provides a principled foundation for analyzing LRM behavior.
2. The use of tractable proxies (monotonicity and compositionality) effectively bridges theoretical hypotheses and empirical evaluation.
3. The paper is well presented.

### Weaknesses
1. LORE-MONO uses only 40 seed questions, which may not generalize across broader reasoning domains or multimodal contexts.
2. Results might not generalize to stronger closed-source reasoning models.
3. While quantitative metrics are thorough, qualitative case studies or failure analyses would make the behavioral claims more intuitive.
4. The authors did not prove why enforcing compositionality leads to stronger reasoning capabilities.
5. The operational definition of question independence (via disjoint concept sets) is heuristic and may not reflect true cognitive independence.
6. More LRMs could be evaluated.

### Questions
1. How robust is the compositionality enforcement to task domain shifts (e.g., from math to commonsense reasoning)?
2. Could the LORE framework be extended to multimodal reasoning tasks (vision-language or embodied reasoning)?

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
3

### Summary
This paper introduces the LoRe together with the LoRe-Bench benchmark, demonstrating that large reasoning models generally exhibit monotonicity but fall short in compositionality. To address this limitation, the authors present SFT-Compo, a fine-tuning approach designed to enforce compositional reasoning and thereby enhance overall reasoning performance.

### Strengths
- This paper proposes LoRe, which characterizes reasoning laws through monotonicity and compositionality, enriching the evaluation dimensions.

- This paper introduces the SFT-Compo method, which is simple yet effective, significantly improving models’ reasoning compositionality and overall performance.

### Weaknesses
1. The experiments in the paper are mainly conducted on closed-source small-scale models, and the current results may be insufficient to demonstrate their applicability to larger-scale or different types of models.

2. In SFT-Compo, the training triplets appear to be generated solely using DeepSeek-R1-Distill-Qwen-14B. Such reliance on a single teacher model may limit data diversity and weaken the robustness of the conclusions. Have the authors considered incorporating multiple teacher models or alternative data sources to enhance generalization?

3. In the General Reasoning Evaluation Results section, the paper only compares the base models, SFT, and SFT-Compo. While this shows that SFT-Compo improves over the original models, it lacks comparisons with other baseline methods (e.g., length-control approaches such as Thinkless or AdaptThink).

### Questions
see weaknesses

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
3

### Summary
This paper introduces the Laws of Reasoning (LORE), a theoretical framework designed to formalize the reasoning behavior of Large Reasoning Models. The proposed framework contains a compute law and an accuracy law. Specifically, compute should scale proportionally with question complexity and accuracy should decay exponentially with increased complexity.
To evaluate this, the authors create LORE-BENCH. Their evaluation finds that while most LRMs satisfy monotonicity, they fail significantly at compositionality
Finally, the paper proposes SFT-Compo, a fine-tuning method that enforces the compute compositionality law. Experiments show SFT-Compo successfully improves compositionality and boosts performance on general reasoning benchmarks as well.

### Strengths
a. This paper is well-written and easy to follow.

b. This paper proposes a theoretical framework focusing on the relationship between problem complexity with accuracy and computation. The authors find that current models exhibit reasonable monotonicity but lack compositionality.

### Weaknesses
a. The evaluation is restricted to models with less than 10 billion parameters. It is a critical question whether significantly larger models inherently follow the proposed compositionality law due to emergent capabilities, or if SFT-Compo remains necessary for them.

b. LORE-COMPO (two totally independent questions followed by an arbitrary combination) is rare in natural language. It contradicts with the example in figure 1. The compositional question is that square the answer of the first question. The two question are clearly related while the question in LORE-COMPO are two isolated problems.

c. LORE-COMPO tests compositionality as two totally independent questions. The model has not seen this type of question during training. And this type of compositionality with two independent question is rare in real world scenarios as they can not be written into a complex question similar to what is shown in Figure 1. This makes the scope of this paper somewhat limited.

### Questions
What are the hyperparameters used for evaluation with AIME. Such as temperature and the how many runs have you runned on the benchmark?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper aims to understand the reasoning behaviour of large reasoning models. First, the authors focus on the incompatibility between the compute used and the complexity of the question (defined as Turing machine operation steps). I.e. sometimes models use more compute than needed for simple questions, yet generating suboptimal results. This should be related to the 'over thinking' problem that has been studied by the field recently.
In addition, the authors also study an accuracy law. 

Since question complexity is hard to quantify, the authors seek to use two surrogates, monotonicity and compositionality, to examine the hypotheses (compute should be linearly related to question complexity). A benchmark is therefore proposed to quantify thee two tractable properties.

Finally, the authors show that most models are reasonable in terms of monotonicity, while lacking compositionality. To this end, the authors proposed a finetuning method to enforce compute-law compositionality.

### Strengths
This paper investigates a practically important problem: How the compute and accuracy of LLM vary with question complexity. In addition, the authors not only evaluate the existing methods, but also propose a novel finetuning method to advance the existing models.

The presentation is clear, and the figures and tables are well organized and visually appealing.

### Weaknesses
The complexity construction of the data is not very consistent with the definition of complexity in the method part. I would say that the Turing machine steps seem to be a more reasonanle way to define the complexity. While the number of steps used in the datasets requires more justification.

### Questions
1. The question complexity is defined based on the concept of Turing machine. During dataset curation, how to justify that the complexity concept used when selecting questions aligns with the complexity concept based on Turing machine steps? For example, the number of matrix operation is used in dataset construction, while this may not be linearly correlated with the Turing machine steps. Otherwise, if 

2. In dataset construction, does each matix operation uses same compute? Different matrix operation may require different compute.

3. In addition to matrix operation, does other 'steps' use same compute. If not, then a question requiring more steps may actually need less compute than a question requiring more steps.

### Soundness
3

### Presentation
3

### Contribution
3
