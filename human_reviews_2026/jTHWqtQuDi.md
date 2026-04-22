# Beyond Truthfulness: Evaluating Honesty in Large Language Models

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 4, 4

## Abstract
As large language models (LLMs) become more capable and agentic, the requirement for trust in their outputs grows significantly, yet at the same time concerns have been mounting that models may learn to lie in pursuit of their goals. To address these concerns, a body of work has emerged around the notion of "honesty" in LLMs, along with interventions aimed at mitigating deceptive behaviors. However, some benchmarks claiming to measure honesty in fact simply measure accuracy—the correctness of a model's beliefs—in disguise. Moreover, no benchmarks currently exist for directly measuring whether language models lie. In this work, we introduce a large-scale human-collected dataset for directly measuring lying, allowing us to disentangle accuracy from honesty. Across a diverse set of LLMs, we find that while larger models obtain higher accuracy on our benchmark, they do not become more honest. Surprisingly, most frontier LLMs obtain high scores on truthfulness benchmarks yet exhibit a substantial propensity to lie under pressure, resulting in low honesty scores on our benchmark. We find that simple methods, such as representation engineering interventions, can improve honesty. These results underscore the growing need for robust evaluations and effective interventions to ensure LLMs remain trustworthy.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper argues that a model's "accuracy" and "honesty" are not strongly related. It introduces the MASK benchmark, which assesses a model's honesty by comparing its responses under "Pressure Prompt" versus in "Belief Elicitation Prompt" situations.

### Strengths
1. Evaluating a model's honesty by checking the consistency between its responses to a "Pressure Prompt" and a "Belief Elicitation Prompt" is a reasonably sound approach, as it helps rule out false accusations of dishonesty caused by inaccurate answers.

2. The insight reported by the authors is highly meaningful: larger models possess more accurate knowledge yet exhibit lower honesty. This indicates that we cannot simply resolve the honesty issue by scaling up models or increasing data size; instead, we should treat this as a critical challenge in AI safety that warrants special attention.

### Weaknesses
1. Treating honesty as the model's internal belief and distinguishing it from accuracy and truthfulness is not a novel contribution of this paper; this distinction has already been discussed and organized in prior survey [1].
2. The authors do not explain why the core finding that "larger models exhibit lower honesty" occurs. Model scale alone cannot serve as a direct cause of dishonesty.
3. The authors appear to consider only one form of dishonesty (when the model’s output contradicts the ground truth), but overlook another important form: dishonesty through incomplete or selectively biased output. For example, a model might produce statements that are factually consistent with its true belief but deliberately omit negative or unfavorable aspects while highlighting only positive ones, thereby distorting the overall meaning.
4. The authors do not consider multilingual settings, despite the fact that honesty can vary significantly across different languages.

[1] A Survey on the Honesty of Large Language Models

### Questions
1. What is the underlying reason for the observation that larger models exhibit lower honesty?
2. If a model’s outputs appear factually correct and not overtly deceptive, but the omission of certain information leads to a significantly distorted or misleading overall meaning, how should such behavior be analyzed?
3. Is the evaluation framework adaptable to multilingual settings?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces MASK (Model Alignment between Statements and Knowledge), a benchmark designed to evaluate "honesty" in large language models by measuring lies of commission - instances where models knowingly make false statements under pressure. The authors distinguish between accuracy (factual correctness) and honesty (consistency between beliefs and statements), proposing a novel evaluation pipeline that elicits model beliefs and compares them against statements made under pressure. The study evaluates 30 LLMs and finds that while larger models are more accurate, they don't become more honest, with many frontier models showing substantial propensity to lie under pressure.

### Strengths
1. The paper addresses a critical and timely problem in AI safety: the common conflation of "honesty".

2. Easy to follow. 

3. The work introduces a novel large-scale dataset of over 1,000 public samples, and the pressure prompts are human-collected. A major strength is that the pressure prompts are human-collected and curated according to thoughtful design principles, such as avoiding unrealistic placeholders ("ABC Company") or clearly fictional settings, which makes the evaluation scenarios more compelling and realistic.

### Weaknesses
1. Potential overclaim. This paper claims that accuracy differs from honesty. However, [1] already proposed similar idea that "Second, honesty is specific to each model, as it requires identifying the model’s known and unknown knowledge, making both its evaluation and improvement challenging."

2. Concern about belief detection. This paper proposes to detect the beliefs of LLMs by consistency in responses. However, we can not rely on responses if we do not know if LLMs are honest or not. It is a circular reasoning trap. One possible way is using probing or the date of collected training corpus, as introduced in [1].

3. The  Low-Rank Representation Adaptation (LoRRA) are evaluated only on smaller Llama 2 7B and 13B models. Performances on larger models are required. 

[1] S. Li et al. A Survey on the Honesty of Large Language Models, TMLR, Mar 2025. arxiv 2409.18786

### Questions
1. How do you address the circular reasoning problem in belief detection? If models might be dishonest in their responses, how can we trust their responses to determine their beliefs?

2. Can you provide validation that pressure prompts create realistic incentives rather than just adversarial conditions that might not reflect real-world deception scenarios?

3. How might alternative belief elicitation methods such as probing techniques or training data analysis compare to the consistency-based approach used here?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces MASK, a large-scale, human-curated benchmark to measure honesty in LLMs by explicitly disentangling honesty  from accuracy. Evaluating 30 LLMs, the authors find that while larger models are more accurate, they are not more honest—and can readily produce lies of commission under pressure. The paper further reports initial honesty interventions (system prompts, representation-engineering) that improve honesty but do not close the gap.

### Strengths
1. Clear concept:  separates honesty from accuracy, addressing a common conflation in prior work;
2. Scale and coverage: ~1.5K carefully curated, realistic, human-written scenarios spanning multiple archetypes, plus 30 frontier models evaluated—strong empirical breadth;
3. Interesting findings: revealing result that scale boosts accuracy but not honesty, motivating research on safety interventions beyond pure capability scaling.

### Weaknesses
1. How “belief” is collected. Using consistent answers under neutral prompts is one choice, but please compare it with other ways to get a model’s belief and discuss other explanations (e.g., different knowledge indentification methods);
2. Prompt sensitivity. Results may change with small wording changes in the pressure or belief prompts. Add tests with paraphrases and different pressure strength to show the results are stable;
3. Missing data details: The paper defines six dishonesty types but doesn’t report how many samples each contains and peformance on different types;
4. Dataset accessibility: As a dataset/benchmark paper, it is recommended to upload a supplementary material or anonymous link for reviewers to check the dataset.

### Questions
See the Weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
2
