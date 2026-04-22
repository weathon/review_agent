# ReasonIE: better LLMs for Scientific Information Extraction with reinforcement learning and data augmentation

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 4

## Abstract
Large Language Models (LLMs) are good at reasoning in math and coding but underperform smaller, supervised models on structured Scientific Information Extraction (SciIE) tasks. This gap rises from a limited domain data and SciIE requires a combination of knowledge memorization and complex reasoning. To bridge this gap, we propose ReasonIE, a novel two-stage training framework. First, we use LLM-driven data augmentation to generate additional domain-specific training data, mitigating data limitation. We then introduce MimicSFT, a supervised fine-tuning method that uses structured reasoning templates to teach logical patterns without human-annotated chains-of-thought, followed by R\textsuperscript{2}GRPO, an RLVR algorithm optimized with a composite reward function that jointly scores factual relevance and logical consistency. Evaluated on SciIE benchmarks, our approach enables a general-purpose Qwen2.5-7B model to become competitive with specialized supervised baselines with less training data, demonstrating that RLVR and LLM-based data augmentation can successfully enhance both the knowledge retention and structured reasoning capacities of LLMs. The implementation is available at: \url{https://anonymous.4open.science/r/R2GRPO-48B5}

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Large language models have achieved good results in mathematics and programming tasks, but their performance on scientific information extraction is still weaker than some smaller models that are fine-tuned with instructions. 

To address this problem, the authors conducted SFT and RL training. Specifically, they used structured reasoning data without human annotations for instruction fine-tuning, and then applied the RLVR algorithm for reinforcement learning. This greatly improved the reasoning ability of the Qwen2.5-7B model.

### Strengths
+ The authors built a set of instruction fine-tuning data for scientific information extraction tasks, and this data does not require human annotation.

+ The authors used an SFT plus RLVR training paradigm to improve the capability of the Qwen2.5-7B model.

+ The method used by the authors is simple and easy to apply, but its novelty may be limited.

### Weaknesses
+ The training paradigm proposed in this paper can be viewed as using synthetic data for instruction fine-tuning, followed by further improvement with RLVR. This paradigm has been mentioned and used many times in previous works. Therefore, the main contribution of this paper is applying the algorithm to a new domain and proposing a new training approach.

+ The poor performance of large models on SciIE tasks may be because this task is relatively new, and the model may not have encountered it during previous training. As a result, the model cannot respond in the correct format, leading to weaker performance. Therefore, would directly applying reinforcement learning or using RFT also achieve good training results?

+ The evaluation uses only a few datasets. Could more evaluation tasks be added?

### Questions
+ Please further explain the contribution of this work. Many previous studies have also used the SFT plus RL training paradigm to improve the reasoning ability of models.

+ In the analysis of Figure 4 and Figure 5, the response length does not seem to help improve model accuracy. Does this mean that the CoT part is not important for the SciIE task?

### Soundness
3

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
5

### Summary
This paper introduces ReasonIE, a two-stage training framework that adapts general-purpose LLMs (specifically Qwen2.5-7B) to Scientific Information Extraction (SciIE) tasks, where they typically underperform due to domain scarcity and the need for structured reasoning. 

The framework includes:
1. LLM-based data augmentation to expand limited training data;
2. MimicSFT, a supervised fine-tuning method that uses structured reasoning templates (pseudo-CoT) instead of human-annotated chains-of-thought;
3. R2GRPO, a reinforcement learning algorithm with a composite reward function that jointly optimizes factual relevance and logical consistency.

### Strengths
S1: This paper identifies a real gap—LLMs underperform in structured SciIE tasks due to data scarcity and lack of reasoning training—and proposes a targeted solution.

S2: This paper combines data augmentation, pseudo-reasoning templates, and RL with composite rewards in a cohesive pipeline. While individual components are known, their integration for SciIE is novel.

S3: This paper outperforms strong supervised baselines (e.g., HGERE, PL-Marker) in RE tasks and achieves competitive performance in NER, even in OOD settings.

S4: This paper provides detailed ablations (e.g., w/ and w/o augmentation), Best@K analysis, and training dynamics, showing that both SFT and RL contribute meaningfully.

S5: Despite being a 7B model, it is 2× faster than smaller BERT-based models like HGERE, showing practical advantages. 

S6: This paper demonstrates that templated reasoning (MimicSFT) and reward-guided RL (R2GRPO) improve both accuracy and generalization, especially in OOD settings.

### Weaknesses
W1: While the integration is thoughtful, the individual components—data augmentation, pseudo-CoT, and RL with composite rewards—are not new. The paper’s algorithmic novelty is incremental.

W2: The method is heavily tailored to SciIE and may not generalize easily to other IE domains or tasks without significant re-engineering of templates and rules.

W3: MimicSFT relies on hand-crafted reasoning templates, which may not scale or transfer well to other domains. Automating or adapting these templates is not explored.

W4: While the paper shows improved F1, it does not deeply analyze what types of errors are fixed by reasoning or RL, or whether the model truly understands vs. mimics patterns.

W5: RL is only trained on 1K samples, which raises questions about scalability and stability of the RL component in larger or noisier settings.

W6: This paper does not compare with other LLM-based IE methods (e.g., GPT-ner, CodeKGC, or other RL-based IE frameworks), limiting the scope of evaluation.

### Questions
Q1: How does the method perform on non-scientific IE domains (e.g., news, biomedical, legal)? Can the templates and rules be easily adapted?

Q2: What happens if the data augmentation model generates noisy or incorrect labels? Is there a filtering or verification mechanism?

Q3: Does the model learn to reason or just mimic templates? How would it behave on inputs that deviate from the template structure?

Q4: Why limit RL training to 1K samples? Would scaling RL to the full dataset improve performance or cause instability?

Q5: How sensitive is the method to template design? Would automated template generation or learning be feasible?

Q6: Can the composite reward function be simplified or generalized for other IE tasks without domain-specific rules?

Q7: In addition, the anonymous code repository is empty when opened.

### Soundness
3

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
3

### Summary
The paper introduces ReasonIE, a two-stage training framework aimed at improving large language models (LLMs) for Scientific Information Extraction (SciIE) tasks. The approach combines two main components: (1) LLM-based data augmentation to generate additional domain-specific training examples. (2)$R^2$GRPO, a reinforcement learning method built on GRPO but enhanced with a composite reward function that considers factual relevance and logical rule consistency. The paper also proposes MimicSFT, a supervised fine-tuning step that mimics chain-of-thought reasoning without requiring human-labeled reasoning traces. Experiments are conducted on the SciER benchmark, showing that ReasonIE improves performance for a Qwen2.5-7B model, achieving competitive results with smaller supervised baselines while maintaining higher inference efficiency.

### Strengths
- Clear structure and motivation. The paper is clearly written and easy to follow. The figures (especially Figure 1 and 2) effectively show how the system integrates augmentation, SFT, and RL.
- Well-designed experimental pipeline. The authors evaluate multiple configurations (SFT, MimicSFT, GRPO, R2GRPO, ReasonIE) on both in-domain and OOD SciIE settings, with metrics like F1 and F1@K. This comprehensive setup strengthens empirical validity.
- Balanced integration of RL and data augmentation. The dual-stage approach of generating synthetic data and refining via RLVR is conceptually coherent and aligns with recent trends in post-training alignment.
- Clarity of analysis. The paper provides decent introspection (e.g., length vs. accuracy, temperature sensitivity) and includes practical insights into efficiency and scalability.

### Weaknesses
- Incremental novelty over GRPO. The main technical addition $R^2$GRPO is presented as a “composite reward” extension of GRPO, but the actual methodological change is relatively minor. The additional reward terms (relevance and rule-based reasoning) resemble standard task-specific reward shaping. There is no deeper theoretical or algorithmic novelty in how the RL framework itself is modified.
- Limited performance gain. While Table 1 shows some improvements (e.g., relation extraction +6–8 F1 over GRPO or MimicSFT), these gains are not dramatic given the increased training complexity. The improvements could be partly due to the data augmentation itself rather than the new RL method. The authors claim state-of-the-art performance, but the results only marginally exceed supervised baselines and are achieved using larger model capacity and synthetic data.
- Weak justification of “scientific reasoning.” Although the paper frequently uses the term “structured reasoning,” the reasoning component here mainly refers to formatted text templates (<reasoning> and <thinking> blocks). There is no convincing evidence that the model genuinely improves in logical reasoning ability rather than following structured prompts.
- Ambiguous contribution separation. The combined use of data augmentation, MimicSFT, and R2GRPO makes it difficult to isolate where the actual improvement comes from. The ablation study briefly touches on this but doesn’t provide quantitative analysis of contribution per component (e.g., RL vs. augmentation vs. structured SFT).
- Modest insight beyond metrics. The discussion remains descriptive rather than analytical. For example, Figure 2’s F1@K analysis is interesting but does not reveal deeper understanding of how RL or reasoning decomposition changes the model’s behavior.

### Questions
- How much of the observed gain (e.g., +7 F1) is due to R2GRPO versus MimicSFT or data augmentation? Can you provide a clearer ablation that isolates these effects?
- The “rule-based” reward seems hand-crafted. How generalizable is it to other IE schemas or domains?
- In Table 1, the improvement on NER is smaller than for RE. Why does the proposed approach help relation extraction more strongly?
- How does the model behave when scaling to larger backbones (e.g., 14B, 32B)? Do gains persist or diminish?
- Given the additional complexity of reinforcement learning, do the modest gains justify the added compute cost?

### Soundness
2

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
3

### Summary
This paper addresses the performance gap between LLMs and smaller supervised models on structured Scientific Information Extraction (SciIE) tasks. The authors attribute this gap to limited domain data and the dual need for knowledge memorization and complex reasoning.
To solve this, they propose ReasonIE, a two-stage training framework. The framework consists of:
1. LLM-driven data augmentation to mitigate data scarcity.
2. MimicSFT, a SFT method that uses structured reasoning templates (Pseudo CoT) to teach logical patterns without human-annotated CoT data.
3. R²GRPO, a RLVR algorithm based on GRPO, which uses a composite reward function that jointly optimizes for factual relevance and logical consistency.
Experiments on the SciER benchmark show that ReasonIE, applied to a Qwen2.5-7B model, significantly closes the performance gap, becoming competitive with or even exceeding specialized supervised baselines, particularly in relation extraction.

### Strengths
1. Important Problem: The paper tackles a critical and practical problem: adapting general-purpose LLMs to specialized, high-precision domains like SciIE.
2. Novel Methodology: The ReasonIE framework is well-designed and innovative. MimicSFT is a pragmatic and effective solution for instilling structured reasoning without costly human-annotated CoT data. The R²GRPO composite reward function is comprehensive, moving beyond simple accuracy to include factual grounding and logical consistency, which are crucial for IE tasks. 
3. Insightful Analysis: The experimental analysis is a key strength. The "Best F1@K" and "Avg@K" analysis (Figure 2) provides valuable insights into the complementary roles of SFT and RLVR. The results compellingly suggest that SFT (especially MimicSFT) is effective at expanding the model's knowledge boundaries (high Best F1@K), while RLVR (R²GRPO) excels at optimizing the model to produce high-quality outputs reliably (high Best F1@1 and Avg@K). This contributes empirical evidence to the ongoing debate about whether RLVR can enhance a model's underlying capabilities.

### Weaknesses
1. A key innovation of this paper is the R²GRPO algorithm, specifically its novel composite reward function. The authors designed this multi-component reward combining F1 score, span precision, factual relevancy, and rule-based consistency to address the specific constraints of SciIE tasks, moving beyond traditional single-metric rewards3. However, a critical weakness is the lack of a necessary ablation study on this core contribution. The paper provides no experiments to isolate the individual contributions of these four reward components or analyze the sensitivity to their weights. Consequently, it is difficult to determine whether the performance gains stem from the novel relevancy and rule rewards or simply from a more tuned optimization of F1 and span metrics. This omission makes it challenging to assess the relative importance of each component and fully understand the true drivers of the model's success
2. Lack of Reward Function Detail. The paper's primary weakness is the vague description of the "Rule-pattern Reward". It is unclear how these "logical reasoning patterns" are defined, detected, and quantified into a reward signal. This detail is critical for understanding the mechanism and for reproducibility.
3. Lack of Data Augmentation Detail. The "rigorous validation" process for the generated data is not well-described. The authors should specify whether this validation was automated or manual, what criteria were used, and what the approximate acceptance/rejection rate of the generated data was.

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
3
