# F1-Reasoner: Synthesizing Verifiable Reasoning Data From Formal Math Statements

- Avg Score: 4.40
- Decision: Reject
- Scores: 6, 6, 4, 4, 2

## Abstract
Recent progress in reinforcement learning with verifiable rewards (RLVR) has substantially advanced the mathematical reasoning ability of large reasoning models (LRMs). However, existing datasets either rely heavily on manual annotation or are synthesized within artificial environments such as logic games. In this work, We propose a data synthesis framework that transforms formal mathematical statements into high-quality verifiable reasoning data. It first performs Statement Collection and Quality Control to obtain high-quality proven statements, then applies Problem Generation to convert them into verifiable math solving problems, and finally leverages RLVR with a verifier for Model Training. Using this framework, we synthesize 19k high-quality mathematical problems at levels 5–10 and train the F1-Reasoner series of models. Across six challenging benchmarks, F1-Reasoner consistently improves upon 3 different open-weight models across different sizes, outperforming models such as SynLogic and Absolute-Zero that are trained on verifiable data from other environments. Moreover, we mix our data with MATH to create F1-Reasoner-Mix, which further boosts performance; notably, F1-Reasoner-Mix-8B surpasses General-Reasoner-14B while using substantially less data. Further analysis shows that F1-Reasoner generalizes to informal theorem proving and exhibits richer thinking behaviors.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces a framework for synthesizing verifiable mathematical reasoning data from formal theorem proving systems. The framework is a pipeline consisting of 4 steps, where theorems are gathered, filtered, transformed into different problems, and used for training a model. The paper evaluates the approach on 6 benchmarks and compares with multiple alternative methods.

**Note:** *I am not very familiar with this field, so I do not know the existing state-of-the-art. This also means I cannot evaluate well how novel the approach is compared to existing methods, nor any claims about this.*

### Strengths
- Thorough experimental evaluation of multiple competitors on 6 benchmarks. Additional experiments investigate "informal theorem proving".
- The results show that the approach generally beats the competitors.

### Weaknesses
- For multiple components of the proposed method, it is unclear why they are 'useful', i.e. no real arguments are given as to what they 'bring' to the method (see Questions 3 and 6). Ablation studies would also help with this.
- It seems an unfair advantage that the approach trains on one of the benchmarks (MATH) while the other approaches seem to not do this?
- There are many typos and language mistakes to the point that it is sometimes hard to understand what is meant (see below)

**Typos and mistakes:**
- line 42: leverage[s]
- line 46: "has" -> "have"
- line 51: "which approach" -> ". This approach"
- line 74: "potentiality" -> "possibility"
- line 86: obtain [the] F1-Reasoner model
- line 99: generalize[s]
- line 100: a[n]
- line 106: proven by [a] formal theorem proving system
- line 162: "the high difficulty problem is" -> "high difficulty problems are"
- line 183: by [the] community
- line 200: question[s]
- line 200: answer[s]
- line 256: "the verified model" -> "a verified model"
- line 297: "()" (I don't know what is missing here)
- line 303: "use combining" -> "combine" or "use"
- line 451: missing space after ")"
- line 481: a[n]

### Questions
1. What do you mean with "inconsistent hypotheses" (Section 2.2)? I presume this means statements where the premise trivially evaluates to false? 
2. What does "by sorry" mean (Figure 2)?
3. Why do you employ the LLM-as-judge strategy in addition to the rejection using the theorem proving (line 160)? The text does not explain why the theorem proving component is insufficient.
4. You only keep statements with difficulty > 5 (line 172). Did you evaluate whether instead keeping these statements would improve the model's accuracy on statements with difficulty < 5?
5. How sensitive are results to the difficulty filtering threshold (> 5)?
6. What is "the provided scorer" (line 178)? Is this part of the Deita selection pipeline? Could you clarify this in the text?
7. Why is the decontamination step useful (line 182)? What does data contamination imply? This is not explained in the text.
8. Why are some competitors only used with some base models (Table 1)?
9. I do not understand this sentence (lines 373-374): "These approaches can generate correct but intermediate problems, their correctness is hard to guarantee as difficulty increases." Do you mean the generated problems of such approaches are too hard to properly evaluate?

### Soundness
2

### Presentation
2

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
This work proposes a data synthesis framework that converts formal math statements (in Lean) into high-quality, verifiable natural language reasoning data. The authors then use this synthesized data to train a model, which ultimately achieves superior performance compared to existing open-sourced models.

### Strengths
The overall quality and clarity of this work are good, presenting its motivation and methodology in detail.

A key distinction of this paper is its novel approach. While many works leverage natural language reasoning data for formal theorem proving, this work explores the alternative direction: synthesizing natural language data from formal data. This method might be very important for the future development of model reasoning.

### Weaknesses
The main problem concerns the motivation of this work. Many formal data sets are originally translated from natural language statements. This paper now suggests translating them back into natural language, a process that seems to offer minimal gain.

(Specifically, if the formal data already has a corresponding natural language version, what is the key difference or added value between the newly "translated-back" data and the original natural language data?)

### Questions
Can you apply your methodology (e.g., quality control) also on natural language data?

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
4

### Summary
The paper proposes a framework for synthesizing high-quality, verifiable informal reasoning data for training LLMs with formal language.  The paper contributes a 4-stage pipeline that addresses the scarcity of such data, which currently relies on limited manual annotation or artificial environments. In the experiments, the models trained on these synthesized data show strong performance. The authors also study the reasoning behaviour of the models trained with this pipeline.

### Strengths
1. The core concept of using a formal proving system as a source for generating informal reasoning problems is excellent.
2. The experimental results are strong.
3. The paper demonstrates the methods clearly.

### Weaknesses
1. The choice of provers for the "Statements Collection" phase is questionable. InternLM2.5-StepProver and STP-Prover are weak and outdated. More powerful open-source ones, such as Kimina-Prover, DeepSeek-Prover-v2, were available well before the ICLR 2026 submission deadline. Therefore, this paper is expected to be based on these new models, since the "Statements Collection" phase is the foundation of the proposed pipeline.
2. The methodology introduces biases to the selected data by using formal provers' success as the filter. The paper aims to improve general informal reasoning, but its training data is filtered through the narrow bottleneck of a specific prover's capabilities. This risks training the LLM on a "prover-friendly" subset of mathematics, excluding many correct and difficult problems that the chosen provers simply failed to solve. While the authors introduce methods like difficulty sampling, this only samples from an already-biased pool and does not mitigate this fundamental issue.
3. The generalizability of the "Problem Generation" step is questionable. While the paper provides one concrete example of math inequalities, it is not clear how this method would apply to other, more diverse types of mathematical problems, especially those with numbers or math terms as their answers, which is common in informal math. Actually, the example provided is really weak. The answer to the generated problem is whether correct or wrong, making it easy to guess.
4. The paper presents an interesting analysis in Section 3.3 (Figures 6 and 7), claiming F1-Reasoner exhibits "richer thinking behaviors" and more efficient reasoning. However, the paper doesn't provide an analysis connecting why its specific data generation method causes this outcome. What properties of the formal-to-informal data pipeline encourage these specific behaviors?

### Questions
Please answer the questions raised in the Weaknesses part.

### Soundness
2

### Presentation
3

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
This paper introduces F1-Reasoner, a framework for synthesizing verifiable reasoning data from formal mathematical statements to train large reasoning models (LRMs). The approach consists of four main components: (1) Statement Collection from formal theorem proving systems, (2) Quality Control through hypothesis rejection, difficulty assessment, and diversity sampling, (3) Problem Generation that converts formal statements into question-answer pairs, and (4) Model Training using reinforcement learning with verifiable rewards (RLVR). The authors synthesize 19k+ high-quality mathematical problems at levels 5-10 and demonstrate that F1-Reasoner consistently outperforms baselines across six challenging mathematical reasoning benchmarks, including models trained on synthetic data from other environments like SynLogic and Absolute-Zero.

### Strengths
1. The use of formal theorem proving systems to ensure correctness of synthetic reasoning data is innovative in this field.
2. The four-stage framework (Statement Collection, Quality Control, Problem Generation, Model Training) is well-designed and addresses multiple aspects of data quality.
3. Consistent improvements across multiple base models (Qwen2.5-7B, Qwen3-4B/8B) and six challenging benchmarks demonstrate the effectiveness of the approach.
4. The paper shows that F1-Reasoner generalizes to informal theorem proving and exhibits richer reasoning behaviors, providing additional validation.

### Weaknesses
1. Limited experimental scale: Experiments are conducted only on models up to 8B parameters, while some baselines use larger models (14B, 32B). This limits the conclusions about the approach's effectiveness at scale.

2. Unclear problem generation process: The conversion from formal statements to natural language problems could be better explained. The examples show generating "opposite" inequalities, but the general principle is not well-articulated.

3. Dataset size limitations: With only 19k problems, the dataset is relatively small compared to other approaches that use hundreds of thousands of examples.

4. Limited theoretical analysis: The paper lacks deeper theoretical understanding of why this approach works better than alternatives. Why do formal statements lead to better reasoning data?

### Questions
1. Can you provide more details on how the "underlying logic" of statements is used to generate questions? The current description is quite high-level.
2. How does the approach scale with larger base models? Have you conducted any preliminary experiments with models larger than 8B?
3. How does the performance compare when using equal amounts of data from different sources (e.g., 19k problems from your approach vs. 19k from MATH dataset)?
4. What is the distribution of mathematical domains in your final dataset? Are certain areas over/under-represented?
5.  Do you have any theoretical insights into why training on formal statement-derived problems leads to better generalization?

Computational cost: What are the computational costs of the statement collection and quality control pipeline compared to alternative data synthesis approaches?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes data augmentation by translating formal mathematical statements, e.g., written in lean. The formal statement, verified by the lean compiler, is transformed to a related non-proof problem with a verifiable answer. The dataset, augmented with natural language training data, is utilized for reinforcement learning. Experiments are conducted on Qwen2.5-7B, Qwen3-4B and Qwen3-8B. The proposed method improves by 0.19% over vanilla RL on natural language training data.

### Strengths
The paper seeks to address the import research question of synthetic data generation, which is increasingly crucial as models scale and data capacity becomes a bottleneck. The writing is concise while clearly conveys its message.

### Weaknesses
1. It is particularly hard to understand the motivation behind translating formal mathematical statements back to natural language. Formalized statements are much more scarce than natural language statements, a significant part of which are actually translated from natural language, such as Lean-Workbook. Augmenting data in the natural language domain is far more flexible than in the formal domain. In fact, formal math provers [1] augment formal statements by synthesizing natural language statements and translating them into the formal domain. 

2. The question synthesis stage is not verified to ensure consistency between formal and informal problems. The synthesized informal problem can be far easier than the original formal one. For Example 1 in section D.3, the translated question can be solved by simply plugging in a=0 into the inequality, then confirms that a sufficient small positive 'a' satisfies the inequality for any S < 250. While the original statement requires a proof for arbitrary a>0. For example 3, proof of an inequality is synthesized into deciding whether there is a condition such that the reversed inequality satisfies. This is an yes/no question which can be well guessed even if the solver fails to provide a correct proof.

3. Given that many lean statements are translated from natural language, it is in particular important to examine the improvement of the augmented dataset with respect to natural language datasets. Therefore, the experiment with simpleRL should have been most convincing. However, the result is only provided on Qwen2.5-7B but not for Qwen3-4B and Qwen3-8B. And the improvement on Qwen2.5-7B at 0.19% is not significant enough.

[1] Lin, Y., Tang, S., Lyu, B., Yang, Z., Chung, J. H., Zhao, H., ... & Jin, C. (2025). Goedel-prover-v2: Scaling formal theorem proving with scaffolded data synthesis and self-correction. arXiv preprint arXiv:2508.03613.

### Questions
Please refer to weaknesses.

### Soundness
1

### Presentation
4

### Contribution
1
