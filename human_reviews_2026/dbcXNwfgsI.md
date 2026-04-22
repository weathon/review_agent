# ScaleDiff: Scaling Difficult Problems for Advanced Mathematical Reasoning

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 6, 2

## Abstract
Large Reasoning Models (LRMs) have shown impressive capabilities in complex problem-solving, often benefiting from training on difficult mathematical problems that stimulate intricate reasoning. 
Recent efforts have explored automated synthesis of mathematical problems by prompting proprietary models or large-scale open-source models from seed data or inherent mathematical concepts.
However, scaling up these methods remains challenging due to their high computational/API cost, complexity of prompting, and limited difficulty level of the generated problems.
To overcome these limitations, we propose ScaleDiff, a simple yet effective pipeline designed to scale the creation of difficult problems.
We efficiently identify difficult problems from existing datasets with only a single forward pass using an adaptive thinking model, which can perceive problem difficulty and automatically switch between "Thinking" and "NoThinking" modes.
We then train a specialized difficult problem generator (DiffGen-8B) on this filtered difficult data, which can produce new difficult problems in large scale, eliminating the need for complex, per-instance prompting and its associated high API costs.
Fine-tuning Qwen2.5-Math-7B-Instruct on the ScaleDiff-Math dataset yields a substantial performance increase of 11.3% compared to the original dataset and achieves a 65.9% average accuracy on AIME'24, AIME'25, HMMT-Feb'25, BRUMO'25, and MATH500, outperforming recent strong LRMs like OpenThinker3.
Notably, this performance is achieved using the cost-efficient Qwen3-8B model as a teacher, demonstrating that our pipeline can effectively transfer advanced reasoning capabilities without relying on larger, more expensive teacher models.
We also observe a clear scaling phenomenon in model performance on difficult benchmarks as the quantity of difficult problems increases. 
Our code is available at the anonymous repository \url{https://anonymous.4open.science/r/ScaleDiff-D053}.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces ScaleDiff, a pipeline designed to address the challenge of creating difficult mathematical reasoning problems at scale for training Large Reasoning Models (LRMs). The core idea is to automate the generation of a large, high-quality dataset of challenging math problems to augment existing training data.

The ScaleDiff pipeline consists of four main stages:
- Identification: It first employs AdaptThink, an existing model, to efficiently identify and filter a subset of difficult problems from a large-scale, high-quality source dataset (AM-Qwen3-Distilled).
- Generation: A dedicated problem generator, named DiffGen-8B, is then trained exclusively on the text of these identified difficult problems to learn their underlying data distribution.
- Distillation: The trained DiffGen-8B generates a massive number of new, difficult problems. A strong teacher model (Qwen3-8B) is then used to generate step-by-step solutions (Chain-of-Thought) for these problems.
- Filtration: The resulting problem-solution pairs undergo a two-stage filtering process: a rule-based filter to remove malformed solutions and a model-based filter to discard problems that the target student model can already solve correctly.

The final augmented dataset is used to fine-tune a Qwen2.5-Math-7B model. The authors demonstrate through extensive experiments that their method yields substantial performance improvements across several challenging math benchmarks, including AIME'24, AIME'25, and HMMT-Feb'25, outperforming strong SFT and RL baselines. Ablation studies are also provided to validate the effectiveness of each key component in the pipeline.

### Strengths
- Addresses a Well-Defined and Important Problem: The paper tackles a significant bottleneck in advancing the reasoning capabilities of LRMs: the scarcity of high-difficulty, high-quality training data. The proposed goal of creating a scalable and automated data generation pipeline is highly relevant and valuable to the community.
- Systematic and Practical Pipeline: The ScaleDiff pipeline is a methodologically sound, end-to-end system. It is well-designed, practical, and demonstrates high efficiency by using a single forward pass for difficulty identification and a dedicated generator for scalable problem creation. This makes the approach a useful and reproducible blueprint for practitioners.
- Thorough Empirical Validation and Ablation Studies: The experimental evaluation is a standout feature of this work. The authors not only demonstrate strong final performance but also conduct detailed ablation studies (Table 3) that convincingly isolate and quantify the positive impact of the "difficult problem identification" and "response filtration" stages. This adds significant credibility to their claims.
- Strong and State-of-the-Art Results: The resulting ScaleDiff-7B model achieves impressive performance on a comprehensive set of mathematical reasoning benchmarks. It surpasses many larger models and strong, contemporary baselines, effectively demonstrating the real-world efficacy of the proposed data augmentation strategy.
- Provides Actionable Insights: The analysis section offers valuable insights, particularly the finding that a smaller teacher model (Qwen3-8B) can be nearly as effective for solution distillation as a much larger one from the same family. This has important practical implications for balancing cost and performance in model development.

### Weaknesses
- Limited Methodological Novelty: The paper's primary contribution lies in the effective integration and application of existing methods rather than the introduction of novel techniques. The difficulty identifier (AdaptThink) is an adopted tool, and the methodology for training the problem generator (DiffGen) is explicitly noted to follow a similar approach to ScaleQuest. While the resulting system is effective, the lack of foundational algorithmic novelty may be a concern from a research perspective.
- Absence of Evaluation on Generated Problem Diversity: The paper successfully shows that the generated problems are "difficult" (88% are classified as such by AdaptThink). However, there is no quantitative analysis of the diversity of these generated problems. It is unclear whether DiffGen is producing genuinely novel problems or merely creating syntactic paraphrases and variations of the problems in its training set (P_diff). This is a critical omission for evaluating the quality of a generative data pipeline.
- Potential for Dataset Contamination and Unclear Generalization: The pipeline starts with the AM-Qwen3-Distilled dataset, which aggregates several public math datasets. This introduces a risk that the training data distribution is already very close to that of the evaluation benchmarks (e.g., AIME), potentially inflating performance metrics. The paper would be strengthened by testing the generalization of its method, for instance, by training on a dataset from one domain (e.g., high school olympiads) and evaluating on a distinctly different one (e.g., college-level mathematics or SAT Math) to prove the pipeline's robustness.
- Ambiguity in the Contribution of the Model Filtering Step: The model filtering step, which discards problems the student model can already solve, is shown to be effective. However, its role could be interpreted as a highly targeted form of rejection sampling. It is not fully disentangled whether the performance gain comes from the sophisticated DiffGen training strategy or simply from having a very large pool of generated problems to apply this effective filtering technique on. The core contribution of DiffGen's training versus a simpler generation-and-filtering scheme is not entirely clear.
- Homogeneity of Teacher and Student Models in Analysis: The teacher model comparison in Section 5.2 uses Qwen3-235B and Qwen3-8B, which are from the same model family. It is highly likely that the 8B model is a distilled or aligned version of the 235B model, thus sharing very similar reasoning patterns. This could explain the minimal performance difference. The claim about the cost-effectiveness of smaller teachers would be far more compelling if validated with a heterogeneous teacher model from a different family (e.g., DeepSeek, Llama, Gemini) to show that the distillation process is robust across different model architectures and training methodologies.
- Lack of Comparison to Simpler Generation Baselines: The paper justifies the complexity of training a dedicated generator (DiffGen) but does not compare it against simpler, potent baselines. For example, applying advanced problem rephrasing or modification techniques (akin to MetaMath) directly to the identified difficult problem set (P_diff) could potentially achieve similar results in terms of difficulty and diversity, but with less computational overhead. Without this comparison, the necessity of the DiffGen component is not fully established.

### Questions
See Weakness.

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
4

### Summary
This paper proposes a new framework, ScaleDiff, to efficiently generate and utilize difficult mathematical problems for training Large Reasoning Models. The method first identifies challenging problems from existing datasets using an adaptive model called AdaptThink, which distinguishes between “thinking” and “no-thinking” modes to infer difficulty with just one forward pass. These difficult examples are then used to train a specialized generator, DiffGen-8B, that produces new hard problems at scale without costly manual prompting. Solutions are distilled from a smaller, cost-efficient teacher model (Qwen3-8B) and filtered for quality, forming the ScaleDiff-Math dataset. Fine-tuning models on this dataset yields significant performance gains—an average accuracy of 65.9% across major math reasoning benchmarks, outperforming strong baselines like OpenThinker3. The paper also demonstrates a clear scaling trend: increasing the quantity of difficult problems systematically enhances reasoning performance, showing that difficulty-aware data generation is a scalable path to stronger reasoning models.

### Strengths
- Efficient difficulty identification via AdaptThink: a single forward pass “Thinking/NoThinking” switch replaces fail-rate loops and brittle LLM-as-judge prompts, with a precise operational criterion (first token).
- Large dataset: ScaleDiff-Math combines original data with 1.15M generated pairs to reach ~1.7M examples—substantial scale for open research.
- DiffGen-8B was trained exclusively on the subset of difficult problems identified by their pipeline. This design intentionally separates the task of problem generation from that of solution synthesis, allowing the model to specialize in producing challenging questions.
- Experiments showed a significant gain from performing SFT on ScaleDiff.

### Weaknesses
- Difficulty is defined by AdaptThink’s first-token decision, making the identifier inherently model-dependent and calibration-dependent, yet the paper provides no robustness checks (e.g., alternative identifiers, threshold sensitivity, cross-model agreement).
-  The generation stage optimizes only over problem tokens (no solution context), which can drift toward ill-posed or underspecified questions. Yet the paper seemed only to mention filtering out easy questions. Also, the cost-saving choice to distill a single solution per problem might cause some noise (e.g., incorrect solution or problematic reasoning).
- The claim "demonstrating that our pipeline can effectively transfer advanced reasoning capabilities without relying on larger, more expensive teacher models" sounds overstated to me. The teacher model (Qwen3-8B) is substantially stronger than the student model (Qwen2.5-Math-7B-Instruct), and the paper uses a 235B model for solution generation.

### Questions
- Have you tested the model on easier datasets like math500 and gsm8k? Disirably, ScaleDiff-7B should outperform baselines not only on the hardest datasets but rather on a whole spectrum of benchmarks. 
- Have you conducted a decontamination or leakage check on the ScaleDiff dataset?
- Have you used ScaleDiff to train Qwen3-8B itself? The contribution would seem more compelling to me if the model (Qwen3-8B) could self-improve by generating harder data, since the whole data generation pipeline is somewhat expensive (using a 235B model).
- The paper uses an off-the-shelf model, AdaptThink, to classify simple/difficult questions. I would suggest an ablation comparing the current method with the existing ones. E.g., this paper: He, Yinghui, et al. "AdaptMI: Adaptive Skill-based In-context Math Instruction for Small Language Models." arXiv preprint arXiv:2505.00147 (2025).

### Soundness
2

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
The authors proposed ScaleDiff, an efficient system to scale the creation of difficult mathematical problems for large reasoning models. The authors proposed to filter the hard problems with a low-parameter system that can efficiently work in one single forward pass. The effectiveness of this system is justified through a pre-experiment on the system. After verification of the evaluation function, a diffcult problem generator is trained and only the difficult generated problems are left. The benchmark results show that the models finetuned with the final augmented dataset outperforms existing baselines.

### Strengths
1. The paper is clearly written and presented.
2. The authors give clear clarification and intuition of why and how the system works.
3. The evaluation results show that the method is quite effective despite its simplicity.

### Weaknesses
The paper shows no significant weakness within its claim.

### Questions
1. The chapter "effectiveness of difficulty" seems to be problematic. The authors trained models on simple/hard divisions of the data with controlled number, and compared the metrics to that with full or random-selected dataset. The difficult subset seems to be effective given it providing similiar performance with smaller dataset size with full dataset. The simple subset, however, looks to me a too easy baseline. For example, what if we mix the simple subset with 20% data from the difficult subset? The peformance drop can happen simply because there is lack of diversity in simple subset, rather than the metric is really meaningful. I hope the authors could clarify this point.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work attempts to generate difficult synthetic problems meant to be used for training powerful mathematical reasoning problem. The approach leverages the AdaptThink model in order to filter out the difficult problems from a seed dataset which are then used for training synthetic problem generation model. After sampling new synthetic data from this problem generator, distilling its solutions and some filtering, the new synthetic problems are combined with the originally filtered seed dataset to form the final training data. Qwen2.5-Math-7B-Instruct trained on this final dataset shows improved average performance across 5 standardly used math reasoning benchmarks.

### Strengths
* The paper in general is clearly written and easy to follow.
* The idea of using dynamic thinking mode switching in AdaptThink as an indicator for difficulty is interesting
* I appreciate the the Abalysis in Section 5
* A good set of ablations has been included

### Weaknesses
I have several concerns about some of the claims made as well as the experimental design of this paper.

* The claim that the thinking vs non-thinking mode switching of AdaptThink can be used as a proxy to judge difficulty is not well substantiated. The authors demonstrate the effectiveness of different subsets of the dataset filtered by AdaptThink as SFT data in Table 1. However, downstream performance cannot be considered as an indicator for the difficulty itself, since it would depend on many factors other than the difficulty of the questions itself, such as but not limited to: diversity of the problems, quality of solution traces, etc. In general, I believe all the factors affecting the usefulness of a problem set as finetuning data are not well understood, thereby introducing confounders. A more direct way of substantiating the claim would be to measure pass@k of the filtered data subsets under multiple models.
* The approach does not involve any form of solution verification, leaving it as future work. However, I believe some basic form of solution verification is necessary to ensure some level of correctness, especially given that 88% of the generated problems are considered to be difficult. This could be something simple like self-consistency based verification.
* Sampling parameters for the solution distillation step are not mentioned?
* Sufficient details for the filtering process carried out after solution distillation are missing. How is verbosity / redundancy in the solutions detected? Do authors use LLM-as-a-judge?
* For model based-filtering, using agreement with the teacher model as a criteria for filtering presumes that all the solutions generated by the teacher model are correct, which is not guaranteed or even attempted to be ensured since there is no solution verification involved. 
* I am skeptical of the sole use of Qwen2.5-7B-Math-Instruct as the base models given the recent results these models tend to improve even without data lacking any correct labels [1, 2] as a result of their heavy math oriented pre / post-training. Ideally, the effectiveness of the data should be demonstrated on multiple (atleast 3) base models, preferable across different model families such as Llama and Gemma family of models. 
* Since this is an approach for synthetic data generation, the baselines should include other synthetic data gen methods. For eg., ScaleQuest [3] is an essential as well as strong baseline which needs to be included since it also involved training specialist synthetic data generation models. Other baselines include persona based synthetic data generation [4], DART-Math [5], MetaMath [1], etc. All these methods have publicly released datasets, making a data-sized match comparison possible.
* Section 4.2 - Comparison with the teacher model performance in not informative in my opinion since the performance would also depend on the base model (Qwen2.5-7B-Math-Instruct in this case). Finetuning a better model as the base can easily lead to performance exceeding that of the base model.
* Most of the results in Table 3 do not seems statistically significant given the small differences between means and high standard deviations. Thus, the conclusions made on the basis of this could be wrong. 
* Section 5.3, point (3): Comparison of the response lengths of D^{L}_{Diff} and D^{S}_{DiffGen} is not a fair since they use two different solution generation models and the response lengths can vary across the two models, thus introducing a confounder.
  
  
### Other Comments (Not affecting my rating)
* Citations missing for AIME, MATH, BRUMO, HMMT in the intro.
* MATH500 citation is wrong. The correct citation should be:  

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023.

  
### References 
[1] https://arxiv.org/abs/2309.12284   
[2] https://openreview.net/forum?id=HM7OkY1jFw    
[3] https://arxiv.org/abs/2410.18693   
[4] https://arxiv.org/abs/2406.20094   
[5] https://arxiv.org/abs/2407.13690

### Questions
* In the first line of Section 4.1, the authors claim that AM-Qwen3-Distilled is "is well-regarded for its high quality and has demonstrated effectiveness in training high quality mathematical reasoning models". Can the authors provide references for this claim?
* Was temperature experimented with during problem generation phase? For thinking models such as Qwen3-8B, a temperature of 0.6 is usually recommended. Is there any reason a temperature of 1.0 was used instead?
* For solution generation - what were the sampling parameters used for Qwen3-8B?
* For solution distillation, what does "reliably" mean? What was the response length of the base model during generation? How was repetition detected? 
* Why are 3 solutions distilled per problem in Section 5.2? How are they used (given the final data size in Table 3 is still 192K)

### Soundness
2

### Presentation
3

### Contribution
2
