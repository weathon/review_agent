# DART: Difficulty-Adaptive Reasoning Truncation for Efficient Large Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
Adaptive reasoning is essential for aligning the computational effort of large language models (LLMs) with the intrinsic difficulty of problems. Current chain-of-thought methods boost reasoning ability but indiscriminately generate long explanations, leading to evident inefficiency. However, existing reinforcement learning approaches to adaptive thinking remain unstable and heavily reward-dependent. Here we propose DART, a supervised Difficulty-Adaptive Reasoning Truncation framework that adjusts thinking length according to problem difficulty. By distilling concise reasoning patterns from stronger models, interpolating them into a continuum of reasoning styles, and curating optimal training data that balances correctness and compactness, DART learns when to ''stop thinking''. Across multiple mathematical benchmarks, experimental results demonstrate its remarkable efficiency while preserving or improving accuracy, achieving a significant 81.2% reasoning truncation (DeepSeek-R1-Distill-Qwen-7B on GSM8K dataset) with 5.33x computational acceleration. DART provides a stable and general paradigm for efficient reasoning, advancing the development of adaptive intelligence in LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The main idea is to propose the DART framework, which implements Difficulty-Adaptive Truncation through a complete SFT framework, enabling the model to answer simple questions quickly and think more deeply about complex problems.

### Strengths
1. The authors empirically identify and formalize the sigmoid-shaped relationship between reasoning length and accuracy, which provides a theoretical motivation for adaptive truncation. This insight contributes a valuable quantitative characterization of the “optimal reasoning length” phenomenon, offering a foundation for subsequent research on reasoning efficiency.
2. Unlike RL-based methods, which are notoriously sensitive to reward shaping and initialization, DART achieves adaptive reasoning entirely through supervised fine-tuning. This yields greater training stability, reproducibility, and easier integration with existing LLM infrastructure, an important practical advantage for scaling and deployment.

### Weaknesses
1. The proposed framework distills reasoning chains and subsequently selects the shortest correct CoT as the sole supervision signal. While this efficiently reduces token redundancy, it inevitably drives the model toward a single canonical reasoning pattern. This may suppress the natural diversity of reasoning trajectories that could otherwise contribute to robustness and creativity. In tasks requiring exploratory or multi-path reasoning, such as open-domain problem-solving or commonsense inference, this rigid supervision could cause premature convergence toward a single logic template, thereby missing potentially valuable intermediate reasoning paths.
2. The interpolation fusion between base and distilled models relies on empirically chosen coefficients (α), which are sampled discretely between 0 and 1. However, the choice of step size and distribution lacks theoretical justification or formal analysis of convergence properties. The paper demonstrates smooth behavior empirically, but does not establish why linear parameter interpolation should produce semantically coherent intermediate reasoning styles. This makes the fusion spectrum somewhat heuristic and architecture-dependent, weakening claims of theoretical robustness or generalizability beyond the tested model families.
3. The adaptive data curation pipeline requires an explicit correctness signal for filtering valid CoTs. This assumption restricts the method to closed-form reasoning tasks with well-defined answers. Consequently, DART cannot be directly extended to open-ended tasks, such as dialogue, writing, or scientific hypothesis generation. The framework would thus benefit from a broader definition of “reasoning sufficiency” that does not rely solely on exact-match evaluation.
4. Since the framework rewards shorter reasoning when correct, there exists a bias toward brevity even when longer reasoning might improve interpretability or error recovery. Without a mechanism to penalize premature truncation, the adaptive model may occasionally terminate too early on out-of-distribution or higher-complexity problems, leading to subtle accuracy degradation or reasoning incompleteness.

### Questions
See the Weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents DART, a method for efficient reasoning for LLMs. It introduces a strategy to curate concise data with varying reasoning lengths for problems of different difficulty levels. Specifically, DART distills short reasoning chains from stronger teacher models, then fuses the long/short-reasoning models to create a continuum of reasoning styles (different lengths), and automatically selects the shortest correct reasoning chain for each problem to build an adaptive training dataset. The final model is then finetuned on this curated data to learn when to stop thinking based on problem complexity. The evaluation across various model sizes demonstrates its effectiveness in compressing generation length by up to 81.2% while maintaining or improving reasoning accuracy.

### Strengths
The written is straightforward and easy to understand.

The paper proposes an angle to train efficient LRM basing on different difficulty level.

The experiments show that the method has some improvements on different models with reduced generation length.

### Weaknesses
It is not very clear what's the advantage of using the extrapolation to generate different lengths of response regarding different difficulty levels. I understand that the extrapolation could help to control the length of the generation, which can be further used to select and include the data used for the final training. It is not clear how this extrapolation based data generation method work compared with using the prompt based method to generate different lengths of response. 

Lack of experimental results. The main idea for this paper is to curate different length of CoT data based on different question difficulty level, and the author noted that methods like tokenskip didn’t consider such difficulties. A most natural baseline to be included is to compare the results with those static methods like tokenskip/lightthinker, which is currently lacking. Another question is why the baselines compared are different on different base models. Only DeepSeek-R1-Distill-Qwen-7B contains the results for other SFT based baselines? It is not clear how the current data curation protocol works without further evaluation on other models compared with other mechanisms. 

lightthinker:Thinking Step-by-Step Compression


Tokenskip Controllable Chain-of-Thought Compression in LLMs

### Questions
See weakness

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
This paper proposes a supervised difficulty-adaptive reasoning truncation framework that enables LLM to adjusts thinking length according to problem difficulty dynamiclly. By distilling concise reasoning patterns from stronger models, interpolating them into a continuum of reasoning styles, and curating optimal training data, that balances correctness and compactness, DART learns to alleviate overthinking problem in LRM.

### Strengths
- The paper addresses an important problem, improving reasoning efficiency for large language models
- The four-step framework (DISTILLING SHORT COTS, interpolation, CREATING A MODEL SPECTRUM, CURATING TRAINING DATA, adaptive training) is clearly structured and easy to follow.
- The experiments cover several standard mathematical reasoning benchmarks and include analyses on certain hyperparameters, such as fusion coefficients and sampling density

### Weaknesses
- Limited novelty. The idea of adaptive, difficulty-aware reasoning is not new, and prior work, such as CoT-Valve, has already explored similar strategies for interpolating model weights and curating adaptive data based on correctness.
- The method appears less effective on DeepSeek-R1-Distill-Qwen-7B. On benchmarks such as GSM8K, MATH-500, and OLYMPAID, the generated token length is reduced, but the accuracy also drops.
- The short-CoT data generated from DeepSeek-R1-Distill-Qwen-7B is important to the framework, yet the paper provides little analysis of its quality or length comparison. Conceptually, the method relies on the same model to generate compressed reasoning traces and subsequently distills itself on this data, but the rationale for why such a self-distillation loop should be effective is unclear. Including a comparison with existing token-compression methods, such as Selective Context used in TokenSkip, would help clarify the discussion.

[1] CoT-Valve: Length-Compressible Chain-of-Thought Tuning

[2] Compressing context to enhance inference efficiency of large language models.

[3] TokenSkip: Controllable Chain-of-Thought Compression in LLMs

### Questions
- Could the authors clarify how the proposed framework differs from prior adaptive reasoning methods such as CoT-Valve?
- For experiments on DeepSeek-R1-Distill-Qwen-7B, token usage decreases, but accuracy also drops. Could this degradation be caused by generating the short-CoT data using the same model?

### Soundness
2

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
4

### Summary
This paper addresses the issue that chain-of-thought (CoT) reasoning often produces unnecessarily long reasoning traces, regardless of the intrinsic difficulty of a problem. To address this problem, the paper proposes DART, a framework that trains a model on optimal reasoning chains collected through a pipeline. Specifically, a base model generates long reasoning chains, while a teacher model converts them into shorter chains. Model fusion is then applied to interpolate between these two models, creating a continuum of models capable of producing intermediate-length reasoning chains. Finally, a training set is constructed from the shortest correct reasoning chains, which is used to train the final model for more efficient reasoning.

### Strengths
The paper focuses on an important problem—the inefficiency of CoT. The proposed framework is presented as modular and is conceptually sound.

### Weaknesses
-	Clarity issues. Some parts of the methodology are not clearly explained. For example, it is unclear how the distillation teacher model shortens long reasoning chains and how this process affects the quality of the reasoning paths. Additionally, the paper does not discuss how the quality of the generated reasoning chains is controlled or verified.
-	Limited novelty. The proposed method essentially involves collecting question-answer pairs with varying reasoning lengths and using this dataset to train a model. While practical, the technical novelty is fairly limited.
-	Computational cost. Using a base model, a teacher model, and creating a continuum of models to obtain reasoning chains of varying lengths appears computationally expensive. It would be useful to consider whether a single model could produce different reasoning lengths using prompting or other steering techniques.
-	Performance drop. While the method reduces the number of output tokens, performance drops on several datasets, suggesting that shorter reasoning chains may not always preserve reasoning quality.
-	Although the approach is described as difficulty-aware, the final trained model does not explicitly identify the difficulty of a given input or adapt the reasoning process accordingly.
-	Limited evaluation. The evaluation is restricted to mathematical reasoning tasks, which limits the generalizability of the findings.

### Questions
-	Would the framework generalize to reasoning tasks beyond mathematics?
-	Would it be possible to use a single model to produce reasoning chains of different lengths using prompting or other steering techniques?

### Soundness
2

### Presentation
3

### Contribution
2
