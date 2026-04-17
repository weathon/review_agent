# Breaking Training Bottlenecks: Effective Reinforcement Learning for Modern Coding Models

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Modern code generation models exhibit longer outputs, accelerated capability growth, and fundamentally changed training dynamics, rendering traditional training methodologies, algorithms, and datasets ineffective for enhancing their performance. To address these training bottlenecks, we propose MicroCoder-GRPO, an enhanced Group Relative Policy Optimization approach with three key innovations: conditional truncation masking to enhance long output potential while maintaining training stability, diversity-determined temperature selection to maintain and encourage output diversity, and removal of KL loss with high clipping ratios to facilitate exploration. MicroCoder-GRPO achieves up to 17.6% relative improvement over strong baselines on LiveCodeBench v6, with more pronounced gains under extended context evaluation. Additionally, we release MicroCoder-Dataset, a more challenging training corpus that achieves 3× larger performance gains than mainstream datasets on LiveCodeBench v6 within 300 training steps, and MicroCoder-Evaluator, a robust framework with approximately 25% improved evaluation accuracy and around 40% faster execution. Through comprehensive analysis across more than thirty controlled experiments, we reveal 34 key training insights across seven main aspects, demonstrating that properly trained models can achieve competitive performance with larger counterparts.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes MicroCoder-GRPO, an improved reinforcement learning optimization framework for code generation tasks based on Group Relative Policy Optimization (GRPO). The method introduces three main innovations: Conditional Truncation Masking, Diversity-Determined Temperature Selection, and Removal of KL loss with high clipping. Additionally, the paper presents MicroCoder-Evaluator. Experimental results demonstrate that MicroCoder-GRPO achieves higher training stability, sustained diversity, and improved accuracy compared with baseline GRPO and LiveCodeBench-based training setups.

### Strengths
1. The paper clearly identifies new training bottlenecks specific to modern coding models (long outputs, altered dynamics, and reasoning-heavy behavior) that traditional RL methods fail to address.

2. The proposed MicroCoder-GRPO introduces three complementary and practically effective modifications (conditional truncation masking, diversity-determined temperature scheduling, and removal of KL loss with high clipping), each addressing a concrete limitation in prior GRPO-style training.

3. The MicroCoder dataset and MicroCoder-Evaluator represent well-designed, practically valuable resources for future RL-based code generation research.

### Weaknesses
1. The proposed modifications (conditional truncation masking, diversity-based temperature scheduling, KL removal with high clipping) are primarily empirically motivated. The paper lacks formal justification or convergence analysis to establish why these strategies work beyond observed trends.

2. Experiments are conducted exclusively on Qwen-based models and LiveCodeBench-derived datasets, raising concerns about generalizability to other architectures.

3. While the combination of improvements is effective, each modification represents a relatively straightforward heuristic extension of prior GRPO/DAPO frameworks.

4. The claim of faster convergence and efficiency is described qualitatively; detailed wall-clock time, GPU cost, or training throughput comparisons to GRPO/DAPO are unclear.

### Questions
1. Could the authors elaborate on how temperature is dynamically updated (formula or algorithmic rule) and whether the same policy generalizes across different model sizes?

2. Have the authors tested MicroCoder-GRPO on other architectures (e.g., CodeLlama, DeepSeek-Coder, or StarCoder)? If not, what aspects of the algorithm might depend on the Qwen-specific tokenizer or output behavior?

3. can the authors clarify how much of the final gain (17.6%) originates from the new algorithm versus the new dataset?

4. Are there particular coding domains or problem types where MicroCoder-GRPO underperforms or shows instability compared to GRPO?

### Soundness
2

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
3

### Summary
This paper talks about MicroCoder-GRPO. It's a new way to make training better for code generation models. It has three main ideas. First, conditional truncation masking. This helps the model produce longer outputs without losing stability. Second, diversity-determined temperature selection. This keeps the outputs varied and interesting. Third, like DAPO, they removed the KL divergence loss and used high clipping ratios. This helps the model explore new options. The authors also share the MicroCoder-Dataset, a tougher training set, and the MicroCoder-Evaluator, a fast and reliable way to check code quality. They tested this on Qwen3 models. The results show up to a 17.6% improvement on LiveCodeBench v6. They also share 34 quick tips from over thirty experiments. These experiments looked at dataset quality, evaluators, temperature, context length, and batch size.

### Strengths
- Comprehensive Methodological Innovation: The paper shows a complete solution by changing three main things: the MicroCoder-GRPO algorithm, the MicroCoder-Dataset, and the MicroCoder-Evaluator. Putting these together gives a strong system for making RL training better for code writing.
- Good Test Results: MicroCoder-GRPO improves performance by up to 17.6% on LiveCodeBench v6. Not only that, but it works even better when testing with longer code contexts, like training on 4K and testing on 8K. This shows it can handle longer tasks well and makes long reasoning easier.
- Deep and Systematic Experimental Analysis:  Comperhensive and diverse experiments provide an invaluable guide for practitioners and researchers in the field of RL for code generation.
- High-Quality Infrastructure: The MicroCoder-Evaluator is a big help. It compares different strategies and works well even if there are errors. It makes evaluation about 25% more accurate and runs 40% faster. This makes rewards better and helps RL training work better.

### Weaknesses
- Missing Dataset and Lack of Details: TThe MicroCoder-Dataset is a main part of the paper. It claims to give 3 times better performance than other datasets. But, the paper doesn’t give a download link or any real details about the dataset. It just mentions a four-step process. This is a big problem. Without access or clear info about the dataset—like where it came from, how big it is, what's in it, or how hard it is—it's impossible to check the paper’s claims. This also means you can’t really reproduce the study.
- Incremental Contribution of KL Loss Removal: The paper mentions removing KL loss and using a high clip ratio. These ideas come from DAPO. It shows they work with the MicroCoder-GRPO method. But, the paper doesn’t clearly explain how much these changes help or how they work with the new masking method. This makes the main idea less original.
- Unclear Temperature Method: The paper says they pick a temperature T(D) based on diversity. But, they don’t explain how they choose T(D). They don’t give the exact rule or formula. This makes it hard to understand or repeat the process.

### Questions
This paper talks about two main things. First, it has a new algorithm called MicroCoder-GRPO. This helps make better code models. Second, it offers 34 insights that give a lot of ideas for training these models. They also built something called the MicroCoder-Evaluator. This is a good tool for checking how well the models work.
However, this recommendation is strictly conditional. A core contribution, the MicroCoder-Dataset, is neither described in detail nor made accessible. This is a significant omission that currently prevents the verification of the paper's central claims regarding dataset quality  and hinders the reproducibility of the entire study.

My final recommendation depends on what the authors give us during the author response time:
1. At least a part of the MicroCoder-Dataset.
2. If they can't share it (maybe because it's proprietary), they need to clearly say why. At a minimum, they should include a detailed description in the appendix. This should cover where the data comes from, how they collected and filtered it, its size, what languages it includes, and how hard it is compared to DeepCoder, with some numbers.
Supporting Arguments and Questions

1. Action Required: MicroCoder-Dataset Access and Description: I need you to do one of these things before I can give my go-ahead. First, share a public link so I can see the MicroCoder-Dataset. If you can't make it public, then give a detailed description in the appendix. This description should include how you collected and filtered the data, where you got it from, how big it is, what languages are in it, and how its problem difficulty compares to the DeepCoder dataset.

2. Clarification on Diversity-Determined Temperature: Clarification on Diversity-Determined Temperature: Can you tell me exactly how you choose the temperature T(D)? For example, how do you change T based on the initial diversity and how it changes over time? 

3. Ablation Study on Conditional Truncation Masking: The mask uses four rules: "reach max length L_{max}," "give wrong answers," "avoid repeating," and "pick randomly with p." To see how each rule helps, maybe an ablation study is required? 

4. Specific Examples for Evaluator Improvement: The MicroCoder-Evaluator gets about 25% better accuracy by switching from "exact match" to "multi-strategy comparison." Can you include some examples in the appendix? Show code outputs that the LiveCodeBench marks as failed but the MicroCoder marks as passed.

### Soundness
3

### Presentation
2

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
This paper explores the diverse training settings of RL in code generation tasks. It first propose MicroCoder-GRPO, an enhancement of original GRPO, incorporating three main innovations. Moreover, the work introduces a training dataset (MicroCoder-Dataset) and an evaluation framework (MicroCoder-Evaluator). The paper presents numerous insights derived from an extensive and empirical analysis.

### Strengths
1. **Comprehensive Analysis:** The work tackles the RL training problem from multiple angles: algorithm refinement, data quality, and evaluation infrastructure. This holistic view is valuable to the community.

2. **Strong Empirical Results:** MicroCoder-GRPO achieves significant relative gains (up to 17.6%) on LiveCodeBench v6, with notable improvements under extended context evaluation, demonstrating superior scalability.

3. **Solid Ablation Study:** The systematic analysis across more than thirty controlled experiments, revealing many training insights. The findings on context length irreversibility, on/off-policy balance, and temperature dynamics are particularly insightful.

### Weaknesses
1. **Overstated Novelty and Contribution Framing:** The paper's framing of its "three key innovations" in the abstract is a significant concern. "Removal of KL loss with high clipping ratios" was explicitly introduced and studied in DAPO [1]. Additionally, the other two proposed innovations, "conditional truncation masking" and "diversity-determined temperature," appear to be common practice for GRPO training, rather than fundamentally new concepts [2, 3]. The authors should revise their paper to clearly and accurately distinguish between the adoption of prior work and their novel contributions.

2. **Presentation of Insights:** The abstract mentions 34 key insights, but these are scattered throughout the text. This diffusion dilutes their impact. The paper would be significantly stronger if it consolidated the most critical and non-obvious insights into a summarized list or table.

3. **Limited Model Generalizability:** The empirical validation is concentrated almost on Qwen-3 models. While this demonstrates effectiveness on a modern architecture, the central claim of breaking "training bottlenecks for modern coding models" would be substantially strengthened by demonstrating performance across a more diverse set of code LLMs.

[1] Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., ... & Wang, M. (2025). Dapo: An open-source llm reinforcement learning system at scale. *arXiv preprint arXiv:2503.14476*.

[2] Mroueh, Y., Dupuis, N., Belgodere, B., Nitsure, A., Rigotti, M., Greenewald, K., ... & Rios, J. (2025). Revisiting Group Relative Policy Optimization: Insights into On-Policy and Off-Policy Training. *arXiv preprint arXiv:2505.22257*.

[3] Liu, Z., Chen, C., Li, W., Qi, P., Pang, T., Du, C., ... & Lin, M. (2025). Understanding r1-zero-like training: A critical perspective. *arXiv preprint arXiv:2503.20783*.

### Questions
See weaknesses.

### Soundness
3

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
4

### Summary
This paper tackles a very relevant problem in today’s code-generation landscape: reinforcement learning methods that worked for math reasoning or earlier coding models don’t translate well to newer models that generate much longer outputs.
The authors introduce MicroCoder-GRPO, a variant of GRPO with three practical tweaks — conditional truncation masking, diversity-based temperature control, and removal of the KL term with high clipping. Together, these aim to stabilize training while preserving long-output diversity.
They also contribute a MicroCoder-Dataset (a harder, more diverse coding set) and a MicroCoder-Evaluator (a more reliable and faster evaluation tool).
Across a broad set of experiments with Qwen3-1.7B and 4B models, the method shows steady gains over GRPO and DAPO, reaching roughly 17% improvement on LiveCodeBench v6.

### Strengths
The paper’s main strength lies in its comprehensiveness. It doesn’t rely on one or two headline results but builds a consistent empirical story across thirty experiments. The insight that early-stage truncation and diversity collapse can have irreversible effects on long-context learning is particularly interesting. The MicroCoder dataset and evaluator also make the contribution more substantial, showing that progress in RL for code depends on both better algorithms and better infrastructure.

### Weaknesses
The main weaknesses are a lack of theoretical framing and limited model diversity — all experiments are done on Qwen models, leaving open whether the same behaviors hold for DeepSeekCoder or StarCoder families. Some analyses, such as temperature scheduling, are presented in great detail but remain somewhat heuristic. Overall, however, these are minor limitations rather than critical flaws.

### Questions
There are a few points that would benefit from clarification in the rebuttal. First, how much of the total improvement can be attributed to each of the three proposed modifications? Second, how exactly is “diversity” measured, and how sensitive are the results to the metric used? Third, is there any risk that removing the KL term and using high clipping could lead to reward overfitting or code-style drift in later stages of training? Finally, it would be helpful to know whether the MicroCoder-Evaluator’s flexible matching rules ever falsely classify incorrect code as correct, since that could partially explain the higher accuracy reported.

### Soundness
3

### Presentation
2

### Contribution
3
