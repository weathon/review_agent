# Visual Compositional Tuning

- Avg Score: 5.67
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6, 8, 4

## Abstract
Visual instruction tuning (VIT) datasets have grown rapidly in scale, yet the informativeness of individual training samples has largely been overlooked. Recent dataset selection methods have shown that a small fraction of such datasets enriched with informative samples can lead to efficient finetuning of Multimodal Large Language Models. In this work, we explore the impact of sample complexity on informative data curation and introduce COMPACT (COMPositional Atomic-to-complex Visual Compositional Tuning), a visual compositional tuning data recipe that scales training sample complexity by combining multiple atomic visual capabilities in a single training example. Concretely, we synthesize rich and informative text questions for each image, allowing us to significantly reduce the number of training examples required for effective visual instruction tuning. COMPACT demonstrates superior data efficiency compared to existing data reduction methods. When applied to the LLAVA-665K VIT dataset, COMPACT reduces the data budget by 90% while still achieving 100.2% of the full VIT performance (compared to only 97.5% by the state-of-the-art method) across eight multimodal benchmarks. Further, training on the COMPACT data outperforms training on the full-scale VIT data on particularly complex benchmarks such as MM-Vet (+8.6%) and MMStar (+2.9%). COMPACT offers a scalable and efficient synthetic data generation recipe to improve on vision-language tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents COMPACT, a data-efficient visual instruction tuning (VIT) framework that synthesizes training samples with controlled compositional complexity. The authors introduce the k-value, representing the number of atomic visual capabilities (e.g., object recognition, spatial reasoning) required to answer a question. By generating high-k samples using Gemini-2.0-Flash and combining them with a small subset of LLaVA-665K for instruction formatting, COMPACT attains 100.2% of the full-dataset performance using only 10% of the data. It substantially outperforms baselines on complex benchmarks such as MM-Vet (+8.6%) and MMStar (+2.9%).

Key contributions:
1. A complexity-aware VIT data recipe leveraging atomic capability composition.
2. Empirical evidence that higher-k samples enhance data efficiency.
3. A scalable synthetic data generation framework that reduces dependence on large-scale datasets.

### Strengths
1. Originality:
COMPACT introduces a novel lens for understanding and curating VIT data by framing compositional complexity through the k-value, which quantifies the number of atomic visual capabilities (e.g., object recognition, spatial reasoning, counting) required to solve a task. This provides a principled and measurable axis for dataset construction, enabling systematic control over the difficulty and diversity of visual-instruction samples—an aspect largely overlooked in prior work.

2. Quality:
The study demonstrates strong methodological rigor, featuring comprehensive evaluations across eight multimodal benchmarks and detailed ablation analyses. These experiments confirm the method’s robustness, showing consistent improvements in both generalization and compositional reasoning while maintaining competitive performance under limited data budgets.

3. Clarity:
The paper presents a well-structured taxonomy of atomic visual capabilities and clearly articulates the data synthesis pipeline, from capability composition to instruction formatting. The process is transparent, reproducible, and amenable to scaling, offering valuable guidance for future research in efficient multimodal data generation.

4. Significance:
By achieving superior performance with only 32K synthetic samples, surpassing models trained on over 665K human-annotated examples, COMPACT directly challenges the prevailing assumption that larger datasets are always necessary for stronger multimodal performance. This redefines the data–efficiency frontier and highlights compositional control as a promising new direction for scaling visual instruction tuning.

### Weaknesses
1. ​​Dependency on closed-source models:​​ Data synthesis via Gemini-2.0-Flash risks reproducibility and may inherit model biases. Experiments with open-source generators would strengthen generalizability.
2. ​​Limited scope of atomic capabilities:​​ Non-visual skills (e.g., knowledge, math) are excluded, limiting gains on benchmarks like OK-VQA. Expanding the taxonomy could broaden applicability.
3. ​​Evaluation of compositional generalization:​​ While high-ksamples improve performance, tests for zero-shot compositionality are lacking.

### Questions
1. How might COMPACT perform if atomic capabilities are expanded to include non-perceptual skills (e.g., commonsense reasoning)? Could this address the modest gains on knowledge-intensive tasks?
2. Have you explored generating data with open-source VLMs to assess reproducibility and reduce reliance on Gemini?
3. Does the benefits of high-ksamples diminish beyond k=3? Is there an optimal complexity ceiling for efficient learning?
4. How does COMPACT handle images where certain atomic capabilities are inherently hard to combine?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper focuses on the curation of informative training data to enhance MLLMs’ finetuning efficiency. It introduces COMPACT, a novel data synthesis approach that generates rich and informative text questions for each image by integrating multiple atomic visual capabilities into a single training sample. Experimental results across various benchmarks demonstrate that COMPACT significantly reduce the required number of training examples while achieving performance comparable to that of full-scale training data, highlighting its efficiency.

### Strengths
Strength:
1. This work defines atomic capabilities essential for general visual reasoning and introduces the k-value metric to quantify task complexity.
2. The proposed COMPACT scales training sample complexity by incorporating multiple atomic visual capabilities within a single data, revealing that increased complexity enhances information utilization.
3. Experiments show the effectiveness of COMPACT, which achieves comparable or even superior performance with just 10% of the training data, improving finetunig efficiency for MLLMs.

### Weaknesses
1. The entire generation, verification, and evaluation process relies on the closed-source Gemini model, which may introduce potential bias and limit reproducibility.
2. The exploration is confined to the LLaVA-v1.5-7B-LoRA model and the LLaVA-665K VIT dataset, leaving the performance of COMPACT with other models and training datasets underexplored, especially considering that the LLaVA-665K dataset exhibits relatively low task complexity.
3. There is a lack of comparison with other data reduction methods in experiments.

### Questions
1. Although the number of required training data decreases, does the incorporation of multiple atomic questions in a single COMPACT question imply that the token count for inputs and outputs hasn't reduced such significantly? 
2. In Figure 3, why do models trained on random data sometimes outperform those trained on the full dataset? Additionally, why does the notably poor performance of COMPACT on the TextVQA dataset?	
3. What is the task complexity of the evaluation benchmarks?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The main motivation is to compress multiple capabilities into a smaller number of data samples to increase sample efficiency, doing more with less data sets that compose multiple atomic capabilities into one.

### Strengths
there are clear taxonomy of capabilities 
there are indeed clear gains over the llava-665k datasets where it was not principally constructed.

### Weaknesses
see the question

### Questions
I think this make sense, but the random baseline is very honest and seem to also suggest that using only 49k out of 665K pretty similar to the COMPACT setup, realistically, SFT is pretty light weight

I would think about improving this work via framing as improving existing answer quality than just data effiency, like many atomic and subjective task here could be used to double check the quality of answers, see if they are correct, or use them in capability-specific abiliations to try to see what task are driving most gains. I would be surprised if color (which seem relatively easy) drive much of the gain

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents COMPACT (COMPositional Atomic-to-Complex Visual Capability Tuning), a new data recipe for visual instruction tuning (VIT) in multimodal large language models (MLLMs). COMPACT introduces the idea of compositional complexity, where each training sample is constructed by combining multiple atomic visual capabilities (e.g., object recognition, spatial reasoning, color, shape). By controlling the number of combined capabilities (“k-value”), the method generates more information-dense and complex questions using Gemini-2.0-Flash, leading to significant data efficiency gains. With only 10% of LLaVA-665K data, COMPACT achieves 100.2% of the full-scale performance across benchmarks such as MM-Vet and MMStar, highlighting the benefit of complexity-aware data curation for MLLMs.

### Strengths
- Conceptually innovative: Introduces a clear and quantifiable notion of compositional complexity in VIT data, shifting focus from scale to information density.

- Strong empirical evidence: Matches or exceeds full-data performance with one-tenth of the samples, demonstrating outstanding data efficiency.

- Thorough evaluation: Comprehensive experiments across major multimodal benchmarks and detailed ablations on complexity levels (k-values) and instruction-tuning ratios.

- The paper is well-organized, with transparent methodology, taxonomy of atomic capabilities, and plans to release the dataset.

### Weaknesses
- Dependency on proprietary models: The reliance on Gemini-2.0-Flash for both question generation and verification is a significant limitation for reproducibility and may introduce unknown biases. The cost ($86.5 for 32K samples) could also be prohibitive for scaling to larger datasets. An analysis using open-source alternatives would strengthen the work.

- While the 10 atomic capabilities are well-defined, the paper acknowledges they are "not expected to be completely orthogonal" but provides limited justification for this specific set. The correlation analysis (Fig. 8) suggests substantial dependencies, yet the implications for the k-value metric are not fully explored. How does correlation between capabilities affect the actual complexity?

- Limited scope of evaluation: The focus is exclusively on vision-centric tasks. The poor performance on knowledge-intensive benchmarks (Table 9) suggests the approach may not generalize to domains requiring external knowledge or reasoning beyond perceptual capabilities. This limits the claim of addressing "general visual reasoning."


- Verification process clarity: The quality verification step (Step 3) uses confidence thresholds and word overlap metrics, but the paper provides limited analysis of failure modes or how often verification rejects generated samples. More transparency about the quality control process would be helpful.

### Questions
The "natural integration" requirement for multi-capability questions is somewhat subjective and relies on the LLM's interpretation

Zero-capability samples (0.9% of LLaVA-665K) are interesting but receive minimal discussion

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
COMPACT (COMPositional Atomic-to-Complex Visual Capability Tuning) introduces a method for generating complex, information-dense Visual Instruction Tuning (VIT) datasets by combining multiple atomic visual capabilities (e.g., color, spatial reasoning, object recognition) into single training examples. This complexity-aware curation improves data efficiency -- achieving 100.2% of full LLaVA-665K performance using only 10% of the data, with notable gains on complex multimodal benchmarks like MM-Vet and MMStar.

### Strengths
- The concept of compositional complexity (k-value) as a controllable metric for VIT data curation is intuitive and well-grounded. The exploratory experiment (Figure 1) effectively demonstrates that increasing k improves performance.

- Strong improvements on compositional tasks with 90 % less data; convincing scaling curves and ablations.

- The paper includes thorough ablations examining complexity ranges, instruction tuning ratios, and complexity distributions. The analysis of LLaVA-665K's complexity distribution (mean k=1.95) provides valuable insights.

- Well-analyzed: Breaks down atomic capabilities, evaluates complexity distributions, and studies instruction-format mixing.

### Weaknesses
- Atomic Capability Definition:
  - The taxonomy appears somewhat arbitrary (why these 10 capabilities specifically?)
  - Capabilities are acknowledged as non-orthogonal (Figure 8), undermining the "atomic" framing
  - Object recognition is implicitly assumed in most questions (Figure 7), suggesting the capabilities may not be properly decomposable

- Evaluation scope: Only one base model (LLaVA-v1.5-7B-LoRA); unclear generality to other architectures or scales.

- Error Analysis:
  - Qualitative examples (Figure 11) cherry-pick favorable cases without systematic error analysis
  - Insufficent analysis of failure modes or systematic biases in generated data

### Questions
- How sensitive are results to the choice of data generator? Have you experimented with other VLMs (e.g., GPT-4V, LLaVA-NeXT)?

- Is there evidence that naturally occurring questions follow a certain k-distribution? How does COMPACT's distribution compare?

- The paper conflates "compositional complexity" with "task complexity" without clearly distinguishing them. 


Minor:
- Figure 3: are you sure its in log scale?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 6

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes COMPACT, where images are from LLaVA-665K, complex instructions are generated by Gemini, to improve data efficiency in multimodal instruction tuning by generating questions that require combinations of atomic visual capabilities. The task complexity is operationalized by the number of atomic capabilities involved (k-value). 
The paper demonstrates that increasing task complexity leads to better use of visual information and yields impressive performance. Experiment results shows that with only 10% of the LLaVA-665K data, COMPACT matches or exceeds the full dataset’s performance across a variety of multimodal benchmarks. 

This paper presents a practically useful and empirically strong method. The idea of compositional capability tuning is promising and clearly validated by experiments. 
However, conceptual and theoretical foundations remain unclear. Several core concepts, such as task complexity, informativeness, information density, and k-value, are used interchangeably without rigorous justification. The mapping from "number of atomic capabilities" to "actual complexity" is assumed rather than demonstrated. In addition, the capability definitions are hand-picked without theoretical or empirical grounding. The analysis section provides statistics but not mechanism-level explanations. As a result, important questions remain unanswered: Why does COMPACT help? Which capabilities benefit? Why does improvement transfer to tasks outside the covered perceptual abilities? 

If supplemented with theoretical evidence, more in-depth analysis and argumentation, this work has the potential to become a very influential contribution, and I will raise your rating.

### Strengths
1. "Atomic-to-complex visual capability tuning" is a novel and valuable problem. This paper addresses a real gap in current data curation approaches for MLLM visual instruction tuning. The method is simple yet impactful.
2. The proposed COMPACT dataset demonstrates significantly higher task complexity and achieves 100% full-data performance using only 10% of the original dataset. The authors validate results on 16 complex multimodal tasks, showing clear improvements over baselines.
3. This paper have good motivation via dataset analysis. The paper conducts a detailed study on the complexity distribution of existing datasets. Especially Figure 1, which shows overrepresentation of low-k samples, provides compelling motivation for why higher-complexity samples are needed.
4. The proposed synthetic data recipe is easy to implement and clearly "works in practice". The method directly offers a more data-efficient way to perform visual instruction tuning.

### Weaknesses
1. The paper repeatedly uses the terms "task complexity", "informativeness", "effective use of information content", and "k-value" as if they were equivalent. Is informativeness = complexity? Is task complexity = number of atomic capabilities? Why is k=3 more “complex” than k=1 in a meaningful sense?
The paper does not provide a theoretical justification nor an empirical validation for these assumptions. As a result, the k-value appears arbitrary and not a reliable measure of complexity.

2. The atomic capabilities define only basic perceptual skills, making the notion of "complex tasks" overly simplistic. COMPACT’s complexity is defined solely as combinations of perception + attributes + spatial relations. This is a very narrow interpretation of "complexity", and does not align with real-world multimodal complexity, which includes OCR, counting, commonsense, math, reasoning, etc.

3. The choice of the categories seems ad-hoc. Table 1 says 

> "We identify 10 atomic capabilities that are necessary for general visual reasoning."

Line 189-190:

> "... but instead provide sufficient coverage of the multimodal task space and to systematically combine tasks of increasing complexities."
 
We don't understand how you concluded that "the 10 atomic capabilities provide sufficient coverage of the multimodal task space". What evidence supports the conclusion? 

The paper does not explain why these are necessary, why others are excluded, and whether the taxonomy is derived from theory, data statistics, or prior work. The capability selection appears subjective and weakens the foundation of the method.

4. No explanation for where the improvement comes from. Benchmarks like MM-Vet and MMStar include tasks requiring OCR, math, logic, world knowledge, far beyond COMPACT’s three perceptual dimensions. Since COMPACT does not train OCR/math/logic abilities, why does it improve these tasks? The paper does not analyze this cross-capability transfer.

5. No breakdown showing whether COMPACT improves perception-only tasks vs. all task categories. Without category-level gains, readers cannot tell whether COMPACT only helps “vision-centric” tasks, or whether simple perceptual compositionality generalizes/transfers to OCR/knowledge/logic. If perceptual complexity generalizes broadly, the claim becomes much stronger. But the paper does not provide the crucial analysis. 

6. The paper only provides descriptive correlations, not causal evidence.
The explanation is high-level and speculative. Causal claim "higher complexity $\rightarrow$ higher information density $\rightarrow$ better learning" remains unproven.

7. Training strategy is insufficiently analyzed. The final training mixture is COMPACT + 5% simple LLaVA data. The paper states that both simple and complex samples are necessary, but does not explain why. This likely involves curriculum learning or optimization stability, but the paper provides no analysis or validation.

8. Analysis section is statistics only, not mechanism. The paper largely reports distributions (k-distribution, capability distribution, correlations) but does not explain the mechanism of improvement.
Readers want to know why this works, not just which ablation performs best.

### Questions
NA

### Soundness
2

### Presentation
2

### Contribution
2
