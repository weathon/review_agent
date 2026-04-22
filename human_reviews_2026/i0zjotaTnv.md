# Not How You Think, It's What You See: Decoupling Perception from Reasoning

- Avg Score: 2.00
- Decision: Reject
- Scores: 0, 4, 2, 2

## Abstract
The ability of Vision-Language Models (VLMs) to reason depends on a complex interplay between visual perception and abstract cognition. While it is widely recognized that perception is a significant bottleneck, systematically diagnosing how it fails and developing methods to unlock latent reasoning capabilities remains a key challenge. To address this, we introduce a cognitively-inspired framework that decomposes VLM behavior through four distinct paradigms: 1) Direct Visual Rule Learning (holistic processing), 2) Deductive Rule Learning (explicit rule extraction), 3) Componential Analysis (CA), which decouples perception by reasoning over task-agnostic textual descriptions, and 4) Interactive Componential Analysis (ICA), which introduces a feedback loop for targeted visual probing. Our framework's emphasis on task-agnostic decomposition and cognitive parallels provides a unique lens for analysis compared to prior decoupling efforts. Applying this framework across an expanded suite of benchmarks, we conduct a comprehensive evaluation on both proprietary and open-source multi-image VLMs. Our results confirm that perception is a primary bottleneck and show that our CA and ICA paradigms yield substantial performance gains, unlocking the latent reasoning abilities of powerful LLMs. Crucially, ICA demonstrates that an interactive loop can resolve fine-grained visual ambiguities that static descriptions cannot, outperforming the non-interactive CA approach. Our work provides a robust diagnostic toolkit for the community and offers concrete architectural insights, demonstrating that interactive, decoupled systems are a promising path toward more general and capable visual intelligence.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper proposes a framework with four different approaches to prompt VLM for visual rule learning tasks, demonstrating that models are primarily bottlenecked by a visual description task rather than a deductive reasoning task. When evaluating specific subsets of the Bongard-OW, Bongard-HOI, and Winoground datasets, it is found that performance can be improved by the approaches that first try to decompose visual rule learning into visual description and deductive reasoning.

### Strengths
The paper investigates some significant limitations of current VLM capabilities.

### Weaknesses
Unfortunately, the paper has several severe weaknesses:

1.  The methodology is not presented well. The motivation for the "cognitively-inspired" prompting strategies is located in the middle of the related work section rather than at the beginning of the introduced method. Additionally, there is no discussion of the necessary assumptions or potential limitations of prompting the VLM to solve presumed subtasks instead of the original task. The decompositionality of a task into such subtasks is a strong assumption and limitation that must be properly discussed and justified. Specifically, A.5.3.1 shows that, for the visual description subtask, the VLM is given many additional context examples of specific visual concepts from the Bongard-OW and Bongard-HOI datasets. This completely undermines the claim of task agnosticism or generalizability of the contribution. Interestingly, it is also an intended feature of the original Bongard problems that are hardly decomposable into perception and reasoning, as a useful description of a single image heavily depends on the patterns found across all of them. (E.g. discussed in Depeweg et al., 2024)
2.  The primary area and use case of the overall contribution are difficult to understand. The "evaluation framework" seems most similar to a benchmark, yet the authors emphasize improving visual intelligence with their prompting strategy. It is clearly not about causal reasoning, which was selected during the submission process.
3.  The curation process of the dataset, which is used for evaluation, appears to be arbitrary. No justification is provided for the selection of the subset of Bongard-OW, and there is no discussion of potential difficulties associated with evaluating a subset of public datasets, which might also conflict with the original design of the dataset.
4.  The paper also lacks novelty, as there are several works explicitly assessing the VLM struggle with visual perception tasks (Geigle et al., 2024; Gou et al., 2024; Kamath et al., 2023; Rahmanzadehgervi et al., 2024; Zhang et al., 2024; Zhou et al., 2023; Wang et al., 2024) as well as evaluating VLM on Bongard problems (Małkiński et al., 2025; Wüst et al., 2025).

If I have completely misunderstood the scope of the paper, I am willing to increase my score; however, at this stage, the paper definitely lacks scientific clarity.

### Questions
1.  Why is only 1/4 of Bongard-OW evaluated?
2.  How are VLM responses to tasks other than classification compared to the ground truth? Is there any form of Human Verification or LLM judging involved?
3.  Where do the captions for the Winoground score (A.4.4 and A.6) come from? If they are ground-truth, what can we conclude from CA outperforming DRL?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a cognitively-inspired framework to decouple perception and reasoning in Vision-Language Models (VLMs), featuring four paradigms: Direct Visual Rule Learning (DVRL), Deductive Rule Learning (DRL), Componential Analysis (CA), and Interactive Componential Analysis (ICA). It evaluates proprietary and open-source VLMs on benchmarks including Bongard-OW, Bongard-HOI, and Winoground, confirming that perception is a primary bottleneck for visual reasoning. The CA and ICA paradigms, which leverage task-agnostic textual descriptions and interactive feedback loops respectively, achieve state-of-the-art performance, unlocking latent reasoning capabilities in LLMs. Key contributions include a diagnostic toolkit for VLMs, a modular decoupling method applicable to diverse architectures, comprehensive empirical validation of the perception bottleneck, and demonstration of interactive systems as a promising direction for visual intelligence.

### Strengths
(1) Proposes an innovative cognitively-inspired framework that models human problem-solving strategies, filling the gap of systematic perception-reasoning decoupling in VLMs.

(2) Conducts comprehensive evaluations across diverse benchmarks (abstract rule discovery, HOI reasoning, compositional grounding) and model types, ensuring robust results.

(3) The CA/ICA paradigms enable multi-image reasoning for single-image architectures and visual reasoning evaluations for text-only LLMs, expanding application scenarios.

(4) In-depth ablation studies isolate perception and reasoning, providing clear insights into model bottlenecks.

(5) Delivers state-of-the-art performance on key benchmarks, demonstrating practical value of the proposed methods.

### Weaknesses
(1) The componential paradigms (CA/ICA) rely heavily on language as an intermediate representation, but the paper lacks sufficient discussion on limitations in tasks requiring non-verbalizable reasoning (e.g., geometric reasoning), making it unclear how the framework performs in such scenarios.

(2) Evaluation of open-source models is incomplete: most open-source VLMs are only tested under CA, with no attempts to adapt DVRL/DRL to their input constraints (e.g., batch processing images), limiting the comparability of paradigm effectiveness across model types.

(3) The selection basis for the 500-case Bongard-OW subset is not clearly explained. The paper mentions it is derived from the first 250 samples with two test cases each, but fails to justify why this subset is representative or whether it introduces selection bias.

(4) Implementation details for prompt engineering are insufficient. While prompts are provided in the appendix, there is no analysis of how minor prompt variations affect performance, which is critical for reproducibility given the sensitivity of VLMs to prompts.

(5) Hardware configuration details are vague. The paper mentions using NVIDIA GPUs (2080Ti, 3090, 6000 Ada) but does not specify which GPU is used for which model, leading to uncertainty about potential performance impacts from hardware differences.

(6) Comparisons with state-of-the-art baselines are incomplete. For example, on Winoground, the paper only compares with a few CoT-based methods but omits recent strong performers (e.g., Claude 3 Opus, Gemini Ultra) or specialized compositional reasoning models.

(7) The number of interaction steps in ICA is not optimized. The paper uses a single feedback loop but does not test whether multiple rounds of probing further improve performance or lead to diminishing returns.

(8) There is no quantitative metric for description quality. The paper claims GPT-4o's descriptions are high-fidelity but does not use objective metrics (e.g., BLEU, ROUGE, or human evaluation) to validate this, making the link between description quality and reasoning performance less rigorous.

(9) Cross-domain generalization is under-tested. The framework is evaluated on natural image benchmarks but not on complex domains like scientific chart interpretation, mathematical reasoning, or medical imaging, leaving its applicability to specialized fields unclear.

(10) Computational cost and latency analysis is missing. The multi-stage CA/ICA paradigms are likely more computationally expensive than end-to-end methods, but the paper provides no data on inference time, memory usage, or scalability for large-scale applications.

(11) Prompt sensitivity is not investigated. The paper uses fixed prompts for all experiments but does not analyze how changes in prompt structure, detail, or tone affect paradigm performance, which is essential for understanding the framework's robustness.

(12) Error analysis is limited. The paper only provides 15 misclassified cases, without systematic categorization of errors (e.g., perceptual vs. reasoning errors) or analysis of whether errors are consistent across models or benchmarks.

(13) Robustness to adversarial examples is untested. The paper does not evaluate how the framework performs when images contain noise, distortions, or adversarial perturbations, which is critical for real-world deployment.

(14) The optimality of CA's JSON description structure is unvalidated. The paper uses a specific hierarchical JSON format but does not compare it with other representation formats (e.g., free-text descriptions, structured lists) to show that JSON is the most effective for reasoning.

(15) ICA's question generation strategy is not detailed. The paper states the reasoning LLM formulates targeted questions but does not explain how questions are prioritized or whether the question generation process itself introduces biases.

(16) Rule complexity analysis is missing. The paper does not quantify the complexity of rules in different benchmarks, making it unclear whether the framework's performance gain varies with rule difficulty.

(17) The scope of text-only LLMs' applicability is undefined. The paper shows text-only LLMs perform well with high-quality descriptions but does not specify the types of visual tasks where this modality transfer fails.

(18) Comparisons with other decoupling frameworks are superficial. The paper distinguishes itself from Prism and CoT methods but does not provide head-to-head performance comparisons or detailed analysis of architectural differences.

(19) Perception module evaluation is absent. The paper focuses on reasoning performance but does not evaluate the perception module in isolation (e.g., accuracy of object detection, attribute recognition), making it hard to quantify the exact impact of perceptual errors.

(20) Reasoning module selection is not justified. The paper uses powerful LLMs (e.g., GPT-4o) as reasoning engines but does not explain why these models are chosen or whether weaker LLMs would still yield effective results.

(21) Multilingual applicability is untested. The framework uses English descriptions and prompts but does not evaluate performance in other languages, limiting its relevance for non-English use cases.

(22) Few-shot learning limits are not explored. The paper uses few-shot prompting but does not test how the framework performs with fewer or zero shots, which is important for low-resource scenarios.

(23) Training data influence is not discussed. The paper does not analyze whether VLMs trained on specific datasets (e.g., more HOI data) perform better under the framework, leaving the impact of training data distribution unclear.

(24) Evaluation metrics are overly simplistic. The paper relies primarily on classification accuracy, without considering other metrics (e.g., precision, recall, F1-score) that are more informative for imbalanced datasets.

(25) Dataset bias analysis is insufficient. Bongard-OW's subset has a dominant category (ID 0, 73%), but the paper does not discuss how this bias affects model performance or generalization.

(26) ICA's feedback loop efficiency is unanalyzed. The paper does not measure how much time the interactive probing adds or whether the added complexity is justified by performance gains.

(27) Image resolution impact is untested. The paper uses default or 1024px images but does not evaluate how lower or higher resolutions affect description quality and subsequent reasoning.

(28) Description consistency is not evaluated. The paper does not check whether the same image generates consistent descriptions across multiple runs, which is important for the framework's reliability.

(29) Rule extraction accuracy is not quantified. The paper evaluates classification accuracy but not the accuracy of the extracted rules themselves (e.g., how well extracted rules match ground-truth rules).

(30) Human reasoning comparison is superficial. The paper mentions human average accuracy but does not compare the framework's reasoning process with human reasoning steps (e.g., sequence of analysis, focus on details).

(31) Open-source model performance gap analysis is inadequate. The paper notes lower CA accuracy for some open-source models but does not deeply investigate whether the gap stems from poor description generation, weak reasoning, or both.

(32) Framework scalability to video is unaddressed. The paper focuses on static images but does not discuss how to extend the paradigms to video reasoning, where temporal dynamics add complexity.

(33) Ethical considerations are missing. The paper does not discuss potential biases in description generation (e.g., racial, gender biases) or how these biases could propagate to reasoning outcomes.

(34) Hyperparameter tuning is not reported. The paper uses zero temperature for all models but does not explain why this choice is optimal or whether tuning temperature affects performance.

(35) Cross-model generalization of paradigms is untested. The paper evaluates a fixed set of models but does not test whether the paradigms perform consistently across newly developed VLMs.

(36) The impact of description length is unanalyzed. The paper does not test whether shorter or longer descriptions affect reasoning performance, leaving the optimal description length unclear.

(37) Benchmark-specific optimizations are not disclosed. It is unclear whether the paradigms are tailored to the tested benchmarks or if they generalize to unseen visual reasoning tasks.

(38) Collaboration between perception and reasoning modules is not modeled. The paper treats the modules as separate but does not explore how bidirectional communication beyond ICA's feedback loop could further improve performance.

(39) Low-resource language model adaptation is untested. The paper uses high-resource models but does not evaluate how the framework performs with low-resource VLMs or LLMs.

(40) Real-world application case studies are absent. The paper demonstrates performance on benchmarks but provides no case studies of applying the framework to practical tasks (e.g., image retrieval, visual question answering).

### Questions
**To facilitate discussions during the Rebuttal phase, authors are advised to respond point-by-point (indicating the question number).**

(1) Could you provide specific examples of the JSON descriptions generated in CA (e.g., 3-5 full descriptions for Bongard-OW samples) to demonstrate the structure and detail of the task-agnostic representations?

(2) How was the number of interaction steps in ICA determined? Have you tested whether 2-3 rounds of probing further improve performance, or does a single round already reach diminishing returns?

(3) For open-source models that do not support large multi-image inputs, have you attempted to adapt DVRL/DRL by batch-processing images or using image embeddings? If not, what technical barriers prevented this adaptation?

(4) What specific criteria were used to select the 500-case Bongard-OW subset? Could you provide statistical evidence (e.g., rule category distribution, difficulty distribution) that this subset is representative of the full dataset?

(5) Could you add comparisons with recent state-of-the-art VLMs (e.g., Claude 3 Opus, Gemini Ultra, Qwen-VL Max) on all benchmarks to better contextualize the performance of your framework?

(6) Have you tested the impact of prompt variations (e.g., changing the level of detail, rephrasing instructions) on paradigm performance? If so, please provide quantitative results; if not, could you explain why prompt sensitivity is not a concern?

(7) Could you provide detailed computational cost data (e.g., inference time per sample, memory usage) for CA, ICA, DVRL, and DRL, and compare them with end-to-end VLM approaches?

(8) Have you used objective metrics (e.g., BLEU, ROUGE, human evaluation scores) to quantify the quality of image descriptions generated by different VLMs? If yes, please share the results; if not, could you conduct such an evaluation to validate the claim of "high-fidelity" descriptions?

(9) Why do performance variations exist across commonsense categories (Table A.7)? For example, why does GPT-4o achieve 100% accuracy on "Taste/Nutrition/Food" while Gemini 2.0 only achieves 85.71%?

(10) Could you extend the error analysis to include systematic categorization of errors (e.g., perceptual errors, rule extraction errors, rule application errors) and report the distribution of error types across models and benchmarks?

(11) Have you evaluated the framework's performance on adversarial examples (e.g., noisy images, distorted objects) or out-of-distribution samples? If yes, please share the results; if not, could you explain the framework's robustness in real-world scenarios?

(12) How does the JSON description format in CA compare to other representation formats (e.g., free-text descriptions, structured bullet points) in terms of reasoning performance? Could you provide a head-to-head comparison?

(13) Could you detail the question generation strategy in ICA? How does the reasoning LLM prioritize which visual details to probe, and how do you ensure the questions are not redundant or irrelevant?

(14) Have you quantified the complexity of rules in the benchmarks (e.g., number of attributes, logical relationships)? If so, how does the framework's performance correlate with rule complexity?

(15) For text-only LLMs, what types of visual tasks do you find the modality transfer fails? Could you provide specific examples and explanations?

(16) Could you conduct a detailed head-to-head comparison with Prism and state-of-the-art CoT methods, including architectural differences, computational costs, and performance trade-offs?

(17) Have you evaluated the perception module in isolation (e.g., accuracy of object detection, attribute recognition, spatial relationship identification)? If yes, please share the results to quantify the impact of perceptual errors on overall performance.

(18) Why did you choose specific LLMs (e.g., GPT-4o, Gemini 2.0) as reasoning engines in CA/ICA? Could you test weaker LLMs (e.g., Llama3-8B, Mistral-7B) to see if the framework's effectiveness is dependent on reasoning module strength?

(19) Have you tested the framework's performance in non-English languages (e.g., Chinese, Spanish)? If yes, please share the results; if not, could you discuss potential challenges for multilingual adaptation?

(20) How does the framework perform with fewer shots (e.g., 1-shot, 2-shot) compared to the reported few-shot setting? Could you provide results to demonstrate its performance in low-resource scenarios?

(21) Have you analyzed how the training data distribution of VLMs affects framework performance? For example, do VLMs trained on more HOI data perform better on Bongard-HOI under your paradigms?

(22) Could you supplement the evaluation with additional metrics (e.g., precision, recall, F1-score, confusion matrices) to provide a more comprehensive view of performance, especially for imbalanced subsets?

(23) How does the dominant category (ID 0) in the Bongard-OW subset affect model training and generalization? Could you test the framework on a more balanced subset to validate robustness?

(24) Could you provide a cost-benefit analysis of ICA's feedback loop, including the additional inference time and memory usage compared to CA, and whether the performance gain justifies the added complexity?

(25) Have you tested the impact of image resolution (e.g., 512px, 2048px) on description quality and reasoning performance? If yes, please share the results; if not, could you discuss how resolution affects the framework?

(26) Have you evaluated the consistency of description generation (e.g., same image, multiple runs)? If yes, please provide quantitative results (e.g., consistency rate); if not, could you explain how you ensure description reliability?

(27) Could you quantify the accuracy of rule extraction (e.g., similarity between extracted rules and ground-truth rules using semantic similarity metrics)? This would strengthen the link between rule quality and classification performance.

(28) Could you compare the framework's reasoning process with human reasoning steps (e.g., sequence of analysis, focus on key details) using qualitative case studies?

(29) For open-source models with low CA accuracy (e.g., Llama-Vision-11B), have you conducted a detailed analysis to determine whether the gap stems from poor description generation, weak reasoning, or both? Could you provide evidence (e.g., replacing descriptions with GPT-4o's for these models)?

(30) How would you extend the paradigms to video reasoning? Could you outline a preliminary design and discuss potential challenges (e.g., handling temporal dynamics, reducing computational cost)?

(31) Have you analyzed potential biases in description generation (e.g., racial, gender, cultural biases) and their impact on reasoning outcomes? If yes, please share the results; if not, could you discuss how to mitigate such biases?

(32) Why did you use zero temperature for all models? Have you tested other temperature values (e.g., 0.3, 0.7) and their impact on paradigm performance?

(33) Could you test the framework on newly developed VLMs (e.g., Llama-4, Mistral Large) to demonstrate its cross-model generalization ability?

(34) Have you analyzed the impact of description length on reasoning performance? For example, do shorter, more concise descriptions perform as well as longer, detailed ones?

(35) Are the paradigms tailored to the tested benchmarks, or do they generalize to unseen visual reasoning tasks? Could you test on a new benchmark (e.g., a custom dataset) to validate generalization?

(36) Could you explore bidirectional communication between perception and reasoning modules beyond ICA's feedback loop? For example, could the perception module proactively flag ambiguous details to the reasoning module?

(37) Have you tested the framework with low-resource VLMs/LLMs (e.g., models with <7B parameters)? If yes, please share the results; if not, could you discuss the framework's accessibility for resource-constrained users?

(38) Could you provide case studies of applying the framework to practical real-world tasks (e.g., image retrieval, visual question answering, medical image analysis) to demonstrate its practical value?

(39) For Winoground, why do the Image Score and Group Score remain lower than the Text Score even with ICA? Could you analyze the specific challenges that prevent further improvements in these metrics?

(40) Could you provide detailed reproducibility instructions, including exact prompt templates, model versions, hardware specifications, and step-by-step evaluation pipelines, to ensure other researchers can replicate your results?

### Soundness
3

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
This paper proposes a cognitively inspired framework that decouples perception from reasoning for multi-image visual reasoning. It introduces four paradigms: DVRL (holistic, all images at once), DRL (rule extraction → rule application), CA (reasoning over task-agnostic textual descriptions), and ICA (CA with an interactive feedback loop where the reasoner can look again). The framework is evaluated on Bongard-OW, Bongard-HOI, and Winoground, showing a consistent trend DVRL < DRL < CA, and additional gains from ICA. The central claim is that many VLMs are limited primarily by perception bottlenecks; when perception is bypassed via high-fidelity descriptions, downstream reasoning can be very strong, even for text-only LLMs.

### Strengths
- Clear problem decomposition. CA creates a clean separation between perceptual description and downstream reasoning; ICA adds a bidirectional loop that lets reasoning guide perception. The design is well motivated and consistently effective across tasks.

### Weaknesses
- The evaluation centers on GPT-4o, Gemini 2.0, and a set of 2024-era open models (Qwen2.5, Llama-Vision, Pixtral, etc.). There is no direct evaluation of 2025 releases such as GPT-5, Gemini-2.5, R1-series variants, or Qwen3-VL, despite the paper positioning itself as a diagnostic/prognostic framework. Given that limitations like perception bottlenecks were already raised by BLINK and PRISM, the absence of recent frontier models makes it hard to judge whether the proposed remedies remain necessary at the current frontier.

### Questions
- Can you include a 2025 refresh (e.g., GPT-5, Gemini-2.5, R1-series 2025, Qwen3-VL) to validate whether the perception bottleneck and CA/ICA gains persist at the current frontier? A small but representative subset on Bongard-OW/HOI and Winoground would suffice.

- Beyond text intermediates. Do you foresee non-linguistic symbolic or visual-token intermediates that retain CA’s separability but cover BLINK-type cases (depth/correspondence) better?

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
3

### Summary
The paper introduces four "cognitively inspired" evaluation paradigms, DVRL, DRL, CA, and ICA, to explicitly decouple visual perception from symbolic reasoning in multi-image reasoning tasks, like Bongard-style benchmarks (OW/HOI) and Winoground.

### Strengths
1. The reported results show clear, monotonic improvements from DVRL to DRL and to CA, and additional boosts with ICA, showing that the framework is practically useful.

### Weaknesses
1. The contribution is primarily a workflow. The paper does not convincingly justify why this workflow design works, like experiments for motivation. Ablations on prompt choices, alternative perception backbones, and end-to-end VLMs under matched constraints are also limited, making the empirical case insufficient for a main-track ML venue focused on algorithmic advances.
2. The evaluation subset of Bongard-OW is built from the first 250 items to produce 500 cases, and the class distribution is highly skewed (73% are class 0). This raises concerns about hidden biases. 
3. The systematic study of description granularity/length/structure are underexplored.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
