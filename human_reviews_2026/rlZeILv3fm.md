# $PhyWorldBench$: A Comprehensive Evaluation of Physical Realism in Text-to-Video Models

- Avg Score: 5.50
- Decision: Accept (Oral)
- Scores: 4, 6, 6, 6

## Abstract
Video generation models have achieved remarkable progress in creating high-quality, photorealistic content. However, their ability to accurately simulate physical phenomena remains a critical and unresolved challenge. This paper presents $PhyWorldBench$
, a comprehensive benchmark designed to evaluate video generation models based on their adherence to the laws of physics. The benchmark covers multiple levels of physical phenomena, ranging from fundamental principles like object motion and energy conservation to more complex scenarios involving rigid body interactions and human or animal motion. Additionally, we introduce a novel "Anti-Physics" category, where prompts intentionally violate real-world physics, enabling the assessment of whether models can follow such instructions while maintaining logical consistency. Besides large-scale human evaluation, we also design a simple yet effective method that could utilize current MLLM to evaluate the physics realism in a zero-shot fashion. We evaluate 10 state-of-the-art text-to-video generation models, including five open-source and five proprietary models, with a detailed comparison and analysis. we identify pivotal challenges models face in adhering to real-world physics. Through systematic testing of their outputs across 1,050 curated prompts—spanning fundamental, composite, and anti-physics scenarios—we identify pivotal challenges these models face in adhering to real-world physics. We then rigorously examine their performance on diverse physical phenomena with varying prompt types, deriving targeted recommendations for crafting prompts that enhance fidelity to physical principles.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a framework for evaluating video generation models in terms of their understanding and generation of physical concepts. The authors establish a comprehensive and fine-grained categorization of physical dimensions, and employ GPT-o1 as the evaluation model to assess ten different models from the perspectives of Semantic Adherence and Physical Commonsense. The evaluation results are shown to be consistent with human judgments.

### Strengths
1. Establishing a comprehensive and fine-grained categorization of physical dimensions which is larger than existing benchmarks.
2. The evaluation results are shown to be consistent with human judgments.

### Weaknesses
1. Apart from introducing a more fine-grained implementation in the dimension categorization, I do not see any genuine innovation in this paper. The authors’ claim regarding Anti-Physics has already been presented and evaluated in paper: Impossible Videos. Another point emphasized by the authors—using prompts of different granularities for the same scenario and concluding that finer-grained prompts lead to better results—has also been thoroughly discussed in paper VBench-2.0.
2. In addition to the lack of novel and valuable conclusions, the evaluation methodology used in this work is much less rigorous compared to previous approaches such as VBench-2.0 and PhyGenBench. Notably, the authors do not consider the impact of camera motion on physical phenomena. For example, in the case of an apple falling, if the camera moves synchronously with the apple, how does the model determine whether the apple is truly falling? The paper merely utilizes a stronger MLLM, GPT-o1, which superficially appears to yield more accurate results. However, the use of a closed-source model introduces a critical flaw: as the model is updated, previous results may become irreproducible, which is a fatal issue for any benchmark.

### Questions
See Weakness.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces PhyWorldBench, a comprehensive benchmark for evaluating physical realism in text-to-video generation models. The benchmark contains 1,050 prompts spanning 10 main physics categories (fundamental, composite, and anti-physics), with each category divided into 5 subcategories and 7 scenarios. The authors evaluate 12 state-of-the-art models (5 proprietary, 7 open-source) and propose CAP (Context-Aware Prompt), a method using MLLMs for automated evaluation. Results show that even the best models (Pika 2.0 achieving 26.2% success rate) struggle significantly with physical realism, particularly with complex interactions and anti-physics scenarios.

### Strengths
- Important and timely problem. The paper tackles a crucial gap in bridging video generation and physical reasoning. The motivation is clear and aligns well with current research trends.
- Comprehensive experimental design. The evaluation is thorough - 12 models tested, 12,600 videos generated, large-scale human annotation via MTurk. They evaluate across a diverse set of tasks, object falling, rolling, fluid dynamics, etc. 
- Practical automated evaluator. The CAP method achieves 80.3% ROC-AUC for semantic adherence and 75.1% for physical commonsense (Table 2), which is pretty solid for zero-shot evaluation. 
- Clear organization and presentation. The hierarchical organization is logical and makes the benchmark easy to understand and extend. The tables and plots are easy to follow.

### Weaknesses
- No end-to-end pipeline for new models. The paper doesn’t provide a unified automatic evaluation framework for new video generation models. While CAP is proposed as an automatic evaluator, there's no clear standalone pipeline or released code/API for researchers to evaluate their own models. The paper mentions "we will open-source our codebase" (reproducibility statement) but it's unclear if this includes an easy-to-use evaluation script. For a benchmark paper, providing a simple evaluate interface would greatly increase adoption. The current workflow seems to require manual video generation then evaluation, which is cumbersome. For practical use, an automated submission or leaderboard system would make the benchmark much more usable.
- Metric definitions design lack justification. A few metrics (like the physics consistency score) are described in high-level terms but not fully formalized. It’s hard to know how reproducible they are from the text alone, e.g., whether they use learned physical estimators or ground-truth simulations. The Yes/No binary evaluation (Section 3.1) might be too coarse-grained - a partial physics violation might deserve a score between 0 and 1. The paper acknowledges models often "rationalize" actions rather than fail outright (Section 4.4), suggesting a more nuanced scoring could capture important phenomena.
- Limited real-world coverage. Most of the benchmark focuses on synthetic data (e.g., MuJoCo or Unity scenes). It would be great to see more real videos or robotic interactions to test generalization.
- The scope of the benchmark is somehow overlapped with previous works, e.g., the inclusion of an "anti-physics" category is similar as “counterfactual prompts” in [1]. 

[1] T2VPhysBench: A First-Principles Benchmark for Physical Consistency in Text-to-Video Generation. 2025.

### Questions
- The paper mentions models sometimes "rationalize" physics violations by generating static scenes. Could you quantify how often this happens across models? This seems like an important failure mode worth analyzing systematically.
- Have you considered releasing a "difficulty rating" for each prompt based on model performance? This could help researchers identify particularly challenging scenarios to focus on.

### Soundness
3

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
4

### Summary
This paper introduces PhyWorldBench, a comprehensive, large-scale benchmark for evaluating physical realism in Text-to-Video (T2V) models. It features 1050 prompts systematically categorized across 10 physics domains and uniquely includes an "Anti-Physics" category to test genuine physical understanding versus mere pattern imitation from training data. The study employed extensive human evaluation (via Amazon Mechanical Turk) on 12 leading models and proposed a novel automated MLLM evaluation method called CAP (Context-Aware Prompting). Findings show that all current T2V models have very low physical success rates (Pika at 26.2% being the highest), and notably, they tend to revert to realistic physical phenomena when instructed to perform anti-physical actions.

### Strengths
1. The benchmark offers the most comprehensive and systematic coverage of physical phenomena
2. he proposed CAP automated MLLM evaluator significantly improves the accuracy of machine-based physical assessment by using a context-aware prompt, offering a cost-effective and reproducible alternative to expensive human studies.

### Weaknesses
1. The automated CAP evaluator suffers from an "aesthetic bias," leading to ranking discrepancies with the human gold standard for high-quality, closed-source models. This limits its reliability as a trustworthy, scalable proxy for future research.

2. Many of the investigated physics phenomena (e.g., fluid viscosity, electromagnetic forces) are challenging to assess accurately by eye in a video. This visual complexity makes human scoring error-prone and may also challenge the perceptual accuracy of the MLLM evaluator.

3. The evaluation primarily focuses on whether the final physical state is correct, but it lacks a detailed metric to measure if the physical process is smooth, coherent, and temporally consistent. This overlooks a critical challenge in dynamic T2V modeling.

### Questions
1. The CAP automated evaluator, while highly accurate in classification, is noted to possess an "aesthetic bias" that causes ranking discrepancies with human judgment; given the non-scalability of human evaluation, what specific plans do the authors have to mitigate or quantify this aesthetic bias in the MLLM, ensuring the CAP tool reliably focuses on physical correctness rather than visual style?

### Soundness
3

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
4

### Summary
Summary:   
This paper addresses the critical limitation of existing text-to-video (T2V) generation models—their inability to adhere to physical laws despite producing photorealistic content—by proposingPhyWorldBench, a comprehensive benchmark for evaluating physical realism in T2V models.  

Contributions:  
（1）Development of a comprehensive physics benchmark: PhyWorldBench fills the gap of lacking holistic tools to evaluate physical realism in text-to-video models. Its hierarchical structure (three levels, 10 main categories, 50 subcategories) covers diverse physical scenarios—from basic motion and energy conservation to anti-physics scenarios—and includes 1,050 well-curated prompts with variations, enabling systematic testing of models’ physical reasoning capabilities.  
（2）Creation of a zero-shot automatic evaluator: The proposed Context-Aware Prompt method resolves the limitations of traditional evaluation. By guiding large language models to explicitly assess AI-generated videos and use chain-of-thought reasoning, it achieves high accuracy in evaluating physical realism, providing a scalable and objective alternative to costly human evaluation.  
（3）Extensive evaluation of state-of-the-art models: The paper conducts large-scale tests on 12 leading text-to-video models, generating 12,600 videos. This evaluation identifies key challenges models face—such as struggling with complex interactions and prioritizing cinematic aesthetics over physics—and quantifies performance gaps across different physics categories, offering clear directions for future model improvement.

### Strengths
Strengths:   
（1）Originality  
It proposes a three-tier (Fundamental, Composite, Anti-Physics) hierarchical structure with 10 main physics categories, going beyond prior benchmarks by using Anti-Physics scenarios to test true physical understanding. The CAP evaluator creatively combines LLMs with domain constraints, separating aesthetics from physics via two-step reasoning to fix traditional evaluators’ inaccuracies. It also offers three prompt variants to test model adaptability, filling gaps of static-prompt benchmarks.  
（2）Quality  
Its benchmark is built via a rigorous three-stage process (literature/expert category definition, LLM-human prompt generation, expert validation) to ensure 1,050 prompts are diverse and accurate. CAP aligns with human evaluation (ROC-AUC: 80.3 for SA, 75.1 for PC) and outperforms baseline LLMs. Experiments cover 12 models (5 proprietary, 7 open-source) with 12,600 videos and systematic analyses to avoid cherry-picking.  
（3）Clarity  
It follows a clear "problem-solution-validation" flow, with the introduction outlining existing benchmark limits and methodology linking components to specific issues. Technical details (CAP’s reasoning, "Yes/No" criteria) are explained in plain language without jargon. Consistent terminology and condensed key findings make it readable for physics and AI researchers.  
（4）Significance  
As a standardized tool, its open benchmark and CAP provide universal metrics for tracking text-to-video physical realism. Prompt design insights (e.g., Physics-Enhanced Prompts boost PC) guide real-world uses like scientific visualization. It shifts evaluation focus to physical correctness, identifies model weaknesses, and reduces educational misinformation risks.

### Weaknesses
Weaknesses:   
（1）CAP Evaluator’s Aesthetic Bias  
The Context-Aware Prompt (CAP) evaluator favors visually polished videos (e.g., smooth lighting, dynamic camera movement) over physically correct ones, which conflicts with the benchmark’s goal of prioritizing physical plausibility. However, the paper doesn’t quantify this bias or fix it. For example, a visually vivid but gravity-violating floating apple video might get a higher score than a plain yet physically correct one.  
（2）Inadequate Niche Fundamental Physics Analysis  
PhyWorldBench focuses on common physics (e.g., free fall) but ignores niche subcategories of fundamental physics, like pressure-dependent phase changes (water boiling at high altitude) or non-uniform heat transfer (a metal rod heating unevenly). It only reports broad category performance, hiding which specific principles models struggle with.  
（3）No Long-Duration Video Validation  
The paper doesn’t specify video duration or test how performance scales with length. Physical inconsistencies (e.g., trajectory drift) worsen in longer videos, which are needed for real uses (e.g., education). Without this data, the benchmark’s real-world value is limited.  
（4）Lack of Specialized Benchmark Comparisons  
The paper only compares to general physics benchmarks (e.g., VideoPhy) but not specialized ones like Morpheus (real-world experiment focus) or T2VPhysBench (first-principles physics). This hides PhyWorldBench’s unique value.

### Questions
Questions:   
（1）Questions About CAP Evaluator’s Aesthetic Bias  
You note the Context-Aware Prompt evaluator prefers visually polished videos but provide no quantitative data on this bias. How often does this preference lead to misclassifying physically incorrect yet visually polished videos as physically plausible? Does a visually vivid video with physical violations score higher than a plain but physically correct one? Have you tested if revising CAP prompts to explicitly ignore aesthetics reduces this bias? Clarifying these points will confirm if CAP’s objectivity is compromised and if adjustments can fix it.  
（2）The comparison with the methods of predecessors is not sufficient.  
A comparison between the dataset and some previous related datasets, such as VideoREPA[1], WISA[2], NewtonGen[3], etc.?  
（3）Niche Fundamental Physics Performance
PhyWorldBench covers 10 main physics categories, but your experiments focus on common phenomena and lack breakdowns for niche subcategories like pressure-dependent phase changes or non-uniform heat transfer. Do you have data on model performance in these niche areas? If not, why were they excluded from detailed analysis? Understanding these gaps will help researchers target specific physical principles models struggle with. If the author's reply is strong and reasonable, I will consider increasing my score.  

[1] VideoREPA: Learning Physics for Video Generation through Relational Alignment with Foundation Models  
[2] Wisa: World simulator assistant for physics-aware text-to-video generation  
[3] NEWTONGEN: PHYSICS-CONSISTENT AND CONTROL-LABLE TEXT-TO-VIDEO GENERATION VIA NEURAL NEWTONIAN DYNAMICS

### Soundness
2

### Presentation
3

### Contribution
3
