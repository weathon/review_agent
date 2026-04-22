# The CoT Encyclopedia: Analyzing, Predicting, and Controlling how a Reasoning Model will Think

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 8, 6, 2

## Abstract
Long chain-of-thought (CoT) is an essential ingredient in effective usage of modern large language models, but our understanding of the reasoning strategies underlying these capabilities remains limited. While some prior works have attempted to categorize CoTs using predefined strategy types, such approaches are constrained by human intuition and fail to capture the full diversity of model behaviors. In this work, we introduce the CoT Encyclopedia, a bottom-up framework for analyzing and steering model reasoning. Our method automatically extracts diverse reasoning criteria from model-generated CoTs, embeds them into a semantic space, clusters them into representative categories, and derives contrastive rubrics to interpret reasoning behavior. Human evaluations show that this framework produces more interpretable and comprehensive analyses than existing methods. Moreover, we show that this understanding translates into measurable improvements on both problem-solving and safety benchmarks. We can predict which strategy a model is likely to use and guide it toward more effective alternatives. Finally, we show that training data format (e.g., free-form vs. multiple-choice) impacts reasoning far more than data domain, highlighting the importance of format-aware model design. In short, the CoT Encyclopedia turns reasoning from a black box into a controllable asset, enabling LLMs that think more clearly, perform more reliably, and act more safely.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces the COT ENCYCLOPEDIA, a novel, bottom-up framework designed to analyze, predict, and control the chain-of-thought (CoT) reasoning in large language models. The method automatically extracts a diverse set of reasoning criteria from model-generated text, uses embedding and clustering to form a coherent taxonomy, and generates interpretable reports. The authors demonstrate the framework's utility through comprehensive human evaluations that show its superiority over predefined analyzers and its ability to measurably improve model performance by steering reasoning toward optimal strategies.

### Strengths
(1) The paper's primary strength is its data-driven, bottom-up methodology for creating a taxonomy of reasoning strategies. This contrasts sharply with prior work using fixed, human-defined categories, allowing the framework to discover emergent and more nuanced reasoning patterns specific to different models and tasks.

(2) Comprehensive Experimental Validation: The claims are supported by extensive experimentation.

(3) Significant Finding on Data Format vs. Domain: The discovery that training data format has a substantially greater impact on reasoning strategies than the content domain is a crucial and non-obvious contribution.

### Weaknesses
(1) All major components of the framework—from the initial ideation of 4,057 criteria (Step 1) to rubric formulation (Step 4) and final classification (Step 5)—are executed through an external proprietary language model (GPT-4o). This design choice embeds a structural reliance on a closed-source system and exposes the analysis to its internal biases. Although the authors acknowledge this limitation and incorporate a multi-evaluator setup to mitigate bias (Appendix B.5), the methodology remains fundamentally shaped by the constraints and opacity of the chosen model.

(2) The paper defines “optimal” reasoning strategies as those correlated with higher frequencies of correct or safe answers within a given dataset (Section 4.1). However, this relationship is correlational rather than causal. It remains ambiguous whether these strategies directly contribute to improved outcomes or merely co-occur with successful reasoning patterns. The experiments stop short of disentangling causation from correlation.

(3) In Section 4.3, classifiers are trained to predict which reasoning strategies perform best for specific question types. These models are developed and evaluated on a fixed set of five benchmarks. Although some cross-domain tests are conducted, the study does not extend to genuinely unseen or out-of-distribution datasets. Consequently, the generalizability and robustness of these classifiers beyond the training domains remain uncertain.

### Questions
Please refer to weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a framework named “CoT Encyclopedia” for analyzing, predicting, and controlling model reasoning patterns from CoT outputs. The authors claim that their approach can automatically extract reasoning criteria, cluster them in a semantic space, and derive rubrics to interpret LLM reasoning strategies. They further argue that this framework provides better interpretability, predictive power, and safety control than previous categorization-based techniques.

### Strengths
- The goal of making LLM reasoning more interpretable and controllable is important and timely.
- The paper provides a large-scale qualitative analysis that could inspire follow-up interpretability research.
- It attempts to link reasoning diversity with data format, a less-explored dimension in CoT studies.
- Presentation is relatively polished and readable, making the high-level idea easy to follow.

### Weaknesses
- The criterion appears to be very important, yet the paper seems to use only a single one. How robust is this criterion to variations from weaker models or the effects of randomness?
- It seems that a prior for a specific reasoning logic has been incorporated for a certain class of questions, and the performance benefits are evident. I am curious how your framework would be applied when the test set does not include CoT outputs.
- Figure 6 is blurry and difficult to read.
- How does the performance of this method compare to instruction optimization techniques such as TextGrad? Additionally, can it lead to the discovery of more effective patterns?
- The finding that "data format is more important than domain" is a valuable insight. However, since these conclusions are drawn primarily from tasks in mathematics and question answering, do these defined "reasoning strategies" remain effective for tasks requiring creativity, open-endedness, or emotional understanding, such as poetry generation or legal argumentation?
- As models continue to evolve, they may develop novel reasoning strategies. This implies that the "Encyclopedia of CoT" is not a one-time creation but requires ongoing updates. What are the anticipated costs and frequency of these updates?
- How are pattern similarity and Q-Q similarity calculated?
- Missing Reference:

[1] Towards reasoning era: A survey of long chain-of-thought for reasoning large language models

[2] Deconstructing long chain-of-thought: A structured reasoning optimization framework for long cot distillation

[3] When more is less: Understanding chain-of-thought length in llms

### Questions
- Given that criterion is presented as highly important, yet only a single one is used, how robust is it against variations from weaker models or the effects of randomness?
- Since a prior for a specific reasoning logic appears to be incorporated for certain questions, leading to clear performance gains, how would the proposed framework be applied to a test set that lacks CoT examples?
- How does the performance of this method compare to instruction optimization techniques like TextGrad, and can it facilitate the discovery of more effective reasoning patterns?
- How are pattern similarity and Q-Q similarity calculated?
- The paper's finding that "data format is more important than domain" is a valuable insight. However, given that this conclusion is based on math and QA tasks, do the defined "reasoning strategies" also apply effectively to tasks that demand creativity, open-endedness, or emotional understanding, such as poetry generation or legal argumentation?
- As models evolve and potentially develop novel reasoning strategies, the "Encyclopedia of CoT" would require continuous updates. What are the anticipated costs and frequency for maintaining and updating this resource?
- Add reference.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces CoT Encyclopedia, a bottom-up, data-driven framework for analyzing and controlling reasoning strategies in large language models (LLMs) performing long Chain-of-Thought (CoT) reasoning. Unlike top-down approaches that rely on predefined strategy types, CoT Encyclopedia automatically extracts reasoning criteria from model-generated CoTs, embeds them into a semantic space, clusters similar dimensions, and builds contrastive rubrics (e.g., top-down vs. bottom-up). It then classifies each CoT response under these rubrics and generates interpretable reasoning reports. ​Experiments across multiple benchmarks (GPQA-Diamond, MMLU-Redux, MATH-500, XSTest, WildGuard, Arena-Hard) show that this framework captures finer-grained reasoning differences across models and tasks, achieving higher interpretability (86% human preference) and consistent performance gains.​ The authors further demonstrate that optimal reasoning strategies can improve both model helpfulness and harmlessness, that question-specific reasoning can be predicted, and that data format (multiple-choice vs. free-form) shapes reasoning behavior more strongly than domain.​ Overall, CoT Encyclopedia provides a scalable taxonomy and control mechanism for reasoning behaviors, contributing to greater interpretability, adaptability, and safety in LLM reasoning.

### Strengths
# 1. Originality and Conceptual Contribution

The work shifts the paradigm from top-down, predefined reasoning taxonomies to a bottom-up, data-driven discovery of reasoning strategies. This formulation is original and theoretically meaningful: it operationalizes reasoning diversity without relying on human-crafted categories, enabling emergent taxonomies directly grounded in model behavior. The introduction of contrastive rubrics (e.g., “bottom-up vs. top-down,” “inductive vs. deductive”) represents an elegant mechanism for interpretable reasoning dimensions—conceptually parallel to semantic factor disentanglement. The finding that data format (MC vs. FF) shapes reasoning more strongly than domain is novel, empirically grounded, and relevant to both cognitive modeling and training data curation.

# 2. Technical Quality and Empirical Breadth

The framework is well-engineered: it integrates LLM-assisted criteria generation, embedding-based clustering, and interpretable classification. The experimental coverage is impressive—spanning six benchmarks across both helpfulness (GPQA-Diamond, MMLU-Redux, MATH-500) and harmlessness (XSTest, WildGuard, Arena-Hard). The authors perform rigorous human evaluations and quantitative analyses, reporting consistent improvements in interpretability and measurable performance gains.
The inclusion of ablation and robustness studies (e.g., embedding choices, random seeds, model scales) shows commendable attention to reproducibility and methodological soundness.

### Weaknesses
# 1. Lack of analysis on classifier choice and sensitivity

​The framework relies on a single LLM (GPT-4o) to perform all classification tasks in the taxonomy pipeline—deciding whether each reasoning trace aligns with one side of a contrastive rubric. Although Appendix B.1 examines benchmark-induced differences (showing that GPQA, MMLU, and MATH benchmarks produce similar criteria while Arena-Hard yields a distinct “User Understanding” dimension), this analysis only reflects task-level variability, not classifier-level robustness. The paper never investigates whether different classifier models (e.g., Claude, Gemini, DeepSeek) or prompting styles would yield consistent categorizations. Consequently, the stability and objectivity of the classification stage remain untested, and taxonomy boundaries may shift under alternate LLMs or small prompt perturbations.

# ​2. Limited human validation of interpretability​

Human evaluation (250 samples, 10 annotators) focuses on plausibility rather than interpretive alignment or consistency, and no inter-annotator reliability metrics are reported.​

# 3.Benchmark scope limited to English text

​All benchmarks are English and text-based; multimodal and multilingual reasoning remain unexplored.

### Questions
# 1. On classifier dependence and taxonomy stability

Your entire classification pipeline—strategy identification, rubric generation, and labeling—relies on GPT-4o. Have you tested whether the same reasoning taxonomy holds when different LLMs (e.g., Claude, Gemini, DeepSeek) are used as classifiers or rubric generators? If not, how do you ensure that the taxonomy reflects general reasoning properties rather than GPT-4o-specific biases?

# 2. On the granularity of reasoning strategy clustering

Your current clustering framework analyzes reasoning strategies at the full CoT–response level, producing high-level dichotomies such as “Inductive vs. Deductive” or “Top-Down vs. Bottom-Up.” However, reasoning trajectories are often compositional: individual reasoning steps may follow distinct micro-strategies that combine to form an overall reasoning pattern. 

Have you considered extending the CoT Encyclopedia to identify atomic reasoning step categories—that is, classifying each reasoning step rather than the entire chain—and investigating whether global reasoning strategies emerge as structured combinations of such atomic units? This could reveal a more mechanistic understanding of how complex reasoning behaviors are composed.

# 3. On the granularity and interpretability of clustered criteria
You report six high-level reasoning dimensions (Table 5). How were the fine-grained criteria merged into these six? Was k = 6 chosen purely based on silhouette scores or adjusted for interpretability by human judgment?

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
4

### Summary
The paper proposes a bottom-up framework to analyze, predict, and steer long chain-of-thought (CoT) reasoning. The pipeline: (i) have an LLM describe the strategies it used, (ii) embed and cluster these criteria, (iii) derive contrastive rubrics (e.g., top-down vs bottom-up), and (iv) classify new CoTs to produce interpretable 'reasoning profiles'. Using the learned rubrics, the authors predict question-specific optimal strategies and inject pattern-based instructions, improving accuracy and safety across multiple benchmarks. They also show that training format (MC vs free-form) shapes reasoning more than domain, and that weight interpolation between MC and FF-trained models smoothly shifts reasoning style. Overall the paper is thorough and the contribution is well articulated. The main limitation of the paper is the lack of a more substantial grounding on the coverage of the underlying categories of reasoning.

### Strengths
- Clear narrative. 

- Clear empirical insights. Demonstrates that format/domain in shaping reasoning patterns, and that merging model weights interpolates strategies. These provide useful guidance for dataset and model design.

- Sensible empirical analysis. Includes ablations on taxonomy construction (embedding, clustering, … ), human evals of report quality, and analyses of stability across families/sizes.

### Weaknesses
- Lack of a rigorous scoping for the problem of reasoning. 

- Lack of an argument for the construction of methods which can deliver a comprehensive feature set which describes the CoT reasoning phenomena.

### Questions
- The title of the approach (CoT encyclopaedia), points the readers in the direction of an approach which is comprehensive and systematic. Yet, it is unclear how the prompts you use to deliver that induction of a set of CoT characteristics are defined. Could you provide additional details and defend on why these prompts have the required properties to deliver the task? These needs to be mechanism/construction-based (not referring to the empirical analysis).

- The term ‘reasoning’ tends to be underspecified and used as whatever the tasks implement. Can you provide a description of what your task corpus is expressing wrt to reasoning and scope your claims accordingly. 

- Wrt validity, how well do rubric-based strategy labels align with expert human coders (definitions, guidelines, IAA)? What are the most common disagreement modes?

- Do results hold when swapping the LLM judge, embedding model, and moderation model? (cross-judge and cross-embed sensitivity analyses)

- How stable are criteria under different k, linkage metrics, and seeds?

- When pattern-based instructions help, how much is due to true strategy change vs prompt priming or longer outputs?

### Soundness
2

### Presentation
3

### Contribution
2
