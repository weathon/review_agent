# MINED: Probing and Updating  with Multimodal Time-Sensitive Knowledge for Large Multimodal Models

- Avg Score: 5.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 8, 4, 4

## Abstract
Large Multimodal Models (LMMs) encode rich factual knowledge via cross-modal pre-training, yet their static representations struggle to maintain an accurate understanding of time-sensitive factual knowledge. Existing benchmarks remain constrained by static designs, inadequately evaluating LMMs' ability to understand time-sensitive knowledge. To address this gap, we propose MINED, a comprehensive benchmark that evaluates temporal awareness along 6 key dimensions and 11 challenging tasks: cognition, awareness, trustworthiness, understanding, reasoning, and robustness. MINED is constructed from Wikipedia by two professional annotators, containing 2,104 time-sensitive knowledge samples spanning six knowledge types. Evaluating 15 widely used LMMs on MINED shows that Gemini-2.5-Pro achieves the highest average CEM score of 63.07, while most open-source LMMs still lack time understanding ability. Meanwhile, LMMs perform best on organization knowledge, whereas their performance is weakest on sport. To address these challenges, we investigate the feasibility of updating time-sensitive knowledge in LMMs through knowledge editing methods and observe that LMMs can effectively update knowledge via knowledge editing methods in single editing scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes MINED, a comprehensive benchmark that evaluates temporal awareness along 6 key dimensions and 11 challenging tasks: cognition, awareness, trustworthiness, understanding, reasoning, and robustness. MINED is constructed from Wikipedia by two professional annotators, containing 2,104 time-sensitive knowledge samples spanning six knowledge types.

### Strengths
1. The paper addresses a highly underexplored area—temporal awareness in Large Multimodal Models
2. The MINED benchmark is multi-dimensional, encompassing 6 core capabilities with comprehensive benchmark design
3. The analysis of the experimental conclusions is very meticulous

### Weaknesses
1. In Section 5, both the knowledge-editing methods and the backbone models are rather outdated. I suggest the authors experiment with more recent backbone models such as Qwen2.5-VL and adopt newer knowledge-editing techniques.
2. To make the experimental conclusions easier to grasp, the authors should provide more detailed case studies, using them to elaborate on the seven observations they draw.

### Questions
Please refer to Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces MINED, a new benchmark designed to systematically evaluate the performance of multimodal large language models on time-sensitive questions and dynamic knowledge updating capabilities. The benchmark assesses six dimensions. Using MINED, the paper evaluates several recent models and further investigate the performance of classical knowledge editing methods on updating time-sensitive knowledge.

### Strengths
MINED offers a comprehensive benchmark of dynamic knowledge understanding in LMMs, with extensive evaluations of recent models and a systematic study of knowledge-editing methods for time-sensitive updates. By jointly leveraging text and images to probe temporal awareness and misalignment, it takes a crucial step toward realistic multimodal evaluation.

### Weaknesses
1. Although the evaluation dimensions have increased, it seems that the proposed dataset is not more challenging than other datasets.
2.Missing inter-annotator agreement (e.g., Cohen’s κ) and conflict resolution details. These should be reported for transparency.
3.The automatically generated misaligned contexts might encode stylistic artifacts; human-written or multi-source variants are needed.

### Questions
1. Is the dataset updated over time?
2. The maximum answer length is 13, but the average answer length is only 2. Are most of the answers single words?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces MINED, a comprehensive benchmark designed to evaluate large multimodal models’ (MLLMs) ability to understand and update time-sensitive knowledge.
Unlike prior static or text-only temporal benchmarks, MINED focuses on multimodal temporal awareness—how models perceive, reason about, and update factual knowledge that evolves over time.

The benchmark contains 2,104 time-sensitive knowledge samples and 4,208 questions, spanning six capability dimensions (Cognition, Awareness, Trustworthiness, Understanding, Reasoning, and Robustness) and six knowledge types (e.g., sport, organization, company).
Extensive evaluations over 15 popular LMMs reveal that even state-of-the-art closed-source models (e.g., Gemini-2.5-Pro) struggle with implicit and misaligned temporal knowledge.
Furthermore, the authors explore knowledge editing methods (FT-LLM, FT-VIS, MEND, SERAC, IKE) to update outdated time-sensitive facts in LMMs, demonstrating that single-editing is effective, while lifelong editing suffers from catastrophic forgetting.

Overall, MINED aims to bridge temporal reasoning, multimodal grounding, and model updating—offering a unified platform for evaluating time-sensitive understanding and editing in LMMs.

### Strengths
1. **Comprehensive benchmark design.**
   - The benchmark is organized into six orthogonal capability dimensions, forming a well-structured framework that captures not only factual recall but also temporal awareness, reasoning, and robustness.
   - The inclusion of *temporal misalignment* and *unanswerable-date* subtasks realistically simulates real-world temporal inconsistencies that often occur in dynamic factual knowledge.
2. **High-quality dataset and clear evaluation protocol.**
   - All data are manually verified, and the benchmark supports *evolvability* through quarterly Wikipedia updates, which ensures long-term relevance.
   - The proposed *Prompt Agreement* scheme is a thoughtful methodological design that effectively mitigates prompt-induced variance during evaluation.
3. **Strong experimental depth and breadth.**
   - Evaluation across 15 large multimodal models (both open- and closed-source) provides rich comparative insights into current model limitations.
   - The experimental analysis includes detailed error breakdowns, cross-model comparisons, and multiple editing paradigms, offering a well-rounded understanding of temporal sensitivity in LMMs.

### Weaknesses
1. **Limited novelty and motivation.**
   - The paper mainly focuses on dataset construction, while the necessity and motivation for studying this particular task could be discussed more clearly.
   - There is little exploration of potential methodological improvements or model-side strategies for enhancing temporal sensitivity beyond the dataset itself.
2. **Relatively small scale of data.**
   - The benchmark includes 4,208 questions divided into seven categories, with only 450 unique images.
   - It remains somewhat unclear whether such a dataset size is sufficient to comprehensively probe multimodal temporal sensitivity, especially given the complexity of real-world temporal reasoning.
3. **Brief analysis of lifelong editing.**
   - The section on “lifelong editing” is rather concise, and the underlying causes of catastrophic forgetting are not deeply analyzed.
   - A more systematic examination of how editing interacts with multimodal representations would strengthen this part.
4. **Evaluation metric limitations.**
   - Heavy reliance on the *Correct Exact Match (CEM)* metric may underestimate partial correctness.
   - Although F1 scores are reported in the appendix, the main results still depend primarily on strict matching, which might not fully reflect model understanding.

### Questions
1. **On the necessity of multimodal extension:**
    The motivation for extending temporal reasoning to multimodal settings could be further clarified.
    If the underlying language model in a multimodal system already possesses temporal sensitivity, does the multimodal extension inherently inherit such ability?
    What makes the multimodal setting particularly challenging or unique in this context?
2. **On the temporal validity of images:**
    Would it be more meaningful to incorporate visual changes over time—such as variations in a person’s appearance (childhood, adulthood, aging) or environmental transformations (seasons, locations)?
    What is the essential difference between the temporal sensitivity problem in MINED and that in purely text-based temporal knowledge benchmarks?
3. **On the simplicity of the knowledge representation:**
    The benchmark relies on a quadruple-based knowledge structure $$(S,H,P,A)$$.
    While this design enables systematic probing along the six dimensions introduced in Section 3.1, it might oversimplify the complexity of temporal evolution in real-world multimodal data.
    Could the authors discuss whether this abstraction limits the benchmark’s ecological validity, and how future work might move toward more realistic, context-rich temporal scenarios?

### Soundness
3

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
In this paper, the authors propose MINED, a comprehensive benchmark that evaluates temporal awareness along 6 key dimensions and 11
challenging tasks: cognition, awareness, trustworthiness, understanding, reasoning, and robustness. MINED is constructed from Wikipedia by two professional annotators, containing 2,104 time-sensitive knowledge samples spanning six knowledge types. The authors also evaluate more than 10 widely used LMMs in the proposed dataset.

### Strengths
1. In this paper, the authors propose MINED, a comprehensive benchmark that evaluates temporal awareness along 6 key dimensions and 11 challenging tasks: cognition, awareness, trustworthiness, understanding, reasoning, and robustness.

2. MINED is annotated by professional annotators, containing 2,104 time-sensitive knowledge samples spanning six knowledge types

### Weaknesses
1. In this paper, I didn't find how to build such a dataset.  The authors only mention that To construct the foundational data for MINED, we employ two professional annotators to gather time-sensitive knowledge from Wikipedia across six domains: Country, Sport, Company, University, Organization, and Competition.

2. Only two professional annotators are invovled in the annotation, How to deal with situations where two people have different opinions?

3. In this paper, how to control the quality of the proposed dataset is unknown

### Questions
NA

### Soundness
2

### Presentation
3

### Contribution
2
