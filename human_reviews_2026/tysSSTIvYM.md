# VALUEFLOW: Toward Pluralistic and Steerable Value-based Alignment in Large Language Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 6, 2, 4

## Abstract
Aligning Large Language Models (LLMs) with the diverse spectrum of human values remains a central challenge: preference-based methods often fail to capture deeper motivational principles. Value-based approaches offer a more principled path, yet three gaps persist--extraction often ignores hierarchical structure, evaluation detects presence but not calibrated intensity, and therefore, the steerability of LLMs at controlled intensities remains insufficiently understood. To address these limitations, we introduce VALUEFLOW, the first unified framework that spans extraction, evaluation, and steering with calibrated intensity control. The framework integrates three components: (i) HiVES, a hierarchical value embedding space that captures intra- and cross-theory value structure; (ii) the Value Intensity DataBase (VIDB), a large-scale resource of value-labeled texts with intensity estimates derived from ranking-based aggregation; and (iii) an anchor-based evaluator that produces consistent intensity scores for model outputs by ranking them against VIDB panels. Using VALUEFLOW, we conduct a comprehensive large-scale study across ten models and four value theories, identifying asymmetries in steerability and composition laws for multi-value control. This paper establishes a scalable infrastructure for evaluating and controlling value intensity, advancing pluralistic and accountable alignment of LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces VALUEFLOW, a unified framework for value-based LLM alignment focused on steerable intensity control. The authors argue that existing methods fail to capture value hierarchies or reliably measure value intensity, leading to unstable evaluations. VALUEFLOW addresses this with three components: HIVES, a hierarchical value embedding space; VIDB, a large-scale database of texts with intensity scores derived from robust ranking aggregation using a Plackett-Luce model; and a stable anchor-based evaluator that ranks outputs against VIDB. Using this framework, the authors conduct a comprehensive study on ten models across four value theories, characterizing model-specific asymmetries in steerability, identifying a "strong-anchor dominance" effect in multi-value control, and providing a scalable infrastructure for pluralistic alignment.

### Strengths
This paper addresses an important and timely question, how to systematically examine and steer the value representations embedded in current large language models under pluralistic and controllable conditions.

It proposes a comprehensive engineering framework that connects extraction, evaluation, and steering into a closed loop, forming an end-to-end pipeline for value-based alignment.

The experiments are large-scale and multi-dimensional, covering 10 models, 4 value theories, and 32 value dimensions, including both single-value and multi-value steering, refusal analysis, and downstream evaluation. These results provide broad and empirically grounded insights into asymmetric steerability and anchor dominance effects.

### Weaknesses
The reliance on LLM-as-a-Judge remains a concern. It would be valuable to include an analysis of inter-model differences in judging behavior, as well as quantitative consistency checks between LLM-based and human annotations.

The paper’s writing and structure are sometimes hard to follow; a clearer organization and smoother transitions would significantly improve readability.

There are two minor typos at Lines 296 and 417, the paragraph titles should be followed by a period.

### Questions
See the **Weaknesses** section.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces VALUEFLOW, a unified framework for value-based alignment in large language models, encompassing three core components: extraction, evaluation, and intensity-controlled steering. The framework includes: (1) HIVES - a hierarchical value embedding space capturing multi-theory value structures; (2) VIDB - a large-scale value intensity database with calibrated intensity estimates via ranking aggregation; (3) an anchor-based evaluator producing consistent intensity scores through ranking rather than rating. The authors conduct large-scale experiments across 10 models and 4 value theories, identifying asymmetries in steerability and composition laws for multi-value control, while demonstrating improved demographic alignment through value-based profiling.

### Strengths
1. Problem Focus and Contribution Scope:
    - Addresses three critical gaps in value alignment research: extraction lacks hierarchical structure, evaluation detects presence but not intensity, and steerability remains insufficiently understood - the problem formulation is clear and important.
    - First to propose "steerability with intensity," extending value alignment from directional control to graded intensity control, opening a new dimension for pluralistic value alignment.

2. Method Mechanism:
HIVES's two-stage training design is well-motivated: Stage 1 aligns intra-theory structure via hierarchical contrastive learning, while Stage 2 unifies heterogeneous theories through cross-theory anchors - the technical approach is sound.

3. Empirical Evaluation:

    - Comprehensive experimental scope: 10 mainstream models (open and closed source) × 4 value theories × 32 value dimensions × 500 prompts, providing broad coverage.
    - Valuable empirical patterns discovered: asymmetry in negative steering, strong-anchor dominance effect, value similarity affecting composition laws, etc.

### Weaknesses
- Technical Correctness: Plackett-Luce model assumes Independence of Irrelevant Alternatives (IIA), but value judgments may exhibit context-dependent effects; the paper does not discuss robustness when this assumption is violated.

- Evaluation Scope: Experiments mainly focus on "short-term prompt-driven value steering" and do not explore the stability of value expression in long-term dialogues (e.g., whether the model deviates from the target intensity after multi-turn interactions). They also fail to test value alignment effects in low-resource language or niche cultural contexts, resulting in limited scenario coverage.

- Loss function weights in two-stage training (λind=0.5, λtheory=1.0) lack ablation study support; the impact of different weight configurations on final representation quality is unknown.

### Questions
- Theoretical Foundation: Why is the Plackett-Luce model suitable for value intensity aggregation? Have other ranking models (e.g., Bradley-Terry, Thurstone) been tested? How does the model perform when value judgments exhibit non-transitivity?

- How do the 274 cross-theory anchor concepts ensure balanced coverage across theories? Is there any theory dominating the anchor set?

- How applicable is the framework to non-English languages or non-Western value systems (e.g., Confucian, Buddhist values)?

- Is the computational cost of ranking evaluation at inference time (k=6, m=3 implies 18 LLM calls) acceptable in real applications?

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
Beyond preference-based methods, this paper accounts for alignment with real human values that are diverse and serve as more stable principles in decision-making. 

It aims to address two main challenges: (1) value extraction, current studies rely on static questionnaires or simple judgments, limiting the ability to capture signals from open-ended conversational contexts and rarely encode the hierarchical nature of values. (2) value evaluation, current studies measure presence rather than strength, overlooking intensity in open-ended outputs. 

For these, they introduce a unified framework VAUEFLOW, with HIVES, a hierarchical value embedding space to capture value profiles, and a ranking-based value evaluator together with a value-intensity database VIDB. Then, they conduct experiments to verify the effectiveness of HIVES and the ranking-based evaluator, as well as the whole framework for value steerability.

### Strengths
1. This paper proposes VALUEFLOW, a unified framework spanning value extraction, evaluation and steering in LLMs, allowing for end-to-end steerable value alignment.
2. To address the challenge of value extraction, it constructs a HIVES method to unify heterogeneous value theories.
3. Accounting for the intensity of values and instability of current rating-based evaluations, this paper builds a value-intensity database and designs a ranking-based evaluation method of intensity.
4. Some experiments and analysis are conducted to demonstrate the usage in value steerability.

### Weaknesses
1. Baselines of value extraction on open-ended conversational contexts are largely ignored both in the Introduction part (Line 52) and Related Work part. I think there are some works on this task.
2. The whole method needs better clarification:
- A structural algorithm is desired to formulate the whole framework, especially how the hierarchical value embedding space is built.
- More descriptions are required for the Sec 4.3 Two Stage Training Process, what are the inputs and what are the outputs? How to obtain the ground truth data for training?
- What is the value steering method for AI used in this paper?
- What are the ground truth and evaluation metrics used for the experiments in Figure 4.
3.  There lack sufficient baselines for comparison in both Table 1 and Table 2, limiting the reliability of effectiveness about your method. There are evaluators in Denevil, ValuePrism, etcs mentioned in this paper, which should be considered for comparison.
4. There are some problematic settings in your method and experiments.
- In Line 259, when constructing the VIDB dataset, you use multiple LLMs to generate the intensity rating label for each text. However, you mentioned in Sec 3.2 that LLMs’ ratings on value intensity are highly unstable. So this would decrease the accuracy of the VIDB dataset.
- In Sec 6.3, you first construct value profiles from the dataset, then use these value profiles as the alignment target, finally compute the accuracy between the alignment with the data which are used for constructing value profiles. This could incur a risk of data contamination, limiting the significance of the experiments.

### Questions
1. Figure 2 is currently a little confusing about which LLM generates the rating score respectively.
2. Figure 8 is hard to understand.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
VALUEFLOW introduces a unified framework for value-based LLM alignment, addressing extraction, evaluation, and steering gaps. It comprises hierarchical value embeddings unifying SVT, MFT, duties, rights, large-scale intensity-labeled texts via ranking aggregation, and a ranking-based evaluator using VIDB anchors for calibrated intensity scores. Experiments across 10 models reveal steerability asymmetries, multi-value composition laws, and improved demographic prediction on OpinionQA (>10% accuracy gains). The framework enables pluralistic, intensity-controlled alignment.

### Strengths
- The ranking-based value evaluation is a novel and timely contribution. It's a promising direction to overcome the reliability and consistency issue of prior evaluation methods.
- Unifying heterogeneous value theories is an interesting and pioneering attempt.
- The large-scale intensity database, upon its open-source, is a significant contribution to the community.

### Weaknesses
The methodological section is hard to follow. For example, it is unclear what the motivation and theoretical basis are for the two-stage training process (Section 4.3). How does the unified taxonomy contribute to the value evaluation? What is the relationship between the anchors in Section 4.3 and those in Section 5.2? In Figure 3, how are parts (a) and (b) used together synergistically?

What is the theoretical justification is for using the Plackett–Luce model among all possible scales?

Is the zero-shot, ranking-based value evaluation reliable? Was the evaluator trained? The experiments in Section 3 seem to focus only on SVT values, rather than on all the values used in this study. How do you validate the accuracy of the value evaluation?

How well do your embedding model and value evaluator generalize across different context lengths?

There appears to be no alignment training, despite the claims in the abstract and introduction.

The evaluations in Sections 6.2 and 6.3 have also been conducted in prior work. How does your evaluation improve upon previous ones? Are there any novel insights?

What is the motivation behind designing a unified framework if the components are not trained synergistically? In what way is the end-to-end workflow superior to prior approaches that design individual components separately?

### Questions
See above.

### Soundness
3

### Presentation
2

### Contribution
3
