# DeepPersona: A Generative Engine for Scaling Deep Synthetic Personas

- Avg Score: 4.67
- Decision: Reject
- Scores: 8, 4, 2

## Abstract
Simulating human profiles by instilling personas into large language models (LLMs) is rapidly transforming research in personalization, social simulation, and human-AI alignment. However, most existing synthetic personas remain shallow and simplistic, capturing minimal attributes and failing to reflect the rich complexity and diversity of real human identities. We introduce DeepPersona, a scalable generative engine for synthesizing narrative-complete synthetic personas through a two-stage, taxonomy-guided method. First, we algorithmically construct the largest-ever human-attribute taxonomy, comprising over hundreds of hierarchically-organized attributes, by systematically mining thousands of real user-ChatGPT conversations. Second, we progressively sample attributes from this taxonomy, conditionally generating coherent and realistic personas, averaging hundreds of structured attributes and roughly 1 MB of narrative text, two orders of magnitude deeper than prior works. Intrinsic evaluations confirm significant improvements in attribute diversity (32% higher coverage) and profile uniqueness (44% greater) compared to state-of-the-art baselines. Extrinsically, our personas enhance GPT-4.1-mini’s personalized Q&A accuracy by 11.6% average on ten metrics, and substantially narrow (by 32%) the gap between simulated LLM ``citizens'' and authentic human responses in social surveys. DeepPersona thus provides a rigorous, scalable, and privacy-free platform for high-fidelity human simulation and personalized AI research.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This work introduces DeepPersona, a two-stage generative engine that synthesizes detailed, diverse, and customizable synthetic persona data. The authors first construct the largest human-attribute taxonomy to date by mining and filtering self-disclosure content from human-LLM interactions. They then employ a progressive attribute sampling approach that iteratively selects diverse attributes and conditions a large language model to generate coherent values and narrative text.

### Strengths
- The motivation is clear and important, pinpointing the problem of "persona depth" in previous persona generation approaches.
- The method the authors use to extract is systematic and thoughtful. 
- Evaluation is done extensively in a multi-faceted manner, ranging from four different downstream tasks.
- Experiments are conducted on many frontier AI models from different sources, further supporting the generality of this work.
- Human experiments are included to complement the possible concerns regarding the instability of LLM judges.
- Most importantly, this work provides a scalable platform for synthetic persona generation, which I think is a significant contribution to the community.

### Weaknesses
I did not spot any significant weaknesses in this paper. One minor regret would be that qualitative examples are limited. It would be great to see qualitative comparisons between previous approaches and DeepPersona.

Also, this is minor, but there are some formatting issues on page 23. Please amend the overflow issue.

### Questions
N/A

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a taxonomy-guided framework that generates synthetic personas by sampling from a human-attribute tree. It addresses the deep and coherent persona generation from existing work. Experiments show higher personalization quality and closer alignment to human distributions on World Values Survey and Big Five benchmarks.

### Strengths
- The synthetic persona generation problem is interesting and timely.
- They provide a large-scale taxonomy of human attributes, which is really beneficial for the literature.
- The experiments are good, showing the advantages of the generated synthetic persona.
- The authors provide a method to diversify selected attributes.

### Weaknesses
- The method is naive. They did break the sampling procedure into two stages (sampling attributes from the taxonomy first and then sampling values from the given attributes), but they are heavily manually engineered.
- Generally, it seems to be a neat paper and can bring benefits to the community, but the novelty is limited. It would be more appreciated if this paper were submitted to the benchmark and dataset tracks instead of the main tracks.

### Questions
- Can we have a learnable way to learn the selector and generator so that it generates personas towards a chosen population? like these methods: PICLe: Eliciting Diverse Behaviors from Large Language Models with Persona In-Context Learning, ICML24 and Mixture-of-Personas Language Models for Population Simulation, ACL25.
- Why do we set the ratio to 5 : 3 : 2 ratio? Will any other combination work?

### Soundness
2

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
This paper introduces DEEPPERSONA, a two-stage generative framework for creating synthetic personas.

### Strengths
I cannot find any strength or contribution in the current manuscript of the paper.

### Weaknesses
The writing is misleading and difficult to follow, suggesting that the paper is not yet in a finished state.

In the abstract, the authors claim to be “mining thousands of real user–ChatGPT conversations.” However, Section 3 shows that no new data were mined; instead, the work relies entirely on existing datasets (Puffin, prefeval_implicit_persona). This inconsistency significantly weakens the claimed novelty.

At the start of Section 3, several important concepts are introduced without any explanation. Terms such as “text mass Narr(P)” and “persona/attribute depth” appear multiple times but are never defined or justified. Similarly, the choice of parameters—such as enforcing k > 10²—is arbitrary and unsupported by analysis or intuition. The authors should explain what k represents, why that threshold was chosen, and how it affects outcomes.

The paper also claims to contribute a dataset or toolkit (Section 3.3), but none of these resources—toolkit, evaluation scripts, or datasets—are publicly accessible. Without open access, the community cannot verify the claims, replicate the results, or assess the contribution’s practical value. As presented, the work lacks transparency and reproducibility.

### Questions
- Are the chosen datasets (Puffin, prefeval_implicit_persona, HiCUPID) demographically balanced?

- How was GPT-4.1-mini’s classification validated? Was any human verification or inter-annotator agreement performed? Were disagreements between GPT-4.1-mini and human judgments analyzed or resolved? If there is human evaluation, what are their background?

- Why is the 5:3:2 sampling ratio (near:middle:far attributes) considered optimal? No ablation study or justification is provided.

- What criteria or heuristics determine the depth budget (k)? How sensitive are the results to this parameter?

- How does the model prevent contradictions among attributes generated across different stages of progressive filling? Could the random traversal introduce bias or unrealistic attribute combinations that rarely occur in real human populations?

### Soundness
1

### Presentation
1

### Contribution
1
