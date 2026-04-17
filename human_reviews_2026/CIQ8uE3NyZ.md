# SELECT: Search-Enhanced Language Models for Analog Circuit Topology Generation

- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
Automating analog circuit topology design is essential to reduce the extensive manual effort required to meet increasingly diverse and customized application demands. Recent advances have applied sequence-to-sequence fine-tuning on pretrained language models to directly generate circuit topologies from user specifications in a single pass. However, these one-shot generation methods failed to generate complex circuits due to their exponentially growing search spaces and limited training datasets. In this paper, we present SELECT, a search-enhanced language model framework that integrates simulator-guided Monte Carlo Tree Search (MCTS) with transformer-based decoding to use test-time computation for improved performance. SELECT introduces novel structural token pruning and P-UCB-based node selection to leverage next-token probability distributions to guide the search process. By combining pretrained priors with simulator feedback at inference time, SELECT converges faster than prior search methods and achieves significantly higher generation success rates, improving by up to 435\% over RL-based search and 145\% over LaMAGIC under a strict tolerance of 0.01.
These results establish SELECT as the first scalable framework for high-fidelity analog topology generation and a practical step toward LLM-driven circuit design automation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This work introduces a search-enhanced framework for automated analog topology generation that integrates simulator-guided Monte Carlo Tree Search (MCTS) with pretrained language-model decoding.

### Strengths
SELECT is the first to incorporate search-based decoding into analog circuit generation
SELECT achieves a 435%, 145% higher success rate under a low tolerance of 0.01 compared to an RL-search method Fan et al. (2021) and LaMAGIC Chang et al. (2024) with sampling and filtering at the same search budget.

### Weaknesses
1. Clarification needed on circuit complexity claims
The characterization of 8–10 component circuits as "complex designs" may benefit from additional context. As reference:

A basic single-stage cascode op-amp (Design of Analog CMOS Integrated Circuits, 2nd Ed., p. 350, Fig. 9.8) uses 8 components.
The classical two-stage op-amp from Ahuja et al. (1983) uses approximately 20 components (Fig. 3b) (An improved frequency compensation technique for CMOS operational amplifiers).
More recent designs, such as the switched op-amp from Young-Ju et al., use 34 components excluding biasing and CMFB (Fig. 5) (A 12 bit 50 MS/s CMOS Nyquist A/D Converter With a Fully Differential Class-AB Switched Op-Amp).

It would be helpful if the authors could clarify their definition of "complex" and provide comparative context relative to typical analog circuit design practice.

2. Design space analysis requires more detailed justification
Figure 1 presents a design space showing 100 topologies for 3-component circuits and 10,000 for 5-component circuits. To better understand this contribution, the authors should address:

What proportion of these topologies are functionally viable?
How many represent meaningful design variations worth including in the training set?
What criteria were used to filter the design space?

3. Novelty claim needs supporting evidence
The claim that this work is "the first time" analog topology generation extends to 7–10 component circuits would benefit from a more comprehensive literature review. Specifically, it would be helpful to see:

A systematic comparison with prior work (AnalogCoder, Lamagic, CktGNN, AnalogGenie, etc.)
Documentation of the maximum circuit complexity handled by these earlier methods
Clear delineation of what constitutes a substantive advance beyond prior work

4. Relationship to existing MCTS-LLM methods needs clarification
MCTS has been widely adopted in LLM-based code generation (e.g., "Planning with Large Language Models for Code Generation," "Large Language Models as Commonsense Knowledge for Large-Scale Task Planning"). The paper would be strengthened by:

Discussing how the proposed approach differs from or builds upon these methods
Clarifying domain-specific adaptations required for analog circuit generation
More explicitly articulating the unique contributions beyond applying established MCTS-LLM techniques

5. Reproducibility concerns
The absence of supplementary materials (code, datasets, or detailed implementation specifications) limits reproducibility. Providing these resources would significantly strengthen the contribution and enable the community to build upon this work.

### Questions
see weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work aims at improving the performance of automatic analog circuit topology design.
The main challenge of existing approaches is that they fail to generate complex circuits due to the exponentially growing search spaces and limited training datasets.
The main idea of this work is to leverage test-time computation to improve performance.
Specifically, it introduces token pruning, leverage next-token probability distribution to guide the search process, and uses simulator feedback at inference time.

### Strengths
- The formulated problem is practical and valuable, which aims to generate complex analog circuit topology with more than six components.
- The proposed method is reasonable and makes sense.
- The writing is clear and easy to follow.

### Weaknesses
- In lines 157-158, it is mentioned that there is an extended pipeline for data collection to build datasets for circuits with 7-10 components. What does extended pipeline mean? Are there any differences to the one used in LaMAGIC?
- There is a runtime illustration for the proposed approach. What is the runtime for baselines? It seems that the proposed method is the most time-consuming one as it introduces model inference and simulator feedback in its workflow.
- The base pretrained language model used in this work is Flan-T5-base, which is quite an old one. I acknowledge that this choice is adopted from LaMAGIC. However, the whole community of pretrained language models evolved very quickly. To make this work solid and up-to-date, new and stronger language models should be adopted (e.g., Qwen or Llama). Besides, a discussion about when base language model becomes stronger, whether the proposed challenges still exist and proposed method still work is valuable and important.
- For the experiment in Section 6.4, are the two methods compared with the same time budget? Will random generation perform better if it was given more time budget?

### Questions
see above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a framework that combines a pretrained Transformer language model with Monte Carlo Tree Search (MCTS) and circuit simulation feedback to automatically generate analog power converter topologies. Unlike prior one-shot or reinforcement-learning methods, SELECT performs search-based decoding, where the language model guides MCTS to explore only the most promising next tokens while simulator feedback evaluates and backpropagates performance rewards. Key innovations include a probability-guided UCB (P-UCB) node selection strategy and structural-token filtering to improve efficiency. Experiments show up to 435% higher success rates than prior methods and demonstrate scalability to 7–10-component circuits.

### Strengths
+ Novel integration of the LLM and search. SELECT is the first framework to combine a pretrained Transformer with Monte Carlo Tree Search (MCTS) for analog circuit topology generation.
+ Extending the topology generation to 7–10-component converters is an important step to make the work more practical.

### Weaknesses
- Overly exaggerated claim. For people who are familiar with analog circuit design, the paper made a serious overclaim of the generality of their method. Throughout the paper, experiment results are only provided for power converters, which are merely one type of analog/mixed-signal circuits. There is no evidence that directly supports or suggests that the proposed data representation or the language model formulation can be extended beyond the converter circuits. Another overclaim comes from exaggerating the accomplishment on extending the previous method from 3~5-component designs to 6~10-component designs. Fewer than 10-component analog circuits have been quite exhaustively studied by analog designers. Although a method to automate the generation of circuits with such a low component count may be of some theoretical interest, it is of little practical use.
- Lack of a clear explanation of the algorithmic method. The paper does not provide sufficient details for the reviewer to fully grasp the relationship between the language model training/pretraining and the Monte Carlo Tree Search (MCTS) method with its various selection/expansion/evaluation steps. It is also very difficult to make sense of the baseline methods being compared with in the evaluation.
- Unclear dataset construction detail. The paper never fully explains how the 7–10-component datasets were generated. For example, the number of rollouts, filtering thresholds, or simulation settings. This omission significantly limits reproducibility.
- Poor writing and organization. The writing of the paper can be improved. In many places, simple typos greatly impede readability. For instance, for the number of components in each topology, the authors sometimes use numeric forms like “345,” sometimes hyphenated ranges like “3–5,” and sometimes words like “six.” Please ensure consistent formatting throughout.

### Questions
Can you explain how the data representation and the training method in this paper can effectively be used for other types of analog circuits? Please give concrete examples.

Can you give examples of 6~10-component converter designs generated by the proposed method? Do the generated designs follow well-known converter topologies? Are there novel topologies in the generated design that are non-obvious to human designers?

### Soundness
2

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
3

### Summary
This paper introduces SELECT, a search-enhanced language model framework that combines transformer-based decoding with simulator-guided Monte Carlo Tree Search to improve analog circuit topology generation, demonstrating substantially higher success rates and scalability compared to prior methods.

### Strengths
- The proposed MCTS-based approach demonstrates superior performance in topology generation experiments, representing a promising direction for future research in this field.
- The experimental evaluation is comprehensive and well-documented.

### Weaknesses
- Figure 2 appears to have been adapted directly from LaMAGIC; it would be more appropriate for the authors to redraw the figure to ensure originality and consistency with their own work.
- The proposed method seems less scalable than prior approaches, as it is only applied to power converter circuit topology generation. Power converters are not necessarily the most representative or critical circuits in analog research. The authors are encouraged to discuss or extend their method to other types of analog circuits.
- Compared to one-shot generation methods, MCTS is computationally less efficient and requires numerous simulation runs, which may limit its practicality for real-world applications.

### Questions
- In Figure 1, why are there no collected topologies for circuits with seven devices?

### Soundness
3

### Presentation
3

### Contribution
2
