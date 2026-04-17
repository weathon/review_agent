# PRICIN: Principle-Centered Inorganic Retrosynthesis

- Decision: Reject
- Scores: 4, 2, 2, 6

## Abstract
Bridging the gap between what is designable by computational discovery and what is synthesizable in the lab remains a central obstacle for closed-loop materials science. We tackle single-step inorganic retrosynthesis and show that explicit chemical principles are potent inductive biases for learning to plan syntheses. We introduce PRICIN, a principle-centered approach that reformulates precursor planning around two laws: elemental conservation and electron balance. PRICIN embeds stoichiometry and oxidation-state semantics directly into the target representation via two pretraining objectives, including an auxiliary oxidation-state supervision that injects charge awareness. At inference, a lightweight element-wise filter first predicts the required number of precursors and then prunes candidates that violate conservation constraints, yielding explainable, chemically consistent precursor sets without external retrieval or rigid templates. Across the Retrieval-Retro (year-split) and Ceder benchmarks, PRICIN attains state-of-the-art performance on Top-$k$ and combination Top-$k$ metrics, improving over the previous best by +5.17 Top-1 and by up to +20.78 percentages on Top-20. Ablations confirm that oxidation-state supervision and conservation-aware filtering are both necessary and complementary, substantially reducing early-rank errors. The code will be released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This work presents a novel *inorganic retrosynthesis* framework based on a principle-centered approach.

### Strengths
- The paper is well-motivated.  
- The principle-centered approach is innovative and has strong potential for guiding inorganic retrosynthesis.

### Weaknesses
### **Major Comments**
1. **Order Invariance**  
   It is unclear whether the proposed method is order-invariant when dealing with different arrangements of elements.

2. **Variable Valence Materials**  
   Can the method handle materials with variable valence states? For example, in Fe₃O₄ there are two Fe³⁺ and one Fe²⁺ ions. How does the model represent and process such mixed-valence compounds?

3. **Missing Details on Element-wise Filter**  
   Section 4 lacks specific implementation details regarding the “Element-wise Filter.” How are these filters defined and applied? A more detailed description or pseudo-code would help readers understand this component.

4. **Clarity of Figure 2**  
   Figure 2 is difficult to interpret. What does the number *118* represent in the figure? Also, in the Task 2 subfigure, the oxidation ground truth also shows *118*. Please clarify its meaning and ensure all symbols are clearly explained.

5. **Number of Predicted Precursors**  
   Do all predicted precursor sets have the same number of materials that predicted by the deviation predictor?

6. **Codebook Construction**  
   Please elaborate on how the *Oxidation Number Codebook* and *Precursor Codebook* are constructed. Are they predefined, learned jointly with the model, or created from external data?


### **Minor Comments**
- Several citation style issues were found. Please use `\citet{}` instead of `\cite{}`  to maintain consistency with the required citation style.

### Questions
see weaknesses

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
A principle-centered approach that reformulates precursor planning around two laws: elemental conservation and electron balance.
The paper argues that existing studies focus on precedent-based learning and have weak modeling of chemical principles.

### Strengths
- Paper is well written so that I can easily follow the work
- Inorganic retrosynthesis is important problem in the field of materials science

### Weaknesses
Lack of Novelty: It appears the only newly added content is the electron-based oxidation number prediction. This is my main concern in this paper that the paper utilizes exisiting ML techniques for inorganic retrosynthesis, and even not new problem formulation

Ablation Study Concerns:
- The filter seems to be the key factor in improving performance in the ablation study. However, there is very little explanation provided about this filter.
- It would be beneficial to apply the filter to all ablated models for a more comprehensive comparison.

Performance of Oxidation vs. Rebuild: There is little performance difference between the oxidation and rebuild components. Can this be explained? The model's main motivation is centered on oxidation, yet it shows only a marginal performance difference from the rebuild component.

### Questions
See Weakness section

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work focuses on inorganic retrosynthesis, specifically predicting the precursor set for a given target material. The authors incorporate strong chemical rules — particularly oxidation number constraints and element filters based on deviation — to enhance model performance. Extensive experiments are conducted to validate the approach.

### Strengths
- The model effectively integrates strong domain knowledge (chemical rules such as elemental conservation and electron balance) directly into the modeling process, leading to improved performance.  
- The paper is well-presented and easy to follow.

### Weaknesses
- The technical contribution appears limited. Although the proposed framework adopts a multi-task learning setup, Task 1—composition reconstruction—has already been explored in prior work (SynthesisSimilarity, He et al.). The main additions seem to be the chemical rule–based element filtering and oxidation number prediction modules.
 - The paper lacks sufficient explanation of how oxidation number prediction contributes to overall performance and how it is modeled.
- Moreover, using known precursor oxidation states to predict the oxidation states of the target material may restrict generalization to in-distribution targets, since many targets can exhibit multiple or mixed oxidation states. This could lead the model to learn only “easy” oxidation states under overly strong supervision.
-  A sensitivity analysis for each training task would be beneficial to clarify their respective contributions.

### Questions
- In the preliminary section, the authors mention by-products. Were by-products explicitly considered during the retrosynthesis task? If so, how were they handled?

- The paper claims strong extrapolation ability to new systems, but there is little explanation or evidence supporting this. There seems to be no specific methodological component designed to improve extrapolation. Since the model mainly builds upon strong chemical-formula-based priors (e.g., oxidation filtering), how does this lead to better extrapolation?

- If oxidation-state prediction is claimed to help handle novel or unseen materials, more detailed discussion is needed: how does this mechanism concretely enable the model to generalize or make accurate predictions for truly new target systems?

### Soundness
2

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
2

### Summary
This work presents PRICIN: a method for inorganic retrosynthesis prediction, built around domain-specific pre-training on several tasks followed by careful filtering during inference. Authors show this yields strong results across two different materials synthesis datasets, including one that employs a (more challenging) time-based split.

### Strengths
**(S1)**: Materials discovery is an important area of scientific pursuit, and it can sometimes be bottlenecked by the ability to synthesize the proposals coming from e.g. the generative models. Hence looking at better synthesis models is an important research direction.

**(S2)**: From an ML perspective, the approach appears to be sound, integrating domain-specific inductive biases in a reasonable way to get better grounding.

### Weaknesses
**(W1)**: There are several aspects of the work that are confusing to me and could benefit from clarification:

- **(W1a)**: Table 1 mentions that PRICIN uses retrieval. How is this performed? I assumed retrieval means that the training data is explicitly stored and can be accessed verbatim during generation (relating it to e.g. RAG in LLMs), yet I missed where this is done in PRICIN.

- **(W1b)**: I'm confused about the "Elemental conservation" paragraph and in particular Equation 2. While Equation 1 does seem to correspond to preserving atom counts under appropriate stochiometric coefficients, Equation 2 seems to suggest only preserving the presence or not of particular atom types. Is therefore PRICIN only enforcing the latter? If yes, it seems somewhat weird to introduce the method as adhering to elemental conservation (e.g. see abstract), as that could be misunderstood.

**(W2)**: The paper can be somewhat hard to parse for a person outside of the materials discovery space, even if they have general chemistry knowledge. This could be improved. For example, oxidation states could be explained in a more accessible way.

---

**Other comments**

**(O1)**: In the ablation study, authors note a synergistic effect between the two ablated model elements, where adding only one does not meaningfully improve performance. This seems counterintuitive to me. I wonder if the authors have any explanation why that would be the case. Being "greater than the sum of the parts" is one thing, but in this case it appears the parts alone bring zero benefit, and combined they bring a substantial one.

**Nitpicks**

Across the paper, parentheses around citations are missing where they should be present, and present where they should be missing. If the citation appears as part of the sentence, e.g. "Author et al have shown that...", it should not be parenthesized, while if it appears as a remark outside of sentence, e.g. "This thing is known (Author et al).", it should be. See e.g. beginning of Section 1, Section 2, and the "Baseline methods" paragraph in Section 5.1.

---

**Note**

I have a lot of experience in AI for Chemistry and in particular organic synthesis, but very limited experience in the materials space and inorganic synthesis. Therefore, I mark my review as lower confidence. While from the ML point of view the approach appears sound, I could not verify the more domain-specific parts.

### Questions
See the "Weaknesses" section above, especially **(W1)**, for specific questions.

### Soundness
3

### Presentation
3

### Contribution
3
