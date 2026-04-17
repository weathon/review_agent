# Context and Diversity Matter: The Emergence of In-Context Learning in World Models

- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
The capability of predicting environmental dynamics underpins both biological neural systems and general embodied AI in adapting to their surroundings. Yet prevailing approaches rest on static world models that falter when confronted with novel or rare configurations. We investigate in-context learning (ICL) of world models, shifting attention from zero-shot performance to the growth and asymptotic limits of the world model. Our contributions are three-fold: (1) we formalize ICL of a world model and identify two core mechanisms: environment recognition (ER) and environment learning (EL); (2) we derive error upper-bounds for both mechanisms that expose how the mechanisms emerge; and (3) we empirically confirm that distinct ICL mechanisms exist in the world model, and we further investigate how data distribution and model architecture affect ICL in a manner consistent with theory. These findings demonstrate the potential of self-adapting world models and highlight the key factors behind the emergence of EL/ER, most notably the necessity of long context and diverse environments. The codes are available at https://github.com/airs-cuhk/airsoul/tree/main/projects/MazeWorld.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
his paper examines how world models can learn and adapt through context, focusing on in-context learning (ICL) within both MDP and POMDP settings. The authors distinguish between two key processes, namely In-Context Environment Learning (ICEL) and In-Context Environment Recognition (ICER). Furthermore, analyze their theoretical properties by deriving error upper bounds that explain when and how each process is effective.

To validate these insights, the paper introduces L2World, a framework designed to study in-context world modeling, and evaluates it on cart-pole and navigation tasks. The experiments show that world models can adapt to new environments through context alone and that both environmental diversity and longer context windows are crucial for enabling ICEL.

### Strengths
- The paper is easy to follow and make clear and consistent concepts. as far as know, the explicit separation between In-Context Environment Recognition (ICER) and In-Context Environment Learning (ICEL) is novel and it made the analysis more intuitive and cleaner. 

-I particularly highly appreciate the theoretical analysis, which adds rigor and depth to the work. I skimmed through the proofs and they seem correct. 

- The proposed L2World framework demonstrates the practical relevance of the theory. Experiments on well-established control tasks (cart-pole and navigation) show that world models can indeed adapt to novel environments via context, supporting the theoretical claims.

### Weaknesses
**W1** ICER vs ICEL:  The distinction is clear conceptually, but the empirical analysis in my perspective does not produce direct causal evidence that the model is performing learning inside context vs recognition + recall. The observed improvement in prediction performance with increasing context length T could be explained by several alternative mechanisms. For instance, the model might simply be recognizing previously encountered environment configurations or recalling stored representations from training, rather than actively updating its internal dynamics to learn new environmental behaviors on the fly. This part is not clear from the experiments. 

**W2** While the derived bounds in Theorem 1 is correct and rigorous. I think the derived claims from the bounds is a bit loose: some of the paper’s main claims rely on comparing these bounds. Upper bounds can be loose; which bound is smaller may not reflect which method has smaller actual error. I think some analysis on the tightness of the bounds is needed in order to solidify the claims.

**W3** The authors mainly report PSNR on pixel-level. This raises questions about whether the proposed framework would scale to more complex, real-world settings or provide tangible gains in downstream task and decision making performance.

### Questions
- 1) Theoretical results assume “sufficiently optimized” models and discrete environment complexity. How realistic are these assumptions for continuous control or visual domains? 

-  2) The analysis of Theorem 1 relies on comparing the bounds in (5). Can the authors comment on the tightness on these bounds? I understand that's hard to do theoretically. But the bounds can be very loose and hence any analysis based on them might not be correct. I think even an empirical analysis of the tightness is needed to solidify the analysis and the claims. 


- 3) Have you conducted any controlled experiments (e.g., context-swapping or corrupted contexts) to isolate ICEL behavior from ICER? maybe gradually corrupt the context  and report how quickly performance collapse? 

- 4)  Why were primarily pixel-level reconstruction metrics (e.g., PSNR) used, and do these metrics accurately reflect the model’s ability to support downstream decision-making or planning tasks?

### Soundness
2

### Presentation
2

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
This paper investigates in-context learning (ICL) in world model training, focusing on how models adapt to novel environments through ICL rather than parameter updates. The authors distinguish two ICL mechanisms: Environment Recognition (ER) and Environment Learning (EL) and derive error bounds characterizing when each emerges. They introduce L2World, a linear-attention architecture for long-context world modeling, and validate their theoretical predictions on cart-pole control and vision-based navigation tasks. The key findings emphasize that environment diversity and long context windows are critical for enabling EL over ER.

### Strengths
S1: This paper extends in-context learning studies to world model learning settings where the model should learn the inference time world, and this construct an adaptive world model. 

S2: The experimental design is thorough, it covers both continuous control (cart-pole) and partial observability (navigation). They authors also careful ablate data diversity, trajectory length and environment complexity.

S3: Verifying that ICL behaves as ER and EL in linear attention models is novel.

S4: The experimental grounding of ICEL for world model learning sets a nice expectation for future experimentalists.

### Weaknesses
W1: Most findings on ER vs EL seems already established in the literature, for example in https://arxiv.org/abs/2412.01003 and https://arxiv.org/abs/2503.05631. It doesn't seem like these are "identified" through this study.

W2: The paper's focus is a little bit unclear. there is a theoretical part, an introduction of a new architecture, and empirical experiments on ICL, it is quite unclear what is the main presented innovation/finding.

W3: The paper discusses larger context length to further enable ICEL. While this claim is likely true, it would be nice to see equal-flop experiments as longer context models get strictly more training.

W4: The presentation of the data is poor. The plots/tables could be enhanced so that, for example, the data diversity's effect can be more directly read out.

### Questions
Maybe worth citing these papers which directly discuss ER vs EL:
https://arxiv.org/abs/2412.01003
https://arxiv.org/abs/2503.05631
https://arxiv.org/abs/2506.17859

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors consider in-context learning of a world model, in particular focusing on the dichotomy between environment-learning and environment-recognition observed in other in-context learning setups. Consistent with theoretical error bounds that they derive, their main contribution is to show that long contexts and environment diversity favor environment-learning in cart-pole and procedural maze navigation tasks.

### Strengths
The paper is clearly written. To my knowledge, the conditions under which ICL task recognition vs task learning appear in world-model-based tasks have not yet been explored though the results are not surprising. The theoretical bounds are consistent with the experiments.

### Weaknesses
I can find no major weaknesses but in my opinion the paper offers few new insights compared to what is already known in the literature. The experimental paradigms considered are sufficiently complicated that it is difficult to draw specific inferences into the mechanisms that lead to ICEL versus ICER. In short, though it is certainly valuable to characterize phenomenology as the authors have done here, it is not clear what question is being answered. While this is likely beyond the scope of the current work, the authors could consider setting up the problem in a simpler, symbolic RL setup that captures the dichotomy between ICEL and ICER, and which potentially lends itself to a deeper mechanistic analysis for what factors determine ICEL vs ICER. 

Highlighting a couple of references that could be included: the ICL regression task was originally introduced by Garg et al, Neurips, 2022 (and not Raventos et al, 2023, line 106) and a minimal theoretical model that explains the origin of the ICL/IWL tradeoff was proposed by Nguyen and Reddy, ICLR 2025 (lines 111-113).

### Questions
I have no questions for the authors.

### Soundness
4

### Presentation
4

### Contribution
3
