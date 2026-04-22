# Multiverse Mechanica: A Testbed for Learning Game Mechanics via Counterfactual Worlds

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 6, 6, 4

## Abstract
We study how generative world models trained on video games can go beyond mere reproduction of gameplay visuals to learning game mechanics—the modular rules that causally govern gameplay. We introduce a formalization of the concept of game mechanics that operationalizes mechanic-learning as a causal counterfactual inference task and uses the causal consistency principle to address the challenge of generating gameplay with world models that do not violate game rules. We present Multiverse Mechanica, a playable video game testbed that implements a set of ground truth game mechanics based on our causal formalism. The game natively emits training data, where each training example is paired with a set of causal DAGs that encode causality, consistency, and counterfactual dependence specific to the mechanic that is in play—these provide additional artifacts that could be leveraged in mechanic-learning experiments. We provide a proof-of-concept that demonstrates fine-tuning a pre-trained model that targets mechanic learning. Multiverse Mechanica is a testbed that provides a reproducible, low-cost path for studying and comparing methods that aim to learn game mechanics—not just pixels.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper employs the formalism of causal learning to formulate the problem of learning game mechanics from visual observations. The paper proposes a framework to describe how interventions on (discrete) game states alter the corresponding image representations under different data generation conditions. This includes an example domain to generate data, including the relevant causal modeling information. To evalute the approach a diffusion model is trained to predict the visual outcomes of causal interventions generated using the framework.

### Strengths
# originality
Provides more formal rigor on defining the task of learning game mechanics from observations compared to prior efforts. The causal learning model here is novel relative to prior formalizations.


# quality
Explicitly seeks to create a data generator for model training that matches the formalism.


# clarity
Introduces the case study domain with a worked example (Table 1).


# significance
Opens the path to investigating causal learning for game mechanics based on discrete states. This could be of interest to studies of world models and their learning and limitations. Also for learning world models for game playing agents from demonstration videos.

### Weaknesses
# quality
The experiments seem mismatched to the objective of formalizing causal learning. The model starts by framing the problem as learning from video, then reduces that to a few input frames. Unfortunately, the evaluation metrics do not touch on the central question of whether the generated data was neccessary or sufficient for causal learning. Under questions I offer some suggestions on what experiments could help directly test this core idea. With development I believe these would be a strong launching point for the ideas to see further development and extension.

# clarity
The notation often was confusing and cluttered, rather than clarifying on the formalism. Perhaps it would help to have some listing of the symbols used and their definitions for easy reader reference.

The causal hierarchy being adopted should be introduced as background to readers. It's a core part of the framework, but never directly explained.

### Questions
# questions
- How are causal hierarchy level-1, level-2, and level-3 defined?
- How does the experiment demonstrate the need for level-3 data? (claims in lines 178-182)
- How does the experiment demonstrate generalization? (claims in lines 92-93)


# suggestions
- The following experiments may be a natural progression to test the claim that the level-2 and level-3 data are required for learning game mechanics. Given the established criteria of how well the model predicts outcomes given information on the intervention:
	- As a base experiment, remove all visual information (or map it as a single pixel indicating each state) and demonstrate the core claims. This setting decouples model representational learning capacity from causal learning. Then demonstrate that the model trained with equivalent amounts of level-1, level-2, and level-3 data shows the expected pattern of succeeding in tests only with level-3 and not the prior levels. Perhaps include a small scaling test for samples required.
		- The outcomes can easily be validated automatically without human evaluation. This bypasses the problems of assessment and directly tests the core claims.
	- Add the current scenario and evaluate on the above metrics.
	- Add the ability to generate scenarios that violate the assumptions required, like (fully or partially) hiding visual representation of the relevant causal variables.
	- Add generation of full videos and require learning on those to support the claim in the formalism around video input.
	- Add procedural variation of assets and placements to the framework to test generalization.

### Soundness
1

### Presentation
2

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
The authors tackle the relevant but understudied problem of determining whether learnable models can correctly learn game mechanics. To do so, the authors contribute Multiverse Mechanica, a video game containing simple visuals, but reasonably complex game mechanics, which could serve as a benchmark for game mechanics learning. Defining characteristics of Multiverse Mechanica include its logic formalism for precisely describing game mechanics, and its ability to mark "impact frames", a sparse set of video frames containing full information on game mechanics which provide a convenient way to perform cheap experimentation. A latent diffusion model is trained on data produced by the synthetic environment, leveraging the established logical formalism to improve model understanding of game mechanics, but evaluation makes it difficult to determine whether this had positive impact on the model. The paper is well written and explains Multiverse Mechanics in great detail. Overall Multiverse Mechanica has potential to serve as a benchmark for game mechanics learning and the logical formalism introduced in the work could spark new ideas in the field, though it remains unclear if employment of such logical formalism will become applicable to more complex models and scenarios of practical interest.

### Strengths
- The paper tackles the relevant but not yet well studied problem of studying the degree to which generative models are capable of understanding game mechanics. In the rise of powerful video generation models investigation of this hard-to-achieve capability can become an interesting research direction.
- The authors contribute a videogame that was carefully designed to probe understanding of relatively complex game mechanics under constrained training budgets. The game is reasonably complex from the perspective of the contained mechanics, requiring reasoning on spell types, elemental affinities, weapon classes and equipment to make decision on taken damages. At the same time, its visual simplicity and ability to produce "impact frames" make it convenient for small scale ablations. This can constitute a good benchmark or tool for ablations for future works.
- The paper seems polished and detailed. Writing is clear and the appendix with its 45 total pages describes in great detail the proposed system and logic framework.

### Weaknesses
- The logic formalism employed in the paper may be unfamiliar to a large portion of readers. The authors make every effort to introduce the formalism, but the paper might not easily accessible to the broad generative models audience.
- The employed logic formalism appears difficult to define even for a relatively simple game as the one created by the authors. A concern is whether the proposed formalism and methodology could eventually scale to modeling the more complex game mechanics encountered in mainstream video game titles. If such possibility is not foreseen, the impact of the method could be limited. I could see a use for the formalism in evaluating in a rigorous way video sequences for the purpose of determining if a model has learned the described mechanic correctly, but I don't see an easy way of scaling training as performed in Sec 5 on a more complex scenario.
- The proof of concept model shown in Sec. 5 is built based on the assumption that contrasts are known, a condition that is satisfied for the proposed synthetic data generator. For example, it proposes a seed consistency loss aligning noise for samples across contrasts and a structure alignment loss. These components would be difficult to leverage in the context of large scale video generation methods, which would be the ultimate models on which to perform evaluation for game mechanics understanding, but that would require large amounts of data for which contrasts would be very difficult to derive.
- The experimental setting with respect to employed metrics and experimental setting is unclear, making it difficult to interpret results. I suggest authors to describe each of the employed metrics and its computation procedure.
- Interpretation of the experimental results is difficult because of the lack of baselines or ablations. The paper does not validate whether any of the introduced components in Sec 5 improves model performance with respect to straightforward diffusion model training.

### Questions
I'd like authors to clarify the aspects highlighted in the weaknesses section with particular regards to the concerns regarding clearer definition of experimental setting, definition of metrics, and creation of a baseline against which to compare the proposed method's performance.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors present a new game environment designed to test the ability for pixel-based game world models to accurately learn game mechanics. The environment is designed to automatically produce causal graphs that represent specific game mechanics to allow for ground-truth checking against model predictions. The authors train a proof-of-concept model in their proposed environment and argue that it points the way towards richer training of other world models.

### Strengths
I think the motivation of this paper is quite compelling. The introduction does a good job both laying out existing perspectives on the nature of pixel-based world models and arguing for them to be made more rigorous and formal. In general, the introduction does a good job of laying out why the paper's contributions would be both novel and timely. The Multiverse Mechanica game, while not currently available, does also seem like a useful contribution to other researchers interested in studying world models. While the technical foundations presented in the paper are at times a bit confusing (see below), they also point towards the authors' deep understanding of the  subject matter.

### Weaknesses
I think the paper suffers somewhat in terms of clarity. I admit that this may be due to my own unfamiliarity with the relevant mathematical underpinnings, but I found the technical descriptions in Section 3 fairly difficult to follow and the accompanying figures too busy to easily parse. Later sections use a variety of evaluation methods without clearly defining them (e.g. exogenous distance and transfer CLIP), which makes it difficult to clearly understand the implications of the proof-of-concept experiment. In particular, I wonder about the claim that “the model successfully generates semantically meaningful counterfactuals” -- this seems like something that would benefit greatly from one or more concrete example outputs. In the absence of any baseline models against which to compare or a dedicated user study to evaluate the outputs, it’s difficult to say what the model has accomplished.

### Questions
- How should we interpret the quantitative results in Table 3? More specifically, what might an unsuccessful model and an oracle model look like?
- Line 161: is this meant to say the rows of Table 1 instead of columns?
- For S3 in Table 1: why is it necessary to condition S = 1? Are there ways for a player to block aside from having a shield, and does blocking have different behavior in that context?

### Soundness
3

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
3

### Summary
The paper proposes a causal framework for making generative models mechanically consistent (that is, able to change only the causal effects of an intervention while keeping all else fixed). To this end, the authors introduce Multiverse Mechanica, a controllable game environment that generates counterfactual video pairs: parallel worlds sharing the same exogenous randomness (non-descendant variables) but differing in a single game mechanic (e.g., shield on/off). Using single impact frames from these videos, they finetune a pretrained T2I diffusion model (OpenJourney-v4) with a new Multiverse Alignment Loss that enforces (a) shared exogenous seed consistency and (b) early-stage structural alignment. Experiments show that the finetuned model preserves global scene structure while correctly toggling mechanic-specific effects, demonstrating a step toward causally consistent generative modeling.

### Strengths
- The paper introduces a causally principled definition of mechanical consistency via counterfactual constraint. Unlike previous works which focused on visual realism, it provides a framework of grounding generative modeling in causal reasoning.

- The Multiverse Mechanica environment provides a controllable, reproducible setting with groundtruth causal graphs and consistent counterfactual pairs. This is valuable for studying causal generalization in generative models. (Although the code and dataset release is remained to be seen.)

- The proposed Multiverse Alignment Loss (seed consistency & structure alignment) is intuitive and conceptually sound. It aligns neatly with the diffusion process’s structure (e.g., early steps for global layout; late steps for fine details).

- In their proof-of-concept experiment, qualitative and quantitative results show that the finetuned diffusion model preserves global scene structure while correctly toggling mechanic-dependent effects, supporting the causal-consistency claim.

### Weaknesses
- The approach is tested only on _single-frame_ impact images, not full temporal sequences. This prevents evaluation of whether the proposed causal consistency generalizes across time, although the authors claim their framework should, in principle, extend to video models.

- All results focus on a single mechanic example ("shield on/off") and one pretrained model (OpenJourney-v4). It remains unclear whether the framework scales to multiple simultaneous mechanics or to other base models. Again, this should work in principle, but empirical evidence is still lacking.

- In terms of empirical validation, the paper also lacks ablation and baseline comparisons. Since the new loss formulation is one of the main contributions, it would be highly beneficial to show what happens when either the seed-consistency (L1) or structure-alignment (L2) term is removed, and to compare against simpler alternatives (for example, standard fine-tuning or contrastive alignment).

- Finally, the evaluation metrics are indirect. While CLIP, PSNR, and SSIM provide useful proxies, they do not directly capture causal correctness. The paper could propose or adopt explicit causal evaluation metrics to better quantify causal consistency.

### Questions
See the above weaknesses.

Besides, I have one question (or suggestion): do you think your framework could be applied beyond the game domain? It seems the same causal-consistency formulation could extend to other areas such as physics simulation or movie generation. If so, adding a short paragraph discussing these broader applications or potential impacts could help widen the paper’s relevance and readership.

### Soundness
2

### Presentation
3

### Contribution
3
