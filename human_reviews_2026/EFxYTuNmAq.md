# Level Up: Defining and Exploiting Transitional Problems for Curriculum Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
Curriculum learning, ordering training examples in a sequence based on difficulty, takes inspiration from human learning but has not gained widespread acceptance. Static strategies for scoring item difficulty produce curricula that are not specific to the learner at hand, and that rely on indirect proxy scores of varying quality. Dynamic approaches base difficulty estimates on gradient information, requiring considerable extra computation during training. We introduce a novel method for measuring the difficulty of individual problem instances directly relative to the ability of a given model, and identify transitional problems that are consistently easier as model ability increases. Applying this method to chess and mathematics, we find that training on appropriately calibrated problems most efficiently "levels up" a model to the next competence tier. These problems induce a natural progression from easier to harder items, which outperforms other training strategies. By measuring difficulty directly relative to model competence, our method yields interpretable transition problems, learner-specific curricula, and a principled basis for step-by-step improvement.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a framework for evaluting the difficulty of data via "Transitional Problems". The authors argue that this approach is more efficient gradient-based approaches and generalized better than domain-specific curriculum. Transition problems require a set of models of different strengths, which are then monotonically ordered according to a performance metric; a sample's difficulty is then rated according to its transition point, i.e. where Model $i$ fails on said point but model $i+1$ (and all subsequent models) succeed on the sample. 
The author's hypothesis is that training a model with level $t-1$ on datapoints with transition point $t$ is effectively the best data to "level-up" the model into a skill equivalent to model $t$. 

The authors provide experiments in both the chess and math domain and show that (1) in Fig 3, starting from multiple model levels, the data subset that will best improve performance is not necessarily the hardest or the easiest data, but really the dataset that's "one-level up". Then, (2) the authors leverage this insight to design a curriculum training approach, where the monotonically increasing levels are presented sequentially to the learner, showing it ouperforms random sampling and reverse curriculum.

### Strengths
The framework presented by the authors is clear, and departs from standard approaches to assess the difficulties of samples and curriculum generation. What I especially like is the tailoring of the curriculum to the model in the math and chess settings. 

The experiments are well designed. The proposed approach has potentially nice applications in the chess setting, where one could leverage this data curriculum to better train human players depending on their level.

### Weaknesses
1. My biggest criticism of this work lies in the "chicken-and-egg" problem to actually use this for model training. This method can design model-specific curricula from having an already trained model. To this end, what's the use of finding the optimal data ordering specific to a model if said model is already trained ?

2. The results for the curriculum learning for math are somewhat inconclusive. It seems that this approach only shows gains in the very limited dataset size regime. This is a far cry from the author's motivation tying curriculum learning to the different training phases of foundation models, which have many (many) orders of magniture more samples. 

3. The claim that this appraoch is more computationally efficient than gradient based approaches is unclear. The authors still require multiple inference calls per datapoint, which is more expansive than computing a single gradient. Could the authors please provide more details for this?

### Questions
1. What happens in the lower-triangular part of Figure 4 and Figure 5 ? For completeness, it would be good to show that training on "levels down" leads to worse performance. 

2. "At Internet scale, the potential for faster convergence with
curriculum learning methods becomes a valuable consideration due to the considerable expense of
training." -> Thi is not a Curriculum learning argument, it should be a continual learning one (cannot assume data distribution is monotonically increasing in difficulty.)

3. Related to my previous comment about the "chicken-and-egg" situation, could you provide other applications or usages of your method beyond model (re) training?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces **transitional problems**, which are examples that weaker models fail but slightly stronger models can always solve. Instead of using human-defined difficulty or extra computation to estimate it, the authors define difficulty based on the model’s actual abilities. They test this idea in chess and math reasoning, showing that training a model on the transitional problems from just one level above its current skill leads to the fastest improvement, beating random training and reverse curricula. The core claim is that curriculum learning works when the data difficulty is aligned with the learner’s real competence. The main limitation is that finding these problems currently depends on having access to a stronger model, which the paper leaves as future work to address.

### Strengths
1. The paper proposes a clear and original way to define task difficulty based on the model’s actual capability, instead of relying on human judgment or handcrafted heuristics. This directly addresses a long-standing issue in curriculum learning.
2. The approach is validated in two very different domains, chess and math reasoning, and shows consistent improvements in both.
3. The experimental setup is well controlled. The comparison between ascending curriculum, i.i.d training, and reverse curriculum makes it easy to see where the gains come from.

### Weaknesses
1. Practical constraints in defining transitional problems
* The approach relies on the existence of a strictly stronger model in order to identify transitional problems. This requirement limits its practicality, as it introduces a non-trivial upfront cost. In effect, the method assumes that one must first perform a less efficient stage of training in order to later enable more efficient learning, which creates an inherent paradox.
2. Limited scale and stability in the math experiments
* While the results in the chess domain are highly stable, the outcomes in the math setting exhibit greater variance and weaker overall gains. This raises concerns about the generality of the proposed method, especially in reasoning-oriented tasks where dataset size and quality remain significant limitations.
* If training on the immediately higher level helps performance, it would be helpful to show how gradual transition learning compares to training with all data randomly mixed.
3. Uncertain separability of capability levels across model checkpoints
* The paper assumes clear performance gaps between successive model versions, yet this assumption may not hold in practice. In the context of LLM fine-tuning in particular, stochastic training dynamics can make it difficult to define a clean, deterministic notion of “one level higher” model capability.
4. The experiments are conducted on only one model, so it is difficult to see whether the proposed explanation applies to all models, especially large foundation models.
5. The description of the IID baseline could be made clearer.
6. The performance gain obtained through transitional learning is less than 1%, which seems very marginal, and there is concern about whether transitional problems can be defined across diverse tasks.

### Questions
* How does the size of the transitional problem set relate to performance gains? The paper does not include experiments that vary or reduce the amount of transitional data, so the scaling behavior remains unclear.
* Is it possible to estimate transition points without access to a stronger reference model? 
* How does the length of the curriculum steps affect performance? A more detailed analysis could clarify whether there is an optimal depth.
* In the chess domain, how much do distributional differences between puzzles and positions influence the reported results?

### Soundness
3

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
This paper proposes a novel framework for curriculum learning by introducing the concept of transitional problems - problems that show a monotonic decrease in difficulty as a model’s competence increases. The authors define formal criteria for identifying such problems within a model series and demonstrate the approach in two domains: chess and mathematical reasoning. Empirically, they find that fine-tuning models on transitional problems from the next competence level yields the most efficient performance improvements (“levels up” the model). Furthermore, curricula constructed from ascending sets of transitional problems outperform i.i.d. and descending curricula. The paper argues that this model-centric notion of difficulty provides a more principled basis for learner-specific curricula compared to human-centric or indirect proxy scoring approaches.

### Strengths
1. The idea of “leveling up” through transitional problems is intuitively compelling and well-motivated. The introduction of transitional problems as a means to measure difficulty relative to model competence is novel. It reframes curriculum learning around model-centric difficulty rather than human intuition.

2. The diagrams (especially Figure 1 and Figure 2) clearly convey the central idea and experimental setup, which help intuitively understand how transitional problems are defined and applied in curricula.

3. The chess experiments make sense and the results show consistent, convincing trends and strong correspondence with the hypothesis.

### Weaknesses
1. While the paper’s model-centric notion of difficulty is novel, it remains empirically motivated. It would strengthen the contribution to show why this definition is theoretically advantageous over other model-based measures (e.g., gradient norm, loss change, or C-score). Currently, the argument for why this formulation is superior is mostly intuitive rather than analytical or empirical.

2. Constructing a model series satisfying Definition 1 (monotonic increasing strength) is non-trivial in general settings, and the definition of transitional problems as those showing "monotonic decrease in difficulty as the model gets more capable" remains somewhat vague without clear guidelines for measuring capability. While chess serves as a strong example with Elo-based strength, extending this to a general ML method requires broader empirical validation across diverse tasks, model sizes, architectures, and domains beyond games and math.

3. The strength function appears ad-hoc and inconsistent across domains: Elo ratings in chess (external benchmark) versus validation accuracy in math (intrinsic metric). This raises concerns about generalizability and soundness. For other tasks, how should one determine an appropriate strength function?

4. The authors state that fine-tuning Qwen3-0.6B-Base on GSM8k improves from 3% to 40%, but the Qwen3 technical report lists 59.59% accuracy for this base model. Could you clarify how the Qwen3-0.6B checkpoints were constructed and why the model performance differs from reported numbers in the official technical report?

5. While the math experiments largely corroborate the trends from chess, some results appear inconsistent with the core hypothesis. For instance, in Figure 6's subplot for Level C3 → C4, the peak performance occurs at a lower difficulty level (Q-2) rather than the hypothesized "level-up" position (green line at Q-4). This deviation, with a noticeable drop in the curve around the expected optimum, suggests potential limitations in the method's robustness for math reasoning. Do you have hypotheses for these deviations?

6. While the paper reads smoothly overall, several minor presentation issues (e.g., inconsistent use of “Fig.” vs. “Figure,” missing figure references, and extra spaces) suggest the paper could benefit from a careful proofreading pass.

### Questions
Same as the Weakness section. I would be happy to discuss these points further during the rebuttal phase.

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
3

### Summary
This paper introduces "transitional problems", a novel method for defining a curriculum based on a model's own learning trajectory. Problems are partitioned into discrete difficulty levels based on which model checkpoint in a series is first able to solve them, and consistently able to solve them after that. The authors show that for a model at competence level $i$, training on problems from level $i+1$ is the most efficient way to improve performance on other held-out problems from that same level. An ascending curriculum built on this principle shows modest gains over baselines.

### Strengths
- Novel approach to define sample difficulty, motivated by concepts from developmental psychology
- The experiments designed to test the "leveling up" hypothesis for a single $i$ -> $i+1$ step are compelling.

### Weaknesses
- The paper proves efficiency for single-step transitions but fails to provide any evidence that chaining these steps leads to a globally optimal or more efficient training process for achieving the best possible final model.
- The evaluation framework is self-referential. Showing that training on "level $k$ problems" makes a model better at "level $k$ problems" is an expected result and not a convincing demonstration of improved general task competence.
- The method requires an existing pre-trained oracle model to define the curriculum, making it inapplicable to the standard use-case of training a model from scratch.

### Questions
- The paper's claim relies on the assumption that a series of locally optimal steps ($i$ -> $i+1$) results in a globally optimal training path. Can the authors provide theoretical or empirical evidence to support this idea, as it is not self-evident?
- How can you be sure that the observed "leveling up" is not just an artifact of in-distribution generalization on a narrow data slice? Could you show, for instance, that training a model $M_i$ on $D_{i+1}$ leads to a greater performance increase on a general, unfiltered test set than training on $D_i$ or $D_{i+2}$?

### Soundness
2

### Presentation
4

### Contribution
3
