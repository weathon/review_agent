# Compositional-ARC: Assessing Systematic Generalization in Abstract Spatial Reasoning

- Decision: Accept (Poster)
- Scores: 8, 4, 2, 6

## Abstract
Systematic generalization refers to the capacity to understand and generate novel combinations from known components. Despite recent progress by large language models (LLMs) across various domains, these models often fail to extend their knowledge to novel compositional scenarios, revealing notable limitations in systematic generalization. There has been an ongoing debate about whether neural networks possess the capacity for systematic generalization, with recent studies suggesting that meta-learning approaches designed for compositionality can significantly enhance this ability. However, these insights have largely been confined to linguistic problems, leaving their applicability to other tasks an open question. In this study, we extend meta-learning for compositionality to the domain of abstract spatial reasoning. To this end, we introduce $\textit{Compositional-ARC}\textemdash{}$a dataset designed to evaluate the capacity of models to systematically generalize from known geometric transformations (e.g., translation, rotation) of abstract two-dimensional objects to novel combinations of these transformations (e.g., translation+rotation). Our results show that a small transformer-based encoder-decoder model, trained via meta-learning for compositionality, can systematically generalize to previously unseen transformation compositions. Notably, despite having only 5.7M parameters, this model significantly outperforms state-of-the-art LLMs$\textemdash{}$including o3-mini, GPT-4o, and Gemini 2.0 Flash, which fail to exhibit similar systematic behavior$\textemdash{}$and performs on par with the winning model of the ARC prize 2024, an 8B-parameter LLM trained via test-time training. Our findings highlight the effectiveness of meta-learning in promoting systematicity beyond linguistic tasks, suggesting a promising direction toward more robust and generalizable models.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper is a straightforward application of a meta-learning method --- developed by Lake and Baroni --- to the task of compositional transformations in 2D grid patterns.  The authors created a dataset of such transformations that involves training and testing "episodes" using different transformation grammars, and Lake and Baroni's meta-learning approach to train a small transformer network to perform such compositional transformations.  This method produces a model (MLC) that outperforms much larger state-of-the-art LLMs (ones that are not specifically trained on this dataset / task).

### Strengths
The paper is well-written, understandable, and offers a solid contribution to the literature on meta-learning and systematicity / compositionality  in AI models.

### Weaknesses
While the results are impressive, it's not clear from the paper how such results could be incorporated in models that use this ability for compositional reasoning in less narrow tasks.  For example, how would the authors use these results in service of, say, solving ARC tasks?  How could the MLC model be used as a component of such a system, or as a way to help train such a system?   It would be useful if the authors discussed this point -- where to go from here.   The authors state ""Our findings suggest that MLC presents a promising direction for enabling systematic generalization in language models across diverse domains."   Please say more about what these directions are.

### Questions
In Figure 1, for level 2, the ordering of primitive transformations matter, correct?  What is this ordering? 

The task addressed in this paper has several constraints: 10x10 grids, 10 colors, only one or two objects, a small repertoire of transformations, and restrictions on those transformations (e.g., rotations are only 90-degrees).   To what extent does MLC rely on (overfit to) these constraints, and to what extent could it generalize beyond them?  If MLC fails on larger grids or on larger numbers of objects, does that mean that it hasn't actually learned any general compositional abilities?  If these restrictions were relaxed, how would that affect the training / success of the network?  A discussion of all this, and discussion of possible future research directions, would be helpful.  

For Figure 3, it would be useful to specify what the primitive transformations are that are illustrated.  This would make the set-up clearer to readers. 

The authors state: "we consider a visual interpretation grammar that associates visual indicators (object  shape, color, or proximity to an indicator object) with specific geometric transformations".  -- Why refer to these patterns as "visual", when the model is given the grids in text format?  Why not just say these are 2D patterns?  It's not clear that the MLC model "perceives" objects, shapes, etc., or even that it needs to in order to perform the task correctly.  In discussing ARC, Chollet is adamant in stating that ARC tasks are not *visual* tasks, but rather 2D pattern-recognition tasks.  It would be useful to comment on this in the paper. 

Why does the model need the "auxiliary copy task" to perform well?  

The authors state, ""we split the data into 82,908 training, 8,546 validation and 8,546
test episodes."   -- Why use these values?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Introduces Compositional-ARC, a new dataset and benchmark to study systematic generalization in abstract spatial reasoning. Focuses on geometric transformations - translation, rotation, reflection, extension and color - and they are applied on 2D grid objects
Uses meta-learning for compositionality (MLC) framework from Lake & Baroni (2023) paper to extend the approach from linguistic domain to the spatial reasoning domain. The evaluations includes both general-purpose and ARC-specific LLMs.

### Strengths
- Systematic generalization is important challenge and the motivation is well explained
- The dataset is carefully formulated and explained in a good detailed manner
- Good empirical analysis and evaluations and ablations are reported as well

### Weaknesses
- Core algorithm (MLC) is adopted unchanged; and is applied to a new domain. It is more of a new dataset/benchmark track? Need some clarification on the position of the study
- The Scope is quite narrow and restrictive - 10x10 grids, rotation angles, number of objects, occlusion, boundaries
It is unclear how insights generalize to richer or naturalistic visual domains (e.g., CLEVR-style reasoning
- Evaluation fairness? - Many comparisons with LLMs use text-formatted grids as prompts. These models are not optimized for symbolic grid reasoning, so the Claim that small MLC model can outperform SOTA LLMs is needs more validation
- Analysis of failure modes? Where it fails and why


[1] CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning

### Questions
- What is the training procedure (meta-learning part). It will be good to have some details in this paper as well, instead of having to refer to the other paper
- Line 266 - can you explain the two special tokens and how they mark the boundaries?
- Auxiliary task - What is intuition behind this? In ablation this is adding more value, which is quite strange
- Some figures are overloaded (Fig. 3) to understand all the actions happening easily

### Soundness
3

### Presentation
4

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
The study proposed a dataset to study the compositionality of transformers. In some way, the proposed Compositional-ARC dataset is similar to the ARC dataset, which contains pixel objects on a grid. The task is to infer the transformation given an input grid. The ground truth transformation depends on the “cues” in the input, such as the shape or color of the object. The authors study whether transformers can learn to compose transformations indicated by the composed cues, and with several extra few-shot composition examples. The authors train a transformer that can achieve this task almost perfectly. However, given the simplicity of the dataset and task, it is hard to infer the actual compositionality ability of transformer models.

### Strengths
The composition ability is one of the most fascinating properties of LLMs. The authors study this property in a controlled toy setting. A small transformer model, after training, can solve this toy task perfectly.

### Weaknesses
I have major concerns that the setup and analysis are too simple, such that the conclusions and implications are limited. 

**Simplicity**

Unlike the original ARC task, the number of possible rules underlying the transformations is small. All the primitive transformations are about shift, rotation, and extension. Given 3-shot examples in some of the testing cases further simplifies the task. It is expected that a small transformer can learn this task almost perfectly. It is hard to say how much of such “systematic generalizability” is in real-world data and models. Though the authors do not provide any theoretical analysis, it may not be difficult to formulate this task such that all the transformation rules are orthogonal in some linear space, then the compositionally is simply a consequence of retrieving the cues and adding them together. The failure of this task in the frontier model could be due to the number of examples provided in the context. Since frontier models are not trained on the task directly, it may not be a fair comparison between them and MLC (author’s).

**overly claimed LLM failures**

The authors overly emphasize the failure case of compositionality and generalizability in LLMs (e.g. Line 115). The cited studies, Ismayilzada et al., 2025; Dziri et al., 2023, use more or less simple synthetic settings. On the large scale, frontier LLMs can consistently show surprising compositionality and generalizability. For example, a user can ask an LLM to answer a question in a specific tone or with other constraints. In a way, it is because of the compositionality and generalizability of LLM that alignment, such as RLHF, can be feasible and effective. Authors should introduce and discuss what is known about LLM’s compositionality and generalizability, rather than introducing only the pessimistic side under simplified unrealistic settings.

**Clearity**

In the introduction, the authors motivate the study with “systematic generalization” in human and AI. Without a definition or more context, it is hard to know what “systematicity” refers to. How is “systematic generalization” different from naive “generalization”?

Similarly, meta-learning for compositionality (MLC) is not defined and formalized. Providing general setups of MLC would be helpful.

**Structure**

The subtitle “Task setup” in the section Background is confusing. Overall, the background section is cumbersome, with too many details on a small number of studies.

### Questions
For my main concerns, see "weakness".

Why is the task considered meta-learning? Is it fair to say this task is supervised learning with online adaptation? The task may be considered as learning a "meta-skill", which is not the same as the common definition of "meta-learning".

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Compositional-ARC: a dataset for evaluating LLMs capacity for systematic generalization & compositionality with a methodology similar to that in Lake & Baroni (2023), but in a visual domain. The authors provide a number of empirical results to support their MLC-based approach.

### Strengths
The paper is focused on an important and relevant task
The paper is clearly written and is a pleasure to read.
The results are promising.

### Weaknesses
The main weaknesses, in my view, are:

- Slightly misleading performance comparisons. The authors compare a model specifically trained for this ARC subtype with more generalist models, or with models designed to perform well on ARC (but not on the new Compositional ARC). Still, Compositional ARC is a reasonable subset of all possible "ARC-lke" problems, so this is not a disqualifying consideration.

- Low practical applicability. While the question of compositionality is fundamental, the authors substantially narrowed the scope of an already artificial dataset (ARC). This limits the potential impact of the paper and introduces the risk of creating models that are increasingly detached from real-world applications. Still, such research has lots of value and so again, this is not a disqualifying weakness.

### Questions
What is your explanation for imperfect color accuracy in systematicity evaluation part of the proposed model? While 97+ is quite high, compared to other models, this component seems to be lacking.

### Soundness
3

### Presentation
3

### Contribution
3
