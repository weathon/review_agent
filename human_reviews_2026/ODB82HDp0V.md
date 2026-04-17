# AssetFormer: Modular 3D Assets Generation with Autoregressive Transformer

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 8

## Abstract
The digital industry demands high-quality, diverse modular 3D assets, especially for user-generated content (UGC). In this work, we introduce AssetFormer, an autoregressive Transformer-based model designed to generate modular 3D assets from textual descriptions. Our pilot study leverages real-world modular assets collected from online platforms.  AssetFormer tackles the challenge of creating assets composed of primitives that adhere to constrained design parameters for various applications. By innovatively adapting module sequencing and decoding techniques inspired by language models, our approach enhances asset generation quality through autoregressive modeling. Initial results indicate the effectiveness of AssetFormer in streamlining asset creation for professional development and UGC scenarios. This work presents a flexible framework extendable to various types of modular 3D assets, contributing to the broader field of 3D content generation. The code is available at https://github.com/Advocate99/AssetFormer.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents a method for generating game assets as a sequence of pre-modeled 3D modules.  The experiments utilize a new dataset including 20k building models with text annotations from GPT-4o, probably the first of its kind.  The model is a decoder only transformer built on a Llama backbone.   

The paper contains interesting comparisons for the ordering in which the modules are generated in the sequence.  Qualitative and quantitative results demonstrate that this ordering plays an important role in generation quality.   A speculative decoding scheme is also presented which is shown to speed up generation with minimal loss of quality.

### Strengths
The dataset used in the experiment is a major contribution if made available to the community.  

The investigation into the ordering of the blocks is very interesting.  Understanding how best to order elements like this has applications to other problems.  It’s very interesting to see that DFS outperformed BFS.  The failure case in the qualitative figure gives a hint as to what goes wrong.  

The “SlowFast speculative decoding” is well motivated and the timings show this is effective.

The ablation studies and comparisons with baselines are sound.  Evaluation of generative models is always tricky.  To the best of my knowledge the authors have done a good job evaluating their model and comparing to other methods.

### Weaknesses
I would have liked more details on the DFS and BFS algorithms.

### Questions
I'm interested to learn more about the DFS and BFS algorithms employed.   
- Given a node with k neighbors which have not yet been visited, how do you choose which of these to add next?  Perhaps you used an off the shelf graph sorting algorithm where the behavior is not well defined.
- I would be very interested to know if some kind of lexicographical ordering in x0, x1, x2 would give a significantly different result over making this choice randomly.  I suspect the way this choice gets made would have a large impact on the results.  Perhaps the DFS is making this decision in some non-random way.  A figure showing an example of the ordering would help spot any pattern.

Line 208
> To ensure valid token set decoding, we filter out unwanted logits and re-normalize the remaining non-zero distribution
It's interesting that this was necessary.  87M parameters should be enough for the syntax of the sequence to be learned with very few invalid sequences, however 16k-20k data is very low and may not have been enough.   I would be very interested to know if this filtering actually had a substantial impact.

### Soundness
3

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
3

### Summary
This paper introduces AssetFormer, an autoregressive Transformer model designed to generate modular 3D assets from text descriptions. Addressing the limitations of traditional 3D representations for user-generated content and gaming—such as large file sizes and difficulty in editing—the authors propose representing assets as a collection of discrete primitives. The core contributions include: 1) a novel framework that models the generation of primitives (class, position, rotation) as a sequential, next-token prediction task; 2) the creation of a large-scale, high-quality hybrid dataset combining procedurally generated assets with real user-created content; 3) an innovative token re-ordering strategy based on graph traversal (DFS) to capture spatial relationships, which is shown to be crucial for generating coherent structures; and 4) the adaptation of efficient decoding techniques like SlowFast decoding for practical application. Experiments demonstrate that AssetFormer generates high-quality, diverse, and structurally sound assets that are directly usable in game engines, outperforming both procedural baselines and showing clear advantages over general-purpose mesh generation models for this task.

### Strengths
1: The decision to focus on modular assets is highly practical and directly addresses key pain points in UGC and game development. This representation is efficient, easy to edit, and perfectly suited for an autoregressive framework. 

2: The creation and combination of a procedural dataset and a real-world user-generated dataset is a major strength. The ablation study convincingly shows that this hybrid approach is superior to using either source alone, providing both structure and diversity. This dataset could be a valuable resource for the community.

### Weaknesses
1: The model relies on a fixed, discrete vocabulary for primitive types, positions, and rotations. This fundamentally limits its creative potential to the predefined set of components and a grid-based layout. It cannot generate novel primitive shapes or place them with continuous precision, which will be a constraint for more organic or complex designs.

2: The method is demonstrated on buildings composed from a specific set of architectural parts. It is unclear how well the approach would scale to a much larger vocabulary of primitives or to entirely different asset categories that may have different structural rules.


3: The text prompts are "phrase bundles" describing high-level features. While effective, this may limit the model's ability to understand more complex, compositional, or spatial instructions in natural language. The quantitative CLIP scores do not show a large separation between methods, suggesting text alignment could be further improved.

### Questions
In a real UGC platform, the library of available primitives might expand over time. How would AssetFormer handle the addition of new primitive types to its vocabulary? Would it require a full retraining, or could methods like vocabulary expansion be adapted?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a modular text-to-3D generation approach by leveraging an auto-regressive transformer model. Specifically, it introduces a new dataset comprising both real-world and procedurally generated homestead modular models, along with text descriptions derived from renderings of the 3D models using GPT-4o. This curated dataset is used to fine-tune a decoder-only transformer model, such as Llama, by applying techniques including discrete tokenization, token reordering, and classifier-free guidance. To decode the tokens generated by the model during inference, the paper proposes a method called SlowFast decoding.

### Strengths
- The use of large language models (LLMs) to generate sequential, modular 3D primitives enhances both efficiency and interpretability.
- If released, the dataset could be a valuable contribution to the community, addressing the current gap in large-scale modular 3D assets.
- The ability to directly integrate the generated 3D models in game engines unlocks numerous real-world applications.
- AssetFormer demonstrates strong generation quality through its proposed techniques, including token ordering, token sampling, and decoding strategies.

### Weaknesses
- The paper is similar in spirit to existing approaches [1,2] that utilize different 3D primitives with autoregressive transformer models for sequential 3D asset generation. While the method is adapted to a homestead-specific dataset, the core technical contribution appears incremental. Additionally, the paper would benefit from including and discussing these relevant prior works.
- Quantitative evaluation is limited, with comparisons restricted to MeshGPT. Broader benchmarking against other LLM-based 3D generation methods (e.g., [1] or [2]) would help clarify whether observed performance improvements are related to the backbone LLM-model or to the way the 3D data is preprocessed for fine-tuning the model.
- The qualitative comparisons in Figure 3(b) may not be entirely fair, as the baseline text-to-3D models are trained on more diverse 3D datasets. A more rigorous evaluation would be to fine-tune these models on the proposed homestead dataset to ensure a fair comparison.
- While the paper discusses the diversity of generated 3D assets, it does not provide qualitative results demonstrating diversity for the same prompt. Including such examples would help assess the model’s ability to produce varied outputs under identical inputs.

Additional references:

[1] MeshLLM: Empowering Large Language Models to Progressively Understand and Generate 3D Mesh

[2] LLaMA-Mesh: Unifying 3D Mesh Generation with Language Models

### Questions
- The paper lacks detailed dataset statistics, particularly regarding the distribution between real-world and procedurally generated (PCG) data. Providing separate breakdowns, similar to those shown in Figure 9, for each data source would help understand the diversity and representativeness of the training data.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
In this work, the authors introduces AssetFormer, which is a new type of autoregressive Transformer based framework. The authors design this for generating 3D assets in a modular way. Their approach focus on how assets are model from primitives and how they learn the distribution of these primitives for generative uses. The authors innovatively adapted token sequencing and decoding methods that is inspired by language models.

### Strengths
- Encouraging results.
- Theoretically well-written paper

### Weaknesses
- Can have more ablations.
- Can be good for the paper to have more expressive examples.

### Questions
What makes autoregressive modelling better than diffusion-based generation?

### Soundness
3

### Presentation
4

### Contribution
3
