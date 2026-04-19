# Can Large Language Models Understand Symbolic Graphics Programs?

- Decision: Accept (Spotlight)
- Scores: 8, 8, 6

## Abstract
Against the backdrop of enthusiasm for large language models (LLMs), there is a growing need to scientifically assess their capabilities and shortcomings. This is nontrivial in part because it is difficult to find tasks which the models have not encountered during training. Utilizing symbolic graphics programs, we propose a domain well-suited to test multiple spatial-semantic reasoning skills of LLMs. Popular in computer graphics, these programs procedurally generate visual data. While LLMs exhibit impressive skills in general program synthesis and analysis, symbolic graphics programs offer a new layer of evaluation: they allow us to test an LLM's ability to answer semantic questions about the images or 3D geometries without a vision encoder. To semantically understand the symbolic programs, LLMs would need to possess the ability to "imagine" and reason how the corresponding graphics content would look with only the symbolic description of the local curvatures and strokes. We use this task to evaluate LLMs by creating a large benchmark for the semantic visual understanding of symbolic graphics programs, built procedurally with minimal human effort. Particular emphasis is placed on transformations of images that leave the image level semantics invariant while introducing significant changes to the underlying program. We evaluate commercial and open-source LLMs on our benchmark to assess their ability to reason about visual output of programs, finding that LLMs considered stronger at reasoning generally perform better. Lastly, we introduce a novel method to improve this ability -- Symbolic Instruction Tuning (SIT), in which the LLM is finetuned with pre-collected instruction data on symbolic graphics programs. Interestingly, we find that SIT not only improves LLM's understanding on symbolic programs, but it also improves general reasoning ability on various other benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors explore the use of symbolic graphical programs (SGPs) as a means to assess the reasoning abilities of large language models (LLMs). SGPs are defined using visual markup languages and can be compiled by symbolic programs to produce images, such as SVGs in 2D or CAD models in 2D/3D. The central hypothesis is that the high-level reasoning skills of an LLM are linked to its capacity to (1) identify complex objects from primitive shapes and answer questions about them (Semantic Understanding) and (2) recognize when different sets of primitive shapes produce the same overall program (Consistency). To test this hypothesis quantitatively, the authors compiled approximately 4,000 program-question pairs for SVG images and 5,000 pairs for CAD programs. Their findings reveal that (1) open-weight models significantly lag behind proprietary ones in performance, (2) fine-tuning using an intermediate representation substantially improves reasoning performance, and (3) all models face challenges in distinguishing MNIST digits presented as SVG schemas.

### Strengths
- SGPs offer an intriguing middle ground between perception and reasoning. While humans can intuitively link rendered images to semantic concepts, associating these same concepts with the underlying SVG schema may demand additional time and coding knowledge.
- The dataset presented is highly scalable, as millions of SVG images and CAD programs with permissive licenses are available online for use in this benchmark.
- The proposed experiments are both well-motivated and effectively executed.
- The presentation is clear, consistent, and transparent.

### Weaknesses
* L108: "highly scalable benchmark." While I agree that SGPs have this potential, the paper does not demonstrate specific contributions toward achieving scalability. For example, benchmarks like [MineDojo](https://minedojo.org/) achieve high scalability by developing tools to efficiently scrape relevant data from platforms like YouTube, Reddit, and Wikia, resulting in a dataset of around 700k Minecraft videos and 6M+ comments. In contrast, the SGP dataset in this paper only contains ~5,000 data points. This is surprising given the abundance of online SVG images (for example, [SVGRepo](https://www.svgrepo.com/) claims to have over 500k SVGs). If SGP-Bench were a "highly scalable" benchmark, I would have expected an order of magnitude more data points or an explanation of the barriers preventing larger-scale sampling. Although this does not diminish the significance of the benchmark (Multimodal reasoning benchmark do not have to be "internet-scale" to be influential: https://arxiv.org/abs/2401.06209), the scalability claims should be adjusted accordingly
* The current definition of symbolic graphical programs might be overly broad. For instance, HTML could technically qualify as an SGP within the given definition since it uses a consistent markup language and rendering engines (like Chrome, Safari, or Firefox). While this isn’t inherently a weakness, it could impact the paper's novelty claims. If the SGP definition admits HTML as well, the authors might need to expand their related work and adjust their novelty claims to include efforts like https://arxiv.org/pdf/2210.03347, https://mbzuai-llm.github.io/webpage2code/, https://arxiv.org/abs/2403.09029 (and other relevant insights from work on action transformers for webpage understanding).  

- [Figure 4] the comparison between OpenAI-o1 (a model likely using Chain of Thought reasoning) and standard zero-shot LLM queries using Llama3.1 70B seems problematic. However, this qualitative example showcases how  CoT reasoning might be a natural way for increasing LLM performance on this benchmark. Is there a particular reason the benchmarked models (Table 1)  do not evaluate the respective models under the CoT (or similar multi-step reasoning) paradigm?  

- What are some potential "reasoning shortcuts" that could enable an LLM to perform well on this dataset without possessing true high-level reasoning abilities? For instance:
  - Data Leakage: Since the SVGs were sourced from the internet, it’s possible that proprietary LLMs were trained on some SVG/caption pairs. The authors propose the semantic consistency experiments to evaluate this. I believe that this partially mitigate this issue (as the jittered images likely wouldn't appear in the training set), but I'm concerned that jittering the image might not modify the underlying XML structure enough to conclusively reject the influence of data leakage. There are many ways to further validate this.
    - Calculate the tree edit distance (The Zhang Shasha algorithm is implemented in `pip install zss`) between the original and jittered SVG. Is the tree-edit-distance above a reasonable minimum threshold? Is the tree-edit-distance correlated with the relative performance difference? If low absolute correlation is observed, then the performance is invariant to the SVG structure, and data leakage might be absent.
    - Evaluate the performance of open-sourced models on obfuscated SVG files (Tools like https://github.com/svg/svgo will minify SVGs). If the minifier significantly alters the XML structure and performance remains unchanged, it could indicate no data leakage. 
    - Of course, the authors may have other effective methods to test this. This isn’t necessarily a criticism of the paper, but rather a recognition that  the benchmark offers a unique opportunity to measure performance gains in LLM reasoning attributed to data leakage.
  
- L396: "No semantic instruction dataset directly over symbolic graphical programs." This paper https://mikewangwzhl.github.io/VDLM/ seems to propose an alternative semantically meaningful intermediate representation. I'd be interested in the authors' opinions on this paper.
- Generally, this paper is missing a discussion on the limitations of this benchmark. Here are some prompting questions: 
  - Are all natural images expressible as SVGs? What does this mean for the comprehensiveness of this benchmark?
  -  Are SVGs / CAD programs representative of all types of symbolic graphical programs? What about visual specification languages like OpenGL, Taichi, etc? What about programming languages like Turtle/LOGO which admit recursive structures?
  - What are the limitations of SGP-Bench's scalability?

In conclusion, I'm currently recommending this paper for a **Weak Rejection**. While the benchmark is undoubtedly interesting to the ICLR community, it is crucial to reconcile the claims in this manuscript with those from the existing literature before the paper is accepted for publication. I am open to discussing these points with the authors and potentially revising my recommendation during the discussion period.

----

After discussions with the authors, I'm revising my recommendation to **Accept**. Overall, SGPs offer a promising avenue to assess the reasoning abilities of LLMs.

### Questions
(Merged with the Weaknesses Section)

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces a benchmark designed to assess LLMs'  comprehension of symbolic graphics programs in SVG format, two 3D CAD formats (DeepCAD and Fusion360), and one 2D CAD format (SketchGraphs). The authors present evaluation results across several LLMs and introduce a symbolic instruction dataset, showing that fine-tuning on this dataset can enhance LLM reasoning capabilities in general.

### Strengths
1. The benchmark dataset is large, with 4,340 questions in the SVG set and 2,400 in the CAD set.
2. Both 2D and 3D objects are evaluated.
3. The paper addresses data leakage issues by utilizing global transformations like translations and rotations.
4. The proposed instruction tuning dataset improves general reasoning ability in LLMs.
5. The paper provides a critical view on current LLMs' capability.

### Weaknesses
1. The human study reports 90% labeling accuracy, leaving 10% mislabeled data, which could significantly impact the results given limited variation of the accuracy in LLM's evaluation. I suggest doing a human filtering across the whole benchmark to improve the accuracy of the ground truth answer.
2. The method for generating questions on 3D dataset from limited rendered views may lead to errors, especially in counting tasks; for example, the second part Figure 7 (row 3, column 3) is confusing. The question "How many visible cylindrical pins are there on the CAD object?" doesn't make sense since "visible" doesn't have a meaning given a code representing a 3D object without perspective information.
3. Several parts of the writing need further clarification (see questions below).

### Questions
1. In L76-L77, the authors claim that "The order of symbolic operations may substantially affect semantic meaning". However, this does not seem applicable to SVG, where objects are defined by coordinates and unaffected by drawing order. Could this be clarified?
2. In Figure 6, what qualifies as an "operation" in vector graphics? For example, does a polyline with multiple vertices count as one operation or several?
3. Inconsistencies in notation between Figure 6(b) (symbol $3D_{Recon}$) and the text (symbol $3D_{complex}$ in L257-L258) need clarification.
4. During fine-tuning, was there any overlap between training and test datasets? Additionally, could more detail be provided on the construction of the symbolic instruction dataset such as whether it includes both SVG and CAD formats?
5. If the enhanced answer extraction with LLM is applied consistently across all experiments or is only applied to models trained on SIT data?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper assesses large language models (LLMs) for their ability to "understand" raw symbolic graphics programs like SVG and CAD without direct visual input. By creating a benchmark (SGP-Bench), the authors test this through tasks requiring semantic-level answers and transformation invariance, providing insight into how LLMs interpret non-image-based graphics data. The authors also introduce Symbolic Instruction Tuning (SIT), a method to fine-tune LLMs with specific symbolic graphics data, show that its enhancing both specialized and general reasoning abilities.

### Strengths
1. Innovative in presenting a benchmark and dataset that adds value to multi-modal LLM and vision foundation model research.  
2. Extensive experimentation with well-designed elements, particularly in the fine-grained categorization of SVG-Understanding. While categories like "color" and "shape" might be addressed through subpattern matching in SVG codes, "Semantics" and "Reasoning" better assess true reasoning capability.  
3. Clear presentation and valuable insights.

### Weaknesses
1. The SGP-MNIST experiments show limited spatial reasoning in LLMs, suggesting that some claims about models’ spatial reasoning or “visual imagery” abilities may be overstated.  
2. The conclusion is somewhat underwhelming, reiterating known principles such as scaling laws and fine-tuning effects.

### Questions
1. Regarding Weakness 1, it appears the model learns from structured data without evidence of internal image rendering, likely due to lack of instruction. It would be interesting to test whether explicitly prompting for “internal rendering” as a Chain-of-Thought (CoT) step impacts performance.
2. I'm curious whether any experiments have been conducted using rendered images as input in multi-modal LLMs to compare performance with raw SVG code inputs, providing insight into how well models perform with visual representations.
3. In Appendix E.2, it’s noted that fine-tuning SIT-55k on Llama3-8B actually degrades performance. Is this due to answers not following the specified template, or are the answers correctly formatted but simply inaccurate?

### Soundness
3

### Presentation
3

### Contribution
3
