# MotionCLR: Motion Generation and Training-free Editing via Understanding Attention Mechanisms

- Avg Score: 5.50
- Decision: Reject
- Scores: 10, 3, 6, 3

## Abstract
This research delves into the problem of interactive editing of human motion generation. Previous motion diffusion models lack explicit modeling of the word-level text-motion correspondence and good explainability, hence restricting their fine-grained editing ability. To address this issue, we propose an attention-based motion diffusion model, namely MotionCLR, with CLeaR modeling of attention mechanisms. Technically, MotionCLR models the in-modality and cross-modality interactions with self-attention and cross-attention, respectively. More specifically, the self-attention mechanism aims to measure the sequential similarity between frames and impacts the order of motion features. By contrast, the cross-attention mechanism works to find the fine-grained word-sequence correspondence and activate the corresponding timesteps in the motion sequence. Based on these key properties, we develop a versatile set of simple yet effective motion editing methods via manipulating attention maps, such as motion (de-)emphasizing, in-place motion replacement, and example-based motion generation, etc. For further verification of the explainability of the attention mechanism, we additionally explore the potential of action-counting and grounded motion generation ability via attention maps. Our experimental results show that our method enjoys good generation and editing ability with good explainability. Codes will be public.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
This paper presents MotionCLR, an attention-based motion diffusion model that enables fine-grained motion generation and editing. The key innovation lies in its explicit modeling of word-level text-motion correspondence through a dedicated cross-attention mechanism while using self-attention to capture temporal relationships between motion frames. 

The model achieves various motion editing capabilities without requiring paired editing data, including motion emphasis/de-emphasis, in-place motion replacement, example-based generation, and style transfer. The authors provide both theoretical analysis and empirical evidence to demonstrate how the attention mechanisms enable these editing capabilities. 

Extensive experiments show that MotionCLR achieves competitive performance on standard motion generation benchmarks while offering editing flexibility. The paper also explores the model's potential for action counting and addressing hallucination issues in motion generation.

### Strengths
- Novel technical contribution in disentangling text-motion correspondence and motion-motion interactions through separate attention mechanisms
- Thorough analysis and validation of how attention mechanisms enable motion control
- Impressive range of editing capabilities achieved without paired editing data
- Strong quantitative results compared to state-of-the-art methods

### Weaknesses
- The cross-attention control method used for motion editing comes from an existing text-to-image editing paper, Prompt-to-Prompt (Hertz et al., 2023). Although this reference is cited, the connection between this submission and Prompt-to-Prompt should be clarified to avoid misunderstanding.
- The authors did not compare the performance of motion editing against other methods. It would be impressive to show that this method performs similarly to those that require paired data and training.

### Questions
- When conducting cross-attention control, do you manipulate the attention weights on all cross-attention layers in all CLR modules or only in some specific layers?
- In the emphasize and de-emphasize cases, why are the weights added and not multiplied by the adjustment?
- We see a significant bump in FID when the editing weight moves beyond +/- 0.5 in Table 2. What might be the reasons behind this phenomenon?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors proposed a method for generating and editing motion using a text-aligned motion generation process. By exploring attention mechanisms, they discovered that manipulating attention weights enables motion editing. Their approach can be applied to downstream tasks such as motion emphasizing and de-emphasizing, motion erasing, and motion sequence shifting.

### Strengths
The paper demonstrated that the cross-attention mechanism plays a crucial role in establishing text-motion correspondence at the word level.
This insight enables a better understanding and manipulation of text-to-motion models with finer granularity.
The proposed method significantly improves the explainability of text-to-motion generation methods.

### Weaknesses
While I believe the paper will play an important role in the development of text-to-motion generation task, I have serious concerns regarding its evaluation method.

In my view, the experimental evaluation relies too heavily on qualitative and empirical results, including attention map visualizations and velocity/trajectory visualizations. The action counting error metric currently used only indicates the number of actions and fails to capture the **temporal correspondence** between the attention map and human motion. Given that the paper's primary contribution is editing motion in by manipulating attention weights, it is crucial that the evaluation demonstrates temporal correspondence between these attention weights and the motion itself. The authors could have proposed an evaluation metric measuring the correspondence or similarity between the attention weights—whether self-attention or cross-attention, depending on the downstream task—and motion intensity over time, which would better elucidate the manipulation. The TMR metric, which calculates the similarity between motion and text, somewhat makes sense for evaluating motion Motion (de-)emphasizing and motion erasing but I don't think it's enough to explain the temporal correspondence. Metrics like TMR and FID are insufficient to fully assess the manipulation either. I believe the quantitative evaluation of motion manipulation in the paper could be improved.

### Questions
In paragraph 5.2, it is stated that TMR-sim values range between 0 and 1, but in Table 2, the results are numbers like 55. This discrepancy requires additional explanation. Considering the inconsistencies among Tables 2, 4, 5, and 6, it appears that the results in Tables 2 and 4 might have been multiplied by 100 to highlight the differences. If so, this scaling should be clearly indicated to ensure consistency and avoid confusion.

### Soundness
2

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
3

### Summary
The research presents MotionCLR, an attention-based motion diffusion model designed to enhance interactive editing of human motion generation. It addresses the shortcomings of previous models that struggle with word-level text-motion correspondence and lack explainability, limiting their fine-grained editing capabilities. MotionCLR utilizes self-attention to evaluate sequential similarities among motion frames and cross-attention to identify specific correspondences between word sequences and motion time steps. The authors present three main applications, each of which includes several specific variations that can be considered separate applications, as follows:
1. Motion emphasizing and de-emphasizing.
    - Motion erasing.
    - In-place motion replacement.
2. Motion sequence shifting.
    - Example-based motion generation.
3. Motion style transfer.

### Strengths
- The authors present many creative applications that utilize simple techniques, open up new applications for text-to-motion.  
- MotionCLR effectively balances control and quality; instead of sacrificing one for the other, the model demonstrates improvements in both aspects.  
- The editing can be performed during inference without the need for specific training for each task.
- The authors leverage existing datasets, allowing for more fine-grained control over motion generation.  
- The paper is well-structured and easy to follow.

### Weaknesses
- This editing is out-of-distribution problem which may limit its performance in real-world applications. More qualitative and quantitative experiments would strengthen the claim of the model’s handling of out-of-distribution motions, providing clearer insights into its capabilities and limitations. (see next point) 
- Lack of a comprehensive experiment discussion
  - It remains unclear how effectively the model can decouple complex actions, as text and motion data tend to be tightly coupled when there are limited samples available.
  - For example, if the dataset contains only one text-motion pair for "turns around and does a cartwheel", can MotionCLR generate a "cartwheel" without performing the prior action in the prompt, i.e. "turns around," using motion erasing (de-emphasizing)?
  - Additionally, let’s say the training set always shows "a man crouches forward" by starting with "standing up" before moving into the crouching position. If we apply "Motion sequence shifting" to place the "crouching" action at the beginning, can the model generate this motion without seeing this during training?

### Questions
In Table 1, is the generation quality is improving from other diffusion models? ("ReMoDiffuse" is retrieval method and MoMask is using masked modeling). If yes, I think this advantage can be emphasis too.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This work introduces MotionCLR, a model designed for versatile human motion generation tasks based on textual input. MotionCLR leverages an attention mechanism to align text with motion, allowing it to handle tasks like in-place motion replacement and motion generation from text without retraining. A key focus is on analyzing how the attention mechanism facilitates alignment between text and motion, enabling effective generation for various tasks. Experimental results demonstrate the method’s effectiveness across a range of motion generation scenarios.

### Strengths
- The paper clearly presents the methodology and contributions, with a well-organized structure that makes it easy to follow.
- Figures are well-designed and effectively illustrate the approach and context.
- Detailed appendices provide additional information that supports readers’ understanding of the work.
- The authors conduct a wide variety of experiments, showcasing the advantages of MotionCLR across different tasks.

### Weaknesses
1. CLR Module Design: The design of the CLR module lacks detailed explanation. Specifically, the paper doesn’t clarify how MotionCLR differs from existing approaches or justify the use of U-Net-style transformer stacks for downsampling and upsampling time dimensions. Additional experiments or analyses are necessary to justify this architectural choice over replacing input condition style to cross attention-style from existing transformer-diffusion models.

2. Dataset Generalizability: Experiment in Sec 5.1 is conducted solely on the HumanML3D dataset for text-based motion generation. To validate generalizability across datasets, additional experiments on the KIT dataset are recommended.

3. Section 3.3 Relevance: Section 3.3 mainly covers widely known concepts without introducing new insights. Condensing it or merging it with Section 3.2 could improve clarity and conciseness.

4. Cross-Attention Explanation: More detailed explanations or illustrations are needed on how cross-attention aligns text with motion. For example:
- How is positional information of the text inserted? Is a positional embedding used?
- How does the network align text with motion tokens, given that text tokens doesn’t inherently contain sequential order information? I cannot find any details for alignment since this is critical when aligning two different sequences. Relative positional embedding or CTC like loss may be required to sequentially align motion and text tokens. Please clarify this point.

5. Longer Text Testing: Testing MotionCLR on longer text inputs with multiple action verbs (e.g., >5 action words) would offer a more practical assessment of its performance in more practical scenarios.

### Questions
Please refer to weaknesses.

### Soundness
3

### Presentation
4

### Contribution
2
