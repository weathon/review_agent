# Real2Code: Reconstruct Articulated Objects via Code Generation

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6, 6

## Abstract
We present Real2Code, a novel approach to reconstructing articulated objects via code generation. Given visual observations of an object, we first reconstruct its part geometry using image segmentation and shape completion. We represent these object parts with oriented bounding boxes, from which a fine-tuned large language model (LLM) predicts joint articulation as code. By leveraging pre-trained vision and language models, our approach scales elegantly with the number of articulated parts, and generalizes from synthetic training data to real world objects in unstructured environments. Experimental results demonstrate that Real2Code significantly outperforms the previous state-of-the-art in terms of reconstruction accuracy, and is the first approach to extrapolate beyond objects' structural complexity in the training set, as we show for objects with up to 10 articulated parts. When incorporated with a stereo reconstruction model, Real2Code moreover generalizes to real-world objects, given only a handful of multi-view RGB images and without the need for depth or camera information.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents Real2Code, a novel approach for reconstructing articulated objects from visual observations. 
Main Contributions:
1. A new method (Real2Code) that reconstructs articulated objects by generating code, using fine-tuned LLMs specialized for this task.
2. A part reconstruction pipeline that combines:
    - Kinematic-aware view-consistent part segmentation model.
    - 3D shape completion model.
    - Fine-tuned LLMs to predict joint articulation.
3. Significant performance improvements over previous methods:
    - First approach to accurately handle objects with more than three parts
    - Generalizes beyond training data (trained on up to 7 parts, works on up to 10 parts)
    - Works with just a few RGB images, without requiring depth or camera information

### Strengths
1. The writing is clear and easy to follow.
2. The proposed pipeline innovatively formulates the articulation reconstruction as code generation, which naturally combine current powerful foundation models (SAM, LLM) for articulated object reconstruction.
3. Real2Code demonstrates significant performance improvements over previous methods. It accurately handle objects with more than three parts and only requires RGB images without requiring depth or camera information.
4. This paper provides details for the training of the whole pipeline, including data preparation and training of key components (SAM, completion model, and CodeLlama), which demonstrates good reproducibility and technical soundness.

### Weaknesses
1. Real2Code demonstrates good performance on trained categories (Laptop, Box, Refrigerator, Storage-Furniture, and Table). However, the performance of unseen categories is not explored. I don't expect the model to generalize well to all other categories, but I do expect some experiments to show whether there is still a problem with category generalization. 
2. There is no discussion of when the model will fail, especially if some of the components of the model fail.  For example, fine-tuned SAM might not segment parts accurately, or the LLM might output an incorrect result under certain circumstances.


These weaknesses don't necessarily diminish the paper's contribution but addressing them would strengthen the work and increase its impact. Many could be addressed through additional experiments and analysis rather than fundamental changes to the method.

### Questions
1. Why is the method from [1] not included in the baseline comparisons, given that it demonstrates better performance than PARIS?
2. How is the cross-category generalization ability of Real2Code, especially on real-world data?
3. How long do the training and inference of the entire pipeline take respectively？
4. How to extract oriented bounding box of each part?

[1] Yijia Weng, Bowen Wen, etc. Neural implicit representation for building digital twins of unknown articulated objects.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper formulates joint prediction as a code-generation problem and adapts LLM to this task, which makes it scale elegantly to process an articulated object with multiple joints. It also introduces a part reconstruction pipeline leveraging 2D part segmentation and part-level shape completion.

### Strengths
- Formulating joint prediction as a code-generation problem provides an elegant way to handle varying numbers of object joints.
- Part-level shape completion makes sense since part structures are much simpler than structures of whole objects. Table 1 demonstrates the effectiveness of the proposed shape completion model.

### Weaknesses
The selection of object categories for evaluation is limited. 
- For part-level shape completion, it would be more compelling to include categories with a greater diversity of part shapes rather than focusing primarily on cuboid-like forms. For instance, objects such as globes and lamps in PartNet Mobility exhibit a variety of shapes, including spherical and cylindrical forms, which provide a more comprehensive basis for evaluation. Additionally, the assumption that 'many common articulated objects consist of cuboid-like parts' is not fully substantiated when considering the full range of object categories in PartNet-Mobility. 
- In articulation prediction, the formulation assumes that 'the position of corresponding revolute joints will lie closely to, if not overlap with, one of the OBB edges'. However, this assumption seems not to be solid enough either. Take “folding chairs” in PartNet-Mobility for example, the revolute joints of many instances lie not close enough to OBB edges (quadrisection point or even trisection point). Do these assumptions restrict the range of categories suitable for evaluation?

### Questions
Why were only these five object categories chosen from PartNet-Mobility for evaluation? The current formulation relies on assumptions that appear somewhat unsubstantiated. Is this why Real2Code is hard to evaluate in more diverse categories?

### Soundness
3

### Presentation
3

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
The paper focuses on the task of articulated objects reconstruction given only a few images of an object. The pipeline proposed first reconstructs parts from images and then leverages LLM to predict the joint parameters, which generalizes to objects with multiple joints. The method is evaluated on five categories in the PartNet-Mobility dataset and outperforms previous methods.

### Strengths
1. The paper proposes a new pipeline Real2Code to reconstruct articulated shapes from images. It shows promising results on multiple categories with different joint types in the PartNet-Mobility dataset.

2. I find its generalization ability to multiple-joint shapes particularly interesting, which could potentially enable many real-world robot manipulation tasks.

3. The paper is overall easy to read.

### Weaknesses
1. The proposed pipeline consists of multiple components and, as a result, rather fragile from what I understand, since a failure in any component in the middle can cause the entire pipeline to break down. For example, if the part bounding boxes parameters (segmentation or shape completion) are inaccurate, the joint prediction part will carry these errors. Since the whole procedure is open-loop, I wonder if the method still produces reasonable shapes assuming initial bounding box predictions are inaccurate?

2. The method is only evaluated on five categories, and these categories (Box, Refrigerator, Storage-Furniture and Table) are all quite similar in topology, similar for the real-world examples. So I think it would be helpful to see results on more diverse shapes. In addition, is CodeLlama trained on all categories together? How does CodeLlama handle scale differences of different objects / categories? Or all shapes normalized so the input to the LLM is kind of already normalized?

3. Which component is the bottleneck of the pipeline? Is it the part segmentation or joint prediction of CodeLlama? Ablation studies on this are essential to better evaluate the approach.

4. To better support the claim of being able to extrapolate beyond objects’ structural complexity in the training set, I think it would be important to provide more results. For example, does the trained model generalize to other categories?

### Questions
The results presented in the paper are interesting, but I believe that additional evaluations would strengthen the significance and impact of the work.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Real2Code, a method for reconstructing articulated objects from multi-view images through code generation. The method first reconstructs part geometry using image segmentation and shape completion. Then it predicts joint information as code generation from fine-tuned LLM which takes an object part as oriented bounding boxes. Experiments show that this method outperforms previous method in generating parts with over three parts and can generalize to real object reconstruction by training only on synthetic data.

### Strengths
1. This method formulates joint prediction as a code generation problem, which is different from prior work. The biggest advantage of such a formulation is the ability to scale well with different numbers of parts (prior work works mostly for objects with <=3 parts).

2. The overall pipeline is novel -- it leverages a few different modules including Vision models for part segmentation and completion as well as LLM for code generation. This way, the problem is decomposed into a few smaller steps which is shown solvable with previous methods.

3. The text and figures are overall well-written and easy to follow.

4. Experiments have been conducted to validate each proposed components. Results seem to achieve state of the art, especially on objects with many parts.

### Weaknesses
1. In Sec. 4.2.1, it mentions that "we generate permutations of the set of predicted meshes and take the permutation that results in lowest error; the same logic is used for joint prediction results". I was wondering why this is needed to evaluate this method. Is it because the proposed method is not very stable? How much more time would this cost for the inference of this method?

2. The link to more visualizations included in Sec. 4.4 does not contain any result visualizations -- it seems it only has a method overview figure and an abstract.

3. The content in Tab. 1 is a bit confusing:
(1) what is ``Real2Code+gtSeg``, the paper does not seem to mention / analyze this row anywhere in the text.

(2) If I understand ``Real2Code+gtSeg`` the same way as ``Real2Code+gtBB`` in Tab. 2, it should be an upper bound of ``Real2Code (Ours)``, if so, why does ``Real2Code+gtSeg`` perform worse than ``Real2Code (Ours)`` in a few columns like Whole & Parts for Box, etc.

### Questions
1. Tab. 3 and its corresponding text has some typos: row 2 has "Rot" in out column, but it is referred to as "Rel" in the text (if I understand it correctly).

2. In Tab. 3, the first row has 0 error for "rot" on 3, 4-5, 6-15 parts. Then why the rot error suddenly becomes very big for 2 parts?

3. How do you determine the parent vs. child node / the canonical pose, especially for real-world objects? For example, the two parts of a laptop have very similar geometries/OBB. If a laptop is placed upside down, would this method also instead treats the keyboard part is child and the screen part as parent?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper reconstructs articulated objects from visual observations. The approach utilizes a modular pipeline which first reconstructs part level geometry from segmentation and then uses a codegen LLM to combine the individual parts into an articulate assembled model to be executed in simulation. The paper compares to relevant recent baselines and demonstrate strong improvement. The approach also scales well to increasing number of joints due to its modular approach.

### Strengths
In my opinion, below are the strengths of the paper:

1. The paper scales well to increased number of joints. This has been a major limitation of preceding works and this work address it nicely with a modular approach i.e, part level reconstruction and code-gen integration for the subsequent steps. 

2. Strong quantitative improvement numbers compared to recent state-of-the-art baselines, especially for increasing number of parts. 

3. The presentation of the paper in nice and paper writing is easy to follow.

### Weaknesses
I have some question to the authors. In my opinion, below are the paper's weaknesses:

1. Why does the PARIS baseline struggle a lot? even for 2-part case? Did the authors try to tune their method? Based on the PARIS results' from the paper, it looks like it should reasonably work well for a simpler 2 part setting?

2. Despite good qualitative results, why are the resutls only shown on simpler objects like cupboards and laptop? Does the method work for varied articulated objects like scissors, stapler etc? Is this an inherent limitation of their method they only work for a subset of articulated objects for which they ahve a prior? If yes, that should be clearly stated as other baselines seems to work for more complicated articulated objects as well?

3. What is the timing result of the method? Some of the baselines mentioned i.e. CARTO, follow-up from CenterSnap [1] are very fast and don't require manual SAM prompting i.e. single-shot. This is not discussed very well in the related works. 

4. I didn't find rigorous details on pretraining datasets for shape completion as well as datasets used for finetuning code llama. Those should be helpful to include. Also do authors plan to open-source their code? It looks like that will be helpful as well for the community to build up on?

[1] Irshad et al. CenterSnap: Single-Shot Multi-Object 3D Shape Reconstruction and Categorical 6D Pose and Size Estimation

### Questions
Please see the weakness section above for clarification questions. I look forward to seeing them in the rebuttal.

### Soundness
3

### Presentation
3

### Contribution
3
