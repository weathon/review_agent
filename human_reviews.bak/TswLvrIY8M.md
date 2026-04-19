# Self-Organizing Visual Embeddings for Non-Parametric Self-Supervised Learning

- Decision: Reject
- Scores: 3, 5, 5, 5

## Abstract
We present Self-Organizing Visual Embeddings (SOVE) a new training technique for unsupervised representation learning.
SOVE avoids learning prototypes from scratch and explores relationships between visual embeddings in a non-parametric space.
Unlike existing clustering-based techniques that employ a single prototype to encode all the relevant features of a complex concept, we propose the SOVE method where a concept is represented by many semantically similar representations, or judges, each containing a complement set of features that together can fully characterize the concept and maximize training performance.
We reaffirm the feasibility of non-parametric self-supervised learning (SSL) by introducing novel non-parametric adaptions of two loss functions with the SOVE technique: (1) non-parametric cluster assignment prediction for class-level representations and (2) non-parametric Masked Image Modeling (MIM) for patch-level reconstruction.
SOVE achieves state-of-the-art performance on many downstream benchmarks, including transfer learning, image retrieval, object detection, and segmentation.
Moreover, SOVE demonstrates scaling performance when trained with Vision Transformers (ViTs), showing increased performance gains as more complex encoders are employed.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper presents Self-Organizing Visual Embeddings (SOVE), a technique for unsupervised representation learning that explores relationships between visual embeddings in a non-parametric space. SOVE uses multiple semantically similar representations, or "judges," to capture complementary features of concepts, enhancing training performance. It introduces two novel non-parametric loss function adaptations: (1) non-parametric cluster assignment prediction and (2) non-parametric Masked Image Modeling (MIM). SOVE achieves state-of-the-art performance in various benchmarks, including transfer learning and object detection, and shows improved scalability with Vision Transformers (ViTs).

### Strengths
1. The overall demonstration is good.
2. The proposed SSL method seems to be reasonable.

### Weaknesses
1. I am worried about the support for the claim of the motivation. "However, when views are too dissimilar (due to extensive augmentations), views and anchors will exhibit inconsistent prediction patterns. " Could the authors provide more clear illustration or pilot experimental results for this? This claim is not intuitive.
2. The essence of this method targets on using similar embedding for replacing the original biased feature representation (like IBoT), and the experimental results show that such mechanism could enhance the knn performance, but the fine-tuned performance merely shows marginal enhancement compared to other methods. From another perspective, this method is quite specifically adapted to KNN task? Overall, such a performance elation is not convincing regarding the effectiveness of this method.

### Questions
None

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes an alternative objective function and methodology for enabling multi-anchor representation learning for self-supervision. The authors propose a similarity based method to sample additional anchors, randomly sampling per iteration a set of anchors from previous iterations representations, incorporating additional views or judges from a k-nn of the representation of the anchors, then producing a new probability distribution by scaling this via the original views of the object. This produces a probability distribution for each views similarity to these random anchors, that the authors claim provides alternative feature information that would otherwise be ignored. The method is evaluated on the full range of standard self-supervised benchmarks, comparing against most expected methodologies and achieving good performance improvements across the board.

### Strengths
**Simple approach.**
- The proposed approach is seemingly effective and simple from my understanding, presenting an interesting alternative for increasing the number of anchors with a similarity scoring mechanism to choose highly related yet opposing anchors.
- The MIM is a sensible extension of the proposed method showing the applicability to addressing the full scope of the problem setting explored, and yield significant improvements as shown in ablation studies.

**Improved Empirical performance.**
- Generally, albeit not always, the empirical performance of the method is increased compared to other benchmarked approaches across all settings, tasks and benchmarks. 
- I am highly satisfied in the empirical evaluations, the authors have done a good job in including most standard benchmarks, both in terms of datasets and tasks.
- All results are presented clearly, with mostly good rigour given to the empirical analysis following standard evaluation procedures presented in comparing works.

**Strong ablations.**
- For the most part, the authors do a good job performing ablations and sensitivity analyses of the proposed method. While there are some important and missing ablations that I believe will strengthen the work, the majority of analyses are performed and discussed adequately.

### Weaknesses
**Text Clarity.**
- The clarity of the writing is relative poor, much of the explanations are overly complex and could be shortened. For example the paragraph from line 200. This makes the work feel bloated, with the majority of text being filler. Pages 2 - 4 could be made more concise and halved with little to no impact on the narrative.
- Following this, the justification of the method is formulated in-terms of semantic features, which although seeing the authors rationale, I feel this is somewhat misleading or obscure as no evidence is provided to back up these claims or hypothesis. There is no guarantee that the procedure captures these other features of interest as a whole, rather additional similar “intermediate or in-between” anchors (for the lack of a better term) are introduced. Please correct me if my I am misunderstanding this.
- Figure 2 needs further annotations, or description, it is not clear to the reader what it is conveying, notably the middle section.
- You mention in the abstract that your method achieves sota however, this should be clarified as being sota under this objective function formulation. There exists other SSL methods with greater performance.

**Missing details and reproducibility instructions.**
- There needs to be a more concrete description of how E is produced, updated, etc. Furthermore, details such as hyperparamterisation are missing, what random distribution are the anchors selected, this makes a huge difference on l2 normalised space? Please correct me if I’m have missed details, as in relation to my point of clarity some of these details may be lost/misunderstood.
- Key details are missing from the results section, for example how many anchors are chosen for Table 1?

**Fair comparison to other approaches with increasing anchors or prototypes targets.**
- How does your method compare to the DINO or MSN method when the number of class anchors you sample is the same as the number of prototypes? For example your method could be seen as being somewhat similar to these methods, however, I understand you are simply using a memory bank akin to MoCo to sample more anchor classes and scale the probability distribution via similarity of other judges. My question is however, is your method more performant simply because you increase the number of anchors, this can be tested by directly comparing the anchors you use to the number of classes or prototypes in these other methods (DINO or MSN).
- This above comparison would show if the performance gains are attributed to your novel selection and scaling process or simply by increasing the number of anchor points.

**Justification of shared features.**
- While I generally agree with the justification made regarding shared features, this is simply a product of the augmentation strategy involving crops. You had identified this in your conjecture, therefore it would be nice to see some additional analysis with varying crops, from your justification, you should observe a smaller delta in performance gain as the crops increase in size. As such you are proposing a method to address a self-introduced issue. I understand in practice this is not the only reason, but your justification and narrative is built on this observation.

**Minor:**
- Less of a weakness, rather a suggestion. Figure 1 could be expanded to include other results, currently it tells part of the story and given the other results are also positive it would be worth showing top-1 performance also to fairly compare agains the most common benchmark metric.
- The narrative of the introduction could be improved, for example, why does the first paragraph not address the problem statement, while the second paragraph’s justification is not as compelling as the ones presented later on. Arguably the justification in the introduction is a purposeful design choice of SSL to ensure generalised representations are learnt, while I agree with the sentiment of the authors, this section could be improved to ensure the justification is consistent. 
- The motivation for MIM could be extended with a more rigorous justification.  
- Weaknesses and limitations of the work are not discussed in detail, at the very least computational performance should be addressed.

### Questions
1. Is the size of E defined as all features of all images from the previous epoch, or is it a subset of the full dataset features? If the latter what impact does the size of E play?
2. What is the random distribution used to select the anchors? Is its uniform to ensure maximal information is preserved on the sphere?
3. Have you experimented with other strategies for selecting the anchors?
4. What is the convergence of such methods, the selection process would imply that convergence would take longer as initial bad representation anchors early in training would negatively impact this stage of training?
5. From the above question, in practice does this occur?
- In addition, I have inherently implied questions in the above weaknesses.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The paper presents are non-parametric representation learning framework, SOVE.  This framework is aiming to ameliorate a key issue with SOTA techniques which use data augmentation to generate multiple views of the same piece of data (image).   The problem is that often this data-augmentation forces too much invariance, e.g. forcing two non-overlapping crops of the same image to be close together, even if, visually they are not, which makes the model focus on limited/irrelevant set of features.  The proposed solution is to increase the number of "judges" so that each image would have multiple shots at being assigned to a given "concept". Additionally, a non-parametric version of Masked Image Modeling is employed (perhaps reminiscing of Non-Local Means?).   The experimental results show modest improvements over SOTA methods across several standard tasks.

### Strengths
1. The use of hand-designed data-augmentations have long been considered suspect, exactly for the reasons outlined in the paper.  The solution proposed here -- don't try to force all views into the same representation -- makes a lot of sense, and indeed goes all the way back to the cognitive science, e.g. Eleanor Rosch work on porotype theory, exemplar theory, etc.   It also nicely connects to some early non-parametric unsupervised learning work of Tomasz Malisiewicz, especially:
Visual Memex: http://www.cs.cmu.edu/~tmalisie/projects/nips09/
and ExemplarSVM: https://www.cs.cmu.edu/~tmalisie/projects/iccv11/
as well as the Dosovitsky's ExemplarCNN: https://arxiv.org/abs/1406.6909
I suspect that connecting the present work to these older efforts would make the paper easier to understand/situate. 

2.  The non-parametric MIM seems very reasonable.  Again, it would be good to understand how related it is to Non-Local Means. 
 
3. The improvements over SOTA are modest but, given the new method is quite different, I think that is fine.

### Weaknesses
My main issue with the paper (and the reason I might have gotten it quite wrong?), is that it is very difficult to read:
1.  What are "concepts", "porotypes", "anchors", and "judges".  They are not clearly defined.   Especially difficult is the definition of "concept" -- in a self-supervised method it can't really be semantic (where would the semantics come from?), and yet, in the paper it's talk about as if it is semantic.  
2.  The paper talks about "many non-parametric judges", but then it seems like there are only two judges for every anchor, so the number of judges is O(K), not O(N).   The definition of non-parametric model is a model which scales with data size N, whereas here it scales with number of anchors K.   
3. Th "illustrative example" in Sec 2.0 is a good story, but is it actually true?   It sounds intuitively reasonable, but I don't think this is enough -- there should be some supporting evidence / experiments / visualizations on real data.   Indeed, it's possible that story is not true after all, we know that many instances of the same class in, say, ImageNet, look nothing like each other, and yet the network does not actually "discard" any features.   Indeed, networks are happy to learn even random labelings: https://arxiv.org/abs/1611.03530
Also, I really don't understand Figure 2 -- I think it needs to be explained more carefully as there is a lot happening there. 
4.  Bolding convention in Table 1 is unclear.  Somehow only SOVE is bolded, even when other methods perform as well or better.

### Questions
1.  I might be misunderstanding something, but wouldn't a good baseline be to just increase the number of anchors (say, by 3x, so it's the same as anchors+judges)?  
2.  Since the main problem this paper is trying to solve is the fact that multiple views of the same image might not have anything to do with each other, why not get rid of multiple views all together?   Instead, try to align any image to any other image -- if it works, great, if it doesn't too bad.  This seems like a more principled way to advance a non-parametric story?  Indeed, maybe just the non-parametric MIM might already be performing well enough? 

My score and confidence are pretty low right now since I feel like I might be missing something key.  Looking forward to explanation from the authors.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper presents Self-Organizing Visual Embeddings (SOVE), a non-parametric loss design for self-supervised representation learning. To be specific, SOVE explicitly assumes multiple anchors that contribute to representing a single concept and aggregate them to represent a single soft label for self-supervised learning. Moreover, it introduces a non-parametric design of masked image modeling loss for joint usage in training. Throughout extensive experiments, the proposed method achieves superior performances across various downstream tasks, including transfer learning, image retrieval, object detection, and segmentation.

### Strengths
- Extensive experiments show that SOVE consistently outperforms baselines across various benchmarks, including image classification, retrieval, and segmentation tasks. This broad evaluation shows its robustness and versatility across both global and dense prediction tasks.

- Strong motivation. The authors claim the limitation of the previous self-supervised learning methodologies that share similar structures. 

- Non-parametric versions of self-supervised clustering and masked image modeling loss designs are valuable.

### Weaknesses
- Overall, writing is uncomfortable to read and many details missing. For example, there is no notation for the dimensionality of the spaces $A, E, D, Y$, and corresponding vectors. What is the size of $E_c$ and $A$?

- The motivation of additional judges is unclear. Unlike the authors claimed $\textit{Existing solutions employ a single judge (usually learnable) to decide upon the membership of a view to a concept}$, DINO, specifically, leverages a soft-label from K prototypes. One key difference between DINO and SOVE (without MIM) is that SOVE explicitly assigns multiple prototypes for each target concept and aggregates their labels using $Y$. On this line, I am not convinced the proposed method is much more effective or superior to DINO. Table 7 also shows the close performances of DINO and SOVE without MIM (even worse than IBOT). Furthermore, it is also unclear whether the proposed non-parametric approach is actually effective. I suspect if the authors do a variant of DINO following SOVE (i.e. $E$ are learnable prototypes with enough numbers considering the size of $E$ in SOVE), the result also would be comparable.

- As shown in Table 7, MIM is a key ingredient of SOVE. However, the authors do not compare SOVE with DINOv2 which is the state-of-the-art SSL baseline using both clustering and MIM. I recommend to include a comparison with DINOv2. 

- There are no full ImageNet fine-tuning comparisons.

- There are no explanations about the trade-off between the parametric and non-parametric approaches. How big is the proposed feature set and computationally expensive compared to parametric approaches like DINO?

### Questions
Please see the weakness section.

### Soundness
2

### Presentation
1

### Contribution
2
