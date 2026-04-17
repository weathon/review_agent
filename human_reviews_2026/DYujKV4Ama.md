# Your Discriminative Model is Secretly a Generative Model

- Decision: Reject
- Scores: 8, 2, 4, 6

## Abstract
Although discriminative and generative models are fundamentally equivalent in understanding data distributions, bridging these paradigms -- especially transforming off-the-shelf discriminative models into generative ones -- remains challenging. In this paper, we introduce a universal framework that unlocks the generative potential of any discriminative model by directly leveraging the data manifold encoded in its parameter space. Drawing inspiration from the score function used in diffusion models, which measures the distance between a sample and the data manifold in probability space, we generalize this concept to the functional domain. 
To achieve this, we introduce the Discriminative Score Function (DSF), which quantifies the functional distance between a sample and the data manifold by mapping both into a shared functional space using the Loss Tangent Kernel (LTK), a variant of the Neural Tangent Kernel.
Our framework is architecture- and algorithm-agnostic, as evidenced by various architectures such as ViT, ResNet, and DETR on tasks including object detection, classification, and self-supervised learning (\eg, CLIP and DINO). Additionally, our approach extends to applications in image editing, inpainting, and explainable AI (XAI).Finally, we demonstrate the promising potential of DSF by adopting diffusion model techniques for enhanced generation quality.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces the Discriminative Score Function (DSF), a principled framework that enables any off-the-shelf discriminative model (e.g., ResNet, ViT, DETR, DINO, CLIP) to perform generative tasks without architectural changes or retraining.
By leveraging the Loss Tangent Kernel (LTK)—a variant of the Neural Tangent Kernel that incorporates loss gradients—the authors reinterpret the functional space of a trained discriminative model as an implicit representation of the data manifold.
They show that gradient-based updates in this functional space can mimic the behavior of score-based diffusion models, effectively turning discriminative models into generators.
The approach supports unconditional and conditional image generation, editing, inpainting, and even explainable AI (XAI) visualization. Experiments demonstrate impressive qualitative results across multiple architectures and datasets (ImageNet, COCO, LVD-124M).

### Strengths
- Novel conceptual framework bridging discriminative and generative paradigms through kernelized functional mapping.
- Architecture-agnostic and requires no retraining or modification, an elegant practical feature.
- Demonstrates broad applicability (classification, detection, self-supervision).
- Qualitative results (Figs. 3–8) are surprisingly strong for a method derived from non-generative networks.
- Provides interpretability and XAI potential, revealing biases and feature entanglements in pretrained models.
- Theoretical formulation draws an interesting connection to score matching and diffusion modeling.

### Weaknesses
- Lack of quantitative evaluation: No FID, IS, or precision/recall metrics, making it difficult to assess generation fidelity.
- Limited theoretical rigor: The equivalence between DSF and diffusion score functions is stated rather than proved; convergence properties are untested.
- Experimental depth: All results are qualitative; the method’s computational cost, convergence behavior, and sensitivity to hyperparameters are unexplored.
- Clarity: Mathematical notations are nonstandard, and derivations in Sec. 4 are hard to follow.
- Ablation studies (e.g., with vs. without LTK, or different surrogate losses) are missing, which limits interpretability of where the generative capability arises.
- Comparisons with other classifier-based generation approaches (DeepInversion, Energy-Based Models) are mostly descriptive, not empirical.

### Questions
1. Can you provide quantitative metrics (e.g., FID, IS, or perceptual distance) to substantiate DSF’s generation quality?
2. How stable is the iterative generation process with respect to step size (ϖₜ) and initial noise distribution?
3. Does DSF generalize beyond vision tasks—e.g., to language models or reinforcement learning—as suggested in Sec. 7?
4. How sensitive is DSF to the choice of surrogate loss (augmentation invariance)? Would other unsupervised losses (e.g., contrastive) yield similar results?
5. Can you clarify the computational complexity compared to a standard diffusion sampling process?
6. Is DSF guaranteed to converge to the data manifold, or can it produce divergent artifacts?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
- The authors propose a new general method for turning discriminative models into generative models. The method uses their discriminative score function (DSF), a novel distance which leverages loss tangent kernels to quantify the distance between a sample and the empirical data manifold. From DSF, samples can be generated through a diffusion-like process (by using DSF as a score function). In addition, the authors show that by modifying the loss function used in the kernel, their methods allows for unconditional generation as well.
- Experimentally, the authors show their method works on large commonly used discriminative models including DINOv2 and LVD-124M on various datasets, for both unconditional and conditional generation. While the generated images are full of artifacts, the underlying objects are identifiable.
- Finally, the authors show that the use of diffusion techniques improve generation quality substantially and that their method allows for increased interpretability of these discriminative models.

### Strengths
- The method is novel and has advantages over existing methods (particularly with respect to unconditional generation)
- The paper shows experiments on multiple large-scale commonly used models, demonstrating the method's ability to scale
- The results are interesting, the qualitative difference between the models, particularly between the SSL methods and image classification/ImageNet ones, is notable
- The ability to use techniques from diffusion models (and their apparent efficacy) offers an interesting direction for future work

### Weaknesses
- Several parts of the paper were unclear to me, I had many
- The experimental evidence could be more convincing. There are no quantitative evaluations, just a fe generated images for each experiment.
- The paper would benefit from further comparisons to existing work.
- The method relies on using a loss function in the LTK that could be justified further
   - "We select augmentation invariance loss as it naturally decreases during training, mirroring label dependent loss behavior" isn't fully convincing.
   - Have additional ablations been run on different choice of loss functions?



### Writing
- The writing in the paper needs to be substantially improved. There are many unsubstantiated/unclear statements, especially in the introduction
	- "Discriminative and generative models are theoretically equvalent as they both aim to understand the true data distribution." What does this mean?
	- "They learn data distributions implicitly, causing their distributional knowledge to become entangled with training objectives." This is unclear.
	- "on generation that incorporates these training objectives." Should be elaborated further
- There are also many typos (e.g. even in the first sentence "equvalent"),  weird capitalization, poor grammar and informal language (e.g. "its shape is very curvy", "turns the impossible to possible", "The answer to the question turned out to be ‘yes’.").
- The paper could motivate earlier the value of this line of research (e.g. what's done in "Position of our paper").

### Questions
- How computationally expensive is the generation procedure?
- Are the displayed images cherry-picked or randomly selected from the generated images.
- Have you tested on any robust classification methods? Would this perhaps reduce some of the visual artifacts?
- Could you explain further the last part of the derivation of Eq.7-10. It seems the removal of the summation is crucial for the scaling of this method.
- For conditional generation, is there a reason the discriminative model is not used to guide generation?
- The global explanation part seems interesting. Do you have more thorough evidence (e.g. multiple images of the clocks)?
- Do you have any quantitative metrics for evaluating the resulting generative models?
- (Lee et al., 2024) seems to get similar qualitative results. How different is their method and how fundamental is its limitation that it can only do conditional generation?
- If you just used the original loss in the LTK, would this be equivalent to any existing methods?

Ultimately, I am on the fence for this paper. I believe the idea and results are interesting but the paper and especially the writing need significant work. I am willing to increase my score if my questions (particularly the last two) are answered and the authors demonstrate the writing has been noticeably improved.

### Soundness
2

### Presentation
1

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
This paper explores the generative potential of discriminative models by investigating how to convert them into generative models. Inspired by the use of score functions in diffusion models to measure the distance between samples and the data manifold, they propose a criterion for estimating this distance within the functional space of discriminative models. Based on this criterion, they develop a corresponding iterative generation process, demonstrating theoretical innovation. Experimental results confirm the proposed model's capabilities in both unconditional and conditional generation, and showcase its applications in image inpainting and editing. However, the experimental outcomes are not particularly outstanding.

### Strengths
The paper presents a novel method for converting discriminative models into generative models. The approach is architecture- and algorithm-agnostic, it is simple to implement and requires no modifications or additional training. Its versatility, demonstrated across multiple practical applications, offers valuable guidance for future research.

### Weaknesses
It fails to provide a complexity analysis of the proposed method.
The experimental results lack quantitative metrics and sufficient comparative analysis against existing baselines.
The evaluation of the method's effectiveness across various applications is inconclusive.
The writing also suffers from formatting issues.

### Questions
1. How is the generalization from the training to the test distribution, as stated in relation to Eq. (3), ensured? Furthermore, is the method limited by the i.i.d. assumption and thus inapplicable under domain shift?
2. Regarding Eq. (11), it would be more appropriate to swap the positions of x_t and A(x_t). Have you considered making this change?
3. Regarding Eq. (12): calculation based on the gradient for all model parameters appears computationally intensive. Could you clarify if the gradient with respect to x propagates through the entire network f(x_t;θ)?
4. What is the required number of timesteps t for the generation process? what's the effect of t on the final performance?
5. How does the choice of initial noise x0 affect output diversity? Can similar or close noise vectors generate distinct images?
6. The manuscript lacks quantitative evaluation of the proposed method.
7. The presented applications (e.g., inpainting, editing) fail to demonstrate a clear advantage over existing methods or articulate the method's practical value.
8. Formatting issues are present (e.g., Line 432).

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
- This paper proposes a way to **project any noisy input (potentially any input)** onto the **learned manifold of any discriminative model**.
- The core idea is to obtain a **score-function-like quantity**, called the **Discriminative Score Function (DSF)**, from a discriminative model. This DSF can then be used as a score function to sample the closest image on the manifold—i.e., refine noise until it lies on the manifold.
- The DSF is defined based on the **Loss Tangent Kernel**, where the “distance to the manifold” is computed as a sum over all training data points in the form
    
    $\sum_i \text{kernel(current, data}_i)$
    
    - The projection is performed by minimizing this distance metric.
    - A beautiful contribution of this paper is how it simplifies this **non-parametric distance** (slow to evaluate) into a **parametric form** (fast to compute).
    - A major advantage of this definition is that it’s **purely derived from the model itself**, requiring fewer external regularizations (such as blurring or frequency constraints).
    - In practice, however, some augmentations (at least horizontal flipping) are still used during optimization.
- The applications are similar to **DeepDream-like feature visualizations**, but this method appears to work **unconditionally**, whereas previous methods required label information.
- The results are **comparable to DeepDream-style methods**—perhaps not strikingly so to an untrained eye—but the approach comes with **nice theoretical justification**.
- However, direct quantitative comparisons to prior methods are lacking.

### Strengths
*(Note: my background is neither in NTK nor feature visualization; I read this paper from an outsider’s point of view.)*

- I like how the paper connects **NTK (specifically the Loss Tangent Kernel)** to feature visualization. Although some doubts remain (see Weaknesses), this connection feels **fundamental** and well-motivated within the rich NTK literature. Feature visualization has long relied on external regularizations to work; since NTK is an intrinsic property of the network, it provides a **plausible and principled justification** for defining such a distance metric.
- From a quick literature check, I tend to agree with the paper’s claim that this is the **first method demonstrating unconditional projection/generation**, which is a nontrivial achievement.
- The unconditional generation quality is **reasonable**, given that it’s a gradient-based visualization method with minimal generative priors.
- By casting the projection operation as a **score function**—hence the name *DSF*—the method conceptually aligns with **diffusion and flow models**, potentially enabling future extensions.
    - This connection is **ingenious**, allowing seamless integration with **conditional generation** (Sec. 5.2), including **CLIP-guided results**. These outputs, while not state of the art, look impressive.
    - Section 6.4 introduces sampling tricks that seem promising, though still underexplored.
- Overall, I genuinely **learned something new** from this paper—specifically, how NTK can be used as a distance measure to project arbitrary points onto a learned manifold.

### Weaknesses
- Since optimization appears to start from $\mathcal{N}(0, I)$, it’s unclear whether the proposed distance metric is **well-defined everywhere** consider that most discriminative models never train on noisy inputs and may behave in an arbitrary manner under such inputs. The paper doesn’t discuss this, and I think it deserves attention.
- Currently, an augmentation (at least flipping) is required to obtain the objective function (Eq 11). To make this method truly universal, we should also the case where the model isn’t trained with any augmentation. How can one justify using such augmentation in the objective function?
- The paper claims relevance to **explainable AI**, but the evidence provided is weak. While the visualizations are interesting, the claim isn’t substantiated in real-world scenarios. For example, one could test whether DSF helps uncover known dataset or model biases, and quantitatively measure how well it reveals them.
- From an untrained eye, the visual improvement over **DeepDream-style** results isn’t very clear. The comparison in Appendix Fig. 11 doesn’t use the same model/dataset across methods, making cross-method evaluation impossible.
- The visualization quality is still **inferior to methods using generative priors**—though this might not be a real drawback, as DSF deliberately avoids such priors and thus remains “truer” to the discriminative model.
- The paper fails to convey **intuition behind the DSF metric**—what does “close” or “far” mean under this metric? What kind of image changes correspond to those distances? A section probing the **properties of the DSF metric** would be helpful.
    - For instance, how does changing the initial noise condition affect the final output? Does colored (biased) noise lead to colored outputs?
    - How diverse are the outputs DSF can produce? Some **qualitative or t-SNE-like visualizations** would be helpful to show coverage and variability.
- Overall, the paper makes a **solid and original contribution**, but falls short in **discussion and interpretability**.

### **Minor Suggestions**

- Line 34: “*as we cannot claim to understand what we cannot create*.” — This isn’t a factual statement. Consider changing the tone or adding quotation marks.
- Include an **algorithm block** summarizing the full projection process to make replication easier.
- Clearly list **which augmentations** are used (is it only flipping?).

### Questions
- Is the distance metric introduced by the paper well-defined everywhere, especially when optimization starts from a random initialization (e.g., $\mathcal{N}(0, I)$)?
- How can the paper substantiate its claims about explainability or interpretability in real-world scenarios (e.g., detecting dataset/model biases) better?
- Can the paper include visualization comparisons on the same model/dataset for easy comparison against other methods?
- What is the intuition behind the DSF metric?
    - What does it mean for two points to be “close” or “far” under this metric?
    - What kind of mental picture should readers have when trying to understand DSF geometrically or perceptually?
    - How does changing the initial noise condition affect the final projected output?
- How diverse or mode-covering are the outputs that DSF can produce?
- What role do augmentations perform during optimization?
    - Is there a way to generalize this method to models that are not trained with augmentations?

### Soundness
4

### Presentation
3

### Contribution
4
