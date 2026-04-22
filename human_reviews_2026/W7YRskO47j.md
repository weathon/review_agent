# CLUTCH: Contextualized Language model for Unlocking Text-Conditioned Hand motion modelling in the wild

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Hands play a central role in daily life, yet modeling natural hand motions remains
underexplored. Existing methods that tackle text-to-hand-motion generation or
hand animation captioning rely on studio-captured datasets with limited actions
and contexts, making them costly to scale to “in-the-wild” settings. Further,
contemporary models and their training schemes struggle to capture animation
fidelity with text–motion alignment. To address this, we (1) introduce ‘3D Hands
in the Wild’ (3D-HIW), a dataset of 32K 3D hand-motion sequences and aligned
text, and (2) propose CLUTCH, an LLM-based hand animation system with two
critical innovations: (a) SHIFT, a novel VQ-VAE architecture to tokenize hand
motion, and (b) a geometric refinement stage to finetune the LLM. To build 3D-
HIW, we propose a data annotation pipeline that combines vision–language models
(VLMs) and state-of-the-art 3D hand trackers, and apply it to a large corpus of
egocentric action videos covering a wide range of scenarios. To fully capture
motion in-the-wild, CLUTCH employs SHIFT, a part–modality decomposed VQ-
VAE, which improves generalization and reconstruction fidelity. Finally, to improve
animation quality, we introduce a geometric refinement stage, where CLUTCH is
co-supervised with a reconstruction loss applied directly to decoded hand motion
parameters. Experiments demonstrate state-of-the-art performance on text-to-
motion and motion-to-text tasks, establishing the first benchmark for scalable
in-the-wild hand motion modelling. Code, data and models will be released.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the problem of text-conditioned hand motion generation in unconstrained "in-the-wild" settings. The authors make three primary contributions: (1) a novel data annotation pipeline combining VLMs and 3D hand trackers to create the 3D-HIW dataset with 32K hand motion sequences and aligned text descriptions, (2) SHIFT, a part-modality decomposed VQ-VAE tokenizer that separately encodes trajectory and pose for left and right hands, and (3) CLUTCH, an LLM-based system featuring a geometric refinement training stage that applies reconstruction losses directly to decoded motion parameters. Experiments demonstrate improvements over baselines including HumanMDM, MotionGPT, and T2M-GPT on both text-to-motion and motion-to-text tasks.

### Strengths
1. The focus on in-the-wild hand motion generation addresses a significant gap in the literature, moving beyond studio-captured datasets with limited diversity.
2. The 3D-HIW dataset with 32K sequences represents approximately 10× the scale of GRAB and ARCTIC, offering substantially greater diversity in actions (1045 verbs, 1355 objects) and scenarios.
3. The two-stage approach using Parallel Chain-of-Thought prompting followed by closed-vocabulary refinement is well-motivated and demonstrates improved GPT-scores (6.9) compared to existing methods.

### Weaknesses
1. Dataset quality concerns:
- The reliance on HaWor for 3D reconstruction may introduce systematic errors or artifacts that propagate through the entire pipeline.
- The filtering criteria (80% hand visibility, acceleration thresholds) may introduce biases toward certain types of motions.
- No human evaluation or validation of the reconstructed 3D motions is provided.

2. Geometric refinement stage lacks clarity:
- The Gumbel-Softmax formulation is mentioned but not detailed.
- The balance between cross-entropy and reconstruction loss (α, λ) appears critical but hyperparameter sensitivity is not thoroughly analyzed.
- The comparison to EgoLM's soft-blending approach (Table 5) shows only marginal improvements, raising questions about whether the added complexity is justified.
3. The paper primarily compares against methods designed for full-body motion (HumanMDM) or general motion (MotionGPT). Comparisons with recent hand-specific methods or SOTA human motion generation method (MoMask) are missing.

### Questions
1. Have you conducted human evaluation of the generated annotations? What is the agreement between your automated annotations and human-written descriptions?
2. How does the model perform on motions significantly different from the training distribution? Can it generate novel compositions of actions?

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
4

### Summary
The work introduces a large-scale dataset of human hands automatically extracted from a large-scale egocentric video dataset, consisting of motion sequences and accompanying textual activity descriptions. Additionally, a method for text-to-hand trajectory generation and hand trajectory-to-text description is proposed, benefiting from a novel hand tokenization scheme. The proposed method outperforms multiple baselines.

### Strengths
The scale of the hand motion dataset is unprecedented. The work provides a thorough analysis of the introduced dataset. The proposed method outperforms multiple baselines on the introduced dataset. The design choices of the method and filtering of the dataset are quantitatively supported by ablation studies. The proposed method supports both the text-to-motion and motion-to-text tasks simultaneously.

### Weaknesses
For a work introducing a novel dataset as its main contribution, more qualitative examples of the generated trajectories as well as hand poses and coarse/fine-grained textual descriptions are necessary. This is a major weakness, especially coupled with the following concern:
The noun distribution in Figure 11 shows several undesirable entries being common in the dataset, e.g. "hand" (hand touching a hand?) and "cut" (a verb?). This raises questions about the quality of the dataset's noun/verb annotations.
The efficacy of the method is supported by its strong performance against baselines. However, at no point does the work mention any human verification of the generated dataset. As such, it is difficult for a reviewer to ascertain its quality. A human study would have greatly benefited the work. The contribution from the dataset side is thus limited for me.

In addition to insufficient qualitative examples of the dataset, more qualitative examples of the method's output must be included.

The proposed method was only evaluated on the introduced dataset, and not on other datasets such as GigaHands or ARCTIC.

It would be good to add qualitative examples of trajectories to the rightmost t-SNE plot in Fig. 6. Merely covering a broader t-SNE range of hand poses could also be achieved by a large fraction of erroneous poses in the dataset.

The object is not at all considered in the proposed dataset and method, limiting their usefulness.

The work could benefit from a table comparing it to existing datasets in terms of scenario count, total length, diversity (number of objects), etc.

### Questions
In Section 3.3, what is meant by "top-200" and "top-3000"?

What is the purpose of the introduced "testing" split if numbers are reported on the validation split only?

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
The researchers present two main contributions:
1. A large-scale dataset, 3D Hands in the Wild (3D-HIW): They construct this dataset using 3D hand trackers to extract hand trajectories from egocentric videos and leverage vision-language models to generate corresponding textual annotations.
2. The CLUTCH model: A transformer-based model trained on 3D-HIW, introducing two key innovations: (a). SHIFT: A novel tokenization method that decomposes hand motions into separate trajectory and pose components for each hand, improving generalization and yielding more accurate motion reconstructions. (b). Geometric Refinement: A fine-tuning stage applied to the language model that enhances the geometric accuracy and realism of the generated animations. This includes a reconstruction loss directly applied to the decoded 3D motion parameters.

### Strengths
1. The motion dataset represents a highly valuable contribution to the field.
2. The SHIFT mechanism is well-motivated, effectively decoupling trajectory-level movements from fine-grained finger motions, and ablation studies demonstrate its effectiveness. Also, enabling bidirectional motion–text decoding is an innovative design, and it is noteworthy that this approach performs successfully in practice.

### Weaknesses
1. Since the hand motions are synthesized from a single textual description, how is motion diversity ensured? Are there mechanisms in CLUTCH to generate varied hand trajectories or poses from the same caption?
2. While the proposed approach is effective for isolated hand motions and the datasets are centered on hand-only movements, its omission of object interactions could constrain its applicability to more realistic, object-involved settings.

### Questions
1. What "Div →" means ?
2. The presentation could be improved, for example, inconsistent title capitalization (Line458), text size in tables, and line-breaking logic (Line271).
3. Because text is discrete while motion is continuous, it may be more natural to model hand motions using architectures like TransFusion, similar to how VLMs such as Pi0?

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
4

### Summary
This work presents a system for text-conditioned 3D hand motion generation and captioning in in-the-wild settings. The two primary contributions:
1. A large-scale dataset of 32,000 3D hand-motion sequences paired with textual descriptions, sourced from egocentric videos (Ego4D). 
2. An LLM-based system that models motion tokens. A new VQ-VAE tokenizer that decomposes hand motion into separate codebooks for trajectory and pose
The authors demonstrate their experimental results on their new benchmark for both text-to-motion and motion-to-text tasks.

### Strengths
1. The 3D-HIW dataset is a good contribution. The proposed VLM-based annotation pipeline is a clever and scalable approach to captioning this in-the-wild data.
2. The paper does a good job of validating its design choices with the ablations for the SHIFT tokenizer and the training stages.
3. The paper is well-written, the figures are informative, and the core ideas are articulated clearly.

### Weaknesses
1. Lack of Hand-Object Interaction (HOI) is the most significant limitation. The paper frames its work as "in-the-wild"  yet the model only generates 3D hand motion. It does not model the objects being interacted with. True in-the-wild motion is almost entirely defined by HOI, which is explicitly left as future work.
2. The model is trained and evaluated exclusively on the authors' new dataset. It is unclear how CLUTCH would perform on other public benchmarks.

### Questions
1. How much computational resource and time does it take to generate a dataset of this size? It's key to show whether the proposed method is scalable.
2. The paper says currently their current method doesn't support HOI generation. How is Figure 1 generated? Do you manually align the hand trajectory to fit the objects?

### Soundness
3

### Presentation
3

### Contribution
3
