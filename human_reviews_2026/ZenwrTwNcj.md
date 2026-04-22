# Midway Network: Learning Representations for Recognition and Motion from Latent Dynamics

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Object recognition and motion understanding are key components of perception that complement each other.
    While self-supervised learning methods have shown promise in their ability to learn from unlabeled data, they have primarily focused on obtaining rich representations for either recognition or motion rather than both in tandem.
    On the other hand, latent dynamics modeling has been used in decision making to learn latent representations of observations and their transformations over time for control and planning tasks.
    In this work, we present Midway Network, a new self-supervised learning architecture that is the first to learn strong visual representations for both object recognition and motion understanding solely from natural videos, by extending latent dynamics modeling to this domain.
    Midway Network leverages a _midway_ top-down path to infer motion latents between video frames, as well as a dense forward prediction objective and hierarchical structure to tackle the complex, multi-object scenes of natural videos.
    We demonstrate that after pretraining on two large-scale natural video datasets, Midway Network achieves strong performance on both semantic segmentation and optical flow tasks relative to prior self-supervised learning methods.
    We also show that Midway Network's learned dynamics can capture high-level correspondence via a novel analysis method based on forward feature perturbation.
    Code is provided at https://github.com/agentic-learning-ai-lab/midway-network.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors, in this work, introduce a new self-supervised learning (SSL) architecture called Midway Network.

- They extend the concept of latent dynamics modeling, used in control and decision-making, to the visual perception domain.

- Midway network is a self-supervised video model designed to learn both: Object recognition (semantic understanding) and Motion understanding (how things move over time) purely from natural, unlabeled videos — without relying on curated image datasets or external motion labels like optical flow.

- Midway Network outperforms or matches the best SSL baselines (like DINO, DoRA, CroCo v2) on both tasks: Semantic segmentation and Optical flow estimation.

### Strengths
### 1. Conceptual Soundness
- The authors ground their approach in predictive coding and latent dynamics modeling, both well-established ideas in neuroscience and machine learning.

- The theoretical motivation, that perception arises from predicting sensory changes is sound.

- The authors identify a clear gap: self-supervised models typically learn either recognition (DINO, iBOT) or motion (CroCo, FlowE), not both.

### 2. Architectural Novelty
- The architecture is a careful combination of: 1. Inverse dynamics (midway path) to infer motion, 2. Forward prediction (dense feature-level) to learn temporal coherence, and 3. Hierarchical refinement (inspired by optical flow networks e.g., PWC-Net, RAFT).

- The design is justified by analogies to biological systems and prior SSL hierarchies (e.g., Ladder Networks, DINO).

- The inclusion of gating units in transformer residuals is thoughtful as it prevents trivial identity mappings, a known issue in predictive models.

### 3. Experimental Soundness
- Comprehensive evaluations across recognition (semantic segmentation) and motion (optical flow).

- Extensive comparison against relevant baselines: DINO, DoRA, PooDLe, CroCo v2, DynaMo, etc.

- Consistent use of natural video datasets (BDD100K and Walking Tours) supports the claim of "learning from natural data only."

- Sufficient ablation studies test each architectural component's impact on both semantic and motion metrics.

### 4. Interpretation Soundness
- The "forwarded feature perturbation" method is a novel and interpretable way to visualize what the dynamics model learns.

- It qualitatively demonstrates non-trivial motion correspondence — a rare strength in SSL papers.

### Weaknesses
### 1. Conceptual Limitations
- While conceptually coherent, "latent dynamics" is borrowed from control/planning literature and adapted here somewhat heuristically — the paper lacks a strong theoretical derivation connecting latent dynamics to semantic learning in videos.

- The link between motion prediction and semantic invariance is intuitive but not formally analyzed.

### 2. Architectural Limitations
- The midway and backward paths add significant complexity; it’s unclear if all components are essential (though the ablation studies help).

- The architecture might overfit to short temporal correlations (1-second gaps between frames) and may not generalize to longer-horizon motion.

### 3. Experimental Limitations
- Only two pretraining sources (BDD and WT-Venice). That’s small compared to large-scale pretraining regimes (e.g., Kinetics-700, Ego4D). Therefore, it is hard to tell if results scale to diverse or indoor/outdoor mixed environments.

- The authors do not measure performance over multiple time steps (e.g., predicting motion 10 frames ahead).

- Some of the baselines (e.g., PooDLe) use higher resolution or external flow networks, making cross-comparison imperfect.

- Few metrics report standard deviation or multiple seeds.

### 4. Interpretation Limitations:
- The forward feature perturbation analysis is qualitative. There’s no quantitative measure of how well it aligns with ground-truth motion.

- The theoretical justification for why this analysis reflects "high-level correspondence" is intuitive but not formalized.

### Questions
Please see the discussion in weakness section.

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
This paper introduces Midway Network, a self-supervised learning framework that jointly learns object recognition and motion understanding from natural videos through latent dynamics modeling. The key idea is to integrate inverse and forward dynamics modules within a hierarchical architecture, where motion latents describe transformations between consecutive frames and are refined top-down across feature levels. Experiments show that Midway Network achieves strong performances in both semantic segmentation and optical flow tasks.

### Strengths
S1. Unified SSL framework for both object recognition and motion understanding: The paper presents a coherent approach that bridges predictive coding theory with modern self-supervised Transformers, achieving representation learning for both semantics and motion within a single framework.

S2. In-depth analysis: The analysis of learned motion latents in Section 4.4 (forwarded feature perturbation) is technically sound and creative. It provides interpretable evidence that the model captures spatial correspondences.

S3. Diverse Evaluation Domains: The proposed method is evaluated on several tasks and benchmarks.

### Weaknesses
W1. Incremental novelty: the proposed inverse + forward dynamics framework for learning latent motions follows a well-established approach used in latent world models and video prediction literature [a, b, c]. Moreover, the hierarchical refinement design is conceptually similar to the Spatial Dynamics Module (SDM) used in PooDLe..

W2. No guarantee of motion: The paper does not theoretically ensure that the learned latent $m_t$ encodes motion rather than directly leaking target feature information ($𝑧_{𝑡+1}$). Although empirical analyses (optical flow results, Sec. 4.4 perturbation study) suggest motion-like behavior, the model could still “hack” the objective by embedding target information in $𝑚$ making the representation less interpretable.

W3. Limited quantitative analysis of motion learning:
Section 4.4 provides compelling qualitative examples, but the conclusions would be stronger with quantitative validation. For instance, comparing perturbation-based correspondence against pseudo ground-truth from an off-the-shelf tracker could more rigorously substantiate the claimed motion alignment.

W4. Marginal effect of design components: As seen in Table 3 (e.g., 2→7, 3→8, 4→9 comparisons), adding design components (e.g., backward layers, gating) only yields small incremental gains, which questions their relative contribution to the overall improvement.

W5. (Minor) Lack of planning or world-model evaluation: Despite the discussion of potential applications to planning and world modeling, no experiments demonstrate this capability, slightly limiting the perceived impact of the proposed approach. 

[a] Bruce et al., “Genie: Generative Interactive Environments,” ICML, 2024.\
[b] Ye et al., “Latent Action Pretraining from Videos,” ICLR, 2025.\
[c] Gao et al., “AdaWorld: Learning Adaptable World Models with Latent Actions,” ICML, 2025.

### Questions
Q1. Why are optical flow results for PooDLe not reported in Tables 1–2?
Was there a fundamental limitation in adapting its architecture for flow prediction, or were these experiments omitted for other reasons?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents a novel self-supervised learning architecture for learning visual representations that jointly capture object recognition and motion understanding from video inputs. The proposed Midway Network achieves this by approximating a hierarchical latent embedding to model motion signals, thereby removing the need for an external motion predictor. Extensive experiments on object segmentation and optical flow tasks demonstrate the effectiveness of the approach.

### Strengths
1. By integrating latent optical flow estimation within a self-supervised framework, the proposed method enables simultaneous learning of motion information and content features through a single encoder. This design allows motion cues to be naturally incorporated into semantic representations.

2. The joint learning of motion and semantic features leads to consistent improvements in both semantic segmentation and optical flow performance.

### Weaknesses
1. While the paper contains several strong ideas, the overall framework is somewhat difficult to follow at first. It took multiple readings to fully grasp how the individual components contribute to the overall system. This could be improved by adding a concise, high-level overview of the architecture—perhaps at the beginning of Section 3—along with a summarizing figure that highlights the key components and their interactions.

2. Since one of the main contributions is the joint learning of motion and semantic representations from video, it would strengthen the paper to include a comparison with MC-JEPA to better contextualize performance gains.

### Questions
1. Figure 1 is referenced in the text but not included in the main manuscript.

2. Were all the additional components used during inference, or only in training?

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
2

### Summary
This paper introduces a new self-supervised learning framework that jointly learns object recognition and motion understanding from natural videos. It extends latent dynamics modeling to the video representation learning domain. The model combines an inverse-dynamics midway path, a dense forward prediction objective, and a hierarchical refinement structure to learn both semantic and motion features from unlabeled data.

### Strengths
1.The paper tackles an important and underexplored problem, learning both recognition and motion representations from unlabeled videos
2. It’s conceptually well-grounded, drawing inspiration from neuroscience to motivate the overall framework.
3. The architecture itself is Innovative. The midway path for motion latent inference feels like a fresh and thoughtful design choice.

### Weaknesses
1. It’s hard to disentangle architecture gains from data scale and model capacity
2. While the combination is novel, many components are incremental extensions of ideas from CroCo, DynaMo, or PooDLe.

### Questions
1. In Table 1, ViT-B only modestly improves flow over ViT-S and CroCo v2 still wins. What’s the failure mode for Midway at that scale?
2.The forwarded feature perturbation analysis is cool, could it be extended into a quantitative metric?

### Soundness
3

### Presentation
3

### Contribution
3
