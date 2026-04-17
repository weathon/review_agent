# Features Emerge as Discrete States: The First Application of SAEs to 3D Representations

- Decision: Accept (Poster)
- Scores: 2, 6, 4

## Abstract
Sparse Autoencoders (SAEs) have found human-interpretable features in LLM activations, clarifying how LLMs transform input to output. However, they have rarely been applied outside of text, limiting explorations of feature dynamics. We present the first application of SAEs to the 3D domain, analyzing the features found in 53k 3D objects encoded by a state-of-the-art 3D reconstruction VAE. We observe that the model encodes discrete rather than continuous features, leading to our key finding: the model's feature activations approximate a discrete state space, driven by phase-like transitions. Through this state space framework, we address three otherwise unintuitive behaviors — the preference for positional encoding features, the sigmoidal relationship between feature ablation and reconstruction loss, and the bimodal distribution of phase transition points. This final observation suggests the model redistributes superposition interference to prioritize the high-importance features. Our work not only catalogs and explains unexpected feature dynamics, but also provides a framework to explain the model's learning dynamics. The code is available at https://feature3d.github.io/Dora-SAE/.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This is a poorly written paper that seems to claim to be the first study on the application of SAEs on 3D datasets and presents corresponding findings and interpretations of the learned features.

### Strengths
Unfortunately, given the current organization and writing of the main content in this paper, it is extremely difficult to identify any valuable insights for readers to learn.

### Weaknesses
(W1) The contributions of this paper are vague and poorly discussed. For example, in line 41, the author states that "the scope of data domains has been limited — recent feature interpretability studies have focused on discrete and structured data, like image and text, rather than continuous or unordered data, ..." However, I don't see why the solution to this would be to study SAE applications on 3D data, as the authors mentioned in lines 48–49 and highlighted in the abstract. Why don't the authors try continuous and unordered data in 1D and 2D spaces first? Besides, a contradictory and confusing point is that the 3D point cloud data used in this paper's experiments is also discrete.

(W2) The writing of the methods section is very confusing and difficult to follow. It seems that the author proposes a summary formula for SAE works in Eqn (2). However, first, it is not clear how the authors derive and denote the well-known representation learning formula of Eqn (1) into Eqn (2). Second, it is unclear how the authors convert the related SAE works into their formula. For example, in lines 113–114, the authors state that "Recent LLM studies (Cunningham et al., 2024) use a sparse autoencoder (SAE) to approximate this decomposition with the assumption that $\bf{α}$ is sparse." However, it seems that the SAE formula in Cunningham et al. (2024) has nothing to do with the left part of Eqn (3) of this paper, and their α is a hyperparameter in the final loss function to control the sparsity of the reconstruction. And it has a completely different meaning from the proposed set of scalars $\bf{α}$ in this paper. Similar confusing descriptions and links to related works are everywhere in the methods section. Lastly, it is unclear how the authors "apply" an SAE to Dora-VAE in their major 3D data application. What is FPS in Eqn (7)? Where are the proposed E and $\bf{α}$ notations in Section 3? How exactly do the authors modify a VAE framework using an SAE structure?

### Questions
In generanl, the authors are encouraged to re-setup the goal and scope of this research and rewrite the whole paper for the future submissions.

Please also respond and provide clear explanations for the confusions I described above, which may change my opinions.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work provides the first application of Sparse Autoencoders (SAEs) to 3D data, decomposing the internal representations of a 3D Variational Autoencoder (VAE). The paper's central finding is that the model learns features as discrete states rather than continuous values, which are activated via phase transitions. This claim is supported by a large-scale analysis of 848k feature ablations, which reveal that: (1) High-impact features have a unimodal (single-peak) distribution of transition points. (2) Low-impact features have a bimodal (two-peak) distribution of transition points. The authors hypothesize this bimodality is evidence that the model actively manipulates superposition interference at inference time, pushing interference onto low-impact features to preserve the saliency of high-impact ones.

### Strengths
- Interpretability for 3D data is underexplored and interesting!
- Lots of experiment runs (e.g., 848k feature interventions) which makes the results robust.
- The theoretical explanation of how models see features as a decomposition of presence and identity is also interesting. 
- The bimodal experiments (Fig 5) and visualization (Fig 3) are insightful. 
- Validating the threshold t with max slope experiments is really nice!

### Weaknesses
- The learning dynamics and 3D contributions seem completely disjoint (although both interesting). Moreover, the paper makes broad claims about "a generally applicable, state-based feature framework." However, all the evidence is derived from a single model architecture (Dora-VAE) on a single data modality (3D point clouds). It's impossible to know if these findings (especially the bimodal transitions) are a fundamental property of feature learning, or a specific quirk of the Dora-VAE architecture e.g., and its cross-attention mechanisms.
- I am having difficulty understanding Figure 6 and more generally, the explanation for the bimodality. I generally get the intuition that the model would put more interference with the low-impact concepts, but have difficulty following the explanation in Lines 459-475. 

Small stuff

- Figure captions should be more detailed and explain (i) why this plot is being shown and (ii) what takeaways the reader should make. Not just a short description of what is shown.

### Questions
- Do these findings hold across different models and data types? Or is this only for Dora-VAE and 3D pointcloud data? Showing these findings hold on another model/domain would increase the contribution strength for the learning dynamic theory section a lot. Or, this could be added as a weakness/limitation to the paper since it is only applied to 3D data (even though applying SAEs to 3D data is a contribution itself, it does not mean all general findings on 3D data will hold to other domains).
- What is Figure 6 showing? It is not clear if it is a real experiment or just a visual aid. It is just mentioned in the first line of Line 454 and 459, but the text and caption are insufficient to parse the figure. 
- Moreover, could the authors kindly explain in more clear terms how the bimodal transition arises? I think this section could even be dropped, or written as a 'hypothesis' rather than making stronger claims about superposition/interference.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
1

### Summary
This paper investigates the internal features of a 3D reconstruction model by being the first to apply Sparse Autoencoders (SAEs) to the 3D domain. The authors use an SAE to analyze the latent vectors of a Dora-VAE (a 3D reconstruction VAE) trained on the Objaverse dataset.  The central finding is that the Dora-VAE's internal features are **discrete, not continuous**.

This paper proposes a novel theory framework oo explain "unintuitive" behaviours of VAEs. They analyze the gradient and decompose it into two components: the feature's **"presence** ($\alpha_j$) and its **"identity"** ($e_j$). The framework suggests that the model prefers to learn discrete features because the learning signal for a feature's *identity* is scaled by its *presence*. Specifically, the **unimodal** (single-peaked) distribution of transition points for **high-impact features** is explained by the feature's "presence" term. In contrast, the **bimodal** (two-peaked) distribution observed for **low-impact features** is explained by the model's active manipulation of superposition. The authors further apply SAEs into VAEs for 3D data and explain the unusual properties.

### Strengths
1. New Theoretical Framework: The decomposition of the learning gradient into "presence" ($\alpha_j$) and "identity" ($e_j$) provides a new and powerful lens for why features emerge in a certain way, moving beyond just observing what features are learned.
2. Well-motivated. The application of SAEs to the 3D domain is motivated by the theory analysis to extend interpretability.
3. Strong Explanatory Power: The proposed framework successfully connects several counter-intuitive empirical observations (discrete features, sigmoidal loss curves, and bimodal transition points) into a single, cohesive theory.

### Weaknesses
1. The writing can be improved for clarity, such as the definition of ARC and loss difference. It is not clear how to calculate them. What the colored points representing are also not depicted. 
2. The findings are based on the Dora-VAE architecture, which processes 3D models by sampling points and using point features and positional encodings. The introduction of SAE is an incremental contribution.
3. The paper uses the terms "discrete state space" and "phase transition" heavily. While this is a helpful analogy for the observed sigmoidal loss curves, it may overstate the case.

### Questions
How important it is for the application of SAE to 3D data? What is performance of the proposed method in Dora-bench?

### Soundness
3

### Presentation
1

### Contribution
3
