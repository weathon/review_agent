# SimpleDesign - A Joint Model for Protein Sequence and Structure Codesign

- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Proteins are fundamental to biological processes, with their function determined by the complex interplay between the amino acid sequence and the three-dimensional structure. Developing generative models capable of understanding this intrinsically multi-modal relationship is crucial for fields like drug discovery and protein engineering. Existing models often rely on a multi-stage training process where autoencoders that tokenize data into latent representations are trained in a first stage. Secondly, a generative model is trained on the latent representation of the autoencoder(s), i.e., generative modeling in a latent space. We hypothesize that this multi-stage training process is not required to obtain performant co-design models and thus present SimpleDesign, an effective multi-modal protein design model trained directly in the raw data space. SimpleDesign leverages a simple end-to-end training objective with two terms, a discrete cross-entropy for protein sequences and a continuous flow-matching regression objective for protein structures. 
In order to better model the sequence and structure modalities, we develop a Mixture-of-Transformer architecture that allows modality-specific processing while keeping global self-attention over both modalities.
We train SimpleDesign on 1.8M sequence-structure pairs achieving strong performance across co-design and unconditional sequence/structure generation benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces SimpleDesign, a protein sequence and structure co-design model, that leverages the Mixture of Transformers architecture. The model is trained end-to-end directly in the raw data space, explicitly avoiding the structural tokenization and multi-stage pipelines common in existing methods. The authors demonstrate strong, competitive results against leading existing co-design methods such as DPLM2 and ProtPardelle.

### Strengths
**Originality**: The paper does include a new architecture for co-design, based on the mixture-of-transformers. It is a simpler architecture, avoiding some of the very commonly seen modules in protein folding and structure designs (e.g. pairformer)

**Quality**: The performance is competitive with other methods, and comes out on top in some of the chosen metrics. 

**Clarity**: The writing is clear and comprehensive. The introduction provides a good review that covers most of the relevant literature. The authors demonstrate a good understanding of the problem at hand, and some of the state-of-the art approaches. The figures, particularly figures 1 and 3 are very well designed, and clearly explained the model architecture, and the idea behind MoT. 

**Significance**: The idea of simplifying architectures for protein design is important and useful.

### Weaknesses
While the paper has some strong points highlighted above, there are some major issues. 

## Conceptual Points
- The paper focuses on unconditional protein design. While this is an interesting problem, there has been a lot of work towards protein design in the last few years (as the papers do highlight in their introduction), and a lot of this work has shown the difficulty of going from unconditional, to conditional protein design. At the same time, it is clear that there is little practical use of unconditional design models. For downstream scientific applications, designing generic proteins is not particularly helpful. Scientific use requires designing proteins to perform specific functions and/or bind to specific targets. Therefore, while these results would have merited conference acceptance 2/3 years ago, the competitive landscape means that unconditional design is (in this reviewer's opinion) not enough. 
- SimpleDesign can jointly generate protein backbones and sequences. However (and again, the authors point this out in their introduction) there are now existing models that do all-atom generation. This is not only also co-design (as specifying all the atoms means implicitly defining the protein sequence) but also a much harder problem. It would have been interesting to see how SimpleDesign could combine the designed $C_{\alpha}$ backbone and sequence into an all-atom structure, and how that compares with existing all-atom models such as La Proteina. 
- It is concerning to me that the model is trained only on the filtered AFESM dataset, a massive corpus of predicted/synthetic structures derived from AlphaFold and ESMFold synthetic structures (unless I have misunderstood). The authors do highlight throughout the paper, and clearly illustrate in figure 2, the relationship between co-design and folding. However, their training data comes entirely from folding models. This to me indicates that any issues or flaws of the folding models would also be learned by the model. It is not clear why experimental data for the PDB was not also considered during training, in addition to synthetic data. 
- Finally, the paper talks extensively about the simplicity of their architecture. In my opinion, there are three possible arguments in favor of simplicity: It reduces computational cost, it might lead to improved performance (by adding the right inductive biases and/or preventing overfitting), or it is just for academic/learning purposes. It is unclear to me if the reason the authors wanted a simpler architecture was reduced cost, or if they actually expected improved performance because of the simpler architecture. In the latter case, it is not clear to me that the chosen architecture adds any inductive biases that would lead to improved performance (like symmetries in the 3D generation). It would be very useful to get some more clarity on why a practitioner would want to choose this simpler model, and not a bigger one, when they had to design proteins. 

## Other issues 
- In line 105 onwards, a number of models are mentioned to generate backbone atoms. These, however, include La Proteina, which is an all-atom method (it is in fact cited as such later in line 110). I would also suggest adding Latent-X to the all-atom methods cited. 
- The results do not include any all-atom methods (which as discussed, are also doing co-design). In fact, given the choice of metrics, that results tables 1 and 2 could include 3D backbone design methods (RFDiffusion, FoldFlow, etc) combined with inverse folding (PMPNN for example). And, based on results from these backbone design papers, their numbers might be better than anything in tables 1 or 2. 
- Performance is not bad, but there doesn't seem to be a unique design choice for SimpleDesign that actually surpasses all other models considered. This makes it harder to recommend the paper for publication

### Questions
- The model uses two times ($t$ and $t'$) to corrupt the sequence and structures. Would it be possible to use a unified time index, that is discretized for the sequences? 
- In the results, the noise scale $\gamma$ seems to play a very important part, but there is no discussion in the main text about the choice of noise scale, or noise schedule. In fact, nowhere in the main text is it mentioned what $\gamma$ is. Given the importance of this parameter in the final results, it would be great to see more discussion on how it is chosen, and the tradeoff between TM score and diversity that comes from changing it. 
- The co-design metric is, in fact, very similar to the designability metric often used in 3D backbone design, with the difference that backbone design requires inverse folding. This should be mentioned. Furthermore, the paper says that "one can use either scRMSD < 2 or scTM > 0.9", but it does not say which of these is actually used in the tables and figures.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
Protein function depends on the interplay between amino acid sequences (discrete) and 3D structures (continuous), but existing generative models for protein co-design rely on multi-stage training (e.g., pretrained autoencoders for structure tokenization) that adds complexity and inefficiency. To address this, the authors propose SIMPLEDESIGN, a minimalist end-to-end multi-modal generative model that directly processes raw sequence and structure data without tokenization.

### Strengths
1. SIMPLEDESIGN breaks from prior multi-modal protein design paradigms by eliminating structural tokenization and multi-stage training—a core limitation of models like ESM3 and DPLM2.
2. The MoT architecture is also innovative: unlike fully decoupled or naive fused models, it explicitly models modality-specific signals (e.g., sequence symbolicity, structure geometry) while enabling cross-modal interaction, aligning with biological priors of sequence-structure coupling.

### Weaknesses
1 .SIMPLEDESIGN is evaluated exclusively on proteins of length 100–500, but many biologically critical proteins (e.g., fibrous proteins, multi-domain enzymes) exceed 500 residues. Additionally, it relies on 3D coordinates (Cα atoms) and secondary structure signals, making it unable to handle intrinsically disordered proteins (IDPs)—~30% of eukaryotic proteins that lack fixed structures but play key roles in signaling. This limits its applicability to the full proteome.
2. No experiments test if generated sequences fold into functional structures (e.g., enzyme catalytic activity, ligand binding).

### Questions
1. The MoT uses separate QKV projections for sequence and structure. Did the authors test if sharing some projections (e.g., value vectors) could reduce parameters without degrading performance?
2. Why it is name SimpleDesign？ why it is simple?
3. Ablate the number of modality-specific FFN layers (1 vs. 2 vs. 3) and report performance—this could reveal if 1 layer is sufficient to capture modality signals, reducing model size.
4. Collaborate with experimental teams to express 5–10 generated proteins (e.g., enzymes) in vitro and test for activity (e.g., hydrolysis assays)

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
SimpleDesign leverages a mixture of transformer architecture to directly learn protein co-design over the the raw structure and sequence space. A central focus of SimpleDesign is the avoidance of structural tokenization, latent representations, and attention pair bias commonly found in prior protein design works. SimpleDesign performs competitively to prior tokenization-based co-design models.

### Strengths
- Competitive performance to tokenization based methods. Compared to DPLM2 the model is competitive but does not require any tokenization. This represents another modality of taking off the shelf protein LLM's and extending them to generative tasks through a simple objective with a lot of data.
- Evidence shows that pure transformers be trained for de novo protein generation for both continuous and discrete data types.
-MoT architecture is a very fitting design choice to preserve scalability but along couple these conditional data types.

### Weaknesses
- Overall outside of the new architecture the technical novelty is low when compared to MultiFlow which uses a nearly identical training objective.
- I find the simplicity angle quite interesting but the tangible benefits are not reported (speed, size, data consumption/need, accuracy)
- Novelty seems really high accompanied with low diversity (Table 1). Is there any concern about memorization or overfitting?
- Missing benchmarks
  - MultiFlow is cited yet not compared to
  - Protpardelle (1-c) is an all atom model yet La-Proteina is cited and not compared to
  - Similarly P(all-atom), and PLAID are not referenced/compared to.
  - FoldFlow-2 would also be good comparison for backbone only and sequence conditioned generation
- Competitive performance to tokenization based methods but those methods are far below state-of-the-art as seen in P(all-atom) and La-Proteina which both can be evaluated in CA/seq co-design mode as Protpardelle and the recent Protpardelle 1-c were used.
- Initializing weights with an existing LLM (ESM2-650M) is not very different than using the embeddings themselves (PLAID) or training a separate autoencoder.

### Questions
- SimpleDesign seems to generate very non diverse structures and sequences with high similarity toward the training data yet does not have high co-designability nor designability. Any thoughts as to what could be the reason for this?
- What happens if the model is not initialized with ESM2 weights?
- How much does the dataset impact the results? If SimpleDesign was trained on just PDB does the simple architecture still generalize or is the simple architecture require large data?
- How is the speed and memory usage compared to prior methods? I would assume the simple architecture performs quire strongly here but it is not reported.
- MultiFlow being the first data space CA/seq co-design method also evaluated on forward folding and inverse folding. Does SimpleFold perform well in these cases?
- Backbone only numbers from Table 5 do not match the long length model reported for Proteina and Genie2. 
- Any thoughts as to why the mmseqs sequence diversity is low for SimpleDesign in Figure 8?
- For the co-generated samples in Table 1 what is the amino acid distribution? Is there any over population of any of the amino acid types?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces SimpleDesign, a multi-modal generative model designed to jointly create novel protein amino acid sequences and their corresponding 3D structures. The core problem it addresses is that existing models often rely on complex, multi-stage training pipelines, particularly the use of autoencoders to tokenize continuous structural data into a discrete format. The authors hypothesize that this tokenization step is unnecessary. SimpleDesign instead operates directly on discrete sequences and continuous coordinates. It employs a Mixture-of-Transformer architecture that allows for modality-specific processing while still enabling joint self-attention across both sequence and structure data. The model is trained end-to-end with a simple, combined objective containing a discrete cross-entropy loss for the protein sequence and a continuous flow-matching regression loss for the protein structure. Experimental results show that SimpleDesign achieves competitive performance in co-generation and separate structure-only and sequence-only generation tasks, often outperforming its more complex, tokenized counterparts.

### Strengths
1. Proposing a novel SimpleDesign that eliminates the need for a separate structure tokenizer by operating directly on continuous 3D coordinates. This simplifies the entire training process into a single end-to-end objective.
2. SimpleDesign demonstrates a high co-designability score, surpassing other models. This suggests its joint objective and architecture are highly effective at capturing the correlation between the two modalities.
3. The authors leverage Mixture-of-Transformer (MoT) architecture, which effectively handles the multi-modal data by applying modality-specific and cross modal processing.

### Weaknesses
1. Across all unconditional generation tasks (co-design, structure-only, and sequence-only), the proposed method consistently exhibits low diversity. This suggests a significant risk of mode collapse.
2. The paper proposes using a Mixture of Transfomer architecture to better process modality-specific and cross modal information. However, the results in Table 4 are contradictory, showing that the MoT-based model achieves lower designability performance.
3. The evaluation is only limited to unconditional generation. The paper lacks results for conditional generation tasks (e.g., inverse folding and protein folding), which are more aligned with practical protein design scenarios and are essential for comprehensively demonstrating the proposed method's multimodal generation capabilities.
4. For the co-generation task, the authors should provide a comparison against more baselines such as [1][2][3][4].
5. SimpleDesign concatenates sequence and structure inputs along the sequence dimension, which is likely to introduce substantial computational overhead, especially when dealing with longer protein sequences.


[1] Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design

[2] P(all-atom) Is Unlocking New Path For Protein Design

[3] Elucidating the Design Space of Multimodal Protein Language Models

[4] La-Proteina: Atomistic Protein Generation via Partially Latent Flow Matching

### Questions
The paper models 3D coordinates directly rather than using structure tokenization. Could the authors provide a detailed discussion on the advantages of this direct modeling approach compared to using discrete structure tokens? Beyond the benefit of a simpler pipeline that already mentioned in the manuscript, are there other significant advantages to this design choice?

### Soundness
2

### Presentation
3

### Contribution
3
