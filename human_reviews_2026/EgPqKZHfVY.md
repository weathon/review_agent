# SLAE:  Strictly Local All-atom Environment for Protein Representation

- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Building physically grounded protein representations is central to computational biology, yet most existing approaches rely on sequence-pretrained language models or backbone-only graphs that overlook side-chain geometry and chemical detail. We present SLAE, a unified all-atom framework for learning protein representations from each residue’s local atomic neighborhood using only atom types and interatomic geometries. To encourage expressive feature extraction, we introduce a novel multi-task autoencoder objective that combines coordinate reconstruction, sequence recovery, and energy regression. SLAE reconstructs all-atom structures with high fidelity from latent residue environments and achieves state-of-the-art performance across diverse downstream tasks via transfer learning. SLAE's latent space is chemically informative and environmentally sensitive, enabling quantitative assessment of structural qualities and smooth interpolation between conformations at all-atom resolution.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes training an all-atom protein environment encoder to learn general-purpose embeddings through reconstruction and energy prediction objectives. The model adopts an encoder–decoder architecture based on SE(3)-equivariant graph neural networks, where the encoder captures local atomic environments and the decoder reconstructs global structural properties.

### Strengths
The paper is well written and clear.
The authors created a pipeline that utilizes both structure and energy to get meaningful encodings both structurally and chemically.
The authors showed the usefulness of their approaches on some important downstream tasks.

### Weaknesses
1. Previous works have shown that MLFF model embeddings (e.g., MACE, ORB, Egret) also perform with high success on downstream tasks such as chemical shift prediction. Comparing SLAE embeddings to these models would be most appropriate in this paper’s context, as both approaches use similar techniques to address the same problem. The training of SLAE’s encoder is similar to that of these MLFF models, except for the added structure reconstruction and the local-encoder/global-decoder design.  Such a comparison would clarify whether SLAE’s success stems from its new training scheme or from the inclusion of energy prediction, which has been explored before.  

2. The authors emphasize the importance of representing proteins at all-atom resolution but ultimately produce residue-level embeddings.  Previous studies have shown that side-chain placement is highly correlated with backbone atom positions, explaining the success of side-chain packing methods.  Since SLAE’s training phase uses inter-residue (rather than inter-atomic) energy terms, it would be informative to test whether the same pipeline without side chains performs comparably. This would reveal whether the all-atom representation truly adds value, given that the encoder and decoder can likely infer side-chain positions from the backbone.  

3. The authors claim to achieve state-of-the-art results on various downstream tasks, stating that they  
   - “achieve state-of-the-art on diverse downstream tasks with transfer learning,” and  
   - “SLAE achieves state-of-the-art or on-par performance.”  
  However, they do not provide comparisons to true state-of-the-art models on several key downstream tasks, such as chemical shift prediction [1,2] and protein–protein interaction prediction [3].  
   It might also be the case for the other reported tasks, though these are outside my area of expertise. The fact that the comparisons for chemical shift and protein–protein interaction are not against the actual SOTA raises concerns about whether similar issues exist for the other downstream benchmarks as well.

[1] Li, J., Bennett, K. C., Liu, Y., Martin, M. V., & Head-Gordon, T. (2020). UCBShift: Accurate prediction of chemical shifts for aqueous protein structure on “real-world” data.

[2] Bojan, M., Vedula, S., Maddipatla, A., Sellam, N. B., Napoli, F., Schanda, P., & Bronstein, A. M. (2025). Representing Local Protein Environments with Atomistic Foundation Models. 

[3] Pourmirzaei, M., Han, Y., Esmaili, F., Alqarghuli, S., Chen, K., Pourmirzaei, M., & Xu, D. (2025). EXTENDING PROT2TOKEN: Aligning Protein Language Models for Unified and Diverse Protein Prediction Tasks.

### Questions
See Weaknesses

### Soundness
2

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
2

### Summary
The authors propose SLAE, a new architecture for learning protein representations. Minimal, equivariant graph encoders handle local interactions (which only see atom positions and no handcrafted torsion angle features, etc.) whereas a standard transformer operates on the outputs of these encoders to model long-range dependencies. Models are trained with a custom objective that includes a sequence recovery task as well as energy prediction. The authors show that the resulting model performs well across diverse benchmarks and boasts an interpretable latent space.

### Strengths
The paper is clearly written (I appreciate the "design motivation" paragraphs) and well-motivated. The experiments are diverse, and the method performs well throughout. Section 5 is pretty nifty too and further helps motivate design choices.

### Weaknesses
In no particular order:

1. Why is the model trained exclusively on synthetic structures? Is it possible that the method only achieves such low RMSDs because these predicted structures have consistent biases? I don't see this justified anywhere in the paper, and would love to see some e.g. PDB experiments, if possible.
2. The paper states that the model is trained for 30 epochs across 300k proteins on a single A100. I know that the transformer component is minimal, but this seems a bit ambitious to me. Can the authors share the wall clock time of a full training run?
3. Claims of generalizability to complexes and RNA are unsupported by experiments in the paper.
4. Section 3.4 is confusing me; in section 1, it's stated that "the ground-truth residue identities" are provided as inputs, but these are the targets in section 2. It's unclear to me that leakage isn't occurring here and I'll need some clarification from the authors.

Minor nits:

a. The citation for FAPE is given as (Anishchenko et al., 2024); I think the authors meant to cite (Jumper et al., 2021), to which SmoothLDDT is incorrectly attributed in turn.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
They pretrain an all-atom autoencoder (where the encoder does message passing within spatially local atom environments and the decoder uses a global context transformer + ROPE) to reconstruct structure, sequence, and energy values. Using the pretrained encoder as a backbone, they then train several prediction heads for downstream tasks (fold classification, binding affinity prediction, single-point mutation thermostability prediction, and chemical shift prediction). They also find that linear interpolation in latent space corresponds to interpolation between physcially coherent structures.

### Strengths
- This is one of the few autoencoders, if not the only one, that recovers all aspects of the input (structure, atom identities, and energy values) and that operates on an all-atom level.
- They ablate several design choices, including the choice of continuous vs discrete latent representations (as well as the type of quantization), the effect of the additional energy term in their reconstruction loss, cutoff radii (though only 6A and 8A were compared), and the use of FAPE.
- They achieve competitive results on downstream prediction tasks.

### Weaknesses
- I am doubtful that the latent space interpolation property is unique to this architecture. This probably applies to other models as well.
- How good are your representations for downstream generative tasks? You can do two sets of generative tasks here, one with continuous latent representations and the other with a discrete codebook. In both cases you can do something like latent diffusion. In the case of discrete codebook you can also do something like autoregressive/masked language modeling token generation. 
- Asides from the novel objective function, not much else strikes me as being very novel in this approach.

### Questions
- How important is spatial locality for reconstruction and downstream performance? Your ablation seemed to only compare 8A with 6A cutoffs. Using a wider range of cutoff sizes can probably give you a better sense of the impact of locality on performance.
- How important is the all-atom representation? My guess is you can apply the same architecture, but at the residue-level, to get a sense of whether the all-atom resolution is indeed important for reconstruction + downstream performance.
- Any notes on the efficiency of your encoder architecture and its scalability to longer sequences (in case you need to consider a bigger all-atom neighborhood)?

### Soundness
3

### Presentation
3

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
SLAE is a physically grounded all-atom protein autoencoder. It first constructs a structure-sequence that encodes each residue as its local all-atom neighborhood consisting of atomic geometries and atom types. The residues are tokenized then passed through a decoder to predict sequence identities, 3D coordinates and energies. The SLAE encoder is evaluated on fold classification, protein-protein binding affinity prediction, variant effect prediction and chemical shift prediction.

### Strengths
This paper introduces a powerful abstraction between protein structure and sequence. It is able to encode all of the atoms present in the protein structure in an organized sequence representation that enables convenient decoding of downstream phenotypes.

### Weaknesses
The paper lacks important ablations that validate the effectiveness of the all atomic resolution. In particular, it fails to compare against sequence-structure co-embedding works such as SaProt and ESM3. It is critical that SLAE outperforms these structure-sequence models that encode the structural embeddings without the same resolution.

There are other works that reason about the all atomic representations of local structures, such as Distilling Structural Representations into Protein Sequence Models, Ouyang-Zhang et al. The paper does not discuss how SLAE compares against such works.

### Questions
How does SLAE perform against SaProt and ESM3?

How does the all atom autoencoder architecture compare to that in Distilling Structural Representations into Protein Sequence Models, Ouyang-Zhang et al?

### Soundness
2

### Presentation
3

### Contribution
3
