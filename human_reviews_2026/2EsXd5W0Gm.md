# Bridging Protein Structure to Sequence via Local Structure for Inverse Folding

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
The design of protein sequences based on given structures, known as inverse folding, has important applications in protein engineering. Protein structures are inherently hierarchical, composed of local structures (e.g., α-helices and β-sheets) connected by loops and coils. However, most existing methods treat inverse folding as a direct 3D structure to 1D sequence task, ignoring this crucial hierarchical information embedded in local structures. In this work, we propose Hier-IF, a controllable inverse folding model that explicitly incorporates structural hierarchy. Hier-IF reformulates the task as a “Tertiary Structure (TS) to Local Structure (LS) to Sequence (Seq)” process by first generating the sequence tokens corresponding to local structures and then building the connecting loops and coils. We introduce classifier-free guidance for controllable hierarchical generation and employ a bidirectional structure-sequence reconstruction loss during the training process. In the sampling process, we design a remask strategy that enables controllable generation following the structural hierarchy. When evaluating Hier-IF across multiple datasets, it surpasses other baselines and achieves high structural fidelity in local structures. Visualizations on generation results and ablation studies in different experimental settings further validate the effectiveness of our approach and provide interpretability in protein hierarchical inverse folding.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the important inverse folding problem in protein design. The authors propose Hier-IF, which reformulates the task as a “Tertiary Structure to Local Structure to Sequence” process. Hier-IF applies classifier-free guidance for controllable generation.

### Strengths
- This paper is well-written and easy to follow.
- Hier-IF introduces a trained mask generator to determine the generation order.

### Weaknesses
- The citation format is unclear.
- As Hier-IF is a diffusion-based model, several important diffusion baselines should be included, such as Grade-IF[1] and Bridge-IF[2].
- The performance improvement is marginal. Hier-IF achieves comparable performance with KW-Design. Both Hier-IF and KW-Design leverage pre-trained knowledge.
- As Hier-IF tokenizes the structure with ESM3 and conducts bidirectional reconstruction. What's the main difference between Hier-IF and ESM3?

[1] Graph denoising diffusion for inverse protein folding

[2] Bridge-if: Learning inverse protein folding with markov bridges

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Hier-IF, a novel model for protein inverse folding that explicitly models the hierarchical nature of protein structures. The method reformulates the task as a "Tertiary Structure (TS) to Local Structure (LS) to Sequence (Seq)" process, aiming to first generate sequences for local structures (helices, strands) and subsequently for connecting loops. The model uses an ESM-3-based structure tokenizer and a conditional masked diffusion language model. Key contributions include a bidirectional structure-sequence reconstruction loss to enforce consistency (using a forward-folding model) and a DSSP-based remasking strategy during sampling to guide the hierarchical generation. The model is trained using CATH classification labels as supervisory signals. Results on CATH4.2 and CATH4.3 benchmarks show competitive performance against existing baselines.

### Strengths
The "TS-LS-Seq" pipeline is a novel and biologically-motivated approach to inverse folding. Explicitly modeling the hierarchy of local structures before loops, rather than as a direct structure-to-sequence problem, is a sound and interesting contribution.

The introduction of a bidirectional structure-sequence reconstruction loss is a strong point. Using a forward-folding model to assess the structural plausibility of the generated sequence and adding it as a loss term is a good strategy for improving sequence-structure consistency.

The analysis of the generated proteins' secondary structure composition is thorough. The results suggest that Hier-IF produces a distribution of helices, strands, and coils that more-closely matches the natural CATH distribution compared to baselines like ProteinMPNN.

### Weaknesses
CATH classification labels are used in the model, which seems to make the model only applicable for CATH benchmark datasets.



Hier-IF seems do not surpass the KW-Design baseline, which is acceptable but overclaimed to “surpasses other baselines.”

The figures are not self-explained, especially figure 1. The double-ended arrow in figure 1 (B) is also confusing for me.

the abbr. DSSP is used multiple times before it is introduced, making the relevant part difficult to follow.

A typo: Bridge-IFZhu et al.

Missing references: SPDesign, BC-Design, MMDesign

### Questions
Could the authors explain how the model can be used in real applications, since it requires CATH classification labels? For example, evaluate the model on TS50 and TS500.

“To determine the generation order, we design a DSSP-based masking strategy, where the mask is directly generated by the model.”Could the authors explain what is the mask for? Does it mask the sequence, the input structure, or the secondary structure?

Could the authors showcase the quality of generated sequences changes in the process of iterative generation and remasking (e.g., a curve depicting how the recovery changes by interactions) , which can help demonstrate the necessity of this mechanism?

Interpretability is claimed to be a feature of the method in the abstract and introduction, but related analysis or explanation seems missing. Could the authors explain why the method has a good interpretability? Can we have a better understanding of inverse folding through this model?

Could the authors explain why a iterative algorithm means controllable generation? In my understanding, users cannot control the inverse folding prediction by adding constraints to this model.

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
5

### Summary
The paper introduces Hier-IF, a hierarchical inverse-folding model that generates protein sequences from structures by mimicking how proteins fold—first secondary structures (helices, strands), then assembling the loop to generate a global structure. It uses ESM-3-derived structure tokens, a masked-diffusion decoder with time-aware masking/remasking, and a bidirectional loss that enforces both sequence and structure reconstruction. Results on CATH benchmarks show improved sequence recovery, lower perplexity, and more realistic secondary-structure distributions than baselines like ProteinMPNN and ESM-IF.

### Strengths
## Biologically grounded design: 
- The hierarchical local-to-global process mirrors real protein folding. The explicit modeling of secondary-structure formation (through DSSP-guided masking) is novel and well-motivated.

## Training loss: 
- Consistency between structure and sequence by using Bidirectional Structure-Sequence Reconstruction Loss. Combining a masked diffusion language model with a time-aware remasking generator is a clever design that stabilizes generation and improves realism.

## Controllability: 
CATH conditioning allows steering generation by fold class. Visualization of intermediate steps (local-to-global assembly) gives intuitive interpretability of the generation process.

### Weaknesses
## Limited evaluation scope:
- The benchmarks focus on recovery/perplexity, but there’s no validation with physical or functional assays (e.g., in silico stability, Rosetta ΔΔG, or AlphaFold2 re-folding consistency).
- Structural reconstruction quality relies heavily on ESM-3 as a black-box teacher, so it’s unclear whether improvements stem from Hier-IF’s hierarchy or the pretrained backbone.

## Moderate methodological novelty:
- While the hierarchical idea is fresh, the individual components (masking schedules, diffusion decoding, remask refinement) are not novel. 

- The approach depends on a large pretrained ESM-3 encoder — more of a fine-tuning or re-framing than a fully new generative backbone. They authors also dont evaluate other structure token models like VQ-VAE from foldseek or from Implicit Structure Model (ISM). It's unclear if ESM3 encoder is frozen or fine-tuned. 


In table 1, Hier-IF is bold when a baseline demonstrates superior sequence recovery.

### Questions
Is the appendix missing? I dont see anything after the references. 


Are ESM-3 features frozen or jointly trained?
Could you run an ablation using other structure token tool other than ESM3? For example, ISM, FoldSeek, and others. 

Figure 2 and Figure 3 poorly demonstrate any improvements by Hier-IF. Is there a way for you to quantify the improvement and demonstrate a statistically significant improvement result in secondary structure sampling or distribution? 

Results are primarily on CATH recovery and perplexity metrics. Could you expand on whether Hier-IF improves downstream protein engineer or design tasks?

You only demonstrate C conditioning but what about CAT conditioning? Does your method enable generating backbones/sequences with the same topology but with increased diversity?

### Soundness
3

### Presentation
3

### Contribution
3
