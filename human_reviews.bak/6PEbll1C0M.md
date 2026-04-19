# Generating All-Atom Protein Structure from Sequence-Only Training Data

- Decision: Reject
- Scores: 6, 5, 6, 5

## Abstract
Using generative models for protein design is gaining interest for their potential scientific impact. However, biological processes are mediated by many modalities, and simultaneous generating multiple biological modalities is a continued challenge. We propose **PLAID (Protein Latent Induced Diffusion)**, whereby multimodal biological generation is achieved by learning and sampling from the *latent space of a predictor* from a more abundant data modality (e.g. sequence) to a less abundant data modality (e.g. crystallized structure). Specifically, we examine the *all-atom* structure generation setting, which requires producing both the 3D structure and 1D sequence, to specify how to place sidechain atoms that are critcial to function. Crucially, since PLAID **only requires sequence inputs to obtain the latent representation during training**, it allows us to use sequence databases when training the generative model, thus augmenting the sampleable data distribution by $10^2×$ to $10^4×$ compared to experimental structure databases. Using sequence-only training further unlocks more annotations that can be used to conditioning model generation. As a demonstration, we use two conditioning variables: 2219 function keywords from Gene Ontology, and 3617 organisms across the tree of life. Despite not receiving structure inputs during training, model generations nonetheless exhibit strong performance on structure quality, diversity, novelty, and cross-modal consistency metrics. Analysis of function-conditioned samples show that generated structures preserve non-adjacent catalytic residues at active sites, and learn the hydrophobicity pattern of transmembrane proteins, while exhibiting overall sequence diversity. Model weights and code are publicly accessible at `[redacted]`.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The manuscript proposed a latent-diffusion framework that can simultaneously generate protein sequence all-atom structure. The framework is called PLAID (Protein Latent Induced Diffusion). The framework leverages pre-trained protein representation and folding model (ESMFold) that generates latent representations containing both sequence and structure information of proteins. It also uses an autoencoder to further compress the latent representation for more efficient diffusion training. Such design makes training the model on larger sequence-only dataset possible. The framework also allows conditional generation.

### Strengths
1. The idea of compressing the learned representation from ESMFold and perform diffusion training in a more efficient manner is valuable, especially when generating larger protein is desired. 
2. I generally agree with the authors that the representation from ESMFold contains both sequence and structure information, giving this work a rather solid foundation. The design in this work allows the model to be trained on larger sequence-only dataset is important.
3. Generating sequence and all-atom structure simultaneously is a relatively less-explored area. This paper is an interesting experiment.

### Weaknesses
1. The CHEAP autoencoder used in this work does not seem like a variational autoencoder. Therefore, the latent space may not be very smooth and can make diffusion model training difficult. I did not see an potential solution to that in the framework proposed by authors.
2. One advantage claimed by authors is that the framework allows the model to be trained on much larger sequence-only dataset. However, the unconditional generation performance of the model after training on Pfam is somewhat lackluster compared with MultiFlow. I also believe that an experiment showing how the performance of the proposed model scale with more data can be insightful.
3. The latent diffusion choice of the framework can be really beneficial in terms of speed when generating very long sequences (>600 aa). The authors did not show the performance of the model when generating very long proteins.

### Questions
1. Figure 2C is only a schematic of how CHEAP autoencoder is used in the framework, but not showing "without the normalization and compression post-processing steps in CHEAP, noise added in the latent space does not affect sequence and structure until the final timesteps in forward diffusion" as described in line 212-213. Is there another figure to visualize the effect of CHEAP autoencoder?
2. Conditional generation: how are conditions added to the model? Can authors elaborate or visualize?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces PLAID (Protein Latent Induced Diffusion), a diffusion model that can simultaneously generate protein sequence and all-atom structure, while only requiring sequence inputs during training. PLAID leverages the latent space of an existing protein structure prediction model, ESMFold, to capture the joint distribution of sequence and structure. By defining the training data distribution based on sequence databases rather than structural databases, PLAID can access a much larger and more diverse set of protein data, increasing the available annotations and enabling controllable generation along axes like protein function and organism of origin. PLAID avoids the need for alternating between sequence and structure generation steps, and can directly sample from the joint distribution of sequence and all-atom structure.

### Strengths
- Leveraging the latent space of existing structure prediction models for protein structure (all-atom) and sequence co-design is a refreshing and innovative idea.
- The paper presents a relatively clear discussion of the main components of the methodology.

### Weaknesses
- My main concern with the paper lies in its underwhelming performance. While I generally believe the approach is feasible, the current experimental results suggest that more effort is needed to thoroughly explore effective strategies for co-design within the latent space of structural prediction models.
- The lack of support for motif scaffolding raises concerns about the controllability of the current model. Although function and organism terms can be used as conditioning, this form of conditioning may still fall short in achieving precise control.
- The paper lacks a comprehensive discussion of related work. For instance, as a method for protein design utilizing structure prediction models, it overlooks hallucination models [1] that also manipulate structure prediction models in similar ways.
- Several important details critical for understanding the paper are not explained:
   - There is no explanation provided on how the model integrates function and organism conditioning. The related model design strategies require further clarification.
   - The design of the sequence decoder $\phi^{-1}_{\text{ESM}}$ also requires further elaboration.
   - The paper does not provide an explanation of how FIDs, originally designed as a metric for image generation, is adapted for application in protein design.
- There are numerous errors in the citations of images and text within the paper:
   - The reference to Figure 2 between lines 188 and 216 does not align with the textual description and needs to be corrected.
   - The description of the probabilistic decomposition for different methods in line 173 contains errors.
   - The yellow and green dots in Figure 4D lack a legend explanation.

[1] Anishchenko I, Pellock S J, Chidyausiku T M, et al. De novo protein design by deep network hallucination[J]. Nature, 2021, 600(7889): 547-552.

### Questions
- If we focus less on the structures obtained from experiments and instead accept those predicted by models, the available structural data is still substantial and does not exhibit the vast gap claimed in the paper. In fact, many current studies already incorporate predicted structures in their training processes. Therefore, the paper's statements may not accurately reflect the current reality.
- Does designing based on structure prediction models introduce an unnatural bias toward certain protein structures? For example, AF2 has been found to predict overly "clean" structures, often lacking the unstructured regions that are crucial for capturing protein dynamics.
- Is there a more detailed explanation or experimental validation available for using the CHEAP module?
- Why is a design result that finds more homologous sequences considered more novel in the context of defining Hit metrics?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper provides a way to generate all atom protein structure using only sequence data during the training without requiring intermediate structural inputs.

### Strengths
The training does not require sequence data. By training on sequence data alone, PLAID can scale across larger datasets compared to models constrained by experimentally resolved structures.

### Weaknesses
1. The model’s reliance on ESMFold’s latent space and pre-trained weights, while advantageous, could be limiting, and main factor contributing the limited performance. 
2) The paper can benefit from improved writing, I think some parts of the paper need more explanation.

### Questions
1) The statement in line 178, p(x) = \phi(s) = \phi(\omega), I am not sure if this is correct, p(x) supposed to be the joint while the other two is individual distributions. 
2) how the ESMFold first component that maps the sequence to the latent captures evolutionary prior? and what is the intuition behind that prior?
3) The equation at the end of the line 187, isn't the structure module takes actually x and map it to the strucuture \omega, the equation written as if it takes the structure as input?
4) line 195, f_ESM is not defined. I am not sure if I understood the part the sequence decoder need to be separately trained? The block B ( in figure 2) is used off the shelf or it is also trained?
5) the line 199 ( the inference ... is shown in Figure 2B, is it typo? the inference is shown in D in figure 2?
6) if I summarize the training of this model, one need a pre-train ESM2 model and CHEAP, then during training, the sequence of protein go through ESM2, mapped to x, then (not sure how the x_norm is obtained), x_norm goes through CHEAP encoder, gives us more compressed rep x_0 and there we train a diffusion model , So only trainable params are the diffusion model in the space of x_0,  and inference time,  we use the frozen CHEAP decoder and ESM structure and sequence decoder, is that right what I summarize?
7) Any intuition behind the why the compressed latent first dimension set L/2?
8) I am wondering what happens if one apply the diffusion model directly on the latent of the ESM2 model, I think the line between 211 and 215 tried to answer that but not sure if I understood.
9) The figure 4D has beed resolution, and  color of the round dots are a bit misleading.
10) what is the red vertical line in the figure 4 B represents?
11) I am wondering what happens if one trains the full Model in figure 2C instead of freezing some parts?

### Soundness
2

### Presentation
2

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
The authors propose a new method for generating all atom protein structures from sequence-only training data which they call PLAID (Protein Latent Induced Diffusion). For this, they leverage the pretrained ESMFold latent space as well as the compressed CHEAP embeddings and train a Diffusion Transformer over these CHEAP embeddings and also create a sequence decoder from the ESMFold latent space. With that, at inference time they generate first a CHEAP embedding, decode that to an ESMFold embedding and then use the ESMFold structure decoder as well as the newly trained sequence decoder to recover structure and sequence. Via classifier-free guidance they allow conditional generation with GO ID and organism labels and compare the performance of their method to other co-generation methods and to reference datasets from the PDB.

### Strengths
1. **Novelty of approach**: While latent diffusion models have been described before, this is the first work that describes the usage of a latent diffusion model over protein language model embeddings to generate both sequences and all-atom structures. This makes dealing with varying lengths of side chains easier than in previous all-atom generation methods like Protpardelle and allows leveraging capabilities from pre-trained models like ESMFold. 
2. **Leveraging compute and data scale**: The compressed latent space as well as the standard Diffusion Transformer architecture allow the model to train and generate efficiently. The fact that only sequence data is used for the actual diffusion model training also allows scaling to larger datasets as well as leveraging potentially more diverse training data. It also allows the integration of sequence labels as conditioning information as presented via GO IDs.

### Weaknesses
1. **Designability/Diversity Performance**: The model underperforms quite strongly on Cross-Modal consistency and diversity compared to Multiflow. Especially the Co-Designability number of 40% shows that the model is not performing well on the intended task of protein structure and sequence generation compared to work that is out there such as Multiflow.
2. **Novelty Performance**: It is not really clear why the authors bold their novelty Hit% number, implying that their model performs best on novelty. Multiflow outperforms on 1-TM score (but its number is not bolded), and if I understand correctly Hit% numbers indicate how often homologs can be found in a sequence database; a higher value here does not imply that the model is producing more novel samples, if anything it is the opposite. Also in the SeqID% column, it is not clear why the 83.6% number of the natural reference set is bolded since the authors claim a lower number on these metrics is better and the natural reference set is just a reference set and not a baseline model. It would be helpful if the authors review the table again and include consistent bolding to make the information in that table more accessible to the reader. It would also be beneficial to describe in more detail why 1-TM score and Hit% are useful metrics and how higher/lower values reflect the performance of the evaluated models.
3. **Missing control experiment for claims**: The PLAID model is only evaluated via cross-modal consistency, i.e. refolding accuracy using the sequence generated from the model. However, past publications have shown that in many cases the sequence generated from the model yields lower self-consistency values than a sequence generated from the generated structure via ProteinMPNN. To truly evaluate whether their model has cross-modal consistency, the authors should compare to this baseline and show that their consistency is higher than when they use ProteinMPNN (see the MultiFlow paper for experiments that show these kind of results).
4. **Naturalness of sequences**: The authors claim that their model can generate more natural sequences than other models. First of all, it is not clear why the molecular weights are a lot higher for all methods compared to the natural reference set? I would assume that to compare fairly one should sample proteins with similar lengths compared to the reference set, but this does not seem to be the case here. And while the model has values that are more similar to natural proteins in terms of isoelectric point, gravy and charges (charge at pH does not note which pH is used for the calculation here?), the samples seem to be a lot more instable compared to MultiFlow's and the baseline samples, which seems one of the most important properties in practical applications.
5. **Mistakes/missing figures**: There seem to be several typos/mistakes in the paper that makes following the flow difficult to understand, for example 
   1. L173: Protpardelle and ProteinGenerator have the same factorization although the one for ProteinGenerator should be switched around.
   2. L191: Diffusion training is depicted in Fig 2C and not 2A.
   3. L199: All-atom sampling is depicted in Fig 2D and not 2B.
   4. L212: "Figure 2C shows that without the normalization and compression post-processing steps in CHEAP, noise added in the latent space does not affect sequence and structure until the final timesteps in forward diffusion". Figure 2C does not show this, it is just a illustrative graphic of the training process. The mentioned content regarding noise not affecting sequence and structure until final timesteps is not mentioned anywhere else and seems to be missing from the paper. It would be helpful if the authors conduct a thorough review of all figure references and ensure that all mentioned content is actually included in the paper. This would help improve the overall clarity and coherence of the manuscript.
   5. L214: "(SNR and log-SNR curves shown below)". There are no SNR and log-SNR curves to be found in the paper.
6. **GO term FID score**: The authors claim that their model can generate realistic proteins with a given GO ID and measure this by an FID score. However, FID score (Frechet Inception Distance) is a metric for judging image quality for conditional image **generation** tasks where Inception refers to the model used to embed the images and compare the embeddings against the reference set. Since using the Inception model in this context does not make sense for proteins, the FID score as presented here does not make much sense. If a different model is used in this paper for embedding and comparing these embeddings against a reference set, the authors should mention the model that was used there, the reference set that was used and show validation experiments to demonstrate that their proposed new metric (which would not be an FID score anymore) has any relevance/utility in the protein domain. It would be helpful if the authors can provide a detailed explanation of how they adapted the FID score for proteins, including specifics on the embedding model used and the reference set. Additionally, validation experiments or justification for using this metric in the protein domain would strengthen the evaluation approach.

### Questions
1. In Table 1, the authors imply that their model can perform two GO term and organism conditioning, while the other methods according to the table cannot do conditioning, even though for example RFDiffusion can do a lot of conditioning constraints that PLAID cannot (symmetric oligmers, binders, ...) which for practical applications have arguable more relevance. Did the authors explore these other conditioning approaches and could the table be updated accordingly?
2. Since the model is trained on protein domains in Pfam only, does this limit the generation capabilities to single-domain proteins?
3. In newer work like ESM3, the language model is directly used for structure generation via explicit structure token embeddings. How does this compare to the approach proposed in the paper and what are potential advantages and disadvantages?
4. It is stressed in the paper that the model can sample more natural sequences than other models. Besides questions about this claim itself (see weaknesses section), should naturaleness in terms of aromaticity etc be a target for a "good" protein generative model? For many applications these models are used for non-natural properties like higher melting temperatures etc might be very useful.
5. The authors show case studies for conditional generation with GO IDs. But while they show low sequence identiity, is there any way of judging how similar the global structure is, not just the sequence? In addition, how well do these case studies work across the board beyond the two examples shown in the paper?
6. In protein generation it is often useful to employ structure constraints such as motifs, specific secondary structure or binding interfaces for generating samples. Is this possible in a latent framework such as the one proposed here?

### Soundness
2

### Presentation
2

### Contribution
3
