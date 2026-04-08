## Human Reviewer 1

### Summary
This paper presents a novel method for model merging where the size of the models do not need to match. In order to do so, the authors use variational auto-encoders, optimal transport, linear interpolation,  and projections down to a merged weight space. The results beat strong baselines (including very recently published methods) on a wide variety of tasks. I’ll keep this reviews short because, overall, I would be very happy to see this published in the conference.

### Strengths
Though the paper is dense and covers a lot of complex topics, it is well-written and easy to follow. In addition, things are presented clearly, such as the caption of Figure 1.

There are a lot of experimental results with very strong baselines that they beat.

It is an interesting idea intellectually and I would have liked to have seen the paper published even if the results had not beaten the baselines – but they did.

### Weaknesses
Perhaps I missed it, but I think a bit more background on OT could be useful for the presentation to the reader. The paragraph at the end of section 3 could be expanded a bit more. I ended up needing to look at one of the cited papers. However, most of the other parts of the paper explained complex topics very well.

### Questions
The method seems like it could be a bit computationally expensive - and this is mentioned in the limitations. How expensive is it exactly? I don't have a particular way I'd like to see this question answered, but whatever makes sense. For instance, maybe time on a GPU? What sort of GPU (which is definitely dependent on the models)? Maybe percent of compute needed compared to doing something from scratch? Or comparison to another method (i.e. AIM)?

### Soundness
4

### Presentation
4

### Contribution
4

### Rating
8

### Confidence
3

---

## Human Reviewer 2

### Summary
LS-Merge is a method that uses a Transformer-based VAE to compress chunks of weight matrices from various LLMs, enabling heterogenous merges where models have different architectures. THhe VAE encodes weight chunks into lower dimensional representations, aligns them in latent space, interpolates them, and then decodes them back into model space. This method can improve performance of single models, improve merges of multiple LoRA adapters, and perform competitively with direct weight space merging methods.

### Strengths
1. By applying the LS-Merge VAE on pre-trained Gemma models, the output model can improve on benchmarks like GSM8K and MMLU without additional gradient steps on the model weights, just on encoding and decoding the weights with the pretrained VAE. This represents a unique and nice result of improving singular model.  
2. The proposed method of using weights as data to learn via a VAE is novel, interesting, and a potential direction to expand upon in future work, especially given interesting results like point 1. Also, it appears that only a few models worth (2 Gemma models) is sufficient to train on to achieve useful results. 
3. This method can achieve heterogenous merging, mostly on intra-family based merges.

### Weaknesses
1. While the results are impressive, the description of the method is not entirely clear at times and detracts from the contribution of the paper. For example, it is not clear how heterogenous rescaling occurs according to the description, and it is not when the OT based alignment occurs in the latent space training. The contribution of the paper seems solid, but its presentation seems rushed and the paper does not seem reproducible or easily understandable in its current state. Another example is the statement that the evaluation of cross family evaluation is performed using lm-eval due to issues with the llama model when using previous evaluation code. What does this mean here? 
2. Despite good results, this work lacks some analysis of what is learned by the VAE model, as well as some ablations of key choices. It does not seem clear why this method works, or what about it makes it work. What is the latent size used in this work? And what is the chunked size? And how were these values set? 

This paper is quite interesting but unfortunately its presentation and polish is very lacking, which brings into question the correct execution of this work. I think this paper could be impactful and a nice contribution, but in its current form I cannot recommend accepting it as it seems only partially finished.

### Questions
1. In section 3.1, is the v_proj included in this analysis? It is missing from the description of the moment analysis of the key LLM weights. 
2. What exactly is a layer matrix in section 3.1 line 150 and section 3.2 line 204? Is it a single weight like q_proj or up_proj or is it the entire Transformer layer?
3. Is the embedding from line 206 part of the transformer encoder? Or is it separate?
4. What is the exact operation for the rescaling procedure described for heterogenous mapping? It is not clear how the value r is used in this mapping. 
5. What is the model used as base in Table 5? And which model is weighted 0.1 in the mixture?

Typos
1. Line 192, porjection
2. Line 139 up_porj, down_porj
3. Line 233 artihmetic 
4. What is the "fixed of 2" mean on line 321?
5. Line 796 is missing a reference to a figure in Latex. 
6. Line 403 familly
7. Line 404 "is perform"

### Soundness
3

### Presentation
1

### Contribution
3

### Rating
4

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper propose a novel method to replace fragile weight-space merging with LSMerge: a latent space merging technique. It is an encode–align–decode pipeline that operates on a latent space of model weights learned by a transformer VAE. Specifically: the method 
(1) encodes parameters into latents via layer-aware chunking, 
(2) aligns heterogeneous models’ latent distributions (depth/width mismatches) using Optimal Transport (OT), and 
(3) decodes an interpolated latent back to weights. 
Evaluations demonstrates performance gain in (i) self-merging of a single model’s latents, (ii) LoRA expert fusion vs. model-soup/SLERP baselines, and (iii) heterogeneous merges (within-family size changes and cross-family).

### Strengths
1. Principled formulation: The core idea of encoding model weight into a shared latent manifold and aligning distributions via OT is right on target to resolve the pain point of current merging methods in shape matching requirements. This paper gives a more grounded understanding of how to merge models by aligning the model's representation geometry.

2. Comprehensive experiment setting: The same recipe can be used for many use cases, including self-merging, LoRA-expert fusion, and cross-family merging. This shows that this method is a more fundamental framework rather than a "trick".

### Weaknesses
1. Lack of analysis on reconstruction error: The main component of the merging system is a VAE which introduces lossy compression during reconstruction, yet there isn't an analysis of how that bottleneck correlates with downstream task accuracy. This matters as it could confound where the gains are from. I'm curious if the author can ablate on compression level vs. KL-weights. 

2. Lack of understanding of successful merging condition: The experiments are diverse in terms of merging various types of models (self, LoRA etc.) but it is unclear what is the boundary condition of successful latent space merging? I'm curious to see if the authors have found any failure case and if so, have done any analysis of what's the key difference between merges that fails/succeeds.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
4

### Rating
8

### Confidence
3