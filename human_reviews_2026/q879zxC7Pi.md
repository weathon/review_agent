# Boosting In-Silicon Directed Evolution with Fine-Tuned Protein Language Model and Tree Search

- Avg Score: 2.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 2, 4

## Abstract
Protein evolution through amino acid sequence mutations is a cornerstone of life sciences. While current in-silicon directed evolution algorithms largely focus on designing heuristic search strategies, they overlook how to integrate the transformative protein language models, which encode rich evolutionary patterns, with reinforcement learning to learn to directly evolve proteins. To bridge this gap, we propose AlphaDE, a novel framework to optimize protein sequences by harnessing the innovative paradigms of large language models such as fine-tuning and test-time inference. First, AlphaDE fine-tunes pretrained protein language models using masked language modeling on homologous protein sequences to activate the evolutionary plausibility for the interested protein class. Second, AlphaDE introduces test-time inference based on Monte Carlo tree search, which effectively evolves proteins with evolutionary guidance from the fine-tuned protein language model. Extensive benchmark experiments show that AlphaDE remarkably outperforms previous state-of-the-art methods even with few-shot fine-tuning. A further case study demonstrates that AlphaDE supports condensing the protein sequence space of avGFP through computational evolution.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors claim to contribute:
- a new framework for doing in silico directed evolution which takes advantage of pre-trained protein language models as large evolutionary priors. 
- a test-time inference procedure using MCTS on top of a fine-tuned PLM to further optimize protein function.
- present improved in silico directed evolution of protein compared to a variety of design methods on a variety of assay-based protein predictors.

### Strengths
- Idea to bring test-time inference like MCTS from LLMs to protein language models seems novel (I’m not super sure though, not super well contextualized). 
- Results (Table 1) look substantially better than other methods on most of the problems. I’m wondering if it’s a fair comparison given this, some almost seem to good to be true. I don’t personally understand why I would expect such substantial gains over various baseline methods, and over zero-shot.
- Experiments compare to many methods for protein design, though some relevant ones are missing.
- Ablations and hyperparameters are documented pretty rigorously in the appendix.

### Weaknesses
- Poorly motivated — other work exists using pre-trained PLMs for protein design / directed evolution; these aren’t even cited (e.g., [1], [2], [3], [7]). Methods specifically for multi-round design, like [6], aren’t considered. 
- It's not really made clear why anything about AlphaDE is specific or particular to directed evolution (which is iterative) and not just general protein design. 
- The background section, which appears to serve as a related works section, lists some prior works that can be used to do in silico protein design. However, the authors don’t explain why their method might be a better idea than any of these. They only cite wanting to use PLMs for directed evolution, and don’t motivate this convincingly. Only the last sentence of the background explains the difference from previous works. It also doesn't explain why fine-tuning without using any supervised data is a desirable strategy, when labels are available for all of the test cases they consider in experiments (Table 1). Wouldn't one expect using labels to be helpful?
- Experiments don't compare to other methods using PLMs, like [1], [2], and [5].
- It looks like for their experiments (Table 1), they chose to use PEX instead of the stronger PEX+MuFacNet without explanation, even though both are shown in the plots where they take their baseline results directly from [4]. For avGFP, AAV, and TEM problems, PEX+MuFacNet performs better than PEX but this isn't reported.
- Fig. 2 doesn’t show how good uniform sampling is as baseline, or the base pre-trained model. Even after 1 fine-tuning epoch, AlphaDE looks pretty good, so one might expect the base model to also be pretty good. It's also unclear whether AlphaDE is generalizing substantially, or just sampling from sequences in the fine-tuning dataset--their fitness distribution should also be shown.
- It’s not surprising that given avGFP’s chromophore (Sec 4.4 / Fig. 5), a PLM can generate a protein very similar to the wild-type. These models have been trained on avGFP. I’d want a comparison to without fine-tuning, or without MCTS to be convinced that their AlphaDE pipeline contributes to this.

I would recommend rejection primarily because their method is not well motivated and situated within related works. Other work exists using pre-trained PLMs for protein design and directed evolution; these aren’t even cited (e.g., [1], [2], [3], [5], [7]). Methods specifically for multi-round design, like [6], aren’t considered. The authors don’t provide a convincing explanation as to why one should expect fine-tuning a PLM on sequences found from a simple homology search, and then using MCTS would outperform all these other design methods. A PLM-based prior could be easily integrated into many of the baseline methods (e.g., CbAS, CMA-ES, BO, probably others that I’m less aware of) for a more equal comparison, but this doesn’t seem to have been done. If the takeaway is primarily that using a PLM as a prior is useful, then the paper should have been written differently. There are no error bars for any of the tables although they claimed to be over replicates. Plots showing the mean value of sequences sampled by each method at each iteration would be more transparent but these are not shown.

More clearly contextualizing your method as motivated by previous works would help it be much more understandable. Currently, it reads as though you’ve presented a method that works quite well but the reasons aren’t clear, and a reader is forced to take your word for it and just try it out. Adding more intuition throughout the paper for why your method should be better than alternative approaches could help remedy this.


Citations: 

[1] Jiang, Kaiyi, et al. "Rapid in silico directed evolution by a protein language model with EVOLVEpro." _Science_ 387.6732 (2024): eadr6006.

[2] Yang, Jason, et al. "Steering Generative Models with Experimental Data for Protein Fitness Optimization." arXiv preprint arXiv:2505.15093 (2025).

[3] Tran, Thanh VT, and Truong Son Hy. "Protein design by directed evolution guided by large language models." IEEE Transactions on Evolutionary Computation (2024).

[4] Ren, Zhizhou, et al. "Proximal exploration for model-guided protein sequence design." International Conference on Machine Learning. PMLR, 2022.

[5] Nisonoff, Hunter, et al. "Unlocking guidance for discrete state-space diffusion and flow models." ICLR (2025).

[6] Yang, Jason, et al. "Active learning-assisted directed evolution." _Nature Communications_ 16.1 (2025): 714.

[7] Wang, Chenyu, et al. "Fine-tuning discrete diffusion models via reward optimization with applications to dna and protein design." ICLR (2025).

### Questions
What are the fitness values for the homologous sequences that the PLM is fine-tuned for each protein dataset? What part of your method exactly is yielding a substantial gain—it appears to be the fine-tuning from Table 2, but that doesn’t make sense given that you wrote that the homologous sequences are generated from the worst sequence in the dataset. If the homologous sequences fine-tuned on really are low-value, why is it so helpful to fine-tune on them? Would it similarly help to just fine-tune on the 100 worst sequences in the dataset?
How are the baseline methods initialized? Completely randomly (which may be unfair), or with samples from a zero-shot version of the PLM used for AlphaDE? It seems like even without fine-tuning, the PLM has a max-value greater than or similar to many of the baselines, which doesn’t really seem fair.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The work proposes to fine-tune a protein language model to guide the directed evolution for protein design. The work first fine-tune a protein language model with masked language model. Then the fine-tuned model is able to propose mutations. Then the model is used as a policy in an RL framework, where MCTS is performed to find protein sequences with a good fit. From the perspective of machine learning research, the methods in this work are well-known in the machine learning community.

### Strengths
The paper is well written and easy to understand. [Though I could not evaluate the experiment results because I am not from the area]

### Weaknesses
The most significant issue with the work is the lack of novelty. Most content of the work before the experiment section is known to the field. Specifically, section 3 describes the proposed method, but all the content is known to the community: the problem 3.1 is a well-known problem [1]. The masked language modeling in 3.2 is popularized by BERT [2] and is widely used in network training. The MCTS in 3.3 is popularized by AlphaGo. Therefore, I don't find the innovation in this work. 

I could not evaluate the significance of the experiment results as I am not from this area. Even if the results are much better than the state-of-the-art, ICLR might not be the proper venue for this work. 

[1] Yang et al. Machine-learning-guided directed evolution for protein engineering. Nature Methods. 2019.
[2] Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers). 2019.
[3] Silver, David, et al. "Mastering the game of Go with deep neural networks and tree search." nature 529.7587 (2016): 484-489.

### Questions
I hope the author could better justify the contribution of leveraging "protein language models into directed evolution for effective exploration".

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
AlphaDE finetunes PLMs using MLM on homologus protein sequences, then uses this as a policy network to do MCTS towards high-fitness mutations. A value network is also trained online to accelerate rollouts. Evaluation is done on 8 tasks, against other baselines that follow the "fitness landscape exploration" approach to DE tasks.

### Strengths
* Low-N fitness prediction is important to enable.
* Formalizing how RL approaches can be used in PLMs is timely and can lead to productive future works.
* On the benchmarks explored, performance looks favorable, e.g. on the TEM task.

### Weaknesses
* Though this idea of RL post-training for PLMs holds promise, given the current state of the LLM field, the execution becomes quite important, and I think the paper can do better on this in terms of rigor and following through on failure cases. I personally don't think the idea itself is super novel, and I think what makes a paper like this shine would be to really help readers get intuition on how RL post-training will differ for PLMs. Even in terms of base execution, there are some decisions that don't entirely make sense. For example: why use ESM-1b oracle rather than a more recent model? I get there's a desire for consistency with baselines, but I think it's more important to execute well, and reimplement the baselines if needed. 
* Finetuning on homologous sequences is not a new idea; it’s been done since earlier ML for protein design works (Alley et al., 2020, Biswas et al., 2021) as well as recent works (Gordon et al., 2025). This limits the novelty of the work and the completeness of the discussion.
* Computational costs is a lot higher. Appendix L reports that AlphaDE takes 4.74 hours, vs 0.69 hours for EvoPlay.
* The simulated landscapes are very toy settings, limiting its applicability to the real world. This is inherent to fitness landscape exploration type works, and not unique to this paper, but nonetheless limits the ultimate impact.
* Nit: presentation - avoid “impressively” and qualifying words in scientific writing, in line 58.


Alley et al, 2020: https://pubmed.ncbi.nlm.nih.gov/33828272/
Biswas et al., 2021: https://pmc.ncbi.nlm.nih.gov/articles/PMC7067682/
Gordon et al., 2025: https://www.biorxiv.org/content/10.1101/2024.10.03.616542v1

### Questions
* IIUC from the Appendix C2, the process stops when the current mutated sequence fitness is lower than the wildtype. Is that overly greedy? If you take out this termination requirement, how often would you see the fitness climb back up? Given how rugged protein landscapes are, this is type of investigation might be interesting.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes AlphaDE, an in-silico directed evolution framework that integrates a fine-tuned protein language model (PLM) with Monte Carlo Tree Search (MCTS) for protein sequence optimization. The PLM is fine-tuned on homologous protein sequences to learn domain-specific evolutionary constraints, while the MCTS performs iterative exploration of the sequence space to identify beneficial mutations guided by the model’s probabilities. Experiments on eight benchmark protein engineering tasks demonstrate that AlphaDE achieves higher fitness improvements than several baselines, including TreeNeuralTS, TreeNeuralUCB, PEX, and AdaLead, even under limited fine-tuning data.

### Strengths
- The paper is well written and clearly structured, providing both background and algorithmic details.

- Combining fine-tuned PLMs with MCTS is conceptually sound and leverages recent progress in both protein modeling and search algorithms.

- Strong empirical evaluation on multiple protein datasets with reproducible settings.

- Demonstrates few-shot fine-tuning results, suggesting potential data efficiency.

### Weaknesses
- Limited novelty: The central idea closely overlaps with existing works on ML-guided directed evolution using protein LMs, particularly "Protein Design by Directed Evolution Guided by Large Language Models" (IEEE Transactions on Evolutionary Computation) [1], which already proposed LLM-based mutation guidance; and "LatentDE: Latent-based Directed Evolution for Protein Sequence Design" (Machine Learning: Science and Technology) [2], which introduced latent-space optimization for protein design using pretrained models. The proposed fine-tuning and search mechanisms appear incremental rather than fundamentally new.

- The integration of tree search does not significantly advance beyond prior reinforcement-learning or latent-search frameworks (e.g., LatentDE).

- Lacks biological validation or wet-lab evidence to confirm improved protein functionality.

- No clear ablation to isolate contributions of PLM fine-tuning vs. MCTS itself.

- Evaluation largely depends on oracle models; real-world applicability remains uncertain.

*** References:

[1] Trong Thanh Tran and Truong-Son Hy, Protein Design by Directed Evolution Guided by Large Language Models, IEEE Transactions on Evolutionary Computation (Q1, Impact Factor = 14.3), vol. 29, no. 2, pp. 418-428, April 2025, DOI 10.1109/TEVC.2024.3439690.
URL: https://ieeexplore.ieee.org/document/10628050

[2] Thanh V. T. Tran, Nhat Khang Ngo, Viet Thanh Duy Nguyen, and Truong-Son Hy, LatentDE: Latent-based Directed Evolution for Protein Sequence Design, Machine Learning: Science and Technology (Q1, Impact Factor = 6.3), Volume 6, Number 1, DOI 10.1088/2632-2153/adc2e2.
URL: https://iopscience.iop.org/article/10.1088/2632-2153/adc2e2/pdf

### Questions
How does the proposed AlphaDE fundamentally differ in principle or expected outcome from prior LLM-guided directed evolution frameworks such as "Protein Design by Directed Evolution Guided by Large Language Models" [1] and "LatentDE" [2]? Specifically, could you clarify what new insights or capabilities are gained by combining fine-tuned PLMs with Monte Carlo Tree Search beyond improved sampling efficiency?

### Soundness
3

### Presentation
3

### Contribution
2
