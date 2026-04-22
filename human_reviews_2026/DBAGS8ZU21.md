# Prot2Token: A Unified Framework for Protein Modeling via Next-Token Prediction

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
The diverse nature of protein prediction tasks has traditionally necessitated specialized models, hindering the development of broadly applicable and computationally efficient Protein Language Models (PLMs). In this work, we introduce Prot2Token, a unified framework that overcomes these challenges by converting a wide spectrum of protein-related predictions—from sequence-level properties and residue-specific attributes to complex inter-protein interactions—into a standardized next-token prediction format. At its core, Prot2Token employs an autoregressive decoder, conditioned on embeddings from pre-trained protein encoders and guided by learnable \texttt{task tokens}, to perform diverse predictions. This architecture uniquely facilitates multi-task learning, enabling general-purpose decoders to generalize across five distinct categories. We present extensive experimental validation across a variety of benchmarks, demonstrating Prot2Token's predictive power in different types of protein-prediction tasks. In 3D structure prediction, Prot2Token delivers substantial speedups (up to $\sim$1000$\times$ faster than AlphaFold2 with MSA on the same hardware) while, across other numerous tasks, matching or surpassing specialized methods. Beyond that, we introduce an auxiliary self-supervised decoder pre-training approach to improve spatially sensitive task performance. Prot2Token thus offers a step towards standardizing biological prediction into a generative interface, promising to accelerate biological discovery and the development of novel therapeutics.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors present Prot2Token, an interesting approach to multitask learning for protein-related tasks. This is a difficult problem: the variety of tasks spans distinct input and output subspaces, so learning how to bring it all together is tricky. The authors went for a maximalist approach where every task becomes next-token prediction. The alternative would have been more custom prediction heads fed from shared encoder representations. I'm not convinced the next-token approach works well, and the evaluations (including some missing ablations) underscore this. The framework shows promise as a unified scaffold, but the "everything is auto-regressive token prediction" philosophy seems to create more problems than it solves.

Let me start with the ProteinGym evaluation, which is a red flag. I don't buy that they're outperforming a well-tested benchmark by this magnitude (Spearman $\rho=0.93$ vs. prior best $\sim 0.61$). This makes me think there's substantial data leakage happening. Could it be that the correlations aren't per-assay but are instead pooled? The latter would artificially inflate results through inter-assay signal.

This is compounded by the odd setup for ligand binding prediction. Most DTI tasks aren't designed to treat each ligand as a separate classification task. For metal ions, I could imagine that, but for most SMILES strings, the formulation should handle arbitrary chemical compounds as input. Their approach uses fixed task tokens for 41 specific ligands (Section A.4.4), making it impossible to generalize to new compounds.

These issues feed into my overall sense that cross-task learning hasn't been evaluated properly. The whole point is showing across-the-board benefits. While the authors found episodic benefits (training on tasks B, C, D helps task A), the right experiment would systematically evaluate: what happens if I train on everything except this task? They show cherry-picked combinations (Tables 2, 7) but no comprehensive task influence matrix or negative transfer analysis.

Even if these evaluations were fixed, my fundamental concerns remain about how expressive next-token prediction can be for diverse tasks. Binding site prediction via sorted indices is fragile; I saw no analysis where binding sites were provided in reordered settings. The authors acknowledge that tokenizing continuous values as ASCII characters is a choice and defend it, but it's still deeply unsatisfying-- they don't seem to have tried even a right-to-left tokenization? The decoder pretraining is clever but feels like a band-aid on a fundamentally problematic design.

Additional Concerns:

- Inconsistent architectural choices: Different tasks unfreeze different numbers of encoder layers (4, 6, or 8 blocks) without justification or ablation studies examining this crucial hyperparameter.

- Structure prediction quality gap: TM-score of $0.54$ vs. AF2's $0.88$ is presented as a speed win, but the model isn't usable for practical applications. If the VQ-VAE is bottleneck, maybe look at the tokenization of newer PLMs like ESM-3, which might offer better results?

- Missing principled baselines: Many of the baselines are simpler, less-strong ones rather than the strongest task-specific baselines around.

### Strengths
see above

### Weaknesses
see above

### Questions
see above

### Soundness
2

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
4

### Summary
This paper introduces Prot2Token framework which tries to handle various protein tasks using a unified decoder, including classification, regression, binding-site prediction, and even sequence-to-structure generation.
The key idea is to turn every task into a next-token prediction problem, all targets are serialized into tokens, and each task is triggered by a learned “task token”.
The framework uses ESM2 as the protein encoder, optionally fused with a SMILES encoder.
Experiments across benchmarks like DeepLoc, ProteinGym, and CAMEO show reasonable performance and strong throughput.

### Strengths
1. this paper addresses a challenging issue in the downstream application of PLM: additional head across each tasks and architectures
2. this paper leverages a straightforward idea that turns everything into a next-token prediction framework training
3. the experiments spans multiple benchmarks including both cls/reg-level and sequence-level
4. paper writing is clear about its goals and experimental setup.

### Weaknesses
My concern mainly falls in the conceptual depth of "unified decoding" proposed in this paper, with several minor issues.
1. The "unified decoding" claimed in this paper mostly coms from designing a token vocabulary for task tokens. Functionally, this isn’t very different from instruction- or prefix-based tuning, it’s essentially another form of prompt learning in the view of learnable tokens. While the paper criticizes prior works for relying on prompt engineering, it ends up doing the same thing in a more structured way. The unification happens at the tokenization level, not at a modeling or representation level.
2. Some of the comparison models don’t use ESM-2 as their encoder. Since the proposed method builds on ESM-2, including ESM-2-based baselines would make the comparison fairer and the paper more convincing.
3. The description of the self-supervised warm-up and multi-task training is a bit confusing. It’s not clear whether the warm-up applies only to the binding-site prediction task, or whether all tasks benefit from it. Likewise, the paper doesn’t clearly state if the pretraining and multi-task learning are done in one continuous run or in separate stages.

### Questions
All my concerns about this paper are listed in "Weaknesses" section, please refer to the weakness part for rebuttal/discussion.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Prot2Token, a unified framework that reformulates various protein-related tasks—such as classification, regression, and structure prediction—into a next-token prediction problem. The authors combine pre-trained protein encoders (mainly ESM2) with an autoregressive decoder conditioned on task tokens. The idea is to create a single model capable of handling diverse prediction formats through a shared tokenization and decoding process.

While the paper claims to present a unifying paradigm similar to GPT-style modeling in NLP, the method mainly reuses existing building blocks (ESM encoder, cross-attention fusion, VQ-VAE structure tokenizer) without introducing substantial architectural or algorithmic novelty.

### Strengths
- A good motivation: to build a unified model for every different protein tasks

- Reasonable engineering effort to connect multiple existing components: for example, they unify different downstream tasks with one model.

- Consistent writing: the writing is clear

### Weaknesses
- **Claim is big, but paper is unable to support the claim**: the biggest issue is that, this paper claims that they try to advance a huge step in the protein field, by unifying different tasks into one model, with some prompt tokens. However, the method part significantly lacks  novelty. It just stated what they have used for model module building. Just stacking together without any deeper insights (either theoretical or empirical). Even worse, the performance tables show very limited baselines (some definitely out-of-dated). For example, table 3 shows the performance in PEER's benchmarking paper from 2023, which is already beaten by a lot of methods later on. 

- **Illustration figures look like workshop quality instead of conference quality**: many specific details without high-level deep insights.

### Questions
Please see the weakness. Sincerely suggest to incorporate much more comprehensive benchmarking results (baselines, metrics) to support your strong and big claim, to make it empirically solid, given that this is more like an application driven paper.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents Prot2Token, a method for unifying multiple downstream classification tasks using an autoregressive decoder. A pre-trained protein language model (PLM) is used to generate representations of an input protein sequence, which is then fused with ligand representations from a small molecule foundation model, if applicable, and fed to an autoregressive decoder that generates tokenized outputs specific to a given task. A task token is also learned and is fed as a precursor to the decoder to make it aware of the task under study. Numerical results demonstrate that Prot2Token outperforms other baselines across various benchmarks.

### Strengths
- Representing diverse downstream classification and regression tasks for protein sequences using a universal tokenization scheme is, in my opinion, interesting and novel.
- Performance improvements over prior methods is significant in some benchmarks.

### Weaknesses
1. To me, the very shortcoming that the paper is trying to address is its main weakness. As the authors allude to, a universal model that can support any given downstream task is computationally prohibitive. Therefore, the method has been limited to a few downstream tasks only.
2. Following (1), it seems that different versions of Prot2Token were trained on different (combinations of) downstream tasks (Prot2Token-A/B/C/D). Is my understanding correct? If so, it defeats the purpose of having a universal model. If not, what do the auxiliary tasks in Tables 2 and 7 imply, and how are they selected?
3. Comparisons with baselines are slightly unconvincing to me. At the very least, I would suggest including the results of a frozen ESM-2 model, combined with a separate linear probe for every task.
4. Following (1) and (3), the main novelty of the approach is the universal decoding engine, which tries to overcome the alternative, which is separate, heterogeneous downstream prediction heads specific to each task. I wonder if having multi-layer downstream prediction heads that share the first few layers and differ only in their last layer would also reap the same benefits as multi-task learning, as seen in your method.

### Questions
- When including auxiliary tasks (e.g., in Tables 2 and 7), does the performance on those auxiliary tasks improve, too? Or is the improvement only observed in the target task?
- Could you please provide more details on the task tokens and whether they could generalize to a task unseen during training (e.g., in a meta-learning way)? What is $m$ on lines 208-210?

### Soundness
2

### Presentation
3

### Contribution
2
