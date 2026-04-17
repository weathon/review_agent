# SHE-LoRA: Selective Homomorphic Encryption for Federated Tuning with Heterogeneous LoRA

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Federated fine-tuning is critical for improving the performance of large language models (LLMs) in handling domain-specific tasks while keeping training data decentralized and private.
However, prior work has shown that clients' private data can actually be recovered via gradient inversion attacks.
Existing privacy preservation techniques against such attacks typically entail performance degradation and high costs, making them ill-suited for clients with heterogeneous data distributions and device capabilities.
In this paper, we propose SHE-LoRA, which integrates selective homomorphic encryption (SHE) and low-rank adaptation (LoRA) to enable efficient and privacy-preserving federated tuning of LLMs in cross-device environments.
Based on model parameter sensitivity assessment, heterogeneous clients adaptively negotiate and select a subset of model parameters for homomorphic encryption. 
To ensure accurate model aggregation, we design a column-aware secure aggregation method and customized reparameterization techniques to align the aggregation results with the heterogeneous device capabilities of clients.
Extensive experiments demonstrate that SHE-LoRA maintains performance comparable to non-private baselines, achieves strong resistance to state-of-the-art attacks, and significantly reduces communication overhead by 99.71\% and encryption time by 99.87\%, compared to HE baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a new framework for privacy-preserving federated fine-tuning of large language models (LLMs) that balances efficiency, privacy, and heterogeneity among client devices. The proposed SHE-LoRA integrates SHE with LoRA to allow clients with diverse computational capacities and data distributions to securely participate in federated fine-tuning. The method adaptively encrypts only the "most sensitive" parameters based on a parameter sensitivity analysis, using a global negotiation mechanism to determine a subset of the parameters to be encrypted. Besides, the adaptive aggregation (on server side) and reparametrization (on client side) techniques are proposed to enable the FL cycle.

### Strengths
1) The paper proposes a framework that provides a strong privacy protection guarantee for the models.
2) The proposed algorithm can save computation cost significantly compared to the existing methods.
3) The proposed algorithm provides some insight about how to define and select important parameters.

### Weaknesses
1) The parameter subset selection and negotiation may remain unclear for readers. For example,
* Why is $S_j$  defined in the paper a reasonable metric to evaluate the importance? 
* What does "features aggregated across L tokens" mean in the context of the paper? 
* If the importance of the parameters changes in the training process, does the selection dynamically adapt to such changes? 
2) It is unclear what privacy protection strength this algorithm can provide in the worst case.
* Do the most sensitive parameters equivalent to those that should be protected for privacy protection?
* It seems there is no guarantee that the most sensitive parameters of a client will remain after negotiation. Does it mean the privacy of such a client, if it exists, is more vulnerable than that of others?
* The attack experiments only consider reconstruction attack, but no membership inference attack (which is more closely related to the privacy definition people commonly believe).
3) Given that the encrypted parameters are only a very small fraction, it is unclear whether directly removing those parameters from training can also produce similar model performance. If so, it may become arguable that the 
* Some ablation studies about the parameter may help.
4) The writing of the paper can be further improved.
* Some notations and definitions are missing in the paper. For example, the notation of "batch" uses the same notation as the matrix $B$ in LoRA.
* Too many details are omitted in the main text, which makes readers hard to follow.

### Questions
Please refer to the question in the Weaknesses section.

### Soundness
3

### Presentation
2

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
The paper proposes SHE-LoRA, a framework that combines selective homomorphic encryption (SHE) with low-rank adaptation (LoRA) for cross-device federated fine-tuning of LLMs. The core ideas are: (1) negotiation across heterogeneous clients using order-preserving encryption to choose columns to encrypt based on channel-wise sensitivity; (2) column swapping to cluster encrypted vs. plaintext columns for efficient batching and to obfuscate positions; (3) column-aware adaptive aggregation that aggregates plaintext and ciphertext parts separately; and (4) reparameterization via SVD so each client recovers LoRA factors at its local rank. Experiments show accuracy comparable to non-private heterogeneous LoRA while reducing communication by up to 99.71% and encryption time by up to 99.87% vs. full-HE baselines, and strong robustness to DAGER gradient inversion.

### Strengths
1. Massive reductions in HE time and bandwidth vs. full HE and MaskCrypt, with stable per-client times due to column clustering and budget control.
2. Comparable to non-private Flex-LoRA across GLUE/MMLU and vision tasks; sometimes better on subsets.

### Weaknesses
1. OPE preserves order information; while only rankings are revealed, the paper does not quantify leakage from order disclosure or compare with order-revealing encryption alternatives.
2. For 30B/70B models, ciphertext size/time per parameter increases notably and requires larger key sizes. While still workable at small budgets, the practicality at higher budgets or longer runs is unclear.

### Questions
1. How does SHE-LoRA fare against membership/property inference or reconstruction attacks that use auxiliary priors? Any reasons to expect similar robustness?
2. How should a,b,c be chosen in practice? Could you provide an adaptive rule and an ablation on real datasets?
3. For highly dynamic Non-IID clients, how often should renegotiation occur? Can you report end-to-end time or energy overheads for different periods?

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
2

### Summary
The proposed method SHE-LoRA integrates selective homomorphic encryption (SHE) with LoRA-based federated fine-tuning under heterogeneous client resources and Non-IID data. It aims to tackle the challenge of adaptive SHE for heterogeneous clients and the expansion of encrypted subsets on SHE. Specifically, this is done through the proposed HE subset negotiation mechanism and selective encryption and column-aware aggregation. Empirical results have shown that SHE-LoRA is resistant to privacy attacks, communication-efficient, and performant.

### Strengths
- The proposed method is empirically effective, which shows advantages in privacy-preserving, communication cost, and model performance.
- The problem of vulnerable and heterogeneous LoRA updates is motivated well.
- Principled design of the proposed algorithm.

### Weaknesses
- Limited novelty in the adoption of SHE methods to LLM LoRA fine-tuning.
- Column-wise weighted averaging is proposed, but the choice of weights (e.g., proportional to client data size or sensitivity) is not formally justified or compared.
- The negotiated global HE subset is claimed to optimally balance privacy and HE overhead per client, but lacks formal optimality guarantees or approximation bounds (e.g., submodular coverage, budgeted max coverage).
- The paper argues that encrypting A is sufficient, but the threat model and Section 2.4 discuss expanded encrypted subsets due to BA multiplication. It is not clear whether encrypting only A’s sensitive columns is enough under all considered attacks.
- Presentation could be improved by fixing the inline citations, defining the exact weighting scheme in column averaging, and reporting the Non-IID partitioning method.

### Questions
See weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a privacy-preserving method for fine-tuning large language models in federated learning. Traditional LoRA-based methods risk data leakage through shared gradients, while full homomorphic encryption is too costly. SHE-LoRA introduces selective homomorphic encryption and designs a column-level negotiation and aggregation mechanism to handle heterogeneous clients with different computing resources and privacy needs. The approach includes four key stages: (1) clients privately report column sensitivity to jointly select a global subset for encryption; (2) encrypted columns are reordered for efficient HE operations; (3) aggregation aligns plaintext and ciphertext parts column-wise; (4) results are re-factorized into new low-rank parameters. Experiments on NLP tasks show SHE-LoRA maintains model accuracy while reducing encryption and communication overhead by up to 99% compared with full HE, and it effectively defends against gradient inversion attacks like DAGER. The system scales to large models and non-IID data, proving that partial, structured encryption can offer strong privacy with minimal cost in federated LoRA training.

### Strengths
1. The framework explicitly supports clients with different hardware capabilities, network conditions, and privacy budgets.
2. This paper introduces selective homomorphic encryption at the column level of LoRA matrices, encrypting only the most privacy-sensitive components.
3. Experiments on NLP and vision datasets demonstrate that SHE-LoRA achieves accuracy comparable to or better than state-of-the-art methods (e.g., Flex-LoRA) under heterogeneous and Non-IID conditions.

### Weaknesses
1. The privacy guarantees of selective encryption, the convergence behavior of federated training under mixed plaintext/ciphertext updates, and the optimality of the HE subset negotiation are not formally proved.
2. SHE-LoRA method can't adapt to heterogeneous LoRA approaches like FLoRA.
3. The experiments rely on relatively small base models and simple benchmark tasks, which limits the generalizability of its results to large-scale or more complex real-world scenarios. It is recommended to evaluate the method on stronger base models such as Qwen3 and Llama 3.2, as well as on more challenging benchmark tasks including MMLU-Pro, GPQA, MuSR, MATH, IFEval, and BBH.

### Questions
See Weakness.

### Soundness
3

### Presentation
2

### Contribution
3
