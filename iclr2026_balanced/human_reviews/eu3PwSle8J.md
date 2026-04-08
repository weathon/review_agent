## Human Reviewer 1

### Summary
This paper addresses prompt injection attacks in LLMs by proposing Augmented Intermediate Representations (AIR), a defense mechanism that injects instruction hierarchy (IH) signals across all decoder layers rather than only at the input layer. The authors argue that existing defenses (using delimiters or input segment embeddings) suffer from signal degradation through the network. Experiments across three models show AIR reduces attack success rates by 1.6-9.2× on gradient-based attacks compared to

### Strengths
1. Clear motivation and well-identified limitation: The observation that IH signals degrade through decoder layers (Figure 3) is compelling and provides solid motivation for the proposed approach. The parallel to positional embeddings (RoPE) is insightful.

2. Comprehensive experimental evaluation: The paper evaluates multiple models (3B, 7B, 8B parameters), training methods (SFT, DPO), and attack types (static and gradient-based), demonstrating thoroughness.

3. Minimal overhead: The additional parameters (0.005% for Llama3.1-8B) and inference compute are negligible, making the approach practical.

4. Well-structured presentation: The paper is clearly written with good use of figures and tables to convey results.

### Weaknesses
1. Limited theoretical justification: While Figure 3 shows cosine similarity increases across layers, this alone doesn't conclusively prove that IH signal degradation is the limiting factor. Alternative explanations could include:
- The difficulty of learning from input-only signals during training
- The specific architecture's tendency to homogenize representations
- The paper would benefit from ablation studies showing what happens with AIR at only some layers, or from analyzing attention patterns to demonstrate that AIR helps maintain privilege distinctions.

2. Inconsistent performance across training methods: AIR-SFT sometimes shows lower utility than the None baseline (Figure 8b), particularly for Qwen-2.5-7B and Llama-3.1-8B. This is concerning and inadequately explained. The paper should:
- Investigate why this degradation occurs specifically with SFT
- Provide guidance on when to use DPO vs. SFT with AIR
- Discuss potential mitigation strategies

3. Model-specific hyperparameter sensitivity: The need for different initialization strategies for Qwen (σ=0.1 vs. σ=0.02 for Llama) raises concerns about generalization:
- How sensitive is performance to this choice?
- What guidance can be provided for applying AIR to new model families?
- The lack of hyperparameter tuning "due to computational constraints" is unsatisfying for a defense mechanism intended for practical deployment.

4. Limited evaluation scope:
- Only single-turn interactions are tested (acknowledged in limitations)
- No evaluation on real-world prompt injection scenarios beyond AlpacaFarm and SEP
- No comparison with detection-based defenses mentioned in Appendix D
- Static attack evaluation less informative: Since all three IH mechanisms achieve near-perfect defense against static attacks (Table 1), these results don't effectively differentiate approaches. More emphasis should be placed on adaptive attacks.

### Questions
Please refer to weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
4

### Confidence
3

---

## Human Reviewer 2

### Summary
The paper introduces a new defense against indirect prompt injections, a type of attack where the attacker injects malicious content into the input context. The authors propose Augmented Intermediate Representations (AIR), which, unlike prior approaches that apply Instruction Hierarchy (IH) information only at the input of the transformer (e.g. using delimiter tokens or IH embeddings), adds IH embeddings at every transformer block. They hypothesize that previous methods lose IH signal strength as it propagates through model layers, which motivates their method. The paper benchmarks AIR on the AlpacaFarm and SEP datasets using Llama-3.2-3B, Qwen-2.5-7B, and Llama-3.1-8B, trained with either SFT or DPO, under both static and gradient-based attacks. On these benchmarks, AIR achieves favorable results.

### Strengths
- **Relevant topic**: The paper addresses a relevant, timely, and practical problem.
- **Clear motivation and presentation**: The paper is well-written and clearly explains why the method is needed.
- **Simple approach**: The method is easy to implement and introduces only a relatively small number of additional parameters.
- **Strong empirical section**: The evaluation is extensive and covers multiple base models, attack types, and training regimes (SFT, DPO). See Weaknesses for comments regarding datasets.

### Weaknesses
- **Validation of motivating hypothesis**: The hypothesis requires stronger validation. It remains unclear whether existing methods fail due to IH signal degradation, as claimed in section "Limitations of Existing Defenses". Measuring cosine similarity across layers is not sufficient, especially for delimiter-based methods. A simple linear probing experiment (as done e.g. in ASIDE) to test IH separability would strengthen the claim substantially. This is particularly important since identifying this limitation of input-only methods is listed as one of the papers three main contributions.

- **Static attacks fail**: The reported improvements come mainly from gradient-based attacks. Static attacks appear to fail even for the naive baseline, so robustness improvements there seem less meaningful. Including more difficult or diverse attack benchmarks would make the results more convincing.

### Questions
- The authors may want to look into ASIDE, which proposes a closely related defense method, also addressing a similar “IH signal degradation” issue in ISE. While not being obligatory, a comparison between AIR and ASIDE would be scientifically valuable, as they both target a similar goal but try to achieve it with different methods: ASIDE enforces IH separation via orthogonal rotations at the input layer, whereas AIR reinforces the IH signal throughout the network with IH embeddings.
- L135 Spelling error: "Ig nore"
- Figure 4 seems to be wrong as a single decoder block contains two masked self-attention computations. 

 **References**:
- Zverev et al. “ASIDE: Architectural Separation of Instructions and Data in Language Models.” ICLR 2025 Building Trust in LLMs and LLM Applications Workshop (non-archival), 2025.

### Soundness
2

### Presentation
3

### Contribution
3

### Rating
4

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper proposes Augmented Intermediate Representations (AIR), a defense mechanism against prompt injection attacks in LLMs. Existing defenses introduce an Instruction Hierarchy (IH) signal, indicating the privilege level of input tokens, but inject it only at the input layer. The authors hypothesize that this limits the model’s ability to maintain hierarchical distinctions as information propagates through layers. To address this, AIR injects layer-specific, trainable embeddings encoding IH signals into every decoder block of the model. This recurrent injection ensures that privilege information remains accessible throughout the network.

### Strengths
1.	The paper proposes a new architectural idea that injects Instruction Hierarchy signals at every layer of the model rather than only at the input, offering a fresh approach to prompt-injection defense distinct from earlier input-based methods.
2.	The experiments cover different model sizes and both SFT and DPO training setups, and consistently show gains in robustness.
3.	The problem addressed, protecting models from prompt injection in settings involving untrusted data and agent workflows, is timely and highly important.

### Weaknesses
1.	Utility is measured mostly via AlpacaEval win rates. There is no assessment of factual accuracy or reasoning (e.g., MMLU), so it is difficult to judge whether AIR affects model quality in benign settings. Including standard benchmarks would strengthen the claims.
2.	Although the paper suggests AIR can be applied to direct prompt attacks and agent settings, no experiments verify this. Multi-turn, retrieval-augmented, and user-as-attacker (jailbreak) scenarios remain unexplored.
3.	There is no visualization of how AIR changes attention patterns across layers. Such analysis could clarify whether AIR genuinely preserves hierarchical separation or merely adds regularization noise.
4.	The SFT models are fully fine-tuned, while DPO models use LoRA. This mismatch may explain the utility drop seen in Figure 8. A controlled comparison (full-FT vs LoRA in both settings) would clarify this.

### Questions
1.	Qwen required a much larger initialization scale for the IH embeddings. Can the authors provide a systematic study of initialization and stability? Otherwise, AIR’s robustness may depend on model-specific tuning rather than a generalizable method.
2.	Utility is measured mainly via AlpacaEval win rate. Can the authors report additional evaluations (e.g., MMLU, BLEU, factual consistency, human preference) to confirm that AIR does not subtly degrade model quality in benign settings?
3.	Since the paper reimplements prior defenses, can the authors verify that the reproduced baselines match the original papers’ reported performance, or provide an error margin? This would ensure a fair comparison and strengthen the empirical claims.
4.	The method uses trainable embeddings for privilege levels. Could the authors verify that learning these embeddings is necessary? For example, how does AIR perform if fixed random vectors are used instead? A comparison would clarify whether the improvement comes from the learned hierarchy signal or simply from adding noise/perturbations at each layer.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4