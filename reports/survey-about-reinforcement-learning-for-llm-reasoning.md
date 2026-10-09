# Reinforcement Learning for LLM Reasoning: A Survey

## TL;DR
- Reinforcement learning for LLMs began with RLHF-style post-training: supervised demonstrations, a reward model trained from human comparisons, and PPO fine-tuning with KL regularization to control policy drift [1][2].
- Newer reasoning-oriented systems use rule-based or verifier-based rewards to reduce dependence on dense human annotation; recent papers report gains on logic, math, and instruction-following benchmarks, but also note shortcutting, reward hacking, and sensitivity to setup [3][4][5].
- A major trend is replacing or simplifying the classic reward-model-plus-PPO pipeline, either with direct preference objectives such as DPO or with weaker supervision built from program checks, synthetic problems, or automatically generated feedback [6][7][8].
- Surveys and reviews consistently frame RL as complementary to supervised fine-tuning, distillation, inference-time scaling, and self-improvement rather than a universal substitute [9][8][10].

## Background
Reinforcement learning entered LLM post-training primarily through RLHF, where a model is first trained on supervised demonstrations, then a reward model is fit from human preference comparisons, and finally policy optimization is run with PPO [1]. PPO itself is the core optimization algorithm behind much of this work: it uses a clipped surrogate objective to stabilize updates and can also be implemented with a KL penalty to limit divergence from a reference policy [2]. In the instruction-following setting, this design is meant to improve human preference scores while reducing unwanted drift from the supervised model [1].

The classic RLHF pipeline is important for reasoning research because it established the basic machinery now reused in more targeted forms: reward modeling, KL-constrained policy updates, and preference-driven selection of outputs [1][2]. Later work showed that the reward model plus PPO stack can be simplified. DPO reframes preference alignment as a direct classification-style objective over human comparisons, explicitly avoiding separate reward-model fitting and RL fine-tuning while aiming to match or exceed PPO-based methods on several alignment tasks [6].

## From RLHF to direct preference optimization
The early RLHF recipe is an alignment method rather than a reasoning-specific method, but it supplies the template that later reasoning systems adapt. InstructGPT’s pipeline uses human demonstrations, a reward model trained on ranked outputs, and PPO with a KL term relative to the SFT policy; the paper reports human evaluators preferred the 1.3B InstructGPT model over 175B GPT-3 outputs [1]. That result matters for reasoning surveys because it showed that post-training can substantially improve model behavior without changing the base architecture.

DPO is a key comparison point because it shows a non-RL route to preference optimization. The paper describes existing RLHF as reward-model plus PPO training and replaces that with a direct objective over preferences [6]. The reported appeal is simplicity and stability, but the tradeoff is conceptual: DPO removes explicit reward maximization and therefore changes how one thinks about credit assignment and exploration in reasoning-focused settings [6].

## Rule-based and verifier-based reinforcement for reasoning
Recent reasoning work shifts from broad alignment to task-specific reward signals. Logic-RL studies rule-based reinforcement learning on synthetic logic puzzles with controllable complexity and straightforward answer verification [3]. The paper reports that a 7B model develops reasoning behaviors such as reflection, verification, and summarization, and that training on 5K logic problems generalizes to AIME and AMC math benchmarks [3]. It also warns that naive training can collapse into degenerate solutions, motivating strict format rewards and prompt design [3].

Verification engineering pushes this idea further by combining programmatic checks with model-based verification. VerIF uses both rule-based code verification and LLM-based verification, and the associated VerInstruct dataset contains approximately 22,000 instances with verification signals [4]. The reported result is significant improvement on representative instruction-following benchmarks and competitive performance among comparable models, with generalization to unseen constraints [4]. The same source also flags a limitation: code-only verification can be too weak, while LLM-only verification can be easier to fit or hack [4].

These approaches share a common theme: they try to make reward signals cheaper and more reliable than dense human judgments. In reasoning tasks, correctness can often be checked by execution, symbolic rules, or other verifiers, so RL can be driven by objective feedback instead of subjective preference [3][4]. That shift is central to current reasoning-oriented RL because it makes trial-and-error learning feasible at scale.

## Synthetic data and self-awareness in RL for reasoning
Another line of work uses synthetic problem generation to expose model weaknesses. SwS, or self-aware weakness-driven problem synthesis, identifies deficiencies in the model and turns them into new training problems rather than generating more problems indiscriminately [7]. The paper reports average gains of 10.0% and 7.7% on 7B and 32B models across eight reasoning benchmarks, plus gains of 16.7% and 13.3% on AIME24 and AIME25 with Qwen2.5-7B when initialized with MATH-12k [7]. It also reports that the approach can recover up to 20.0% more problems the model had consistently failed [7].

This kind of method is important because it connects RL to curriculum design. Instead of assuming the problem set is fixed, the system actively searches for weaknesses, then uses those weaknesses to guide both data generation and policy improvement [7]. The downside is that it depends on accurate weakness detection and on synthetic data that remains useful rather than merely larger in quantity [7].

## Comparisons with supervised fine-tuning, distillation, and self-improvement
Recent surveys describe RL for reasoning as one component in a broader toolkit. One review says recent reasoning breakthroughs use inference-time scaling, reinforcement learning, supervised fine-tuning, and distillation together, not one in isolation [9]. Another survey on reward models frames learning from rewards as a unified paradigm spanning RLHF, RLAIF, DPO, GRPO, reward-guided decoding, and post-hoc correction [8]. That survey also highlights process rewards as finer-grained supervision for intermediate reasoning steps, while noting that step-level human annotation is expensive [8].

The main comparison is between outcome-level and process-level supervision. Outcome rewards are cheaper but weaker, because they score only the final result; process rewards can guide intermediate reasoning but are harder to obtain at scale [8][10]. Reinforced reasoning surveys argue that automated data construction and LLM-driven search are increasingly used to reduce dependence on human step-by-step annotation [10]. In that framing, RL is not a replacement for supervised learning, but a way to add targeted feedback where standard next-token prediction does not capture reasoning quality [9][10].

## Trends and open problems
The literature points to several unresolved issues. First, RL for reasoning is still sensitive to implementation details, and a recent deep dive reports fragmented understanding, inconsistent experimental settings, and the absence of standardized guidelines [5]. Second, verifier quality remains a bottleneck: rule-based verifiers may be brittle, and LLM-based verifiers can be manipulated or overfit [4][5]. Third, the field is still searching for scalable supervision that does not rely on expensive human labels, especially for multi-step reasoning [9][8][10].

A broader open problem is deciding when RL is worth the complexity relative to alternatives. Some tasks may be better served by supervised fine-tuning, distillation, or better test-time search, while others benefit from explicit reward optimization [9][8]. The surveys also emphasize that progress depends on better benchmark design, clearer training recipes, and stronger understanding of what the model is actually learning when RL improves reasoning traces [5][9][10].

## References
[1] Training language models to follow instructions with human feedback. arxiv. https://arxiv.org/abs/2203.02155 (2022-03-04)
[2] Proximal Policy Optimization Algorithms. web. https://v1.endtoend.ai/papers/proximal-policy-optimization-algorithms.pdf (unknown)
[3] Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning. arxiv. https://arxiv.org/abs/2502.14768 (2025-02-20)
[4] Verification Engineering for Reinforcement Learning in Instruction Following. web. https://arxiv.org/abs/2506.09942 (unknown)
[5] Part I: Tricks or Traps? A Deep Dive into RL for LLM Reasoning. hf-search. https://huggingface.co/papers/2508.08221 (2025-08-11)
[6] Direct Preference Optimization: Your Language Model is Secretly a Reward Model. web. https://www.cs.columbia.edu/~blei/fogm/readings/RafailovSharmaMitchellErmonManningFinn2023.pdf (unknown)
[7] SwS: Self-aware Weakness-driven Problem Synthesis in Reinforcement Learning for LLM Reasoning. hf-search. https://huggingface.co/papers/2506.08989 (2025-06-10)
[8] Sailing by the Stars: A Survey on Reward Models and Learning Strategies for Learning from Rewards. web. https://arxiv.org/abs/2505.02686 (2025-06-12)
[9] Reasoning Beyond Limits: Advances and Open Problems for LLMs. web. https://arxiv.org/abs/2503.22732 (2025-03-26)
[10] Towards Large Reasoning Models: A Survey of Reinforced Reasoning with Large Language Models. web. https://arxiv.org/abs/2501.09686 (unknown)
