# Efficient Inference and Small Language Models: A Survey

## TL;DR
- Early compression work established the core playbook for efficient inference: prune, quantize, and distill large models into smaller or cheaper-to-run forms [1][2].
- For autoregressive decoding, speculative decoding and related draft-and-verify schemes reduce token-by-token latency without changing outputs, and they can deliver multi-fold speedups in some settings [3].
- Modern post-training quantization and systems work show that inference efficiency is shaped by both algorithms and hardware: SmoothQuant targets activation outliers for W8A8 execution, while PowerInfer uses a GPU-CPU hybrid design around hot and cold neurons [4][5].
- Small language models are now studied as a distinct regime, with surveys finding that evaluation must cover reasoning, latency, memory footprint, and energy rather than parameter count alone [6][7][8].

## Background
Efficient inference has long been driven by the need to shrink model size, memory traffic, and compute cost while preserving task performance. Distillation transfers behavior from a large teacher or ensemble into a smaller student for deployment [1]. Deep Compression combined pruning, trained quantization, and Huffman coding into a three-stage pipeline that reduced storage substantially while preserving accuracy in the reported vision examples [2]. These ideas remain foundational because they separate model capability from deployment cost.

For language models, the same objectives reappear in autoregressive decoding. Speculative decoding accelerates generation by proposing several tokens with a smaller model and verifying them with the target model, avoiding architecture changes and retraining [3]. Later systems work and surveys emphasize that runtime also depends on memory bandwidth, context length, batch size, and hardware-specific bottlenecks, so evaluation must include latency, throughput, power, and memory footprint rather than accuracy alone [7][9][8].

## Foundations of model compression for deployment
Distillation and compression are the oldest route to smaller effective models. Distillation was originally framed as a way to preserve a learned function while moving it into a smaller model form, and the same high-temperature soft-target recipe is still echoed in later language-model distillation work [1]. MiniPLM modernizes that idea for pre-training by moving teacher inference offline so multiple students can reuse the same distilled data, reducing pre-training computation and improving downstream performance across tasks [10].

Quantization and pruning attack a different bottleneck: memory and arithmetic cost per parameter. Deep Compression reported aggressive storage reductions using pruning, trained quantization, and Huffman coding together, with no accuracy loss in the cited vision models [2]. For large language models, GPTQ and SmoothQuant show how post-training methods can make low-bit execution practical: GPTQ uses approximate second-order information for one-shot weight quantization, while SmoothQuant moves activation difficulty into weights via a mathematically equivalent transformation to enable W8A8 inference [4][11].

## Faster decoding and draft-verify inference
A separate line of work focuses on decoding rather than model size. Speculative decoding speeds up autoregressive generation by drafting multiple tokens in parallel and then verifying them with the target model, which can preserve exact outputs while reducing wall-clock time [3]. Self-speculative decoding removes the need for an external draft model by skipping selected intermediate layers of the same LLM during drafting, again keeping outputs identical to the original model [12].

These methods matter because decoding is often the dominant cost at deployment time. Surveys of efficient inference emphasize that the best method depends on whether a workload is memory-bound, compute-bound, or constrained by KV-cache growth, which varies with prompt length, generation length, and batch size [9][8][13]. The practical lesson is that draft-verify methods are complementary to compression: one reduces per-token work, the other reduces the cost of the work itself.

## Small language models as a distinct deployment regime
Recent surveys treat small language models as a separate category rather than simply scaled-down LLMs. One survey of open-source SLMs covers models in the 100M–5B parameter range and examines commonsense reasoning, in-context learning, mathematics, coding, latency, and memory footprint [6]. A broader survey of SLMs frames the goal as preserving accuracy and adaptability under constraints from hardware, bandwidth, and generation time, and organizes the field around lightweight architectures, efficient attention, NAS, training/fine-tuning, and compression [14].

The empirical message from recent releases is that “small” does not automatically mean “weak.” Hugging Face’s SmolLM release describes 135M, 360M, and 1.7B parameter models trained on curated data and positioned for local deployment [13]. MiniPLM adds that smarter distillation can boost students on downstream tasks while cutting pre-training compute [10]. Together, these results suggest that data quality, training recipe, and architectural choices can matter as much as raw scale in the SLM regime.

## Systems and hardware-aware inference
Many modern efficiency gains come from systems co-design rather than model changes alone. PowerInfer exploits activation sparsity patterns, splitting frequently used hot neurons from cold neurons and executing them on a GPU-CPU hybrid system; the paper reports large serving-speed improvements over llama.cpp in its tested setup [5]. Surveys of serving systems similarly emphasize model placement, request scheduling, storage management, disaggregation, multiplexing, and CPU/system-memory offloading as deployment levers [15].

Hardware-aware analysis also changes how researchers interpret benchmark numbers. Roofline-based surveys argue that inference performance must be read through peak compute, peak memory bandwidth, arithmetic intensity, and hardware setting, because the same model can be compute-bound in one regime and memory-bound in another [9][8]. The SLM survey likewise reports that longer context increases KV-cache and compute-buffer memory, and that vocabulary size and attention design affect edge-device runtime [6].

## Trends and open problems
Three trends stand out. First, efficiency work is moving from isolated algorithmic tricks toward pipelines that combine compression, quantization, and serving-system design [4][5][15]. Second, evaluation is broadening from accuracy to latency, memory, energy, and deployment realism, especially for on-device and edge use [6][7][13][15]. Third, SLM progress is increasingly benchmarked as a Pareto problem: gains in one metric often come with costs in another, so the right question is which tradeoff is acceptable for a target device or service-level objective [6][9][8].

Open problems remain substantial. Surveys of SLMs and efficient inference repeatedly point to unresolved effects of architecture choice, quantization method, hardware type, context length, and serving stack on true runtime cost [6][9][14][15]. Low-bit methods still face accuracy loss in some settings, and long-context deployment stresses KV-cache and memory buffers [6][8][14]. More broadly, the field still lacks a universal yardstick for comparing “small” and “efficient” models across heterogeneous hardware, workloads, and quality goals.

## References
[1] Distilling the Knowledge in a Neural Network. arxiv. https://arxiv.org/abs/1503.02531 (2015-03-09)
[2] Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding. arxiv. https://arxiv.org/abs/1510.00149 (2016-02-15)
[3] Fast Inference from Transformers via Speculative Decoding. arxiv. https://arxiv.org/abs/2211.17192 (2023-05-18)
[4] SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. arxiv. https://arxiv.org/abs/2211.10438 (2024-03-29)
[5] PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU. arxiv. https://arxiv.org/abs/2312.12456 (2023-12-16)
[6] Small Language Models: Survey, Measurements, and Insights. hf-search. https://huggingface.co/papers/2409.15790 (2024-09-24)
[7] Towards Coarse-to-Fine Evaluation of Inference Efficiency for Large Language Models. hf-search. https://huggingface.co/papers/2404.11502 (2024-04-17)
[8] LLM Inference Unveiled: Survey and Roofline Model Insights. web. https://arxiv.org/pdf/2402.16363v6.pdf (unknown)
[9] A Survey on Efficient Inference for Large Language Models. web. https://arxiv.org/html/2404.14294v2 (unknown)
[10] MiniPLM: Knowledge Distillation for Pre-training Language Models. arxiv. https://arxiv.org/abs/2410.17215 (2024-10-22)
[11] GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. web. https://www.alphaxiv.org/abs/2210.17323 (2023-03-22)
[12] Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding. arxiv. https://arxiv.org/abs/2309.08168 (unknown)
[13] SmolLM - blazingly fast and remarkably powerful. web. https://huggingface.co/blog/smollm (2024-07-16)
[14] A Survey on Small Language Models. web. https://aclanthology.org/anthology-files/pdf/ranlp/2025.ranlp.1.93.pdf (unknown)
[15] Taming the Titans: A Survey of Efficient LLM Inference Serving. web. https://aclanthology.org/2025.inlg-main.32.pdf (unknown)
