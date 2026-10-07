# Can Sparse Mixture-of-Experts Modularity Mitigate Catastrophic Forgetting in Class-Incremental Learning?

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tucozz/smoe_incremental_learning/blob/main/smoe_continual_learning.ipynb)

Code for the paper by **Arthur Estefanato Lopes** and **Filipe Mutz**, Departamento de Informática, Universidade Federal do Espírito Santo (UFES). Anais da Escola Regional de Informática do Espírito Santo (ERI-ES), 2026.

## Summary

Modular networks are often expected to resist catastrophic forgetting, since isolated components could learn new tasks without overwriting old ones. We test this expectation on a sparse Mixture-of-Experts (SMoE) in class-incremental learning, comparing it with a dense MLP on MNIST and CIFAR-10.

- **Modularity alone offers no retention advantage.** Under the same output masking, the SMoE retains *less* of the first task than the dense MLP on MNIST, with two and with three tasks.
- **Freezing experts trades plasticity for retention**, and its plasticity cost does not come from a failure to learn: the new task is learned as well as without freezing, but old classes win the final decision.
- **Routing does not stabilize.** The router groups digits by shape, but repeating tasks erases this specialization instead of consolidating it, and the shared input projection, never frozen, keeps moving the latent space.
- **With three tasks, two distinct forgetting mechanisms appear:** the earliest task collapses through accumulated drift, while the intermediate one loses to the most recent (recency bias).
- **The shared input projection is the main source of forgetting.** Removing it, at the same number of parameters, raises retention by about 30 points in every SMoE variant, at a cost in plasticity.
- **With the projection, an oracle router that fixes one expert per class is the only variant that surpasses the dense baseline without sacrificing plasticity**, which points to learned routing, not modularity itself, as the main bottleneck.

## Repository structure

```
smoe_continual_learning.ipynb   single notebook reproducing every figure and table in the paper
results/                        raw results of every experiment (JSON), loaded by the notebook
figures/                        figures as they appear in the paper (Figure 1a, smoe_layer.png, is a diagram)
requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt
jupyter notebook smoe_continual_learning.ipynb
```

The notebook also runs on Google Colab. MNIST and CIFAR-10 are downloaded automatically.

By default (`RETRAIN = False`), experiments whose results are in `results/` are loaded instead of retrained, so the notebook runs in a few minutes and redraws every figure and table. Set `RETRAIN = True` to retrain everything; the notebook lists the approximate time of each experiment.

On Colab, the notebook downloads the saved results from this repository before running, so the "Open in Colab" badge above loads everything in minutes. Training is saved one seed at a time, so an interrupted run resumes where it stopped; for long retraining sessions on Colab, `USE_DRIVE = True` keeps the results on Google Drive. A final section compares every computed number with the one published in the paper.

## Reproducibility

All runs are deterministic: `torch.use_deterministic_algorithms(True)` is enabled, and the SMoE combines expert outputs without indexed accumulation, which has no deterministic GPU implementation. The same seed yields the same numbers across sessions. The paper's results were computed on CPU (`FORCE_CPU = True`, the default); CPU and GPU agree up to the last decimals. Main comparisons use eight seeds; the evaluation diagnosis and the additional analyses use three, and the routing analyses one. `results/environment.json` records the device and library versions behind each result.

## Main results

MNIST, two tasks (digits 0–4, then 5–9), macro F1-score (%), mean ± standard deviation over eight seeds:

| Configuration | Retention | Plasticity |
|---|---|---|
| MLP | 0.00 ± 0.00 | 97.70 ± 0.22 |
| SMoE | 0.00 ± 0.00 | 97.80 ± 0.27 |
| MLP (masked) | 55.49 ± 6.10 | 97.27 ± 0.48 |
| SMoE (no freeze) | 28.64 ± 7.05 | 97.64 ± 0.23 |
| SMoE (freeze) | 58.25 ± 6.79 | 68.63 ± 8.30 |
| SMoE (ideal router) | 61.81 ± 5.74 | 97.67 ± 0.16 |

Three tasks grouped by digit shape, retention averaged over the two earlier tasks: MLP (masked) 41.64%, SMoE (no freeze) 20.25%, SMoE (freeze) 25.54%, SMoE (ideal router) 48.04%.

## Additional analyses

Analyses summarized in the paper's discussion, with full results in the notebook (Section 5.6):

- **Capacity reservation.** Keeping a share of experts free for later tasks, proportional to the number of remaining tasks, does not recover the intermediate task (23.83% versus 25.54% retention with three tasks).
- **Input projection.** At the same number of parameters, removing the shared projection raises retention by 31 to 33 points in every SMoE variant with two tasks (e.g. 28.64% to 61.24% without freezing), at a cost in plasticity. Without it, the unfrozen SMoE reaches the masked MLP.
- **Morphological grouping.** Grouping digits by shape makes no measurable difference: against a control with the same task sizes in numerical order, retention changes by at most 4.8 points, and the gap between the SMoE and the masked MLP, which is what the hypothesis concerns, shifts by less than 4 points, below the spread across seeds.
- **Number of experts, top-k, balancing weight α.** More experts do not isolate tasks better (62.6%, 56.4% and 52.1% retention with 4, 10 and 20 experts). k = 2 retains as much as k = 4 and more than k = 1, with more plasticity than k = 4. α changes retention by less than its spread across seeds; α = 10 gives the most stable plasticity.
- **Masking only from the second task onward.** Raises retention by 7.6 points but lowers plasticity by 39.3, which is why masking is applied in every task.

## Citation

```bibtex
@inproceedings{lopes2026smoe,
  title     = {Can Sparse Mixture-of-Experts Modularity Mitigate Catastrophic Forgetting in Class-Incremental Learning?},
  author    = {Lopes, Arthur Estefanato and Mutz, Filipe},
  booktitle = {Anais da Escola Regional de Informática do Espírito Santo (ERI-ES)},
  year      = {2026}
}
```

## License

Released under the MIT License; see `LICENSE`.
