<h1 align="center"><img src="docs/assets/fox_hero_200.png" alt="ProgramBench logo" width="120"><br/>ProgramBench</h1>

<p align="center"><em>Can Language Models Rebuild Programs From Scratch?</em></p>

<p align="center">
Given only a compiled binary and its documentation, AI agents must architect and implement a complete codebase that reproduces the original program's behavior.
</p>

## Links

- [Website](https://programbench.com)
- [Paper](https://programbench.com/static/paper.pdf)
- [HuggingFace](https://huggingface.co/datasets/programbench/ProgramBench-Tests)
- [Leaderboard](https://programbench.com)
- [Usage Guide](docs/README.md)

## Quickstart

We recommend [uv](https://docs.astral.sh/uv/getting-started/installation/) for managing Python environments.

```bash
# Run without installing
uvx programbench --help

# Or install into a project
uv pip install programbench

# Or with pip
pip install programbench
```

For development:

```bash
git clone https://github.com/facebookresearch/programbench.git
cd programbench
uv sync  # installs editable + dev dependencies
```

> [!NOTE]
> For more details, please refer to the [Usage Guide](docs/README.md).

## Citation

If our work was useful for you, please cite it:

```bibtex
@preprint{yang2026programbench,
  title={ProgramBench: Can Language Models Rebuild Programs From Scratch?},
  author={John Yang and Kilian Lieret and Jeffrey Ma and Parth Thakkar and Dmitrii Pedchenko and Sten Sootla and Emily McMilin and Pengcheng Yin and Rui Hou and Gabriel Synnaeve and Diyi Yang and Ofir Press},
  year={2026},
}
```

## License

ProgramBench is licensed under the terms of the license found in [LICENSE](LICENSE).
