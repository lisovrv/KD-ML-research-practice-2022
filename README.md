# KD-ML-research-practice-2022
Repository for research experiments on ML research practice 2022 course

## Contributing

Please note that since `40f1400212eb81df2969a9718ae2ebfde3e552d5` (April 19, 2022) this repository uses the [pre-commit](https://pre-commit.com/) (see `.pre-commit-config.yaml`).

In order to utilize the `pre-commit`, first, you need to ensure that your work is rebased onto the specified revision of the `main` branch. The easiest thing to do that is to run `git pull && git rebase origin/main` command while on your own branch. Note that this may require you to fix the rebase conflicts. Wherever applicable, it might be a good idea to squash your own commits first in order to avoid multiple fixes.

Once you have `.pre-commit-config.yaml` in your branch, you need to install the `pre-commit` itself (e.g. with `pip install pre-commit`) and run the `pre-commit install` in the root of your repository. It might also be a good idea to run `pre-commit run --all-files` in order to fix already committed files. After you did this, you are good to go. Note that the first commit after running `pre-commit install` might take some time, don't worry it's a one-of thing.

Additionally, note that the `pre-commit` runs `black` and `isort` on every commit, which means that your commit might fail if any one of those decides to modify your files. Most of the time, this can be fixed with a simple re-commit. The only thing that might stop the second commit is a `flake8` error, such as unused import, which cannot be fixed automatically. In such case it's your job to go and fix the errors. Also consider that every once in a while you might encounter a `flake8` error, which cannot be fixed. In such case you can still commit the code. You just add an inline comment `# noqa: E***` where `E***` is the error code, e.g. `# noqa: E402` in `src/binpord/cross_entropy/cross_entropy.py`.
