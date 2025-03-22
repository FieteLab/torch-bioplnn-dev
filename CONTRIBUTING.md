# Contributing to BioPlNN

Thank you for investing your time in contributing to BioPlNN! Your
contributions are essential in making BioPlNN a robust and innovative project.

In this guide you will get an overview of the contribution workflow from
opening an issue, creating a PR, reviewing, and merging the PR.

Please take a moment to review our guidelines below.

## New Contributor Guide

To get an overview of the project, read the [README](/README.md) file.

If you are new to open source contributions, here are some resources to help you get started:
- [Forking a Repository](https://docs.github.com/en/get-started/quickstart/fork-a-repo)
- [Git basics](https://docs.github.com/en/get-started/using-github/git-basics)
- [Creating a Pull Request](https://docs.github.com/en/pull-requests)

## Getting Started

To begin contributing to BioPlNN, you must first set up your development
environment and install the package from source. To do so, follow these steps:

1. Create a fork of the repository.
   See [Forking a Repository](https://docs.github.com/en/get-started/quickstart/fork-a-repo)
   for details.

2. Clone your fork of the repository
   ```bash
   git clone <your-fork-url>
   ```

3. Create a development branch:
   ```bash
   git checkout -b <your-branch-name>
   ```

4. Setup your development environment and install the package and its dependencies
   according to the [installation guide](/docs/installation.md).

   *Note*: When installing the package, we recommend that you use editable mode (using `-e`) and
   install the `dev` extras, which includes several useful tools for development,
   including the **required** pre-commit hooks:
   ```bash
   pip install -e ".[dev]"
   ```

5. Install the pre-commit hooks. These hooks check and automatically format your
   code to ensure that it adheres to the project's standards. This is **required**
   to ensure that your contributions are merged:
   ```bash
   pip install pre-commit
   pre-commit install
   ```
## Contributing Code

All contributions are welcome! We ask that each contribution be self-contained
and consist of a single, atomic change. In other words, each contribution should
be a single, complete feature or bug fix. This makes it easier to review and merge.
We may ask you to split your contribution into smaller, more manageable pieces if
it addresses multiple issues or if it is too large to be reviewed in a timely manner.

Once you have made your changes, you can commit them and push them to your fork:
```bash
git add --all
git commit -m "<your-commit-message>"
git push origin <your-branch-name>
```
## Testing Your Changes

Before submitting a pull request, please run our tests to ensure your changes
do not break existing functionality. If you have contributed code that is not
tested by the existing tests, please add tests for your changes. We appreciate
thorough testing!

To test your changes, run `pytest` in your development environment (you will
have to have installed the `dev` extras (see [Getting Started](#getting-started)):
```bash
pytest
```
## Opening a Pull Request

When you're ready:
- Submit a pull request (PR) targeting the main branch of BioPlNN from the feature branch in your fork.
- Fill out the PR template to help reviewers understand your changes.
- Link your PR to any relevant issues.
- Allow maintainer edits if requested, to facilitate merges.

For more details on how to open a PR, see [Creating a Pull Request](https://docs.github.com/en/pull-requests).

## Merging and Updates

Once your PR is reviewed and approved, it will be merged. If there are merge conflicts,
please resolve them promptly.

## Opening Issues

If you are not able to contribute directly, we also gladly welcome you to open issues to
report bugs or suggest improvements. If you spot a problem with the docs,
[search if an issue already exists](https://docs.github.com/en/github/searching-for-information-on-github/searching-on-github/searching-issues-and-pull-requests#search-by-the-title-body-or-comments).
If a related issue doesn't exist, you can open a new issue using a relevant
[issue form](https://github.com/FieteLab/torch-biopl-dev/issues/new/choose).
