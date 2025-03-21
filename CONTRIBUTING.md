# Contributing to BioPlNN

## Branch or fork
If you have permission to create branches in the BioPlNN GitHub repository,
create a new branch for your changes. You can then submit a pull request to the
main branch of the BioPlNN repository.

Otherwise, fork the BioPlNN repository. You can then submit a pull request from
your fork to the main branch of the BioPlNN repository.

## Development setup

1. Clone the BioPlNN repository (or your fork)
```bash
git clone https://github.com/valmikikothare/bioplnn.git
```
or
```bash
git clone <your-fork-url>
```
2. Create a new branch (optional if you are using a fork)
```bash
git checkout -b <your-branch-name>
```
3. Install dev dependencies
```bash
pip install -e .[dev]
```
or
```bash
pip install -r requirements/dev.txt
```
4. Install pre-commit hooks
```bash
pre-commit install
```
5. Make your changes and commit them
```bash
git add --all
git commit -m "<your-commit-message>"
```
6. Push your changes to your branch
```bash
git push origin <your-branch-name>
```
7. Submit a pull request
Go to the BioPlNN repository and submit a pull request from your branch or fork
to the main branch. See [GitHub's Pull Request documentation](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)
for more information.

## Testing
