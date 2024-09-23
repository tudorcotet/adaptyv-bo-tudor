Forked from https://github.com/adaptyvbio/ml_recommendations

# ML team: best practices

Here we have some (opinionated) guidelines for writing and maintaining code within the ML team.

## Github

Generally, you should have one repo per project, where the `main` branch is the current stable implementation that is regularly updated and all changes and new features are added in weparate branches that are merged into `main` with pull requests.

### Commits

For commit names [use conventional commits](https://www.conventionalcommits.org/en/v1.0.0/):

- Commit messages should follow the format: `type(scope): description`.
- The `type` should be one of the following:
  - `feat`: A new feature
  - `fix`: A bug fix
  - `docs`: Documentation changes
  - `style`: Code style changes (e.g., formatting, missing semicolons)
  - `refactor`: Code refactorings that don't change functionality
  - `test`: Adding or modifying tests
  - `chore`: Other changes that don't modify code or tests
- The `scope` (optional) should indicate the module or feature area affected by the commit.
- The `description` should be a concise summary of the changes introduced by the commit.

https://github.com/cocogitto/cocogitto makes following this _easy_

The description should be uncapitalized, in imperative, present tense. The total length of the header shouldn't exceed 100 characters. If you want to add more information, it can be put in the body of the message, separated with a blank line.

Examples:

- `'fix(processing): make ProteinDataset work with old style dictionaries'`,
- `'test: test for exclude_chains in test_generate'`.
- ```
  'feat(generating): add ligand support

  Add the option to load ligands, cluster using Tanimoto clustering on smiles,
  and download files based on a pdb_id text file.'
  ```

See the [Conventional Commits guide](https://www.conventionalcommits.org/en/v1.0.0/) for more information.

### Pull Requests

When creating a pull request, please follow the following guidelines:

- Use a capitalized and imperative present tense for the pull request title.
- Provide a clear and informative description of the changes introduced by the pull request.
- If the pull request addresses an issue, reference it in the description using the `#issue-number` syntax.

Even if you're the only person working on the project, it is useful to keep a clear history of the changes.

### Branches

Remove branches after they have been merged. Normally you should only have a `main` branch and one or a few feature branches that you're currently working on. In bigger projects, it is a good idea to protect the `main` branch with required checks that run on pull requests. Those can be pre-commit checks (see below) and/or automated tests. See the [ProteinFlow CI/CD pipelines](https://github.com/adaptyvbio/ProteinFlow/tree/main/.github/workflows) for an example of both.

## Datastructures and types


## Documentation

Try to always document your code as much as possible, e.g. in [NumPy style](https://github.com/adaptyvbio/ProteinFlow/tree/main/.github/workflows).
Keep in mind that documentation is not for the _what_ but for the _why_, i.e., you want to help the user reason about the functions as part of their usage, either with the module documentation or by giving more context to the datastructures used.

You can use [`pydocstyle`](https://www.pydocstyle.org/en/stable/) and/or [`docsig`](https://pypi.org/project/docsig/) to check that the your code is documented correctly.

## Code formatting

In general, python code should always comply with [the PEP8 style guide](https://peps.python.org/pep-0008/). Thankfully, there are tools that can help you keep track of that. In particular, [`black`](https://black.readthedocs.io/en/stable/) can make many changes automatically and [`flake8`](https://flake8.pycqa.org/en/latest/) will catch the remaining issues and report them to you. In addition, [`isort`](https://pycqa.github.io/isort/) fixes the specific issue of sorting the import statements.

## Pre-commit

I highly recommend using [`pre-commit`](https://pre-commit.com/) to check that the code is maintained properly. It is a framework that allows running some automated checks and corrections every time you make a commit. This repo includes sample configuration files, you can copy them in your project, install the package and the pre-commit hooks, and whenever you make a commit it will go over the newly added or modified files and check that they are well documented and formatted and do not contain spelling errors. Note that `pyproject.toml` needs to be configured for your package. The pre-commit config in the repo contains all the tools mentioned above and a few additional ones. In addition, it will also check that your commit message complies with the naming scheme. Do not forget to install both the standard and the commit message hooks with these commands.

```bash
python -m pip install pre-commit
# or simply pip install -e .[dev] if you are keeping the default pyproject.toml
cd $PROJECT_REPO
pre-commit install
pre-commit install --hook-type commit-msg
```

## Packaging

For most bigger projects, at some point there comes a time when you need to package it as a library. It simplifies managing imports, allows you to register commands and to use the code in other projects easily. It is also a good way to release your code, e.g. when publishing a paper. In case you suspect this time will come eventually, I recommend doing that from the beginning in order to avoid having to refactor a large codebase. Please use [`pyproject.toml`](https://pip.pypa.io/en/stable/reference/build-system/pyproject-toml/) format for the build file.

## Module boundaries

It can be useful for organization to enforce strict module boundaries within a code base, in orderto avoid chaotic code. https://github.com/gauge-sh/tach helps with this
