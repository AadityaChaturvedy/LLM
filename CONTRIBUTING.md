# Contributing to Hindi/Hinglish Transformer LLM

First off, thank you for considering contributing to this project! It's people like you that make open-source a great community.

## How Can I Contribute?

### Reporting Bugs
This section guides you through submitting a bug report. Following these guidelines helps maintainers and the community understand your report, reproduce the behavior, and find related reports.

* **Check the issue tracker** to see if the bug has already been reported.
* **Use a clear and descriptive title** for the issue.
* **Describe the exact steps** which reproduce the problem in as many details as possible.
* **Provide specific examples** to demonstrate the steps. Include code snippets or error messages.
* **Describe the behavior you observed** after following the steps and point out what exactly is the problem with that behavior.
* **Explain which behavior you expected** to see instead and why.

### Suggesting Enhancements
This section guides you through submitting an enhancement suggestion, including completely new features and minor improvements to existing functionality.

* **Check the issue tracker** to ensure the enhancement hasn't already been suggested.
* **Use a clear and descriptive title** for the issue to identify the suggestion.
* **Provide a step-by-step description** of the suggested enhancement in as many details as possible.
* **Explain why this enhancement would be useful** to most users.

### Pull Requests
The process described here has several goals:
* Maintain code quality.
* Fix problems that are important to users.
* Enable a sustainable system for maintainers to review contributions.

1. **Fork the repo** and create your branch from `main`.
2. **If you've added code that should be tested**, add tests.
3. **If you've changed APIs**, update the documentation.
4. **Ensure the test suite passes**.
5. **Issue that pull request!**

## Development Environment Setup

1. Clone your fork:
   ```bash
   git clone https://github.com/YOUR-USERNAME/LLM.git
   cd LLM
   ```
2. Create a virtual environment and install the requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Read the documentation in `docs/` to familiarize yourself with the architecture, tokenizer, and evaluation pipeline.

## Code Style
* Please try to follow standard PEP-8 style guidelines for Python code.
* Use clear and descriptive variable and function names.
* Comment complex logic, especially around tensor manipulation, attention mechanisms, and custom tokenization rules.

Thank you for contributing!
