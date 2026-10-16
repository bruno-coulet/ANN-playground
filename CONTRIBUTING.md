# Contributing to ANN Playground
Thank you for contributing to ANN Playground.

This document describes the development workflow, coding standards, testing requirements, and Git process used in this repository.

---

# Development Workflow
All development must be done from feature branches based on develop.

Never work directly on main.

Workflow:

main 
↓ 
develop 
↓ 
feature/* 
↓ 
develop 
↓ 
main

Example:

git checkout develop
git pull

git checkout -b feature/my-new-feature

---

# Installing the Project

Clone the repository:

git clone <repository-url>
cd ANN-playground

Install dependencies:

uv sync --all-groups

---

# Running the Application
Launch Streamlit:

uv run streamlit run src/ann_playground/apps/app.py

---

# Running Tests

Run all tests:

uv run pytest
Run coverage:

uv run pytest \
  --cov=src \
  --cov-report=term-missing

---

# Linting

Check code quality:

uv run ruff check .
Check formatting:

uv run ruff format --check .

---

# Before Every Commit

Always run:

uv run ruff check .

uv run ruff format .

uv run pytest

Or execute:

uv run pre-commit run --all-files

---

# Pull Requests
Before opening a Pull Request:

    * tests must pass
    * CI must be green
    * Docker workflow must pass
    * coverage must remain above the project threshold


Recommended process:

git checkout develop

git checkout -b feature/my-feature

# work

git push origin feature/my-feature


Create a Pull Request:

feature/my-feature
        ↓
develop

---

# Coding Standards

    * Python 3.12+
    * Ruff for linting
    * Ruff Format for formatting
    * Type hints encouraged
    * Unit tests required for new logic

---

# Test Structure

tests/

├── unit/ ├── integration/ └── smoke/

Unit tests:

    * single function
    * isolated behavior


Integration tests:

    * multiple components working together


Smoke tests:

    * import validation
    * application startup validation

---

# Coverage Policy

Current coverage target:

50%+

All new code should include tests.

Coverage regressions should be avoided.

---

# Branch Protection

The main branch is protected.

Changes must be merged through Pull Requests only.

Direct pushes to main are prohibited.

