# CI/CD Setup Documentation

## Overview

This document describes the Continuous Integration and Continuous Deployment (CI/CD) setup for the Clinical Trial Analysis project. The implementation uses GitHub Actions to automate testing, linting, and deployment processes.

## Configuration Files

### .gitignore

A `.gitignore` file has been added to the project root to exclude unnecessary files from version control, including:

- Python cache files and bytecode
- Virtual environment directories
- IDE-specific files
- OS-specific files
- Test coverage reports
- Build artifacts

This ensures that only essential code and documentation are tracked in the repository.

### GitHub Actions Workflows

Two GitHub Actions workflow files have been created in the `.github/workflows/` directory:

#### 1. Python Tests (`python-tests.yml`)

This workflow runs automatically on pushes and pull requests to the main branch and performs:

- Setup of Python environment (versions 3.8 and 3.9)
- Installation of project dependencies
- Code linting with flake8
- Unit testing with pytest and coverage reporting
- Upload of coverage reports to Codecov

#### 2. Deployment (`deploy.yml`)

This workflow runs on pushes to the main branch and manual triggers, handling:

- Setup of Python environment
- Installation of project dependencies
- Building of documentation/reports
- Deployment to GitHub Pages

## Setup Instructions

1. Push the project to GitHub to activate the workflows
2. Enable GitHub Pages in your repository settings:
   - Go to Settings > Pages
   - Set the source to the gh-pages branch
3. For Codecov integration, connect your GitHub repository to Codecov

## Customization

You can customize these workflows by:

- Adjusting the Python versions in the test matrix
- Modifying the build commands for documentation
- Adding additional deployment targets
- Configuring notifications for workflow results

## Benefits

This CI/CD setup provides several benefits:

- Automated testing ensures code quality
- Continuous deployment keeps documentation up-to-date
- Standardized build process improves reproducibility
- Reduced manual effort for repetitive tasks