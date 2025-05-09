# GitHub Actions Workflows

This directory contains GitHub Actions workflow configurations for implementing CI/CD for the Clinical Trial Analysis project.

## Workflows

### Python Tests (`python-tests.yml`)

This workflow runs on pushes and pull requests to the main branch and performs:

- Setup of Python environment (versions 3.8 and 3.9)
- Installation of project dependencies
- Code linting with flake8
- Unit testing with pytest and coverage reporting
- Upload of coverage reports to Codecov

### Deployment (`deploy.yml`)

This workflow runs on pushes to the main branch and manual triggers, handling:

- Setup of Python environment
- Installation of project dependencies
- Building of documentation/reports
- Deployment to GitHub Pages

## Setup Instructions

1. Ensure your repository has the necessary secrets configured:
   - For Codecov integration: Codecov should automatically detect the repository
   - For GitHub Pages deployment: No additional secrets needed as it uses `GITHUB_TOKEN`

2. The first time you push to GitHub, you may need to enable GitHub Pages in your repository settings:
   - Go to Settings > Pages
   - Set the source to the gh-pages branch

## Customization

You can customize these workflows by:

- Adjusting the Python versions in the test matrix
- Modifying the build commands for documentation
- Adding additional deployment targets
- Configuring notifications for workflow results