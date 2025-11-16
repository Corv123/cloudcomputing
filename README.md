# Fake News Detector - AWS Cloud Deployment

## Project Structure

- **docs/**: All documentation files (.md)
- **scripts/**: All PowerShell scripts (.ps1)
- **fakenews/**: Main application code
- **lambda_function.py**: AWS Lambda function

## Quick Links

### Documentation
- [Deployment Complete](docs/DEPLOYMENT_COMPLETE.md) - Full deployment summary
- [Lambda Fix Status](docs/LAMBDA_FIX_STATUS.md) - ML dependencies fix guide
- [API Fix Summary](docs/API_FIX_SUMMARY.md) - API Gateway fixes
- [AWS Console Sign-In](docs/AWS_CONSOLE_SIGNIN.md) - Console access guide
- [AWS Credentials Info](docs/AWS_CREDENTIALS_INFO.md) - Credentials management

### Scripts
- `scripts/fix_ml_dependencies_linux.ps1` - Build Linux-compatible ML dependencies (requires Docker)
- `scripts/fix_ml_dependencies_no_docker.ps1` - CloudShell method (no Docker needed)
- `scripts/attach_ml_layer.ps1` - Attach ML layer after downloading from CloudShell
- `scripts/fix_lambda_dependencies.ps1` - Fix web scraping dependencies
- `scripts/setup_phase*.ps1` - Phase setup scripts (3-10)

## Getting Started

1. See docs/DEPLOYMENT_COMPLETE.md for full deployment status
2. See docs/LAMBDA_FIX_STATUS.md if you need to fix ML dependencies
3. All scripts are in `scripts/` folder - run from project root: `.\scripts\script-name.ps1`

## Current Status

âœ… Infrastructure deployed
âœ… API Gateway configured
âœ… Frontend deployed to CloudFront
âš ï¸ ML dependencies need Linux-compatible build (see docs/LAMBDA_FIX_STATUS.md)
