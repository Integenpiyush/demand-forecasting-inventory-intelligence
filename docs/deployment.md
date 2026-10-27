# Deployment and GitHub handoff

## GitHub

The project repository target is `https://github.com/Integenpiyush/demand-forecasting-inventory-intelligence`. The local GitHub CLI must be authenticated with an account that can push to that repository. To authenticate through GitHub's device flow, run:

```sh
gh auth login --hostname github.com --git-protocol https --web
```

Complete the browser/device-code steps shown by GitHub, then verify with `gh auth status`. Never paste a personal access token or device code into chat.

## Streamlit Community Cloud

The app is deployed at [demand-forecasting-inventory-intelligence-bk8hrfzrqmicjhie9njz.streamlit.app](https://demand-forecasting-inventory-intelligence-bk8hrfzrqmicjhie9njz.streamlit.app/). It runs on Python 3.12 with `dashboard/app.py` as the entry point.

To recreate the deployment:

1. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub.
2. Create an app from `Integenpiyush/demand-forecasting-inventory-intelligence` on branch `main`.
3. Set the app entrypoint to `dashboard/app.py` and Python to 3.12.
4. Leave secrets empty and deploy.

The root `.streamlit/config.toml` provides the theme. The dashboard requires the checked-in aggregate metrics plus the 50-store validation/test quantile artifact for store plots and configurable empirical lead-time scenarios. It does not need raw Kaggle source files or training weights. The sampled forecast artifact includes the target Sales values for its validation/test dates, so it is published with the app; raw train/store CSVs remain local and gitignored. Community Cloud watches the chosen GitHub branch for updates. See [Streamlit's deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) for the current UI flow.
