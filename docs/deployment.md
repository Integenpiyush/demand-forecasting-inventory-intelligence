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

The root `.streamlit/config.toml` provides the theme. The app uses aggregate dashboard artifacts only; the Kaggle raw files and row-level/per-store outputs remain local and gitignored. Community Cloud watches the chosen GitHub branch for updates. See [Streamlit's deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) for the current UI flow.
