# Setting up GitHub Pages for ChronosLink

Due to permission limitations with GitHub Actions, you'll need to manually enable GitHub Pages for this repository before the workflow can deploy your site.

## Steps to Enable GitHub Pages

1. Go to your repository on GitHub.com
2. Click on **Settings** (tab at the top of the repository)
3. Scroll down to the **Pages** section in the left sidebar
4. Under **Source**, select **GitHub Actions** from the dropdown menu
5. Save your changes

## After Enabling GitHub Pages

Once GitHub Pages is enabled, you can:

1. Push changes to the `main` branch or manually run the workflow
2. The GitHub Actions workflow will build and deploy your site
3. After successful deployment, your site will be available at `https://<username>.github.io/<repository-name>/`

## Troubleshooting

If you encounter errors like "Resource not accessible by integration" or "Get Pages site failed", it's likely because GitHub Pages hasn't been manually enabled in your repository settings.

The GitHub Actions workflow cannot automatically enable GitHub Pages due to permission restrictions, so this manual step is necessary.

## Additional Resources

- [GitHub Pages documentation](https://docs.github.com/en/pages)
- [Configuring a publishing source for your GitHub Pages site](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site) 