## Adding a new notebook checklist

[Contributing a notebook on confluence](https://graphcore.atlassian.net/wiki/spaces/PM/pages/3242393645/Contributing+a+notebook) contains full instructions, if you have questions please ask on #internal-paperspace-graphcore by tagging @aie-paperspace, here is the checklist:

- [ ] Your notebook should exist and have been landed in another repository (examples or optimum-graphcore) - (this can be skipped in rare instances)
    - [ ] Make it configurable by environment variables [see notebook technical guidelines](https://graphcore.atlassian.net/wiki/spaces/PM/pages/3098345498/Writing+a+Paperspace+notebook#Reading-configuration-in-notebooks)
    - [ ] Make sure it has a compliant title [See notebook content guidelines](https://graphcore.atlassian.net/wiki/spaces/PM/pages/3094381247/Notebooks+guidelines)
- [ ] Add an entry to `.github/deployment-configs/` to copy it over - this will create a PR with files automatically copied over for you. You will need to merge it into your branch (simply click the merge button on the automated PR, it will do the right thing). The config format is defined in [graphcore/paperspace-automation - deployment](https://github.com/graphcore/paperspace-automation/tree/main/deployment)
    - [ ] remove READMEs (they do not render on Paperspace)
    - [ ] make sure appropriate licence is included (MIT: no action needed, other licenses need to be added to folder)
    - [ ] Once the file structure matches what you want, merge the PR that was automatically created, ask for feedback from #internal-paperspace-graphcore if you are not sure about the file structure to adopt
- [ ] Generate a short link [confluence instructions](https://graphcore.atlassian.net/wiki/spaces/PM/pages/3219194169/Generating+a+short+URL+for+a+Paperspace+notebook) and add a ROG button on the notebook
- [ ] Make minimal Paperspace specific changes
  - [ ] remove relative links in Markdown text (unsupported on Paperspace). Either use full URLs to github, or print the relative path as code e.g. "... the notebook at `../tutorial3/walkthrough.ipynb`"
  - [ ] unpin matplotlib, pandas and numpy requirements
  - [ ] Make sure the graphcore-cloud-tools logger is added
- [ ] Add an entry to test the notebook in `.gradient/notebooks-tests.yaml`
- [ ] Add the notebook to the `README_first.ipynb`
- [ ] Dataset, checkpoint, poplar cache upload ([dataset management - confluence](https://graphcore.atlassian.net/wiki/spaces/PM/pages/3226206448/Paperspace+dataset+management))
  - [ ] Upload any required datasets, checkpoints and caches to `/a/scratch/ai-public-datasets`
  - [ ] Symlink any new datasets by editing `.gradient/symlink_config.json`, symlinks are from the read only `PUBLIC_DATASETS_DIR` to the appropriate read/write equivalent `DATASETS_DIR`, `CHECKPOINT_DIR`, `HF_DATASETS_CACHE`, etc... (see `setup.sh` for possibilities)
  - [ ] if you need new environment variables defined, make changes to `setup.sh`
  - [ ] Download files generated during the CI run which will be cached from AWS ([download - AWS data - confluence](https://graphcore.atlassian.net/wiki/spaces/PM/pages/3226206448/Paperspace+dataset+management#Accessing-artefacts-generated-in-Github-actions))
  - [ ] Upload datasets, checkpoints and other caches to gradient datasets
  - [ ] If you have created a new dataset, add corresponding entry to `.gradient/settings.yaml` 
- [ ] Test on Paperspace: you can trigger a test on Paperspace by using the "workflow dispatch" trigger in Github Actions and changing "Local" to "Paperspace" (you can also do this manually)

Once all this is done, or steps have been agreed to be unnecessary, merge this PR 🙂

Don't forget to tell #internal-paperspace-graphcore that the PR has landed
