[dockerfile.github.io-master (1).zip](https://github.com/Nico070707/Nico070707/files/6900367/dockerfile.github.io-master.1.zip)
[transformers-master.zip](https://github.com/Nico070707/Nico070707/files/6898635/transformers-master.zip)
- 👋 Hi, I’m @Nico070707
- 👀 I’m interested in real Data Humanity to build a better world. A real dream could begin a real life
- 🌱 I’m currently learning DataScience
- 💞️ I’m looking to collaborate on Project for humanity with code
- 📫 How to reach me https://www.linkedin.com/in/nicolas-bourne-635a0033/
corpus = ["I Like Python because I can build AI applications", 
"I like Python because I can do data analytics","The cat sits on the ground","The cat walks on the sidewalk"]
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
sentence = "I like Javascript because I can build web applications"
sentence_embedding = model.encode(sentence, convert_to_tensor=True)
cos_scores = util.pytorch_cos_sim(sentence_embedding, corpus_embeddings)[0]
top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]
print("Sentence:",sentence, "\n")
print("Top", top_k, "most similar sentences in corpus:")
for idx in top_results(0:top_k]:
print(corpus[idx], "(Score: %.4f)" % (cos_scores[idx]))
sentences = ['This framework generates embeddings for each input sentence','Sentence are passed as a list of string, ','The quick brown for jumps over the lazy dog.') 
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
encoded input + tokenizer(sentences, padding=True, truncation=True, max_lengh=128, return_tensors='pt')
with torch.no_grad{) :
model_ouput = model(**encoded_input
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
util.pytorch_cos_sim(sentence_embeddings[1], sentence_embeddings[0],numpy()[0][0]
model = SentenceTransformer('distilroberta-base-paraphrase-v1')
embeddings = model.encode(sentences, convert_to_tensor=True)
cosine_scores = util.pytorch_cos_sim(embeddings,embeddings)
g = nx.from_numpy_matrix(cosine_scores.numpy())
centrality_scores = nx.degree_centrality(g)
most_central_sentence_indices = np.argsort(centrality_scores)
print("\n\nSummary:")
sentence 1 = "I Like Python because I can build AI applications"
sentence 2 = "I Like Python because I can do data analytics"
embedding1 = model.encode(sentence1, convert_to_tensor=True)
embedding2 = model.encore(sentence2, convert_to_tensor=True)
cosine scores = util.pytorch_cos_sim(embedding1, embedding2)
print("Sentence 1:", sentence1)
print("Sentence 2:", sentence2)
print("Similarity score", cosine_scores.itrem())
sentence1 = ["I Like Python because I can build AI applications, "The cat sits on the ground"]
sentence2 = [I Like Python because I can do data anaytics", "The cat walks on the sidewalk"]
embedding1 = model.encode(sentence1, convert_to_tensor=True)
embedding2 = model.encore(sentence2, convert_to_tensor=True)
cosine scores= util.pytorch_cos_sim(embedding1, embedding2)
for i in range(len(sentence1))
for j in range(len(sentence2))
print("Sentence 1:", sentences1[i])
print("Sentence 2:", sentences2[j])
print("similarity Score:", cosine_scores[i][j].item())
print()
OPENSHIFT_SERVER: ${{ secret.OPENSHIFT_SERVER }}
OPENSHIFT_TOKEN: ${{ secret.OPENSHIFT_TOKEN }}
name: Python package
on: [push]
jobs:
build:
runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.6, 3.7, 3.8, 3.9]
steps:
      - uses: actions/checkout@v2
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v2
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install flake8 pytest
          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
      - name: Lint with flake8
        run: |
          # stop the build if there are Python syntax errors or undefined names
          flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
          # exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
          flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
      - name: Test with pytest
        run: |
          pytes
          name: Python package
on: [push]
jobs:
build:
runs-on: ubuntu-latest
    strategy:
      # You can use PyPy versions in python-version.
      # For example, pypy2 and pypy3
      matrix:
        python-version: [2.7, 3.6, 3.7, 3.8, 3.9]
steps:
      - uses: actions/checkout@v2
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v2
        with:
          python-version: ${{ matrix.python-version }}
      # You can test your matrix by printing the current Python version
      - name: Display Python version
        run: python -c "import sys; print(sys.version)"
        steps:
- uses: actions/checkout@v2
- name: Set up Python
  uses: actions/setup-python@v2
  with:
    python-version: '3.x'
- name: Install dependencies
  run: python -m pip install --upgrade pip setuptools wheel
  steps:
- uses: actions/checkout@v2
- name: Set up Python
  uses: actions/setup-python@v2
  with:
    python-version: '3.x'
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    steps:
- uses: actions/checkout@v2
- name: Setup Python
  uses: actions/setup-python@v2
  with:
    python-version: '3.x'
- name: Cache pip
  uses: actions/cache@v2
  with:
    # This path is specific to Ubuntu
    path: ~/.cache/pip
    # Look to see if there is a cache hit for the corresponding requirements file
    key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
    restore-keys: |
      ${{ runner.os }}-pip-
      ${{ runner.os }}-
- name: Install dependencies
  run: pip install -r requirements.txt
  steps:
- uses: actions/checkout@v2
- name: Set up Python
  uses: actions/setup-python@v2
  with:
    python-version: '3.x'
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    pip install -r requirements.txt
- name: Test with pytest
  run: |
    pip install pytest
    pip install pytest-cov
    pytest tests.py --doctest-modules --junitxml=junit/test-results.xml --cov=com --cov-report=xml --cov-report=html
    steps:
- uses: actions/checkout@v2
- name: Set up Python
  uses: actions/setup-python@v2
  with:
    python-version: '3.x'
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    pip install -r requirements.txt
- name: Test with pytest
  run: |
    pip install pytest
    pip install pytest-cov
    pytest tests.py --doctest-modules --junitxml=junit/test-results.xml --cov=com --cov-report=xml --cov-report=html
    name: Python package
on: [push]
jobs:
  build:
runs-on: ubuntu-latest
    strategy:
      matrix:
        python: [3.7, 3.8, 3.9]
steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: ${{ matrix.python }}
      - name: Install Tox and any other packages
        run: pip install tox
      - name: Run Tox
        # Run tox using the version of Python in `PATH`
        run: tox -e py
        name: Python package
on: [push]
jobs:
  build:
runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.6, 3.7, 3.8, 3.9]
steps:
      - uses: actions/checkout@v2
      - name: Setup Python # Set Python version
        uses: actions/setup-python@v2
        with:
          python-version: ${{ matrix.python-version }}
      # Install pip and pytest
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pytest
      - name: Test with pytest
        run: pytest tests.py --doctest-modules --junitxml=junit/test-results-${{ matrix.python-version }}.xml
      - name: Upload pytest test results
        uses: actions/upload-artifact@v2
        with:
          name: pytest-results-${{ matrix.python-version }}
          path: junit/test-results-${{ matrix.python-version }}.xml
        # Use always() to always run this step to publish test results when there are test failures
        if: ${{ always() }}
        # This workflow uses actions that are not certified by GitHub.
# They are provided by a third-party and are governed by
# separate terms of service, privacy policy, and support
# documentation.
name: Upload Python Package
on:
  release:
    types: [published]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.x'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build
      - name: Build package
        run: python -m build
      - name: Publish package
        uses: pypa/gh-action-pypi-publish@27b31702a0e7fc50959f5ad993c78deac1bdfc29
        with:
          user: __token__
          password: ${{ secrets.PYPI_API_TOKEN }}
APP_PORT: ""
        name: Docker Image CI
        on:
        push:
        branches: [ main ]
        pull_request:
        branches: [ main ]
        jobs:
        build:
        runs-on: ubuntu-latest
steps:
    - uses: actions/checkout@v2
    - name: Build the Docker image
      run: docker build . --file Dockerfile --tag my-image-name:$(date +%s)
OPENSHIFT_NAMESPACE: ""
APP_NAME: ""
TAG: ""
branches: [ main ]
openshift-ci-cd:
name: Build and deploy OpenShift
runs-on: ubuntu-18.04
environment: production
name: Build and push Docker images
uses: docker/build-push-action@v2.6.1
name: Log into registry ${{ env.REGISTRY }}
 if: github.event_name != 'pull_request'
 uses: docker/login-action@28218f9b04b4f3f62068d7b6ce6ca5b26e35336c
with:
registry: ${{ env.REGISTRY }}
username: ${{ github.actor }}
password: ${{ secrets.GITHUB_TOKEN }}
name: Extract Docker metadata
id: meta
uses: docker/metadata-action@98669ae865ea3cffbcbaa878cf57c20bbf1c6c38
with:
images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
name: Build and push Docker image
uses: docker/build-push-action@ad44023a93711e3deb337508980b4b5e9bcdc5dc
with:
context: .
push: ${{ github.event_name != 'pull_request' }}
tags: ${{ steps.meta.outputs.tags }}
labels: ${{ steps.meta.outputs.labels }}
outputs:
ROUTE: ${{ steps.deploy-and-expose.outputs.route }}
SELECTOR: ${{ steps.deploy-and-expose.outputs.selector }}
steps:
name: OpenShift
REGISTRY: quay.io/Nico070707
REGISTRY_USER: Nico070707
REGISTRY_PASSWORD: ${{ secrets.REGISTRY_PASSWORD }}
push:
branches: [main]
[dockerfile.github.io-master.zip](https://github.com/Nico070707/Nico070707/files/6899915/dockerfile.github.io-master.zip)
jobs: 
openshift-ci-cd:
name: Build and deploy to OpenShift
runs-on: ubuntu-18.04
environment: production
schedule:
- cron: '20 14 * * *'
push:
branches: [ main ]
# Publish semver tags as releases.
tags: [ 'v*.*.*' ]
pull_request:
branches: [ main ]
REGISTRY: ghcr.io
cron: '20 14 * * *'
push:
branches: [ main ]
name: Docker Metadata action
uses: docker/metadata-action@v3.4.1
name: Docker Login
uses: docker/login-action@v1.10.0
name: Build and push Docker images
uses: docker/build-push-action@v2.6.1
IMAGE_NAME: ${{ github.repository }}
name: Log into registry ${{ env.REGISTRY }}
if: github.event_name != 'pull_request'
uses: docker/login-action@28218f9b04b4f3f62068d7b6ce6ca5b26e35336c
with:
registry: ${{ env.REGISTRY }}
username: ${{ github.actor }}
password: ${{ secrets.GITHUB_TOKEN }}
name: Extract Docker metadata
name: Docker Metadata action
uses: docker/metadata-action@v3.4.1
${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
# https://github.com/docker/build-push-action
name: Build and push Docker image
uses: docker/build-push-action@ad44023a93711e3deb337508980b4b5e9bcdc5dc
context: .
push: ${{ github.event_name != 'pull_request' }}
tags: ${{ steps.meta.outputs.tags }}
labels: ${{ steps.meta.outputs.labels }}
name: Hello world action
    with: # Set the secret as an input
      super_secret: ${{ secrets.SuperSecret }}
    env: # Or as an environment variable
      super_secret: ${{ secrets.SuperSecret }}
      name: Checkout repository
        uses: actions/checkout@v2
        name: Log into registry ${{ env.REGISTRY }}
        if: github.event_name != 'pull_request'
        uses: docker/login-action@28218f9b04b4f3f62068d7b6ce6ca5b26e35336c
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
name: Extract Docker metadata
        id: meta
        uses: docker/metadata-action@98669ae865ea3cffbcbaa878cf57c20bbf1c6c38
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
name: Build and push Docker image
        uses: docker/build-push-action@ad44023a93711e3deb337508980b4b5e9bcdc5dc
        with:
          context: .
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
         - name: Docker Login
  uses: docker/login-action@v1.10.0
$ git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
$ cd YOUR-REPOSITORY
alert("Hello, World!");
$ npm init
  ...
  package name: @YOUR-USERNAME/YOUR-REPOSITORY
  ...
  test command: exit 0
  ... 
  $ npm install
$ git add index.js package.json package-lock.json
$ git commit -m "initialize npm package"
$ git push
name: Node.js Package
on:
  release:
    types: [created]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-node@v2
        with:
          node-version: 12
      - run: npm ci
      - run: npm test
publish-gpr:
    needs: build
    runs-on: ubuntu-latest
    permissions:
      packages: write
      contents: read
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-node@v2
        with:
          node-version: 12
          registry-url: https://npm.pkg.github.com/
      - run: npm ci
      - run: npm publish
        env:
          NODE_AUTH_TOKEN: ${{secrets.GITHUB_TOKEN}}
          @NICO070707:registry=https://npm.pkg.github.com
          $ git add .github/workflows/release-package.yml
# Also add the file you created or edited in the previous step.
$ git add .npmrc or package.json
$ git commit -m "workflow to publish package"
$ git push
name: Log into registry ${{ env.REGISTRY }}
        if: github.event_name != 'pull_request
        - name: Docker Login
  uses: docker/login-action@v1.10.0
with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
          - name: Docker Metadata action
  uses: docker/metadata-action@v3.4.1
 name: Extract Docker metadata
        id: meta
        uses: docker/metadata-action@98669ae865ea3cffbcbaa878cf57c20bbf1c6c38
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
- name: Build and push Docker images
  uses: docker/build-push-action@v2.6.1
  name: Build and push Docker image
        uses: docker/build-push-action@ad44023a93711e3deb337508980b4b5e9bcdc5dc
        with:
          context: .
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          name: Docker
          on:
  schedule:
    - cron: '19 0 * * *'
  push:
    branches: [ main ]
    tags: [ 'v*.*.*' ]
  pull_request:
    branches: [ main ]
    env:
    REGISTRY: ghcr.io
    IMAGE_NAME: ${{ github.repository }}
    jobs:
  build:
runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
steps:
      - name: Checkout repository
        uses: actions/checkout@v2
        name: Log into registry ${{ env.REGISTRY }}
        if: github.event_name != 'pull_request'
        uses: docker/login-action@28218f9b04b4f3f62068d7b6ce6ca5b26e35336c
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
          name: Extract Docker metadata
        id: meta
        uses: docker/metadata-action@98669ae865ea3cffbcbaa878cf57c20bbf1c6c38
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          name: Build and push Docker image
        uses: docker/build-push-action@ad44023a93711e3deb337508980b4b5e9bcdc5dc
        with:
          context: .
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          name: Docker Image CI
          on:
 name: OpenShift
 env:
  REGISTRY: quay.io/NICO070707
  REGISTRY_USER: NICO070707
  REGISTRY_PASSWORD: ${{ secrets.REGISTRY_PASSWORD }}
  OPENSHIFT_SERVER: ${{ secrets.OPENSHIFT_SERVER }}
  OPENSHIFT_TOKEN: ${{ secrets.OPENSHIFT_TOKEN }}
 APP_PORT: ""
 OPENSHIFT_NAMESPACE: ""
 APP_NAME: ""
  TAG: ""
  on:
  push:
 branches: [ main ]
 jobs:
  openshift-ci-cd:
    name: Build and deploy to OpenShift
    runs-on: ubuntu-18.04
    environment: production
outputs:
        ROUTE: ${{ steps.deploy-and-expose.outputs.route }}
        SELECTOR: ${{ steps.deploy-and-expose.outputs.selector }}
steps:
    - name: Check if secrets exists
      uses: actions/github-script@v3
      with:
        script: |
          const secrets = {
            REGISTRY_PASSWORD: `${{ secrets.REGISTRY_PASSWORD }}`,
            OPENSHIFT_SERVER: `${{ secrets.OPENSHIFT_SERVER }}`,
            OPENSHIFT_TOKEN: `${{ secrets.OPENSHIFT_TOKEN }}`,
          };
          const missingSecrets = Object.entries(secrets).filter(([ name, value ]) => {
            if (value.length === 0) {
              core.warning(`Secret "${name}" is not set`);
              return true;
            }
            core.info(`✔️ Secret "${name}" is set`);
            return false;
            });
          if (missingSecrets.length > 0) {
            core.setFailed(`❌ At least one required secret is not set in the repository. \n` +
              "You can add it using:\n" +
              "GitHub UI: https://docs.github.com/en/actions/reference/encrypted-secrets#creating-encrypted-secrets-for-a-repository \n" +
              "GitHub CLI: https://cli.github.com/manual/gh_secret_set \n" +
              "Also, refer to https://github.com/redhat-actions/oc-login#getting-started-with-the-action-or-see-example");
          }
          else {
            core.info(`✅ All the required secrets are set`);
          }
    - uses: actions/checkout@v2
- name: Determine app name
      if: env.APP_NAME == ''
      run: |
        echo "APP_NAME=$(basename $PWD)" | tee -a $GITHUB_ENV
    - name: Determine tag
      if: env.TAG == ''
      run: |
        echo "TAG=${GITHUB_SHA::7}" | tee -a $GITHUB_ENV
    # https://github.com/redhat-actions/buildah-build#readme
    - name: Build from Dockerfile
      id: image-build
      uses: redhat-actions/buildah-build@v2
      with:
        image: ${{ env.APP_NAME }}
        tags: ${{ env.TAG }}
        # If you don't have a dockerfile, see:
        # https://github.com/redhat-actions/buildah-build#scratch-build-inputs
        # Otherwise, point this to your Dockerfile relative to the repository root.
        dockerfiles: |
          ./Dockerfile
    # https://github.com/redhat-actions/push-to-registry#readme
    - name: Push to registry
      id: push-to-registry
      uses: redhat-actions/push-to-registry@v2
      with:
        image: ${{ steps.image-build.outputs.image }}
        tags: ${{ steps.image-build.outputs.tags }}
        registry: ${{ env.REGISTRY }}
        username: ${{ env.REGISTRY_USER }}
        password: ${{ env.REGISTRY_PASSWORD }}
- name: Log in to OpenShift
      uses: redhat-actions/oc-login@v1
      with:
        openshift_server_url: ${{ env.OPENSHIFT_SERVER }}
        openshift_token: ${{ env.OPENSHIFT_TOKEN }}
        insecure_skip_tls_verify: true
        namespace: ${{ env.OPENSHIFT_NAMESPACE }}
- name: Create and expose app
      id: deploy-and-expose
      uses: redhat-actions/oc-new-app@v1
      with:
        app_name: ${{ env.APP_NAME }}
        image: ${{ steps.push-to-registry.outputs.registry-path }}
        namespace: ${{ env.OPENSHIFT_NAMESPACE }}
        port: ${{ env.APP_PORT }}
- name: View application route
     run: |
        [[ -n ${{ env.ROUTE }} ]]
        on:
  schedule:
    - cron: '19 0 * * *'
  push:
    branches: [ main ]
    tags: [ 'v*.*.*' ]
  pull_request:
    branches: [ main ]
    IMAGE_NAME: ${{ github.NICO070707 }}
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
      - name: Checkout repository
        uses: actions/checkout@v2
        - name: Docker Login
  uses: docker/login-action@v1.10.0
   registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
          - name: Docker Metadata action
  uses: docker/metadata-action@v3.4.1
   images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
   - name: Build and push Docker images
  uses: docker/build-push-action@v2.6.1
context: .
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
                  [transformers-master (3).zip](https://github.com/Nico070707/Nico070707/files/6924444/transformers-master.3.zip)





      




