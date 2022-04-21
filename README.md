[longformer-master.zip](https://github.com/Nico070707/Nico070707/files/6980808/longformer-master.zip)
[longformer-master.zip](https://github.com/Nico070707/Nico070707/files/6980810/longformer-master.zip)
[AR-Net-master.zip](https://github.com/Nico070707/Nico070707/files/6946571/AR-Net-master.zip)
[AR-Net-master.zip](https://github.com/Nico070707/Nico070707/files/6946579/AR-Net-master.zip)
[dockerfile.github.io-master (1).zip](https://github.com/Nico070707/Nico070707/files/6900367/dockerfile.github.io-master.1.zip)
[transformers-master.zip](https://github.com/Nico070707/Nico070707/files/6898635/transformers-master.zip)
corpus = ["I Like Python because I can build AI applications", 
"I like Python because I can do data analytics","The cat sits on the ground","The cat walks on the sidewalk"]
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
with torch.no_grad():
    model_output =  model(**encoded_input)
    sentence1 = "I like Python because I can build AI applications"
sentence2 = "I like Python because I can do data analytics"
embedding1 = model.encode(sentence1, convert_to_tensor=True)
embedding2 = model.encode(sentence2, convert_to_tensor=True)
cosine_scores = utilpytorch_cos_sin(embedding1, embedding2)
print("Sentence 1:", sentence1)
print("Sentence 2:", sentence2)
print("Similarity score:", cosine_scores.item())
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers:bert-base-nli-mean-tokens")
model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')
with torch.no_grad():
    model_output =  model(**encoded_input)
sentence_embedding = mean_poolinhg(model-output, encoded-input['attention_mask'])
util.pytorch_cos_sin(sentence_embedding[1], sentence_embedding[0]).numpy()[0](0)
0.71667016
model_output =  model(**encoded_input)
sentence_embedding = mean_poolinhg(model-output, encoded-input['attention_mask'])
util.pytorch_cos_sin(sentence_embedding[1], sentence_embedding[0]).numpy()[0](0)
0.71667016
import networkx as nx
        model = SentenceTransformer('distilroberta-base-paraphrase-v1')
embedding = model.encode(sentences, convert-to-tensor=True)
cosine_scores = util.pytorch_cos_sin(embeddings, embeddings)
g = nx.from_numpy_matric(cosine_scores.numpy())
centrality_scores = nx.degree_centrality(g)
most_central_sentence_indices = np.argsort(centrality_scores)
print("\n\nSummary;")
for idx in most_central_sentence_indices[0:4]:
idx(sentences[idx].strip())
        from sentence_transformers import SentenceTransformer, util
import numpy as np
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('distilroberta-base-parapharse-v1')
embedding = model.encode(sentences, convert_to_tensor=True)
cosine_scores = util.pytorch_cos_sin(embedding, embedding).numpy()
print(cosine_scores)
        corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
sentence = "I like Javascript because I can build web applications"
sentence_embedding = model.encode(sentence, convert_to_tensor=True)
cos_scores = util.pytorch_cos_sim(sentence_embedding, corpus_embeddings)[0]
top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]
print("Sentence:",sentence, "\n")
print("Top", top_k, "most similar sentences in corpus:")
def gradient_descent(objective, derivative,
    solution = bounds[: , 0] + rand(len(bound
    for i in range(n_iter):
        gradient = derivative(solution)
        solution = solution - step_size * gr
        solution_eval = objective(solution)
        print('>%d f(%s) + %.5f' %(i, solut
   return [solution, solution_eval]
   objective function
   def objective(x) :
       return x**2.0
        bounds = asarray([[-1.0, 1.0]])
        for idx in top_results(0:top_k]:
 bounds = asarray([[-1.0, 1.0]])
 model_output =  model(**encoded_input)
        corpus = ["I Like Python because I can build AI applications", 
"I like Python because I can do data analytics","The cat sits on the ground","The cat walks on the sidewalk"]
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
print("Similarity score", cosine_scores.item())
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
- name: Perform Scan
      uses: ShiftLeftSecurity/scan-action@master
      env:
        WORKSPACE: ""
        GITHUB_TOKEN: ${{ secrets.the_cat_is_on_the_sidewalk }}
        SCAN_AUTO_BUILD: true
      with:
        output: reports
        def gradient_descent(objective, derivative,
    solution = bounds[: , 0] + rand(len(bound
    for i in range(n_iter):
        gradient = derivative(solution)
        solution = solution - step_size * gr
        solution_eval = objective(solution)
        print('>%d f(%s) + %.5f' %(i, solut
   return [solution, solution_eval]
   objective function
   def objective(x) :
       return x**2.0
        ...
   r_min, r_max = -1.0, 1.0
   inputs = arange(r_min, r_max+0.1, 0.1)
   results = objective(inputs
   ...
   pyplot.plot(inputs, results)
   pyplot.show()
   from numpy import arange
   from matplotlib import pyplot
   def objective(x):
       return x**2.0
   pyplot.plot(inputs, results)
   pyplot.show()
   from numpy import arange
   from matplotlib import pyplot
   def objective(x):
       return x**2.0
   pyplot.plot(inputs, results)
   "show the plot
   pyplot.show()
   from numpy import arange
   from matplotlib import pyplot
   def objective(x):
       return x**2.0
   r_min, r_max = -1.0, 1.0
   inputs = arange(r_min, r_max+0.1, 0.1)
   results = objective(inputs)
   pyplot.plot(inputs, results)
   pyplot.show()
   def derivative(x):
       return x * 2.0
       import numpy as np
        for idx in top_results(0:top_k]:
 bounds = asarray([[-1.0, 1.0]])
   bounds = asarray([[-1.0, 1.0]])
sentence1 = ["I like Python because I can build AI applications", "The cat sits on the ground"]
sentence2 = ["I like Python because I can do data analytics", "The cat walks on the sidewalk"]
corpus = ["I like Python because I can build AI applications",
          "I like Python because I can do data analytics",
          "The cat walks on the sidewalk"]   
sentences = ['This framework generates embedding for each input semtence',
             'Sentences are passed as a list of string, ',
             'The quick brown fox jumps over the lazy dog, ']
sentences = ['This framework generates embeddings for each input sentence',
             'Each embedding has a point in the semantic space',
             'Sentences are passed as a list of string.] 
[fastText-master (2).zip](https://github.com/Nico070707/Nico070707/files/7356135/fastText-master.2.zip)
[apps-main.zip](https://github.com/Nico070707/Nico070707/files/7356141/apps-main.zip)
[human-eval-master.zip](https://github.com/Nico070707/Nico070707/files/7356143/human-eval-master.zip)
[Polygames-main.zip](https://github.com/Nico070707/Nico070707/files/7356144/Polygames-main.zip)
[simclr-master.zip](https://github.com/Nico070707/Nico070707/files/7356145/simclr-master.zip)
[lightly-master.zip](https://github.com/Nico070707/Nico070707/files/7356147/lightly-master.zip)
[gluon-ts-master.zip](https://github.com/Nico070707/Nico070707/files/7356151/gluon-ts-master.zip)
[transformers-master (4).zip](https://github.com/Nico070707/Nico070707/files/7356177/transformers-master.4.zip)
[Uploading transformers-master (5).zip…]()
[longformer-master (1).zip](https://github.com/Nico070707/Nico070707/files/7356181/longformer-master.1.zip)
[Uploading transformers-master (4).zip…]()
[Uploading transformers-master (6).zip…]()
  [pytorch-master.zip](https://github.com/Nico070707/Nico070707/files/7357442/pytorch-master.zip)
[fastText-master (3).zip](https://github.com/Nico070707/Nico070707/files/7357625/fastText-master.3.zip)
[fmin_adam-master.zip](https://github.com/Nico070707/Nico070707/files/7357627/fmin_adam-master.zip)
[longformer-master.zip](https://github.com/Nico070707/Nico070707/files/6980808/longformer-master.zip)
[longformer-master.zip](https://github.com/Nico070707/Nico070707/files/6980810/longformer-master.zip)
[AR-Net-master.zip](https://github.com/Nico070707/Nico070707/files/6946571/AR-Net-master.zip)
[AR-Net-master.zip](https://github.com/Nico070707/Nico070707/files/6946579/AR-Net-master.zip)
[dockerfile.github.io-master (1).zip](https://github.com/Nico070707/Nico070707/files/6900367/dockerfile.github.io-master.1.zip)
[transformers-master.zip](https://github.com/Nico070707/Nico070707/files/6898635/transformers-master.zip)
   bounds = asarray([[-1.0, 1.0]])<head>
  <script src="https://aframe.io/releases/0.8.0/aframe.min.js"></script>
  <script src="https://unpkg.com/aframe-spe-particles-component@^1.0.4/dist/aframe-spe-particles-component.min.js"></script>
</head>
<body>
  <a-entity particles="texture: /assets/blob.png; color: blue; velocity: 0 10 0; velocity-spread: 2 0 2; acceleration: 0 -10 0"></a-entity>
</body>
   <head>
  <script src="https://aframe.io/releases/0.8.0/aframe.min.js"></script>
  <script src="https://unpkg.com/aframe-spe-particles-component@^1.0.4/dist/aframe-spe-particles-component.min.js"></script>
</head>
<body>
  <a-entity particles="texture: /assets/blob.png; color: blue; velocity: 0 10 0; velocity-spread: 2 0 2; acceleration: 0 -10 0"></a-entity>
</body>
for idx in top_results(0:top_k]:
 bounds = asarray([[-1.0, 1.0]])
   model_output =  model(**encoded_input)
        corpus = ["I Like Python because I can build AI applications", 
"I like Python because I can do data analytics","The cat sits on the ground","The cat walks on the sidewalk"]
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
print("Similarity score", cosine_scores.item())
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
- name: Perform Scan
      uses: ShiftLeftSecurity/scan-action@master
      env:
        WORKSPACE: ""
        GITHUB_TOKEN: ${{ secrets.the_cat_is_on_the_sidewalk }}
        SCAN_AUTO_BUILD: true
      with:
        output: reports
        def gradient_descent(objective, derivative,
    solution = bounds[: , 0] + rand(len(bound
    for i in range(n_iter):
        gradient = derivative(solution)
        solution = solution - step_size * gr
        solution_eval = objective(solution)
        print('>%d f(%s) + %.5f' %(i, solut
   return [solution, solution_eval]
   objective function
   def objective(x) :
       return x**2.0
        ...
   r_min, r_max = -1.0, 1.0
   inputs = arange(r_min, r_max+0.1, 0.1)
   results = objective(inputs
   ...
   pyplot.plot(inputs, results)
   pyplot.show()
   from numpy import arange
   from matplotlib import pyplot
   def objective(x):
       return x**2.0
   pyplot.plot(inputs, results)
   pyplot.show()
   from numpy import arange
   from matplotlib import pyplot
   def objective(x):
       return x**2.0
   pyplot.plot(inputs, results)
   "show the plot
   pyplot.show()
   from numpy import arange
   from matplotlib import pyplot
   def objective(x):
       return x**2.0
   r_min, r_max = -1.0, 1.0
   inputs = arange(r_min, r_max+0.1, 0.1)
   results = objective(inputs)
   pyplot.plot(inputs, results)
   pyplot.show()
   def derivative(x):
       return x * 2.0
       import numpy as np
model = SentenceTransformer('distilroberta-base-paraphrase-v1)
embeddings = model.encode(sentences, convert_to_tensor=True)
cosine_scores = util.pytorch_cos_sin(embeddings, embeddings).numpy()
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
sentence = "I like Javascript because I can build web applications"
sentence_embedding = model.encode(sentence, convert_to_tensor=True)
cos_scores = util.pytorch_cos_sim(sentence_embedding, corpus_embeddings)[0]
top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]
print("Sentence:",sentence, "\n")
print("Top", top_k, "most similar sentences in corpus:")
for idx in top_results(0:top_k]:
 bounds = asarray([[-1.0, 1.0]])
   bounds = asarray([[-1.0, 1.0]])
sentence1 = ["I like Python because I can build AI applications", "The cat sits on the ground"]
sentence2 = ["I like Python because I can do data analytics", "The cat walks on the sidewalk"]
corpus = ["I like Python because I can build AI applications",
          "I like Python because I can do data analytics",
          "The cat walks on the sidewalk"]   
sentences = ['This framework generates embedding for each input semtence',
             'Sentences are passed as a list of string, ',
             'The quick brown fox jumps over the lazy dog, ']
sentences = ['This framework generates embeddings for each input sentence',
             'Each embedding has a point in the semantic space',
             'Sentences are passed as a list of string.]          
[fastText-master (2).zip](https://github.com/Nico070707/Nico070707/files/7356135/fastText-master.2.zip)
[apps-main.zip](https://github.com/Nico070707/Nico070707/files/7356141/apps-main.zip)
[human-eval-master.zip](https://github.com/Nico070707/Nico070707/files/7356143/human-eval-master.zip)
[Polygames-main.zip](https://github.com/Nico070707/Nico070707/files/7356144/Polygames-main.zip)
[simclr-master.zip](https://github.com/Nico070707/Nico070707/files/7356145/simclr-master.zip)
[lightly-master.zip](https://github.com/Nico070707/Nico070707/files/7356147/lightly-master.zip)
[gluon-ts-master.zip](https://github.com/Nico070707/Nico070707/files/7356151/gluon-ts-master.zip)
[transformers-master (4).zip](https://github.com/Nico070707/Nico070707/files/7356177/transformers-master.4.zip)
[Uploading transformers-master (5).zip…]()
[longformer-master (1).zip](https://github.com/Nico070707/Nico070707/files/7356181/longformer-master.1.zip)
[Uploading transformers-master (4).zip…]()
[Uploading transformers-master (6).zip…]()
  [pytorch-master.zip](https://github.com/Nico070707/Nico070707/files/7357442/pytorch-master.zip)
[fastText-master (3).zip](https://github.com/Nico070707/Nico070707/files/7357625/fastText-master.3.zip)
[fmin_adam-master.zip](https://github.com/Nico070707/Nico070707/files/7357627/fmin_adam-master.zip)
[longformer-master.zip](https://github.com/Nico070707/Nico070707/files/6980808/longformer-master.zip)
[longformer-master.zip](https://github.com/Nico070707/Nico070707/files/6980810/longformer-master.zip)
[AR-Net-master.zip](https://github.com/Nico070707/Nico070707/files/6946571/AR-Net-master.zip)
[AR-Net-master.zip](https://github.com/Nico070707/Nico070707/files/6946579/AR-Net-master.zip)
[dockerfile.github.io-master (1).zip](https://github.com/Nico070707/Nico070707/files/6900367/dockerfile.github.io-master.1.zip)
[transformers-master.zip](https://github.com/Nico070707/Nico070707/files/6898635/transformers-master.zip)
   bounds = asarray([[-1.0, 1.0]])
[fastText-master (2).zip](https://github.com/Nico070707/Nico070707/files/7356135/fastText-master.2.zip)
[apps-main.zip](https://github.com/Nico070707/Nico070707/files/7356141/apps-main.zip)
[human-eval-master.zip](https://github.com/Nico070707/Nico070707/files/7356143/human-eval-master.zip)
[Polygames-main.zip](https://github.com/Nico070707/Nico070707/files/7356144/Polygames-main.zip)
[simclr-master.zip](https://github.com/Nico070707/Nico070707/files/7356145/simclr-master.zip)
[lightly-master.zip](https://github.com/Nico070707/Nico070707/files/7356147/lightly-master.zip)
[gluon-ts-master.zip](https://github.com/Nico070707/Nico070707/files/7356151/gluon-ts-master.zip)
[transformers-master (4).zip](https://github.com/Nico070707/Nico070707/files/7356177/transformers-master.4.zip)
[Uploading transformers-master (5).zip…]()
[longformer-master (1).zip](https://github.com/Nico070707/Nico070707/files/7356181/longformer-master.1.zip)
[Uploading transformers-master (4).zip…]()
[Uploading transformers-master (6).zip…]()
  [pytorch-master.zip](https://github.com/Nico070707/Nico070707/files/7357442/pytorch-master.zip)
[fastText-master (3).zip](https://github.com/Nico070707/Nico070707/files/7357625/fastText-master.3.zip)
[fmin_adam-master.zip](https://github.com/Nico070707/Nico070707/files/7357627/fmin_adam-master.zip)
[Java-Machine-Learning-master.zip](https://github.com/Nico070707/Nico070707/files/7357630/Java-Machine-Learning-master.zip)
[Uploading Python-master.zip…]()
- name: Cache
  uses: actions/cache@v3.0.2
  with:
    # A list of files, directories, and wildcard patterns to cache and restore
    path: 
    # An explicit key for restoring and saving the cache
    key: 
    # An ordered list of keys to use for restoring the cache if no cache hit occurred for key
    restore-keys: # optional
    # The chunk size used to split up large files during upload, in bytes
    upload-chunk-size: # optional

