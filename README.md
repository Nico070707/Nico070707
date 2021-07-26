- 👋 Hi, I’m @Nico070707
- 👀 I’m interested in real Data Humanity to build a better world. A real dream could begin a real life
- 🌱 I’m currently learning DataScience
- 💞️ I’m looking to collaborate on Project for humanity with code
- 📫 How to reach me https://www.linkedin.com/in/nicolas-bourne-635a0033/

<!---
Nico070707/Nico070707 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
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
print("\n\nSummary:")
