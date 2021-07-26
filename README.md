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
