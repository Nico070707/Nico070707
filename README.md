- 👋 Hi, I’m @Nico070707
- 👀 I’m interested in real Data Humanity to build a better world. A real dream could begin a real life
- 🌱 I’m currently learning DataScience
- 💞️ I’m looking to collaborate on Project for humanity with code
- 📫 How to reach me https://www.linkedin.com/in/nicolas-bourne-635a0033/

<!---
Nico070707/Nico070707 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
encoded input + tokenizer(sentences, padding=True, truncation=True, max_lengh=128, return_tensors='pt')
with torch.no_grad{) :
model_ouput = model(**encoded_input
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
util.pytorch_cos_sim(sentence_embeddings[1], sentence_embeddings[0],numpy()[0][0]

