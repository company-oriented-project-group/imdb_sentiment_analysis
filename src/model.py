from models.word2vec import Word2Vec

w2v_model = Word2Vec(data_path='./IMDB-example-dataset.csv')
# w2v_model.load_data()
# w2v_model.explore_data()
# w2v_model.set_seaborn_style()
# w2v_model.preprocess_data()
# w2v_model.split_data()
# w2v_model.visualize_word_clouds()
# w2v_model.build_word2vec_model()
# w2v_model.save_word2vec_model("./models/word2vec.model")
w2v_model.test_word2vec_model("./models/_word2vec.model")
