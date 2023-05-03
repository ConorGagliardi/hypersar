install:
        pip install -r requirements.txt

word2vec:
        python main.py --dataset "lastfm" --model "HyperSaR" --num_layer 2 --edge_dropout 0.1 --loss_weight 0.001
bertembeddings:
        python main.py --dataset "lastfm" --model "HyperSaR" --num_layer 2 --edge_dropout 0.1 --loss_weight 0.001 --use_bert
        
help:
        @echo "############### Options: ###############"
        @echo "install: install requirements"
        @echo "word2vec: run without BERT embeddings"
        @echo "bertembeddings: run with BERT embeddings"
        @echo "########################################"