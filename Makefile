install:
	pip install -r requirements.txt

ftext:
	python main.py --dataset "lastfm" --model "HyperSaR" --num_layer 2 --edge_dropout 0.1 --loss_weight 0.001
bert:
	python main.py --dataset "lastfm" --model "HyperSaR" --num_layer 2 --edge_dropout 0.1 --loss_weight 0.001 --bert
        
help:
	@echo "############### Options: ###############"
	@echo "install: install requirements"
	@echo "ftext: run without BERT embeddings"
	@echo "bert: run with BERT embeddings"
	@echo "########################################"
