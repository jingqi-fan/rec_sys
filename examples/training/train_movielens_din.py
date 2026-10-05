from configs.model import DIN_SMALL_CONFIG, OUTPUT_MODEL_PATH
from data.dataset_movie_len import prepare_movie_len_dataset
from model.din import DeepInterestModel
from trainer.simple_trainer import SimpleTrainer


if __name__ == "__main__":
    train_dataset, eval_dataset, movie_index = prepare_movie_len_dataset(history_seq_length=8, reindex=False)
    model = DeepInterestModel(config=DIN_SMALL_CONFIG)

    trainer = SimpleTrainer(model=model, train_dataset=train_dataset, eval_dataset=eval_dataset)
    trainer.train(num_epochs=5)

    trainer.save(model_name="din-movie-len-small", movie_index=movie_index, path=OUTPUT_MODEL_PATH)
