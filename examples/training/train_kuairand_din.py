from configs.model import OUTPUT_MODEL_PATH
from data.dataset_kuairand import KuairandDINDataset, prepare_kuairand_din_dataset
from model.din_kuairand import KuaiRandDINModel
from trainer.din_kuairand_trainer import KuaiRandDINTrainer


def infer_tag_cardinality(data) -> int:
    max_candidate_tag = max(max(tags) for tags in data["candidate_tag_ids"] if len(tags) > 0)
    max_history_tag = max(max(max(tags) for tags in seq if len(tags) > 0) for seq in data["history_tag_ids"])
    return max(max_candidate_tag, max_history_tag) + 1


if __name__ == "__main__":
    history_seq_length = 20
    train_dataset, eval_dataset = prepare_kuairand_din_dataset(
        history_seq_length=history_seq_length,
        eval_ratio=0.1,
    )

    full_data = KuairandDINDataset.prepare_data(history_seq_length=history_seq_length)
    model = KuaiRandDINModel(
        user_cardinality=int(full_data["user_id"].max()) + 1,
        video_cardinality=int(max(full_data["candidate_video_id"].max(), max(max(seq) for seq in full_data["history_video_ids"]))) + 1,
        author_cardinality=int(max(full_data["candidate_author_id"].max(), max(max(seq) for seq in full_data["history_author_ids"]))) + 1,
        tag_cardinality=infer_tag_cardinality(full_data),
    )

    trainer = KuaiRandDINTrainer(model=model, train_dataset=train_dataset, eval_dataset=eval_dataset)
    trainer.train(num_epochs=5)
    trainer.save(model_name="din-kuairand-1k", path=OUTPUT_MODEL_PATH)
