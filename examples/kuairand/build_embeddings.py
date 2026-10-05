import numpy as np
import torch

from data.dataset_kuairand import KuairandTwoTowerDataset
from model.two_tower_kuairand import KuaiRandTwoTowerModel
from trainer.embedding_trainer_kuairand import KuaiRandEmbeddingTrainer


NUM_TRIAL = 3
EVAL_RATIO = 0.05
NUM_EPOCHS = 1


if __name__ == "__main__":
    full_data = KuairandTwoTowerDataset.prepare_data()
    unique_user_active_degree = sorted(full_data["user_active_degree"].dropna().unique().tolist())
    invalid_user_active_degree = [value for value in unique_user_active_degree if value not in [0, 1, 2, 3]]
    print(f"user_active_degree unique values: {unique_user_active_degree}")
    print(f"user_active_degree invalid values: {invalid_user_active_degree}")

    unique_fans_user_num_range = sorted(full_data["fans_user_num_range"].dropna().unique().tolist())
    invalid_fans_user_num_range = [value for value in unique_fans_user_num_range if value not in [0, 1, 2, 3, 4, 5, 6]]
    print(f"fans_user_num_range unique values: {unique_fans_user_num_range}")
    print(f"fans_user_num_range invalid values: {invalid_fans_user_num_range}")

    eval_data = full_data.sample(frac=EVAL_RATIO, random_state=42)
    train_data = full_data.drop(index=eval_data.index)
    train_data = train_data.reset_index(drop=True)
    eval_data = eval_data.reset_index(drop=True)

    train_dataset = KuairandTwoTowerDataset(train_data)
    eval_dataset = KuairandTwoTowerDataset(eval_data)

    best_embeddings = None
    best_video_ids = None
    best_loss = None

    # Train multiple times and keep the best checkpoint by eval loss.
    for trial in range(NUM_TRIAL):
        model = KuaiRandTwoTowerModel.from_dataframe(full_data)
        trainer = KuaiRandEmbeddingTrainer(model=model, train_dataset=train_dataset, eval_dataset=eval_dataset)
        trainer.train(num_epochs=NUM_EPOCHS)
        loss = trainer.eval()

        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_video_ids, best_embeddings = trainer.export_item_embeddings(full_data)
            print(f"On trial {trial}, found best loss {loss:.4f}")

    if best_embeddings is None or best_video_ids is None:
        raise RuntimeError("Failed to train KuaiRand two-tower model and export item embeddings.")

    torch.save(best_embeddings, "artifacts/kuairand_item_embeddings.pt")
    np.save("artifacts/kuairand_item_ids.npy", best_video_ids)
    print("Saved embeddings to artifacts/kuairand_item_embeddings.pt")
    print("Saved video ids to artifacts/kuairand_item_ids.npy")
