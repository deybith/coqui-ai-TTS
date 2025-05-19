from xtts.train_model import train_model

lang='es'
train_csv='/home/ubuntu/projects/coqui-ai-Trainer/data/test/dataset/metadata_train.csv'
eval_csv='/home/ubuntu/projects/coqui-ai-Trainer/data/test/dataset/metadata_eval.csv'
num_epochs=10
batch_size=16
grad_acumm=1
out_path='/home/ubuntu/projects/coqui-ai-Trainer/data/test'
max_audio_length=30

def main():
    train_model(
        lang,
        train_csv,
        eval_csv,
        num_epochs,
        batch_size,
        grad_acumm,
        out_path,
        max_audio_length
      )
                

if __name__ == "__main__":
    main()