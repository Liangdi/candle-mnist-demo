use candle_mnist_demo::{utils::gz_decoder, mnist::{TrainingArgs, LinearModel, training_loop}};
use tracing::info;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let dataset_dir = "datasets/mnist";
    gz_decoder::decompress_dataset(dataset_dir);

    let dataset = candle_datasets::vision::mnist::load_dir(dataset_dir)?;

    info!("train-images: {:?}", dataset.train_images.shape());
    info!("train-labels: {:?}", dataset.train_labels.shape());
    info!("test-images: {:?}", dataset.test_images.shape());
    info!("test-labels: {:?}", dataset.test_labels.shape());


    let training_args = TrainingArgs {
        epochs: 200,
        learning_rate: 1.0,
        load: None,
        save: Some("models/mnist/model.safetensors".to_string()),
    };

    training_loop::<LinearModel>(dataset, &training_args)?;

    Ok(())
}
