use candle_mnist_demo::utils::gz_decoder;
use tracing::info;

fn main() {
    tracing_subscriber::fmt::init();
    info!("Hello, world!");

    let dataset_dir = "datasets/mnist";

    gz_decoder::decompress_dataset(dataset_dir);
}
