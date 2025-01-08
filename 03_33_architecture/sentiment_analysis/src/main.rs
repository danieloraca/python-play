use rust_bert::pipelines::sentiment::{Sentiment, SentimentModel};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Load the sentiment model from Hugging Face's 'nlptown/bert-base-multilingual-uncased-sentiment'
    let model = SentimentModel::new(Default::default())?;

    // Text to classify
    let text = "this is pretty interesting";

    // Classify the text
    let result = model.predict(&[text]);

    // Print the result
    println!("{:?}", result);

    Ok(())
}
