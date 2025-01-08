use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;
use dotenv::dotenv;

#[derive(Serialize, Deserialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize, Deserialize)]
struct OpenAIResponse {
    choices: Vec<Choice>,
}

#[derive(Serialize, Deserialize)]
struct Choice {
    message: Message,
}

#[tokio::main]
async fn main() {
    dotenv().ok();

    // Load the OpenAI API key from the .env file
    let api_key = env::var("OPENAI_API_KEY").expect("API key not found in .env file");

    // Prompt for summarization
    let prompt = r#"Reply with location information:
(location): {location}"#;

    // Few-shot examples
    let examples = vec![
        ("location1", "Romania"),
        ("location2", "United Kingdom"),
        ("location3", "United States"),
    ];

    // Location to query
    let location = "what's location 1 and location 2? Also, what's location 4?";

    // Prepare messages
    let mut messages = Vec::new();
    messages.push(Message {
        role: "system".to_string(),
        content: "You are a snarky assistant.".to_string(),
    });

    for (input, output) in &examples {
        messages.push(Message {
            role: "user".to_string(),
            content: prompt.replace("{location}", *input),
        });
        messages.push(Message {
            role: "assistant".to_string(),
            content: output.to_string(),
        });
    }

    messages.push(Message {
        role: "user".to_string(),
        content: prompt.replace("{location}", location),
    });

    // Make the API request
    let client = Client::new();
    let response = client
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&serde_json::json!({
            "model": "gpt-3.5-turbo",
            "messages": messages,
        }))
        .send()
        .await;

    match response {
        Ok(res) => {
            let res_body: OpenAIResponse = res.json().await.unwrap();
            println!("{}", res_body.choices[0].message.content);
        }
        Err(err) => {
            eprintln!("Error making request: {}", err);
        }
    }
}
