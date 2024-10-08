mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;
use chat::Chat;
use kvcache::KVCache;
use std::io::Write;
use std::io::{self, BufRead, BufReader};
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;
mod chat;

use std::str::FromStr;

// 加载模型和分词器
fn load_model_and_tokenizer() -> (model::Llama<f32>, Tokenizer) {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");

    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();

    (llama, tokenizer)
}

// 处理用户输入命令
fn handle_input(reader: &mut BufReader<io::Stdin>) -> Option<String> {
    let mut input = String::new();
    if reader.read_line(&mut input).is_err() {
        eprintln!("读取输入时出错");
        return None;
    }
    Some(input.trim().to_string())
}

// 创建新的对话
fn create_new_chat(chat_vec: &mut Vec<Chat>, llama: &model::Llama<f32>, id: &mut usize) -> usize {
    *id += 1;
    let new_chat = Chat::new_chat(*id, llama);
   
    println!("创建新的对话, chat-{}", *id);
    *id
}

// 切换对话
fn change_chat_id(chat_vec: &Vec<Chat>, input: &str) -> Option<usize> {
    let id_part = input.split_whitespace().nth(1);
    if let Some(id_str) = id_part {
        if let Ok(c_id) = usize::from_str(id_str) {
            if c_id < chat_vec.len() {
                println!("切换到 chat-{}", c_id);
                return Some(c_id);
            } else {
                println!("错误: chat-id 不存在");
            }
        } else {
            println!("错误: 无效的 chat-id");
        }
    }
    None
}

// 打印历史消息
fn print_history(chat: &Chat) {
    let history = chat.get_message();
    println!("{}", history);
}

// 生成新的回复
fn generate_reply(
    chat: &mut Chat,
    tokenizer: &Tokenizer,
    llama: &model::Llama<f32>,
    input: &str,
    im_start: &str,
    im_end: &str,
    regenerate: bool,
) -> String {
    let mut tmp_ids: Vec<u32> = chat.get_ids();
    let mut re_input_ids: Vec<u32> = chat.get_re_ids();
    let mut message_input;
    let message;

    let tmp_cache;
    if regenerate {
        message_input = chat.get_his_message().to_string();
        message = chat.get_his_message().to_string();
        tmp_cache = chat.get_h_cache();
    } else {
        message_input = format!("{} system-prompt \n {}{}", im_start, input, im_end);
        message = chat.get_message().to_string();
        tmp_cache = chat.get_cache();
    }

    let binding = tokenizer.encode(message_input.clone(), true).unwrap();
    let input_ids = binding.get_ids();
    tmp_ids.extend_from_slice(&input_ids);
    re_input_ids.extend_from_slice(&input_ids);

    let value = Chat::generate(llama, message.clone(), &tmp_ids, tmp_cache.clone());
    let new_cache = Arc::new(value.1);

    let mut output_ids = value.0;
    tmp_ids.extend_from_slice(&output_ids);

    let output = tokenizer.decode(&output_ids, true).unwrap().trim_start().to_string();
    let first_space_or_newline = output.find(|c: char| c.is_whitespace() || c == '\n').unwrap_or(output.len());

    let role = &output[..first_space_or_newline].trim_end().to_string();
    let out = &output[first_space_or_newline..].trim_start().to_string();

    let message_output = format!("{} {} \n {}\n{}", im_start, role, out, im_end);
    println!("{}", message_output);

    chat.set_message(message_output.clone());
    chat.set_his_message(message_input);
    chat.set_cache(new_cache);
    chat.set_input_ids(tmp_ids);
    chat.set_re_input_ids(re_input_ids);

    message_output
}

// 主函数
fn main() {
    println!();
    println!("===============================================================");
    println!("📢 Welcome to the Interactive Chat Program!");
    println!("---------------------------------------------------------------");
    println!("💬 Available Commands:");
    println!("  👉 Type 'farewell'      : Exit the program");
    println!("  👉 Type 'fresh start'   : Begin a new chat session");
    println!("  👉 Type 'switch chat <id>' : Switch to a different chat session");
    println!("  👉 Type 'retry'         : Regenerate the latest response");
    println!("  👉 Type 'recap'         : View chat history");
    println!("===============================================================");
    println!("📝 Let's begin the conversation...");
    println!("<|im_start|> system: \n This is a chat  model \n <|im_end|>");

    let (llama, tokenizer) = load_model_and_tokenizer();

    let mut chat_vec: Vec<Chat> = Vec::new();
    let mut id: usize = 0;
    let mut change_id: usize = 0;

    let stdin = io::stdin();
    let mut reader = BufReader::new(stdin);

    let im_start = "<|im_start|>";
    let im_end = "<|im_end|>";

    // 默认创建 chat_0
    let chat_0 = Chat::new_chat(id, &llama);
    chat_vec.push(chat_0);

    loop {
        println!();
        println!("===============================================================");
        println!("📝 Please enter your input below:");
        println!("---------------------------------------------------------------");

        io::stdout().flush().expect("无法刷新stdout");

        if let Some(input) = handle_input(&mut reader) {
            if input == "farewell" {
                println!("👋 Exiting the program. Goodbye!");
                println!("===============================================================");
                break;
            }

            if input == "fresh start" {
                change_id = create_new_chat(&mut chat_vec, &llama, &mut id);
                continue;
            }

            if input.starts_with("switch chat") {
                if let Some(new_id) = change_chat_id(&chat_vec, &input) {
                    change_id = new_id;
                }
                continue;
            }

            let chat_tmp = &mut chat_vec[change_id];

            if input == "recap" {
                print_history(chat_tmp);
                continue;
            }

            if input == "retry" {
                generate_reply(chat_tmp, &tokenizer, &llama, &input, im_start, im_end, true);
            } else {
                generate_reply(chat_tmp, &tokenizer, &llama, &input, im_start, im_end, false);
            }
        }
    }
}
