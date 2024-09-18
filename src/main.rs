mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use std::path::PathBuf;
use tokenizers::Tokenizer;

fn main() {
    // 获取项目目录
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    // 加载LLaMA模型
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    // 加载分词器
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    // 输入文本
    let input = "Once upon a time";
    // 对输入文本进行编码
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();
    // 打印原始输入
    print!("\n{}", input);
    // 调用模型生成
    let output_ids = llama.generate(input_ids, 500, 0.9, 4, 1.0);
    // 对生成的token
    println!("{}", tokenizer.decode(&output_ids, true).unwrap());
}
