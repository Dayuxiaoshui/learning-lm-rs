use core::slice;
use std::error::Error;

use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::{SafeTensors, View};

pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>,  // (vocab_size, dim)
    // decoder layers
    pub rms_att_w: Vec<Tensor<T>>,   // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,          // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,          // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,          // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,          // (hidden_size, n_heads * head_size) x layers
    // feedforward layers (FFN)
    pub rms_ffn_w: Vec<Tensor<T>>,   // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,        // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,      // (hidden_size, intermediate_size) x layers
    // output layers
    pub rms_out_w: Tensor<T>,        // (hidden_size, )
    pub lm_head: Tensor<T>,          // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Result<Self, Box<dyn Error>> {
        let layers = config.num_hidden_layers;

        // 输出 tensor 的名称，便于调试
        safetensor.names().iter().for_each(|name| {
            println!("{}", name);
        });

        // 提供一个内部通用的函数来获取 tensor，错误时返回默认值，并记录日志
        let get_tensor = |name: &str| -> Result<Tensor<f32>, Box<dyn Error>> {
            match safetensor.tensor(name) {
                Ok(data) => {
                    let p: usize = data.shape().iter().product();
                    // 将原始数据转换为 f32 类型的切片
                    let new_data = unsafe { slice::from_raw_parts(data.data().as_ptr() as *const f32, p) };
                    // 创建新的 Tensor 对象
                    Ok(Tensor::new(Vec::from(new_data), &data.shape().to_vec()))
                }
                Err(_) => {
                    eprintln!("Error reading tensor: {}", name);
                    Ok(Tensor::default(&Vec::new()))
                }
            }
        };

        // 生成模型参数
        Ok(Self {
            embedding_table: get_tensor("lm_head.weight")?,
            rms_att_w: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.input_layernorm.weight")).unwrap())
                .collect(),
            wq: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.self_attn.q_proj.weight")).unwrap())
                .collect(),
            wk: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.self_attn.k_proj.weight")).unwrap())
                .collect(),
            wv: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.self_attn.v_proj.weight")).unwrap())
                .collect(),
            wo: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.self_attn.o_proj.weight")).unwrap())
                .collect(),
            rms_ffn_w: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.post_attention_layernorm.weight")).unwrap())
                .collect(),
            w_up: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.mlp.up_proj.weight")).unwrap())
                .collect(),
            w_gate: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.mlp.gate_proj.weight")).unwrap())
                .collect(),
            w_down: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.mlp.down_proj.weight")).unwrap())
                .collect(),
            rms_out_w: get_tensor("model.norm.weight")?,
            lm_head: get_tensor("lm_head.weight")?,
        })
    }
}
