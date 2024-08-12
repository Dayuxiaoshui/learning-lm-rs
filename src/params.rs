use core::slice;
use std::alloc::LayoutErr;

use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::{SafeTensors, View};
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    // 从 SafeTensors 创建 LLamaParams 的方法
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // 获取配置中隐藏层的数量
        let layers = config.num_hidden_layers;

        // 打印所有的张量名称，方便调试
        for name in safetensor.names() {
            println!("{}", name);
        }

        // 辅助函数：从 safetensor 获取张量并转换为 f32 类型
        let get_tensor = |name: &str| {
            // 使用 match 对结果进行匹配处理
            match safetensor.tensor(name) {
                Ok(data) => {
                    // 计算张量元素的数量
                    let num_elements: usize = data.shape().iter().product();
                    // 使用 unsafe 将数据转换为 f32 的切片
                    let new_data = unsafe { 
                        slice::from_raw_parts(data.data().as_ptr() as *const f32, num_elements) 
                    };
                    // 返回新的 Tensor 对象
                    Tensor::new(Vec::from(new_data), &data.shape().to_vec())
                },
                Err(err) => {
                    
                    println!("警告：未找到张量 {}: {}", name, err);
                    Tensor::default(&Vec::new())
                }
            }
        };

        // 辅助函数：获取每一层的张量
        let get_layer_tensors = |prefix: &str| {
            (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.{}.weight", prefix)))
                .collect()
        };

        // 构建 LLamaParams 对象
        Self {
            embedding_table: get_tensor("lm_head.weight"), // 嵌入表
            rms_att_w: get_layer_tensors("input_layernorm"), // RMS attention 权重
            wq: get_layer_tensors("self_attn.q_proj"), // Query 投影权重
            wk: get_layer_tensors("self_attn.k_proj"), // Key 投影权重
            wv: get_layer_tensors("self_attn.v_proj"), // Value 投影权重
            wo: get_layer_tensors("self_attn.o_proj"), // Output 投影权重
            rms_ffn_w: get_layer_tensors("post_attention_layernorm"), // FFN 权重
            w_up: get_layer_tensors("mlp.up_proj"), // MLP 上升投影权重
            w_gate: get_layer_tensors("mlp.gate_proj"), // MLP 门控投影权重
            w_down: get_layer_tensors("mlp.down_proj"), // MLP 下降投影权重
            rms_out_w: get_tensor("model.norm.weight"), // 输出层归一化权重
            lm_head: get_tensor("lm_head.weight"), // 语言模型头权重
        }
    }
}
