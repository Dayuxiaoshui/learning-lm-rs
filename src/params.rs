use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
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
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        let get_tensor = |name: &str| -> Tensor<f32> {
            let tensor = safetensor.tensor(name).unwrap();
            let data = tensor.data();
            let size_in_bytes = tensor.dtype().size();
            let elem_count = data.len() / size_in_bytes;
            // SAFETY This is safe because we just checked that this
            // was correctly aligned.
            let data: &[f32] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, elem_count) };
            Tensor::new(data.to_vec(), &tensor.shape().to_vec())
        };
        let head_size = config.hidden_size / config.num_attention_heads;
        let embedding_table = if config.tie_word_embeddings {
            let embedding_table = get_tensor("lm_head.weight");
            assert_eq!(
                embedding_table.shape(),
                &[config.vocab_size, config.hidden_size]
            );
            embedding_table
        } else {
            let embedding_table = get_tensor("model.embed_tokens.weight");

            assert_eq!(
                embedding_table.shape(),
                &[config.vocab_size, config.hidden_size]
            );
            embedding_table
        };
        let mut rms_att_w = Vec::new();
        let mut wq = Vec::new();
        let mut wk = Vec::new();
        let mut wv = Vec::new();
        let mut wo = Vec::new();

        let mut rms_ffn_w = Vec::new();
        let mut w_up = Vec::new();
        let mut w_gate = Vec::new();
        let mut w_down = Vec::new();
        for layer in 0..config.num_hidden_layers {
            let rms_att_w_item_name = format!("model.layers.{layer}.input_layernorm.weight");
            let rms_att_w_item = get_tensor(rms_att_w_item_name.as_str());
            assert_eq!(
                rms_att_w_item.shape(),
                &[config.hidden_size],
                "{}",
                format!("{rms_att_w_item_name} shape is not correct")
            );
            rms_att_w.push(rms_att_w_item);
            let wq_item_name = format!("model.layers.{layer}.self_attn.q_proj.weight");
            let wq_item = get_tensor(wq_item_name.as_str());
            assert_eq!(
                wq_item.shape(),
                &[head_size * config.num_attention_heads, config.hidden_size],
                "{}",
                format!("{wq_item_name} shape is not correct")
            );
            wq.push(wq_item);
            let wk_item_name = format!("model.layers.{layer}.self_attn.k_proj.weight");
            let wk_item = get_tensor(wk_item_name.as_str());
            assert_eq!(
                wk_item.shape(),
                &[config.num_key_value_heads * head_size, config.hidden_size],
                "{}",
                format!("{wk_item_name} shape is not correct")
            );
            wk.push(wk_item);
            let wv_item_name = format!("model.layers.{layer}.self_attn.v_proj.weight");
            let wv_item = get_tensor(wv_item_name.as_str());
            assert_eq!(
                wv_item.shape(),
                &[config.num_key_value_heads * head_size, config.hidden_size],
                "{}",
                format!("{wv_item_name} shape is not correct")
            );
            wv.push(wv_item);
            let wo_item_name = format!("model.layers.{layer}.self_attn.o_proj.weight");
            let wo_item = get_tensor(wo_item_name.as_str());
            assert_eq!(
                wo_item.shape(),
                &[config.hidden_size, config.hidden_size,],
                "{}",
                format!("{wo_item_name} shape is not correct")
            );
            wo.push(wo_item);
            let rms_ffn_w_item_name =
                format!("model.layers.{layer}.post_attention_layernorm.weight");
            let rms_ffn_w_item = get_tensor(rms_ffn_w_item_name.as_str());
            assert_eq!(
                rms_ffn_w_item.shape(),
                &[config.hidden_size],
                "{}",
                format!("{rms_ffn_w_item_name} shape is not correct")
            );
            rms_ffn_w.push(rms_ffn_w_item);
            let w_up_item_name = format!("model.layers.{layer}.mlp.up_proj.weight");
            let w_up_item = get_tensor(w_up_item_name.as_str());
            assert_eq!(
                w_up_item.shape(),
                &[config.intermediate_size, config.hidden_size],
                "{}",
                format!("{w_up_item_name} shape is not correct")
            );
            w_up.push(w_up_item);
            let w_gate_item_name = format!("model.layers.{layer}.mlp.gate_proj.weight");
            let w_gate_item = get_tensor(w_gate_item_name.as_str());
            assert_eq!(
                w_gate_item.shape(),
                &[config.intermediate_size, config.hidden_size],
                "{}",
                format!("{w_gate_item_name} shape is not correct")
            );
            w_gate.push(w_gate_item);
            let w_down_item_name = format!("model.layers.{layer}.mlp.down_proj.weight");
            let w_down_item = get_tensor(w_down_item_name.as_str());
            assert_eq!(
                w_down_item.shape(),
                &[config.hidden_size, config.intermediate_size],
                "{}",
                format!("{w_down_item_name} shape is not correct")
            );
            w_down.push(w_down_item);
        }
        let rms_out_w = get_tensor("model.norm.weight");
        assert_eq!(rms_out_w.shape(), &[config.hidden_size]);
        let lm_head = get_tensor("lm_head.weight");
        assert_eq!(lm_head.shape(), &[config.vocab_size, config.hidden_size]);

        LLamaParams {
            embedding_table,
            rms_att_w,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_w,
            w_up,
            w_gate,
            w_down,
            rms_out_w,
            lm_head,
        }
    }
}