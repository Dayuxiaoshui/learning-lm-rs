use std::fs::File;
use std::vec;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators::{self as OP, masked_softmax, matmul_transb, random_sample, rms_norm, silu};
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::path::Path;

pub struct Llama<T> {
    vocab: usize,           // vocab size
    n_layers: usize,        // number of layers
    n_q_h: usize,           // number of heads for q
    n_kv_h: usize,          // number of heads for k and v
    d: usize,               // dimension of hidden states
    dqkv: usize,            // length of a single q, k, or v vector
    di: usize,              // dimension of intermediate states
    eps: f32,               // epsilon for RMS normalization
    rope_theta: f32,        // rope theta for rope initialization
    max_seq_len: usize,     // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    bos_token_id: u32,      // start token id
    eos_token_id: u32,      // end token id
}

impl Llama<f32> {
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).expect("Failed to open config.json");
        let config: LlamaConfigJson = serde_json::from_reader(config).expect("Failed to parse config.json");
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).expect("Failed to read model.safetensors");
        let safetensor = SafeTensors::deserialize(&model_file).expect("Failed to deserialize safetensors");

        // 使用 unwrap 或者 expect 解包结果
        let params = LLamaParams::from_safetensors(&safetensor, &config).expect("Failed to load model parameters");

        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params: params,  // 解包后的 params
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    // 创建新的缓存实例
    pub fn new_cache(&self) -> KVCache<f32> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    // 模型前向传播，使用缓存
    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;

        // 预分配缓存，减少内存分配开销
        let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<f32>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores = Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        let mut x = Tensor::<f32>::default(&vec![seq_len, self.d]);

        // 开始进行模型计算
        // Embedding 查找
        OP::gather(&mut residual, input, &self.params.embedding_table);

        for layer in 0..self.n_layers {
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
            OP::matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
            OP::matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0);
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );

            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)

            self_attention(
                &mut x,
                &mut att_scores,
                q,
                &full_k,
                &full_v,
                self.n_kv_h,
                n_groups,
                seq_len,
                total_seq_len,
                self.dqkv,
            );

            // x = x @ O_weight.T
            OP::matmul_transb(&mut hidden_states, 0., &x, &self.params.wo[layer], 1.0);

            // residual = x + residual
            let len = residual.size();
            assert!(len == hidden_states.size());
            let _r = unsafe { residual.data_mut() };
            let _h = hidden_states.data();
            for i in 0..len {
                _r[i] += _h[i];
            }

            mlp(
                &mut residual,
                &mut hidden_states,
                &mut gate_buf,
                &mut up_buf,
                &self.params.w_up[layer],
                &self.params.w_down[layer],
                &self.params.w_gate[layer],
                &self.params.rms_ffn_w[layer],
                self.eps,
            );
        }

        // 无论 seq_len 多大，输出始终是长度为 vocab 的 1D 向量
        let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);

        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        OP::matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0);

        logits
    }

    // 模型生成
    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32> {
        let mut result = Vec::<u32>::new();
        let mut cache = self.new_cache();
        let mut token: Vec<u32> = Vec::from(token_ids);
        if token[0] != self.bos_token_id {
            token.insert(0, self.bos_token_id);
        }
        let mut input = Tensor::<u32>::new(token, &vec![1, token_ids.len()]);
        loop {
            let output =
                random_sample(&self.forward(&input, &mut cache), top_p, top_k, temperature);
            result.push(output);
            if result.len() >= max_len || output == self.eos_token_id {
                break;
            }
            input = Tensor::<u32>::new(Vec::from([output]), &vec![1, 1]);
        }

        result
    }
}

fn self_attention(
    hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    let _a = unsafe { att_scores.data_mut() };
    let _q = q.data();
    let _k = k.data();
    let _v = v.data();
    let sqrt = (dqkv as f32).sqrt();

    for h in 0..n_kv_h * n_groups {
        for l in 0..seq_len {
            for i in 0..total_seq_len {
                let sum = (0..dqkv)
                    .map(|j| {
                        _q[l * n_kv_h * n_groups * dqkv + h * dqkv + j]
                            * _k[i * n_kv_h * dqkv + h / n_groups * dqkv + j]
                    })
                    .sum::<f32>();
                _a[h * seq_len * total_seq_len + l * total_seq_len + i] = sum / sqrt;
            }
        }
    }

    masked_softmax(att_scores);

    let _a = att_scores.data();
    let _h = unsafe { hidden_states.data_mut() };
    for h in 0..n_kv_h * n_groups {
        for l in 0..seq_len {
            for i in 0..dqkv {
                let sum = (0..total_seq_len)
                    .map(|j| {
                        _a[h * seq_len * total_seq_len + l * total_seq_len + j]
                            * _v[i + h / n_groups * dqkv + j * n_kv_h * dqkv]
                    })
                    .sum::<f32>();
                _h[l * n_kv_h * n_groups * dqkv + h * dqkv + i] = sum;
            }
        }
    }
}

fn mlp(
    residual: &mut Tensor<f32>,      
    hidden_states: &mut Tensor<f32>, 
    gate: &mut Tensor<f32>,          
    up: &mut Tensor<f32>,            
    w_up: &Tensor<f32>,              
    w_down: &Tensor<f32>,            
    w_gate: &Tensor<f32>,            
    rms_w: &Tensor<f32>,             
    eps: f32,
) {
    rms_norm(hidden_states, residual, rms_w, eps);

    matmul_transb(gate, 0.0, hidden_states, w_gate, 1.0);

    matmul_transb(up, 0.0, hidden_states, w_up, 1.0);

    silu(up, &gate);

    matmul_transb(residual, 1.0, &up, w_down, 1.0);
}
