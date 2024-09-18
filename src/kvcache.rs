use std::{usize, vec};
use crate::tensor::Tensor;

pub struct KVCache<T> {
    k_cache: Vec<Tensor<T>>, // (max_seq_len, n_kv_head * dqkv) x layers
    v_cache: Vec<Tensor<T>>, // (max_seq_len, n_kv_head * dqkv) x layers
    #[allow(unused)]
    max_seq_len: usize,
    dim: usize,
    length: usize, // length of the current sequence
}

impl<T: Default + Copy> KVCache<T> {
    // 预分配足够的缓存空间，避免频繁内存分配
    pub fn new(n_layers: usize, max_seq_len: usize, dim: usize, init_len: usize) -> Self {
        KVCache {
            k_cache: (0..n_layers)
                .map(|_| Tensor::default(&vec![max_seq_len, dim]))  // 预分配缓存
                .collect(),
            v_cache: (0..n_layers)
                .map(|_| Tensor::default(&vec![max_seq_len, dim]))  // 预分配缓存
                .collect(),
            max_seq_len,
            dim,
            length: init_len,
        }
    }

    // 只返回从 start 开始的部分，避免重复计算
    pub fn k_cache(&mut self, layer: usize, start: usize) -> Tensor<T> {
        self.k_cache[layer].slice(start * self.dim, &vec![self.length - start, self.dim])
    }

    pub fn v_cache(&mut self, layer: usize, start: usize) -> Tensor<T> {
        self.v_cache[layer].slice(start * self.dim, &vec![self.length - start, self.dim])
    }

    // 增加缓存长度，记录新加入序列的长度
    pub fn increment(&mut self, seq_len: usize) {
        self.length += seq_len;
    }

    // 返回当前缓存的长度
    pub fn len(&self) -> usize {
        self.length
    }

    // 更新缓存的接口，直接替换某一层的 Key 和 Value
    pub fn update_cache(&mut self, layer: usize, k_new: Tensor<T>, v_new: Tensor<T>) {
        self.k_cache[layer] = k_new;
        self.v_cache[layer] = v_new;
    }
}
